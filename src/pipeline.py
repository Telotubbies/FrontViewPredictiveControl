"""
LKA Pipeline — หลักเหตุผล: Perception → Fusion → Reference Path → MPC → Safety
Input: RGB, speed, waypoint state. Output: control + FrameState สำหรับ view
"""
import logging
import time
from typing import Optional, Tuple, List

import numpy as np
import torch

from config import (
    CAM_W,
    CAM_H,
    CTE_TO_METERS,
    LANE_EMA_ALPHA,
    REF_PATH_LOOKAHEAD_M,
    REF_PATH_NUM_PTS,
    SAFETY_MAX_STEER_RAD,
    STEER_SMOOTH_ALPHA,
    USE_TRAJECTORY_PIPELINE,
    USE_CLASSICAL_DETECTOR,
)
from algorithms.fusion import apply_fusion
from state import FrameState
from temporal.lane_lstm import LaneTemporalSmoother
from control.lane_mpc import LaneMPC, MPCConfig
from safety.override import SafetyOverride
from perception.road_perception import BEVRoadPerception

logger = logging.getLogger(__name__)


def get_reference_path(
    cte_m: float,
    head_rad: float,
    curv: float,
    lookahead_m: float = REF_PATH_LOOKAHEAD_M,
    num_pts: int = REF_PATH_NUM_PTS,
) -> List[Tuple[float, float]]:
    """(s_m, lateral_m) path in vehicle frame from current state."""
    return [
        (
            lookahead_m * i / num_pts,
            cte_m
            + head_rad * (lookahead_m * i / num_pts)
            + 0.5 * curv * (lookahead_m * i / num_pts) ** 2,
        )
        for i in range(num_pts + 1)
    ]


class LKAPipeline:
    """
    หนึ่ง step: RGB + speed + waypoint state -> steer, throttle, brake, FrameState.
    ไม่รวม Stuck Recovery (ทำใน main).
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        target_speed_kmh: float,
        use_trajectory_pipeline: Optional[bool] = None,
    ) -> None:
        if use_trajectory_pipeline is None:
            use_trajectory_pipeline = USE_TRAJECTORY_PIPELINE

        self._use_trajectory = use_trajectory_pipeline
        self._target_speed_ms = target_speed_kmh / 3.6

        # Only load UNet if not using classical detector
        if not USE_CLASSICAL_DETECTOR:
            # Use BEV-based perception (more robust for dashed lines)
            self._legacy_perception = BEVRoadPerception(model_path, device)
        else:
            self._legacy_perception = None
            logger.info("Skipping UNet load - using Classical Detector")

        if use_trajectory_pipeline:
            from perception.lane_trajectory import LaneTrajectoryPipeline
            self._trajectory_pipeline = LaneTrajectoryPipeline(
                detector=self._legacy_perception.detector if (self._legacy_perception and not USE_CLASSICAL_DETECTOR) else None,
                use_classical_detector=USE_CLASSICAL_DETECTOR,
                cam_w=CAM_W,
                cam_h=CAM_H,
                v_nominal_ms=self._target_speed_ms,
                bev_top_ratio=0.28,  # ต่ำลง = ใช้แถวบนภาพมากขึ้น → เลนยาวขึ้น
                bev_bot_ratio=0.98,
                bev_top_margin=0.32,
                bev_bot_margin=0.10,
            )
            if USE_CLASSICAL_DETECTOR:
                logger.info("LKAPipeline: using LaneTrajectoryPipeline with CARLA Waypoint Detection (ground truth, no AI model)")
            else:
                logger.info("LKAPipeline: using LaneTrajectoryPipeline with UNet (BEV+Kalman)")
        else:
            self._trajectory_pipeline = None
            logger.info("LKAPipeline: using legacy RoadPerception")

        self._smoother = LaneTemporalSmoother(ema_alpha=LANE_EMA_ALPHA)
        self._mpc_cfg = MPCConfig(
            N=10,
            dt=0.1,
            w_cte=200.0,
            w_heading=85.0,
            w_vel=5.0,
            w_steer=80.0,
            w_steer_rate=350.0,
            max_steer_rate=0.20,
        )
        self._mpc = LaneMPC(self._mpc_cfg)
        self._safety = SafetyOverride({
            "max_speed_kmh": target_speed_kmh + 5.0,
            "max_steering_angle": SAFETY_MAX_STEER_RAD,
            "emergency_brake_enabled": True,
        })

    def reset(self) -> None:
        if self._legacy_perception is not None:
            self._legacy_perception.reset()
        if self._trajectory_pipeline is not None:
            self._trajectory_pipeline.reset()
        self._smoother.reset()

    def process(self, rgb: np.ndarray) -> Tuple[float, float, float, FrameState]:
        """Process method for compatibility - delegates to step with default values."""
        return self.step(
            rgb=rgb,
            speed_ms=0.0,  # Default speed
            wp_state=None,  # Default waypoint state
            prev_steer=0.0,  # Default previous steer
            prev_throttle=0.0,  # Default previous throttle
        )

    def step(
        self,
        rgb: np.ndarray,
        speed_ms: float,
        wp_state: Optional[Tuple[float, float, float]],
        prev_steer: float,
        prev_throttle: float,
        world=None,
        vehicle=None,
    ) -> Tuple[float, float, float, FrameState]:
        """Returns: (steer, throttle, brake, frame_state)."""
        if self._trajectory_pipeline is not None:
            out = self._trajectory_pipeline.process(rgb, world=world, vehicle=vehicle)
            cte_m_lane = float(out.cte)
            head_s = float(out.heading_err)
            curv_s = float(out.curvature)
            lane_conf = float(out.confidence)
            geometry_valid = bool(out.geometry_valid)
            v_ref_traj = float(out.v_ref_at_ego)
            lane_overlay = out.lane_overlay
            bev_window_vis = out.bev_window_vis
            mask_vis = out.mask_vis
            # Peter Moran viz data
            bev_binary = getattr(out, 'bev_binary', None)
            left_px_img = getattr(out, 'left_px_img', None)
            right_px_img = getattr(out, 'right_px_img', None)
            left_curvature_m = getattr(out, 'left_curvature_m', 9999.0)
            right_curvature_m = getattr(out, 'right_curvature_m', 9999.0)
            tracked = getattr(out, 'tracked', False)
            used_completion = getattr(out, 'used_completion', False)
        else:
            cte_raw, heading, curvature, mask_vis, lane_conf, _ = (
                self._legacy_perception.process(rgb)
            )
            sm = self._smoother.update(cte_raw, heading, curvature)
            cte_m_lane = float(sm[0]) * CTE_TO_METERS
            head_s = float(sm[1])
            curv_s = float(sm[2])
            geometry_valid = False
            v_ref_traj = None
            lane_overlay = None
            bev_window_vis = None
            bev_binary = None
            left_px_img = None
            right_px_img = None
            left_curvature_m = 9999.0
            right_curvature_m = 9999.0
            tracked = False
            used_completion = False

        # WP เป็นหลัก; UNet ช่วยจัดกลาง + เช็คเลน (apply_fusion); โค้ง+geometry invalid → WP only
        cte_m, head_s, curv_s, mode = apply_fusion(
            wp_state, cte_m_lane, head_s, curv_s, lane_conf,
            geometry_valid=geometry_valid,
        )

        reference_path = get_reference_path(cte_m, head_s, curv_s)

        if v_ref_traj is not None:
            vt = v_ref_traj
        elif abs(curv_s) > 0.02:
            vt = self._target_speed_ms * 0.5
        elif abs(curv_s) > 0.01:
            vt = self._target_speed_ms * 0.7
        else:
            vt = self._target_speed_ms

        mpc_t0 = time.time()
        steer_rad, accel, solver_status, mpc_trajectory = self._mpc.solve(
            x0=0, y0=cte_m, psi0=head_s, v0=speed_ms,
            v_ref=vt, cte=cte_m, heading_err=head_s, curvature=curv_s,
            confidence=float(np.clip(lane_conf, 0.0, 1.0)),
        )
        mpc_solve_time_ms = (time.time() - mpc_t0) * 1000.0
        s_raw = self._mpc.steer_to_carla(steer_rad)
        steer = STEER_SMOOTH_ALPHA * prev_steer + (1 - STEER_SMOOTH_ALPHA) * s_raw
        throttle, brake = self._mpc.accel_to_carla(accel)

        if speed_ms < 0.6 and brake < 0.1:
            throttle = max(throttle, 0.55)
            brake = 0.0

        steer, throttle, brake = self._safety.apply_safety_override(
            {"velocity": speed_ms}, steer, throttle, brake
        )

        frame_state = FrameState(
            rgb=rgb,
            speed_kmh=speed_ms * 3.6,
            steer=steer,
            throttle=throttle,
            brake=brake,
            cte_m=cte_m,
            heading_rad=head_s,
            curvature=curv_s,
            mode=mode,
            lane_conf=lane_conf,
            reference_path=reference_path,
            geometry_valid=geometry_valid,
            lane_overlay=lane_overlay,
            mask=mask_vis,
            bev_window_vis=bev_window_vis,
            left_px_img=left_px_img,
            right_px_img=right_px_img,
            tracked=tracked,
            used_completion=used_completion,
            # Peter Moran visualization
            bev_binary=bev_binary,
            raw_windows_left=getattr(self._trajectory_pipeline, '_raw_left_wins', None) if self._trajectory_pipeline else None,
            raw_windows_right=getattr(self._trajectory_pipeline, '_raw_right_wins', None) if self._trajectory_pipeline else None,
            filt_windows_left=getattr(self._trajectory_pipeline, '_filt_left_wins', None) if self._trajectory_pipeline else None,
            filt_windows_right=getattr(self._trajectory_pipeline, '_filt_right_wins', None) if self._trajectory_pipeline else None,
            left_px_bev=left_px_img,
            right_px_bev=right_px_img,
            left_curvature_m=left_curvature_m,
            right_curvature_m=right_curvature_m,
            solver_status=solver_status,
            mpc_trajectory=mpc_trajectory,
        )
        frame_state.mpc_solve_time_ms = mpc_solve_time_ms
        return steer, throttle, brake, frame_state

