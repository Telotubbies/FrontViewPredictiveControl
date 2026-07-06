"""
Single LKA algorithm step: Perception → Fusion → Reference → MPC → Safety.
Pure algorithm layer; no CARLA I/O.
"""
from __future__ import annotations

from typing import Optional, Tuple, Any

import numpy as np

from config import (
    CAM_W,
    CAM_H,
    PERCEPTION_LIGHTWEIGHT_VIS,
    CTE_TO_METERS,
    CURVATURE_WP_EMA_ALPHA,
    LANE_EMA_ALPHA,
    PERCEPTION_PATH_MIN_CONF,
    PERCEPTION_PATH_SMOOTH_ALPHA,
    REF_PATH_LAT_SANITY_M,
    REF_PATH_NUM_PTS,
    SAFETY_MAX_STEER_RAD,
    STATE_EMA_ALPHA,
    STEER_MAX_DELTA_HIGH_SPEED,
    STEER_MAX_DELTA_PER_FRAME,
    STEER_RATE_LIMIT_SPEED_KMH,
    STEER_SMOOTH_ALPHA,
    USE_TRAJECTORY_PIPELINE,
    TURN_LOOKAHEAD_MIN_M,
    TURN_LOOKAHEAD_CURV_THRESHOLD,
    # Curve anticipation braking
    CURVE_BRAKE_CURV_MILD,
    CURVE_BRAKE_CURV_MEDIUM,
    CURVE_BRAKE_CURV_SHARP,
    CURVE_BRAKE_CURV_EXTREME,
    CURVE_BRAKE_SPEED_MILD,
    CURVE_BRAKE_SPEED_MEDIUM,
    CURVE_BRAKE_SPEED_SHARP,
    CURVE_BRAKE_SPEED_EXTREME,
    MPC_N,
    MPC_DT,
    MPC_L,
    MPC_MAX_STEER,
    MPC_MAX_STEER_RATE,
    MPC_MAX_ACCEL,
    MPC_MIN_ACCEL,
    MPC_W_CTE,
    MPC_W_HEADING,
    MPC_W_VEL,
    MPC_W_STEER,
    MPC_W_ACCEL,
    MPC_W_STEER_RATE,
    MPC_W_ACCEL_RATE,
    MPC_W_STEER_JERK,
    MPC_USE_ADAPTIVE_WEIGHTS,
)
from state import FrameState
from temporal.lane_lstm import LaneTemporalSmoother
from control.lane_mpc import LaneMPC, MPCConfig, get_mpc_weights, compute_mpc_horizon
from safety.override import SafetyOverride
from perception.road_perception import BEVRoadPerception

from .reference import dynamic_lookahead_m, get_reference_path, resample_path, smooth_path_lat
from .fusion import apply_fusion


def make_fallback_trajectory_out(
    target_speed_ms: float,
    lane_overlay: Optional[np.ndarray] = None,
    bev_window_vis: Optional[np.ndarray] = None,
    cam_h: int = CAM_H,
    cam_w: int = CAM_W,
) -> Any:
    """Minimal trajectory-like object for when perception is stale (e.g. threaded mode)."""
    from types import SimpleNamespace
    return SimpleNamespace(
        cte=0.0,
        heading_err=0.0,
        curvature=0.0,
        confidence=0.0,
        waypoint_only=True,
        geometry_valid=False,
        v_ref_at_ego=target_speed_ms,
        lane_overlay=lane_overlay,
        bev_window_vis=bev_window_vis,
        mask_vis=np.zeros((cam_h, cam_w), dtype=np.uint8),
        tracked=False,
        used_completion=False,
        phase_p2_case="none",
        phase_p3_case="none",
    )


class LKAStep:
    """
    One step of LKA: (rgb, speed, wp_state, prev_steer, prev_throttle)
    → (steer, throttle, brake, FrameState).
    """

    def __init__(
        self,
        model_path: str,
        device: Any,
        target_speed_kmh: float,
        use_trajectory_pipeline: Optional[bool] = None,
    ) -> None:
        if use_trajectory_pipeline is None:
            use_trajectory_pipeline = USE_TRAJECTORY_PIPELINE

        self._use_traj = use_trajectory_pipeline
        self._target_speed_ms = target_speed_kmh / 3.6

        # Use BEV-based perception by default (more robust for dashed lines)
        self._perception = BEVRoadPerception(model_path, device)
        if use_trajectory_pipeline:
            from perception.lane_trajectory import LaneTrajectoryPipeline
            self._trajectory = LaneTrajectoryPipeline(
                self._perception.detector,
                cam_w=CAM_W,
                cam_h=CAM_H,
                v_nominal_ms=self._target_speed_ms,
                bev_top_ratio=0.28,  # ต่ำลง = ใช้แถวบนภาพมากขึ้น → เลนยาวขึ้น
                bev_bot_ratio=0.98,
                bev_top_margin=0.32,
                bev_bot_margin=0.10,
                lightweight_vis=PERCEPTION_LIGHTWEIGHT_VIS,
            )
        else:
            self._trajectory = None
        self._smoother = LaneTemporalSmoother(ema_alpha=LANE_EMA_ALPHA)
        self._mpc_cfg = MPCConfig(
            N=MPC_N,
            dt=MPC_DT,
            L=MPC_L,
            max_steer=MPC_MAX_STEER,
            max_steer_rate=MPC_MAX_STEER_RATE,
            max_accel=MPC_MAX_ACCEL,
            min_accel=MPC_MIN_ACCEL,
            w_cte=MPC_W_CTE,
            w_heading=MPC_W_HEADING,
            w_vel=MPC_W_VEL,
            w_steer=MPC_W_STEER,
            w_accel=MPC_W_ACCEL,
            w_steer_rate=MPC_W_STEER_RATE,
            w_accel_rate=MPC_W_ACCEL_RATE,
            w_steer_jerk=MPC_W_STEER_JERK,
            use_adaptive_weights=MPC_USE_ADAPTIVE_WEIGHTS,
        )
        self._mpc = LaneMPC(self._mpc_cfg)
        self._safety = SafetyOverride({
            "max_speed_kmh": target_speed_kmh + 5.0,
            "max_steering_angle": SAFETY_MAX_STEER_RAD,
            "emergency_brake_enabled": True,
        })

    def reset(self) -> None:
        self._perception.reset()
        if self._trajectory is not None:
            self._trajectory.reset()
        self._smoother.reset()
        if hasattr(self, "_prev_fused_state"):
            del self._prev_fused_state

    def step(
        self,
        rgb: np.ndarray,
        speed_ms: float,
        wp_state: Optional[Tuple[float, float, float]],
        prev_steer: float,
        prev_throttle: float,
        trajectory_out: Optional[Any] = None,
        waypoint_path: Optional[list] = None,
        wp_lookahead_curv: float = 0.0,
    ) -> Tuple[float, float, float, FrameState]:
        """
        Returns: (steer, throttle, brake, FrameState).
        If trajectory_out is provided (e.g. from perception thread), use it; else run perception here.
        """
        # 1. Perception
        if trajectory_out is not None:
            out = trajectory_out
            cte_m_lane = float(out.cte)
            head_s = float(out.heading_err)
            curv_s = float(out.curvature)
            lane_conf = float(out.confidence)
            geometry_valid = bool(out.geometry_valid)
            v_ref_traj = float(out.v_ref_at_ego)
            lane_overlay = out.lane_overlay
            bev_window_vis = out.bev_window_vis
            mask = out.mask_vis
            tracked = getattr(out, "tracked", False)
            used_completion = getattr(out, "used_completion", False)
            left_px_img = getattr(out, "left_px_img", None)
            right_px_img = getattr(out, "right_px_img", None)
            p2_case = getattr(out, "phase_p2_case", "none")
            p3_case = getattr(out, "phase_p3_case", "none")
            path_from_perception = getattr(out, "reference_path_from_perception", None)
            # Peter Moran viz data
            bev_binary = getattr(out, "bev_binary", None)
            left_curvature_m = getattr(out, "left_curvature_m", 9999.0)
            right_curvature_m = getattr(out, "right_curvature_m", 9999.0)
        elif self._trajectory is not None:
            out = self._trajectory.process(rgb)
            cte_m_lane = float(out.cte)
            head_s = float(out.heading_err)
            curv_s = float(out.curvature)
            lane_conf = float(out.confidence)
            geometry_valid = bool(out.geometry_valid)
            v_ref_traj = float(out.v_ref_at_ego)
            lane_overlay = out.lane_overlay
            bev_window_vis = out.bev_window_vis
            mask = out.mask_vis
            tracked = getattr(out, "tracked", False)
            used_completion = getattr(out, "used_completion", False)
            left_px_img = getattr(out, "left_px_img", None)
            right_px_img = getattr(out, "right_px_img", None)
            p2_case = getattr(out, "phase_p2_case", "none")
            p3_case = getattr(out, "phase_p3_case", "none")
            path_from_perception = getattr(out, "reference_path_from_perception", None)
            # Peter Moran viz data
            bev_binary = getattr(out, "bev_binary", None)
            left_curvature_m = getattr(out, "left_curvature_m", 9999.0)
            right_curvature_m = getattr(out, "right_curvature_m", 9999.0)
        else:
            result = self._perception.process(rgb)
            # BEVRoadPerception returns 8 values: cte, head, curv, mask, conf, centerline, bev_vis, lane_overlay
            if len(result) == 8:
                cte_raw, heading, curvature, mask, lane_conf, _, bev_window_vis, lane_overlay = result
            elif len(result) == 7:
                cte_raw, heading, curvature, mask, lane_conf, _, bev_window_vis = result
                lane_overlay = None
            else:
                cte_raw, heading, curvature, mask, lane_conf, _ = result
                bev_window_vis = None
                lane_overlay = None
            sm = self._smoother.update(cte_raw, heading, curvature)
            cte_m_lane = float(sm[0]) * CTE_TO_METERS
            head_s = float(sm[1])
            curv_s = float(sm[2])
            geometry_valid = False
            v_ref_traj = None
            tracked = False
            used_completion = False
            left_px_img = None
            right_px_img = None
            p2_case = "none"
            p3_case = "none"
            path_from_perception = None
            bev_binary = None
            left_curvature_m = 9999.0
            right_curvature_m = 9999.0

        # 2. Fusion (geometry_valid: ในโค้งสูง + invalid → บังคับ WP only)
        cte_m, head_s, curv_s, mode = apply_fusion(
            wp_state, cte_m_lane, head_s, curv_s, lane_conf,
            geometry_valid=geometry_valid,
        )

        # 2b. Post-fusion state EMA — ลด jitter ก่อนส่งเข้า MPC
        if not hasattr(self, "_prev_fused_state"):
            self._prev_fused_state = np.array([cte_m, head_s, curv_s], dtype=np.float64)
        a = STATE_EMA_ALPHA
        self._prev_fused_state = a * self._prev_fused_state + (1.0 - a) * np.array([cte_m, head_s, curv_s], dtype=np.float64)
        cte_m, head_s, curv_s = float(self._prev_fused_state[0]), float(self._prev_fused_state[1]), float(self._prev_fused_state[2])
        # 2c. Optional: smooth curvature from WP (ลดกระตุกตอนเลี้ยว); reset EMA on mode change
        if mode in ("WP_PRIMARY", "WP+UNET_CENTER"):
            prev_mode = getattr(self, "_prev_mode", None)
            if prev_mode is not None and mode != prev_mode:
                self._prev_curv_wp = curv_s
            self._prev_mode = mode
            if not hasattr(self, "_prev_curv_wp"):
                self._prev_curv_wp = curv_s
            curv_s = CURVATURE_WP_EMA_ALPHA * self._prev_curv_wp + (1.0 - CURVATURE_WP_EMA_ALPHA) * curv_s
            self._prev_curv_wp = curv_s
        else:
            self._prev_mode = mode

        # 3. Reference path: Option A perception path (with guard + sanity + smooth); else WP in turn; else parabola
        # Use max of current curvature and waypoint lookahead curvature for anticipation
        effective_curv = max(abs(curv_s), wp_lookahead_curv)
        lookahead_m = dynamic_lookahead_m(speed_ms)
        scale = min(effective_curv / max(TURN_LOOKAHEAD_CURV_THRESHOLD, 1e-9), 1.0)
        lookahead_m = (1.0 - scale) * lookahead_m + scale * TURN_LOOKAHEAD_MIN_M

        use_perception_path = (
            path_from_perception
            and len(path_from_perception) >= 2
            and geometry_valid
            and lane_conf > PERCEPTION_PATH_MIN_CONF
        )
        # Perception path must be (s, lat) in vehicle frame; reject if lateral at ego is out of range (debug)
        if use_perception_path:
            lat_at_ego = path_from_perception[0][1]
            if abs(lat_at_ego) <= REF_PATH_LAT_SANITY_M:
                reference_path = smooth_path_lat(
                    path_from_perception, alpha=PERCEPTION_PATH_SMOOTH_ALPHA
                )
            else:
                use_perception_path = False
        if not use_perception_path:
            # Force waypoint path when lookahead curvature is sharp OR current curvature is high
            if (
                waypoint_path
                and len(waypoint_path) >= 2
                and effective_curv > TURN_LOOKAHEAD_CURV_THRESHOLD
            ):
                reference_path = resample_path(waypoint_path, lookahead_m, REF_PATH_NUM_PTS)
                if not reference_path:
                    reference_path = get_reference_path(cte_m, head_s, curv_s, lookahead_m=lookahead_m)
            else:
                reference_path = get_reference_path(cte_m, head_s, curv_s, lookahead_m=lookahead_m)

        # 4. v_ref for MPC — use effective_curv (includes lookahead) to brake early
        # Progressive speed reduction based on curve severity (configurable)
        if effective_curv >= CURVE_BRAKE_CURV_EXTREME:
            vt = self._target_speed_ms * CURVE_BRAKE_SPEED_EXTREME
        elif effective_curv >= CURVE_BRAKE_CURV_SHARP:
            vt = self._target_speed_ms * CURVE_BRAKE_SPEED_SHARP
        elif effective_curv >= CURVE_BRAKE_CURV_MEDIUM:
            vt = self._target_speed_ms * CURVE_BRAKE_SPEED_MEDIUM
        elif effective_curv >= CURVE_BRAKE_CURV_MILD:
            vt = self._target_speed_ms * CURVE_BRAKE_SPEED_MILD
        elif v_ref_traj is not None:
            vt = v_ref_traj
        else:
            vt = self._target_speed_ms

        # 5. MPC (dynamic horizon + adaptive weights)
        N_desired = compute_mpc_horizon(speed_ms, curv_s)
        self._mpc.set_horizon(N_desired)
        weight_scales = get_mpc_weights(speed_ms, curv_s, lane_conf)
        steer_rad, accel, solver_status = self._mpc.solve(
            x0=0, y0=cte_m, psi0=head_s, v0=speed_ms,
            v_ref=vt, cte=cte_m, heading_err=head_s, curvature=curv_s,
            confidence=lane_conf,
            weight_scales=weight_scales,
        )
        s_raw = self._mpc.steer_to_carla(steer_rad)
        steer = STEER_SMOOTH_ALPHA * prev_steer + (1 - STEER_SMOOTH_ALPHA) * s_raw
        # Steering rate limit: reduce oscillation when perception mode switches (waypoint ↔ UNet)
        speed_kmh = speed_ms * 3.6
        max_delta = (
            STEER_MAX_DELTA_HIGH_SPEED
            if speed_kmh >= STEER_RATE_LIMIT_SPEED_KMH
            else STEER_MAX_DELTA_PER_FRAME
        )
        steer = float(np.clip(steer, prev_steer - max_delta, prev_steer + max_delta))
        throttle, brake = self._mpc.accel_to_carla(accel)

        if speed_ms < 0.6 and brake < 0.1:
            throttle = max(throttle, 0.55)
            brake = 0.0

        steer, throttle, brake = self._safety.apply_safety_override(
            {"velocity": speed_ms}, steer, throttle, brake
        )

        # 6. Phase report (P1..P5) — Back to basic: จับเลนและตีเส้นแบ่ง phase ครบหรือไม่
        phase_p1_ok = lane_conf > 0.01
        phase_p2_ok = p2_case in ("both", "mirror_left", "mirror_right", "completion")
        phase_p3_ok = p3_case in ("from_both", "completion")
        phase_p4_ok = bool(
            np.isfinite(cte_m) and np.isfinite(head_s) and np.isfinite(curv_s)
        )
        phase_p5_ok = bool(mode and mode.strip())

        state = FrameState(
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
            mask=mask,
            bev_window_vis=bev_window_vis,
            left_px_img=left_px_img,
            right_px_img=right_px_img,
            road_symbol_result=None,
            tracked=tracked,
            used_completion=used_completion,
            phase_p1_ok=phase_p1_ok,
            phase_p2_ok=phase_p2_ok,
            phase_p3_ok=phase_p3_ok,
            phase_p4_ok=phase_p4_ok,
            phase_p5_ok=phase_p5_ok,
            phase_p2_case=p2_case,
            phase_p3_case=p3_case,
            solver_status=solver_status,
            # Peter Moran visualization
            bev_binary=bev_binary,
            raw_windows_left=getattr(self._trajectory, '_raw_left_wins', None) if self._trajectory else None,
            raw_windows_right=getattr(self._trajectory, '_raw_right_wins', None) if self._trajectory else None,
            filt_windows_left=getattr(self._trajectory, '_filt_left_wins', None) if self._trajectory else None,
            filt_windows_right=getattr(self._trajectory, '_filt_right_wins', None) if self._trajectory else None,
            left_px_bev=left_px_img,
            right_px_bev=right_px_img,
            left_curvature_m=left_curvature_m,
            right_curvature_m=right_curvature_m,
        )
        return steer, throttle, brake, state

