"""
Comprehensive Telemetry Collector — เก็บข้อมูลทุก frame แบบ ring buffer
Captures vehicle state, control, MPC, perception, waypoints, ADAS, safety, performance.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TelemetryFrame:
    """Single frame of telemetry data."""
    # Timestamps
    sim_time: float = 0.0
    real_time: float = 0.0
    frame_idx: int = 0
    # Vehicle State
    vehicle_x: float = 0.0
    vehicle_y: float = 0.0
    vehicle_z: float = 0.0
    vehicle_pitch: float = 0.0
    vehicle_yaw: float = 0.0
    vehicle_roll: float = 0.0
    speed_ms: float = 0.0
    speed_kmh: float = 0.0
    accel_ms2: float = 0.0
    # Vehicle Control (final sent to CARLA)
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    reverse: bool = False
    # MPC Output
    mpc_steer_rad: float = 0.0
    mpc_accel: float = 0.0
    mpc_solver_status: str = "unknown"
    mpc_solve_time_ms: float = 0.0
    mpc_trajectory: Optional[List[Tuple[float, float]]] = None  # (x, y) pairs
    # Perception
    cte_m: float = 0.0
    heading_err_rad: float = 0.0
    curvature: float = 0.0
    lane_conf: float = 0.0
    geometry_valid: bool = False
    # Waypoint Data
    nearest_waypoint_idx: int = 0
    waypoint_count: int = 0
    path_tangent: float = 0.0
    # Lane Boundaries
    left_boundary_count: int = 0
    right_boundary_count: int = 0
    # Fusion State
    fusion_mode: str = "unknown"
    # Safety
    safety_override_active: bool = False
    steering_clamped: bool = False
    stuck_recovery_active: bool = False
    stuck_recovery_phase: str = "none"
    aeb_active: bool = False
    # ADAS
    adas_active: bool = False
    aeb_active: bool = False
    aeb_ttc: float = -1.0
    aeb_warning_level: str = "none"
    acc_active: bool = False
    acc_target_speed_ms: float = -1.0
    acc_distance_m: float = -1.0
    ldw_state: str = "in_lane"
    ldw_warning_active: bool = False
    ldw_side: str = "none"
    lka_pro_assist: float = 0.0
    bsw_left_alert: str = "clear"
    bsw_right_alert: str = "clear"
    tsr_speed_limit_kmh: Optional[float] = None
    tsr_traffic_light: str = "unknown"
    tsr_action: str = "none"
    tja_state: str = "inactive"
    tja_active: bool = False
    # Performance
    fps: float = 0.0
    loop_time_ms: float = 0.0
    perception_time_ms: float = 0.0
    mpc_time_ms: float = 0.0


class ComprehensiveTelemetryCollector:
    """
    Real-time telemetry collection with ring buffer storage.

    Captures per-frame data from all subsystems for analysis and export.
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        export_interval_frames: int = 1000,
    ):
        """
        Args:
            buffer_size: max frames to keep in ring buffer
            export_interval_frames: auto-export CSV every N frames (0 = disabled)
        """
        self.buffer_size = buffer_size
        self.export_interval_frames = export_interval_frames

        self._frames: deque = deque(maxlen=buffer_size)
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._last_speed = 0.0
        self._last_timestamp: Optional[float] = None

    def record(
        self,
        frame_idx: int,
        vehicle_transform: Optional[Any] = None,
        speed_ms: float = 0.0,
        control: Optional[Any] = None,
        mpc_steer_rad: float = 0.0,
        mpc_accel: float = 0.0,
        mpc_solver_status: str = "unknown",
        mpc_solve_time_ms: float = 0.0,
        mpc_trajectory: Optional[np.ndarray] = None,
        frame_state: Optional[Any] = None,
        adas_out: Optional[Any] = None,
        fps: float = 0.0,
        loop_time_ms: float = 0.0,
        perception_time_ms: float = 0.0,
        sim_time: float = 0.0,
    ) -> TelemetryFrame:
        """Record a single frame of telemetry."""
        real_time = time.time()
        if self._start_time is None:
            self._start_time = real_time

        # Extract vehicle pose from CARLA transform
        vx = vy = vz = vpitch = vyaw = vroll = 0.0
        if vehicle_transform is not None:
            loc = vehicle_transform.location
            rot = vehicle_transform.rotation
            vx = float(loc.x)
            vy = float(loc.y)
            vz = float(loc.z)
            import math
            vpitch = math.radians(rot.pitch)
            vyaw = math.radians(rot.yaw)
            vroll = math.radians(rot.roll)

        # Compute acceleration
        accel = 0.0
        if self._last_timestamp is not None:
            dt = real_time - self._last_timestamp
            if dt > 1e-6:
                accel = (speed_ms - self._last_speed) / dt
        self._last_speed = speed_ms
        self._last_timestamp = real_time

        # Extract control values
        steer = throttle = brake = 0.0
        reverse = False
        if control is not None:
            steer = float(getattr(control, 'steering', 0.0))
            throttle = float(getattr(control, 'throttle', 0.0))
            brake = float(getattr(control, 'brake', 0.0))
            reverse = bool(getattr(control, 'reverse', False))

        # Extract perception data from frame_state
        cte_m = heading_err = curvature = lane_conf = 0.0
        geometry_valid = False
        fusion_mode = "unknown"
        if frame_state is not None:
            cte_m = float(getattr(frame_state, 'cte_m', 0.0))
            heading_err = float(getattr(frame_state, 'heading_rad', 0.0))
            curvature = float(getattr(frame_state, 'curvature', 0.0))
            lane_conf = float(getattr(frame_state, 'lane_conf', 0.0))
            geometry_valid = bool(getattr(frame_state, 'geometry_valid', False))
            fusion_mode = str(getattr(frame_state, 'mode', 'unknown'))

        # Extract ADAS data
        aeb_active = acc_active = False
        aeb_ttc = -1.0
        aeb_warning_level = "none"
        acc_target_speed = -1.0
        acc_distance = -1.0
        ldw_state = "in_lane"
        ldw_warning_active = False
        ldw_side = "none"
        lka_pro_assist = 0.0
        bsw_left = bsw_right = "clear"
        tsr_speed_limit = None
        tsr_traffic_light = "unknown"
        tsr_action = "none"
        tja_state = "inactive"
        tja_active = False
        adas_active = False

        if adas_out is not None:
            aeb_active = bool(getattr(adas_out, 'aeb_active', False))
            aeb_ttc = float(getattr(adas_out, 'aeb_ttc', -1.0))
            aeb_warning_level = str(getattr(adas_out, 'aeb_warning_level', 'none'))
            acc_active = bool(getattr(adas_out, 'acc_active', False))
            acc_target_speed = float(getattr(adas_out, 'acc_target_speed_ms', -1.0))
            acc_distance = float(getattr(adas_out, 'acc_distance_m', -1.0))
            ldw_state = str(getattr(adas_out, 'ldw_state', 'in_lane'))
            ldw_warning_active = bool(getattr(adas_out, 'ldw_warning_active', False))
            ldw_side = str(getattr(adas_out, 'ldw_side', 'none'))
            lka_pro_assist = float(getattr(adas_out, 'lka_pro_assist', 0.0))
            bsw_left = str(getattr(adas_out, 'bsw_left_alert', 'clear'))
            bsw_right = str(getattr(adas_out, 'bsw_right_alert', 'clear'))
            tsr_speed_limit = getattr(adas_out, 'tsr_speed_limit_kmh', None)
            tsr_traffic_light = str(getattr(adas_out, 'tsr_traffic_light', 'unknown'))
            tsr_action = str(getattr(adas_out, 'tsr_action', 'none'))
            tja_state = str(getattr(adas_out, 'tja_state', 'inactive'))
            tja_active = bool(getattr(adas_out, 'tja_active', False))
            adas_active = bool(getattr(adas_out, 'active_features', False))
        elif frame_state is not None:
            # Fall back to frame_state ADAS fields (populated in main.py)
            aeb_active = bool(getattr(frame_state, 'aeb_active', False))
            aeb_ttc = float(getattr(frame_state, 'aeb_ttc', -1.0))
            aeb_warning_level = str(getattr(frame_state, 'aeb_warning_level', 'none'))
            acc_active = bool(getattr(frame_state, 'acc_active', False))
            acc_target_speed = float(getattr(frame_state, 'acc_target_speed_ms', -1.0))
            acc_distance = float(getattr(frame_state, 'acc_distance_m', -1.0))
            ldw_state = str(getattr(frame_state, 'ldw_state', 'in_lane'))
            ldw_warning_active = bool(getattr(frame_state, 'ldw_warning_active', False))
            ldw_side = str(getattr(frame_state, 'ldw_side', 'none'))
            lka_pro_assist = float(getattr(frame_state, 'lka_pro_assist', 0.0))
            bsw_left = str(getattr(frame_state, 'bsw_left_alert', 'clear'))
            bsw_right = str(getattr(frame_state, 'bsw_right_alert', 'clear'))
            tsr_speed_limit = getattr(frame_state, 'tsr_speed_limit_kmh', None)
            tsr_traffic_light = str(getattr(frame_state, 'tsr_traffic_light', 'unknown'))
            tsr_action = str(getattr(frame_state, 'tsr_action', 'none'))
            tja_state = str(getattr(frame_state, 'tja_state', 'inactive'))
            tja_active = bool(getattr(frame_state, 'tja_active', False))
            adas_active = bool(getattr(frame_state, 'adas_override_active', False))

        # Safety/stuck from frame_state
        safety_active = False
        stuck_active = False
        stuck_phase = "none"
        if frame_state is not None:
            safety_active = bool(getattr(frame_state, 'safety_override_active', False))
            stuck_active = bool(getattr(frame_state, 'stuck_recovery_active', False))
            stuck_phase = str(getattr(frame_state, 'stuck_recovery_phase', 'none'))

        # Convert MPC trajectory to list of (x, y) tuples for serialization
        mpc_traj_list = None
        if mpc_trajectory is not None and isinstance(mpc_trajectory, np.ndarray):
            if mpc_trajectory.ndim == 2 and mpc_trajectory.shape[0] >= 2:
                mpc_traj_list = [(float(mpc_trajectory[0, k]), float(mpc_trajectory[1, k]))
                                 for k in range(mpc_trajectory.shape[1])]

        frame = TelemetryFrame(
            sim_time=sim_time if sim_time > 0 else real_time - self._start_time,
            real_time=real_time,
            frame_idx=frame_idx,
            vehicle_x=vx, vehicle_y=vy, vehicle_z=vz,
            vehicle_pitch=vpitch, vehicle_yaw=vyaw, vehicle_roll=vroll,
            speed_ms=speed_ms, speed_kmh=speed_ms * 3.6, accel_ms2=accel,
            steering=steer, throttle=throttle, brake=brake, reverse=reverse,
            mpc_steer_rad=mpc_steer_rad, mpc_accel=mpc_accel,
            mpc_solver_status=mpc_solver_status, mpc_solve_time_ms=mpc_solve_time_ms,
            mpc_trajectory=mpc_traj_list,
            cte_m=cte_m, heading_err_rad=heading_err, curvature=curvature,
            lane_conf=lane_conf, geometry_valid=geometry_valid,
            fusion_mode=fusion_mode,
            safety_override_active=safety_active,
            stuck_recovery_active=stuck_active, stuck_recovery_phase=stuck_phase,
            aeb_active=aeb_active, adas_active=adas_active,
            acc_active=acc_active, acc_target_speed_ms=acc_target_speed,
            acc_distance_m=acc_distance, aeb_ttc=aeb_ttc,
            aeb_warning_level=aeb_warning_level,
            ldw_state=ldw_state, ldw_warning_active=ldw_warning_active,
            ldw_side=ldw_side, lka_pro_assist=lka_pro_assist,
            bsw_left_alert=bsw_left, bsw_right_alert=bsw_right,
            tsr_speed_limit_kmh=tsr_speed_limit,
            tsr_traffic_light=tsr_traffic_light, tsr_action=tsr_action,
            tja_state=tja_state, tja_active=tja_active,
            fps=fps, loop_time_ms=loop_time_ms,
            perception_time_ms=perception_time_ms, mpc_time_ms=mpc_solve_time_ms,
        )

        self._frames.append(frame)
        self._frame_count += 1

        # Auto-export if interval reached
        if self.export_interval_frames > 0 and self._frame_count % self.export_interval_frames == 0:
            logger.debug(f"Telemetry auto-export at frame {self._frame_count}")

        return frame

    def get_latest(self) -> Optional[TelemetryFrame]:
        """Get the most recent frame."""
        if not self._frames:
            return None
        return self._frames[-1]

    def get_frames(self, last_n: Optional[int] = None) -> List[TelemetryFrame]:
        """Get list of frames, optionally only last N."""
        if last_n is None:
            return list(self._frames)
        return list(self._frames)[-last_n:]

    def get_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not self._frames:
            return {}

        speeds = np.array([f.speed_ms for f in self._frames])
        ctes = np.array([f.cte_m for f in self._frames])
        loop_times = np.array([f.loop_time_ms for f in self._frames])
        mpc_times = np.array([f.mpc_solve_time_ms for f in self._frames])

        return {
            "total_frames": self._frame_count,
            "duration_s": self._frames[-1].real_time - self._frames[0].real_time if len(self._frames) > 1 else 0.0,
            "speed_mean_ms": float(np.mean(speeds)),
            "speed_max_ms": float(np.max(speeds)),
            "cte_mean_m": float(np.mean(ctes)),
            "cte_rms_m": float(np.sqrt(np.mean(ctes ** 2))),
            "cte_max_m": float(np.max(np.abs(ctes))),
            "fps_mean": float(1000.0 / np.mean(loop_times)) if np.mean(loop_times) > 0 else 0.0,
            "loop_time_mean_ms": float(np.mean(loop_times)),
            "mpc_time_mean_ms": float(np.mean(mpc_times)),
            "aeb_events": int(sum(1 for f in self._frames if f.aeb_active)),
            "stuck_events": int(sum(1 for f in self._frames if f.stuck_recovery_active)),
        }

    def export_csv(self, filepath: str) -> None:
        """Export all frames to CSV."""
        if not self._frames:
            logger.warning("No telemetry data to export")
            return

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        import csv
        fields = [
            "frame_idx", "sim_time", "real_time",
            "vehicle_x", "vehicle_y", "vehicle_z",
            "vehicle_yaw", "speed_ms", "speed_kmh", "accel_ms2",
            "steering", "throttle", "brake", "reverse",
            "mpc_steer_rad", "mpc_accel", "mpc_solver_status", "mpc_solve_time_ms",
            "cte_m", "heading_err_rad", "curvature", "lane_conf", "geometry_valid",
            "fusion_mode",
            "safety_override_active", "stuck_recovery_active", "stuck_recovery_phase",
            "aeb_active", "aeb_ttc", "aeb_warning_level",
            "acc_active", "acc_target_speed_ms", "acc_distance_m",
            "ldw_state", "ldw_warning_active", "ldw_side",
            "tja_state", "tja_active",
            "tsr_speed_limit_kmh", "tsr_traffic_light", "tsr_action",
            "fps", "loop_time_ms", "perception_time_ms", "mpc_time_ms",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for frame in self._frames:
                row = []
                for fld in fields:
                    val = getattr(frame, fld, "")
                    if isinstance(val, bool):
                        val = int(val)
                    elif isinstance(val, float):
                        val = f"{val:.6f}"
                    row.append(val)
                writer.writerow(row)
        logger.info(f"Telemetry CSV exported to {path} ({len(self._frames)} frames)")

    def export_json(self, filepath: str) -> None:
        """Export summary to JSON."""
        import json
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Telemetry JSON exported to {path}")

    def reset(self) -> None:
        """Reset all collected data."""
        self._frames.clear()
        self._frame_count = 0
        self._start_time = None
        self._last_speed = 0.0
        self._last_timestamp = None
