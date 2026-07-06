"""
MetricsCollector — รวบรวมและบันทึกทุก metric จากระบบ

เก็บข้อมูลทุก frame และสรุปเป็น CSV + JSON เมื่อจบ run
ไม่ต้องการ InfluxDB หรือ server ใดๆ — เก็บใน memory แล้ว flush ลงไฟล์

Metrics ที่วัด:
- Vehicle: speed_ms, speed_kmh, steering, throttle, brake
- Lane/Perception: cte_m, heading_err_rad, curvature, lane_conf, geometry_valid
- MPC: solve_time_ms, solver_status, fallback_count, cost
- Safety: safety_active, safety_reason, stuck_recovery_active
- ADAS: aeb_active, acc_active, ldw_state, tja_state, bsw_alert
- Performance: fps, loop_time_ms, perception_time_ms
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """ข้อมูล metric 1 frame"""
    frame_idx: int = 0
    timestamp: float = 0.0

    # Vehicle
    speed_ms: float = 0.0
    speed_kmh: float = 0.0
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    # Lane / Perception
    cte_m: float = 0.0
    heading_err_rad: float = 0.0
    curvature: float = 0.0
    lane_conf: float = 0.0
    geometry_valid: bool = False
    perception_time_ms: float = 0.0
    lane_phase_p1: bool = False
    lane_phase_p2: bool = False
    lane_phase_p3: bool = False
    lane_phase_p4: bool = False
    lane_phase_p5: bool = False

    # MPC
    mpc_solve_time_ms: float = 0.0
    mpc_solver_status: str = "unknown"
    mpc_cost: float = 0.0
    mpc_fallback: bool = False

    # Safety
    safety_active: bool = False
    safety_reason: str = ""
    stuck_recovery_active: bool = False

    # ADAS
    aeb_active: bool = False
    aeb_ttc: float = -1.0
    acc_active: bool = False
    acc_distance_m: float = -1.0
    ldw_state: str = "in_lane"
    ldw_warning_active: bool = False
    lka_pro_assist: float = 0.0
    bsw_left_alert: str = "clear"
    bsw_right_alert: str = "clear"
    tsr_speed_limit_kmh: float = -1.0
    tsr_traffic_light: str = "unknown"
    tja_state: str = "inactive"
    tja_active: bool = False

    # Performance
    fps: float = 0.0
    loop_time_ms: float = 0.0


class MetricsCollector:
    """
    รวบรวม metric ทุก frame เก็บใน memory
    เมื่อจบ run เรียก save() เพื่อ export CSV + JSON summary
    """

    def __init__(self, output_dir: str = "metrics_output"):
        self.output_dir = Path(output_dir)
        self.frames: List[FrameMetrics] = []
        self._start_time = time.time()
        self._fallback_count = 0
        self._total_frames = 0

    def record(self, metrics: FrameMetrics) -> None:
        """บันทึก metric 1 frame"""
        self.frames.append(metrics)
        self._total_frames += 1
        if metrics.mpc_fallback:
            self._fallback_count += 1

    def record_from_frame_state(
        self,
        frame_idx: int,
        speed_ms: float,
        steer: float,
        throttle: float,
        brake: float,
        frame_state: Optional[Any] = None,
        mpc_solve_time_ms: float = 0.0,
        mpc_solver_status: str = "unknown",
        mpc_cost: float = 0.0,
        loop_time_ms: float = 0.0,
        perception_time_ms: float = 0.0,
        safety_active: bool = False,
        safety_reason: str = "",
        stuck_recovery_active: bool = False,
        adas_out: Optional[Any] = None,
    ) -> None:
        """สร้าง FrameMetrics จากข้อมูลใน main loop แล้วบันทึก"""
        m = FrameMetrics(
            frame_idx=frame_idx,
            timestamp=time.time() - self._start_time,
            speed_ms=speed_ms,
            speed_kmh=speed_ms * 3.6,
            steering=steer,
            throttle=throttle,
            brake=brake,
            mpc_solve_time_ms=mpc_solve_time_ms,
            mpc_solver_status=mpc_solver_status,
            mpc_cost=mpc_cost,
            mpc_fallback=(mpc_solver_status != "Solve_Succeeded"),
            loop_time_ms=loop_time_ms,
            perception_time_ms=perception_time_ms,
            safety_active=safety_active,
            safety_reason=safety_reason,
            stuck_recovery_active=stuck_recovery_active,
            fps=(1000.0 / loop_time_ms) if loop_time_ms > 0 else 0.0,
        )

        if frame_state is not None:
            m.cte_m = float(getattr(frame_state, 'cte_m', 0.0))
            m.heading_err_rad = float(getattr(frame_state, 'heading_rad', 0.0))
            m.curvature = float(getattr(frame_state, 'curvature', 0.0))
            m.lane_conf = float(getattr(frame_state, 'lane_conf', 0.0))
            m.geometry_valid = bool(getattr(frame_state, 'geometry_valid', False))
            m.lane_phase_p1 = bool(getattr(frame_state, 'phase_p1_ok', False))
            m.lane_phase_p2 = bool(getattr(frame_state, 'phase_p2_ok', False))
            m.lane_phase_p3 = bool(getattr(frame_state, 'phase_p3_ok', False))
            m.lane_phase_p4 = bool(getattr(frame_state, 'phase_p4_ok', False))
            m.lane_phase_p5 = bool(getattr(frame_state, 'phase_p5_ok', False))
            m.solver_status = str(getattr(frame_state, 'solver_status', 'unknown'))
            m.ldw_state = str(getattr(frame_state, 'ldw_state', 'in_lane'))
            m.ldw_warning_active = bool(getattr(frame_state, 'ldw_warning_active', False))
            m.lka_pro_assist = float(getattr(frame_state, 'lka_pro_assist', 0.0))
            m.bsw_left_alert = str(getattr(frame_state, 'bsw_left_alert', 'clear'))
            m.bsw_right_alert = str(getattr(frame_state, 'bsw_right_alert', 'clear'))
            m.tsr_speed_limit_kmh = float(getattr(frame_state, 'tsr_speed_limit_kmh', -1.0) or -1.0)
            m.tsr_traffic_light = str(getattr(frame_state, 'tsr_traffic_light', 'unknown'))
            m.tja_state = str(getattr(frame_state, 'tja_state', 'inactive'))
            m.tja_active = bool(getattr(frame_state, 'tja_active', False))
            m.aeb_active = bool(getattr(frame_state, 'aeb_active', False))
            m.aeb_ttc = float(getattr(frame_state, 'aeb_ttc', -1.0))
            m.acc_active = bool(getattr(frame_state, 'acc_active', False))
            m.acc_distance_m = float(getattr(frame_state, 'acc_distance_m', -1.0))

        if adas_out is not None:
            m.aeb_active = bool(getattr(adas_out, 'aeb_active', m.aeb_active))
            m.aeb_ttc = float(getattr(adas_out, 'aeb_ttc', m.aeb_ttc))
            m.acc_active = bool(getattr(adas_out, 'acc_active', m.acc_active))
            m.acc_distance_m = float(getattr(adas_out, 'acc_distance_m', m.acc_distance_m))
            m.ldw_state = str(getattr(adas_out, 'ldw_state', m.ldw_state))
            m.ldw_warning_active = bool(getattr(adas_out, 'ldw_warning_active', m.ldw_warning_active))
            m.lka_pro_assist = float(getattr(adas_out, 'lka_pro_assist', m.lka_pro_assist))
            m.bsw_left_alert = str(getattr(adas_out, 'bsw_left_alert', m.bsw_left_alert))
            m.bsw_right_alert = str(getattr(adas_out, 'bsw_right_alert', m.bsw_right_alert))
            m.tsr_speed_limit_kmh = float(getattr(adas_out, 'tsr_speed_limit_kmh', m.tsr_speed_limit_kmh) or -1.0)
            m.tsr_traffic_light = str(getattr(adas_out, 'tsr_traffic_light', m.tsr_traffic_light))
            m.tja_state = str(getattr(adas_out, 'tja_state', m.tja_state))
            m.tja_active = bool(getattr(adas_out, 'tja_active', m.tja_active))

        self.record(m)

    def get_summary(self) -> Dict[str, Any]:
        """สรุปทุก metric เป็น dict"""
        if not self.frames:
            return {"total_frames": 0, "duration_s": 0.0}

        duration = self.frames[-1].timestamp - self.frames[0].timestamp if len(self.frames) > 1 else 0.0

        cte_vals = [f.cte_m for f in self.frames]
        heading_vals = [f.heading_err_rad for f in self.frames]
        conf_vals = [f.lane_conf for f in self.frames]
        speed_vals = [f.speed_ms for f in self.frames]
        fps_vals = [f.fps for f in self.frames if f.fps > 0]
        solve_times = [f.mpc_solve_time_ms for f in self.frames if f.mpc_solve_time_ms > 0]
        loop_times = [f.loop_time_ms for f in self.frames if f.loop_time_ms > 0]
        steer_vals = [f.steering for f in self.frames]

        geometry_valid_count = sum(1 for f in self.frames if f.geometry_valid)
        p1_ok = sum(1 for f in self.frames if f.lane_phase_p1)
        p2_ok = sum(1 for f in self.frames if f.lane_phase_p2)
        p3_ok = sum(1 for f in self.frames if f.lane_phase_p3)
        p4_ok = sum(1 for f in self.frames if f.lane_phase_p4)
        p5_ok = sum(1 for f in self.frames if f.lane_phase_p5)

        aeb_count = sum(1 for f in self.frames if f.aeb_active)
        acc_count = sum(1 for f in self.frames if f.acc_active)
        ldw_warn_count = sum(1 for f in self.frames if f.ldw_warning_active)
        safety_count = sum(1 for f in self.frames if f.safety_active)
        stuck_count = sum(1 for f in self.frames if f.stuck_recovery_active)

        summary: Dict[str, Any] = {
            "total_frames": self._total_frames,
            "duration_s": round(duration, 2),
            "fallback_count": self._fallback_count,
            "fallback_rate": round(self._fallback_count / max(1, self._total_frames), 4),

            # Perception
            "perception": {
                "cte_mean_m": round(float(np.mean(cte_vals)), 4),
                "cte_std_m": round(float(np.std(cte_vals)), 4),
                "cte_max_abs_m": round(float(np.max(np.abs(cte_vals))), 4),
                "cte_rmse_m": round(float(np.sqrt(np.mean(np.square(cte_vals)))), 4),
                "heading_err_mean_rad": round(float(np.mean(heading_vals)), 6),
                "heading_err_std_rad": round(float(np.std(heading_vals)), 6),
                "heading_err_max_abs_rad": round(float(np.max(np.abs(heading_vals))), 6),
                "lane_conf_mean": round(float(np.mean(conf_vals)), 4),
                "lane_conf_std": round(float(np.std(conf_vals)), 4),
                "lane_conf_min": round(float(np.min(conf_vals)), 4),
                "geometry_valid_rate": round(geometry_valid_count / max(1, self._total_frames), 4),
                "phase_p1_ok_rate": round(p1_ok / max(1, self._total_frames), 4),
                "phase_p2_ok_rate": round(p2_ok / max(1, self._total_frames), 4),
                "phase_p3_ok_rate": round(p3_ok / max(1, self._total_frames), 4),
                "phase_p4_ok_rate": round(p4_ok / max(1, self._total_frames), 4),
                "phase_p5_ok_rate": round(p5_ok / max(1, self._total_frames), 4),
            },

            # Control / MPC
            "control": {
                "mpc_solve_time_mean_ms": round(float(np.mean(solve_times)), 3) if solve_times else 0.0,
                "mpc_solve_time_max_ms": round(float(np.max(solve_times)), 3) if solve_times else 0.0,
                "mpc_solve_time_p95_ms": round(float(np.percentile(solve_times, 95)), 3) if solve_times else 0.0,
                "steer_mean": round(float(np.mean(steer_vals)), 4),
                "steer_std": round(float(np.std(steer_vals)), 4),
                "steer_max_abs": round(float(np.max(np.abs(steer_vals))), 4),
                "speed_mean_ms": round(float(np.mean(speed_vals)), 2),
                "speed_std_ms": round(float(np.std(speed_vals)), 2),
                "speed_max_ms": round(float(np.max(speed_vals)), 2),
            },

            # Performance
            "performance": {
                "fps_mean": round(float(np.mean(fps_vals)), 2) if fps_vals else 0.0,
                "fps_min": round(float(np.min(fps_vals)), 2) if fps_vals else 0.0,
                "fps_std": round(float(np.std(fps_vals)), 2) if fps_vals else 0.0,
                "loop_time_mean_ms": round(float(np.mean(loop_times)), 2) if loop_times else 0.0,
                "loop_time_p95_ms": round(float(np.percentile(loop_times, 95)), 2) if loop_times else 0.0,
                "loop_time_max_ms": round(float(np.max(loop_times)), 2) if loop_times else 0.0,
            },

            # Safety
            "safety": {
                "safety_active_count": safety_count,
                "safety_active_rate": round(safety_count / max(1, self._total_frames), 4),
                "stuck_recovery_count": stuck_count,
                "aeb_trigger_count": aeb_count,
                "acc_active_count": acc_count,
                "ldw_warning_count": ldw_warn_count,
            },
        }

        return summary

    def save(self, prefix: str = "run") -> Dict[str, str]:
        """
        บันทึกผลเป็น CSV (per-frame) + JSON (summary)
        Returns: {"csv": path, "json": path}
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        # CSV — per-frame data
        csv_path = self.output_dir / f"{prefix}_{ts}.csv"
        if self.frames:
            fieldnames = list(asdict(self.frames[0]).keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for fm in self.frames:
                    writer.writerow(asdict(fm))
            logger.info("Metrics CSV saved: %s (%d frames)", csv_path, len(self.frames))

        # JSON — summary
        json_path = self.output_dir / f"{prefix}_{ts}_summary.json"
        summary = self.get_summary()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Metrics JSON summary saved: %s", json_path)

        return {"csv": str(csv_path), "json": str(json_path)}

    def reset(self) -> None:
        """ล้างข้อมูลทั้งหมด"""
        self.frames.clear()
        self._start_time = time.time()
        self._fallback_count = 0
        self._total_frames = 0
