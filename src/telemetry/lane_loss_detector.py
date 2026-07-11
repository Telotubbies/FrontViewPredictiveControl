"""
LaneLossDetector — ตรวจจับและบันทึก "สาเหตุ" ที่ lane detection ล้มเหลว

หลักเหตุผล: BehaviorLogger บอกได้ว่า "lane lost" เกิดขึ้น แต่ไม่บอกว่า "ทำไม"
LaneLossDetector ขยายรายละเอียด: ระบุ phase ที่ล้มเหลว (P1-P5), confidence curve
ก่อนหน้า, บริบทรถ (speed/steer/curvature), วิธี recovery, และ severity

ใช้คู่กับ BehaviorLogger:
- BehaviorLogger = high-level behavioral events (lane lost / recovered / drift)
- LaneLossDetector = root-cause analysis เฉพาะ lane loss (ทำไมหาย / กลับมายังไง)

วิธีใช้:
    detector = LaneLossDetector()
    for frame_state in pipeline:
        incident = detector.update(frame_state, context={"frame_idx": i, ...})
        if incident:
            print(incident.to_ai_text())
    summary = detector.get_summary()
    print(detector.to_ai_readable_text())
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────────────

class RootCause(str, Enum):
    """หมวดสาเหตุหลักที่ทำให้ lane detection ล้มเหลว"""
    LOW_CONFIDENCE = "low_confidence"        # P1 ล้มเหลว: confidence ตก
    BOUNDARY_LOST = "boundary_lost"          # P2 ล้มเหลว: ไม่ได้ left/right boundary
    CENTERLINE_FAILED = "centerline_failed"  # P3 ล้มเหลว: สร้าง centerline ไม่ได้
    GEOMETRY_INVALID = "geometry_invalid"    # P4 ล้มเหลว: geometry ไม่ valid
    MODE_FALLBACK = "mode_fallback"          # P5 ล้มเหลว: mode เปลี่ยนเป็น fallback
    UNKNOWN = "unknown"                      # ระบุไม่ได้


class RecoveryMethod(str, Enum):
    """วิธีที่ lane detection กลับมาทำงานอีกครั้ง"""
    CONF_RESTORED = "conf_restored"    # confidence กลับขึ้นเหนือ threshold
    PHASE_RESTORED = "phase_restored"  # phase P1-P5 กลับ ok ก่อน conf ขึ้น
    MODE_CHANGE = "mode_change"        # mode เปลี่ยน (เช่น fallback → normal)
    NOT_RECOVERED = "not_recovered"    # ยังไม่ recover ตอนจบ run


class Severity(str, Enum):
    """ระดับความรุนแรงของ lane loss incident"""
    LOW = "low"          # สั้น, CTE น้อย
    MEDIUM = "medium"    # ปานกลาง
    HIGH = "high"        # นานหรือ CTE สูง
    CRITICAL = "critical"  # นานมาก + CTE สูงมาก (เสี่ยง off-track)


# ── Dataclass: LaneLossIncident ────────────────────────────────────────────────

@dataclass
class LaneLossIncident:
    """
    1 เหตุการณ์ lane loss ตั้งแต่เริ่ม lost จนถึง recover (หรือจบ run)

    ฟิลด์ครบทั้ง "ทำไมหาย" และ "กลับมายังไง" เพื่อให้ AI วิเคราะห์ root cause ได้
    """
    # Timing
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    frame_idx_start: int = 0
    frame_idx_end: int = 0
    duration_frames: int = 0
    duration_seconds: float = 0.0

    # Root cause
    root_cause: RootCause = RootCause.UNKNOWN
    failed_phase: str = ""           # "P1" | "P2" | "P3" | "P4" | "P5" | ""
    phase_case: str = ""             # phase_p2_case / phase_p3_case ที่เกี่ยวข้อง
    root_cause_detail: str = ""      # คำอธิบายเพิ่มเติม

    # Confidence
    conf_at_loss: float = 0.0
    conf_at_recovery: float = 0.0
    conf_curve: deque = field(default_factory=lambda: deque(maxlen=30))  # conf 30 frame ก่อน loss

    # Lane state ณ จังหวะ lost
    cte_at_loss: float = 0.0
    heading_at_loss: float = 0.0
    curvature_at_loss: float = 0.0

    # Lane state ระหว่าง lost (ค่าสูงสุด/เฉลี่ย เพื่อบอกทิศทาง drift)
    cte_max_during_loss: float = 0.0
    cte_mean_during_loss: float = 0.0
    drift_direction: str = "none"    # left | right | none

    # Vehicle context ณ จังหวะ lost
    vehicle_speed_at_loss: float = 0.0   # km/h
    vehicle_pose_at_loss: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x,y,z,yaw

    # Environmental context (speed, steering, road curvature)
    environmental_context: Dict[str, Any] = field(default_factory=dict)

    # Mode tracking
    mode_before: str = ""
    mode_during: str = ""
    mode_after: str = ""

    # Recovery
    recovery_method: RecoveryMethod = RecoveryMethod.NOT_RECOVERED
    recovery_detail: str = ""

    # Severity
    severity: Severity = Severity.LOW

    # ── Helpers ────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_start": round(self.timestamp_start, 3),
            "timestamp_end": round(self.timestamp_end, 3),
            "frame_idx_start": self.frame_idx_start,
            "frame_idx_end": self.frame_idx_end,
            "duration_frames": self.duration_frames,
            "duration_seconds": round(self.duration_seconds, 3),
            "root_cause": self.root_cause.value,
            "failed_phase": self.failed_phase,
            "phase_case": self.phase_case,
            "root_cause_detail": self.root_cause_detail,
            "conf_at_loss": round(self.conf_at_loss, 4),
            "conf_at_recovery": round(self.conf_at_recovery, 4),
            "conf_curve": [round(float(c), 4) for c in self.conf_curve],
            "cte_at_loss": round(self.cte_at_loss, 4),
            "heading_at_loss": round(self.heading_at_loss, 4),
            "curvature_at_loss": round(self.curvature_at_loss, 6),
            "cte_max_during_loss": round(self.cte_max_during_loss, 4),
            "cte_mean_during_loss": round(self.cte_mean_during_loss, 4),
            "drift_direction": self.drift_direction,
            "vehicle_speed_at_loss": round(self.vehicle_speed_at_loss, 2),
            "vehicle_pose_at_loss": list(self.vehicle_pose_at_loss),
            "environmental_context": self.environmental_context,
            "mode_before": self.mode_before,
            "mode_during": self.mode_during,
            "mode_after": self.mode_after,
            "recovery_method": self.recovery_method.value,
            "recovery_detail": self.recovery_detail,
            "severity": self.severity.value,
        }

    def to_ai_text(self) -> str:
        """สรุป incident เดียวเป็น natural language สำหรับ AI"""
        cause_txt = {
            RootCause.LOW_CONFIDENCE: f"confidence dropped (P1 failed)",
            RootCause.BOUNDARY_LOST: f"boundary lost (P2 failed, case={self.phase_case})",
            RootCause.CENTERLINE_FAILED: f"centerline failed (P3 failed, case={self.phase_case})",
            RootCause.GEOMETRY_INVALID: f"geometry invalid (P4 failed)",
            RootCause.MODE_FALLBACK: f"mode fallback (P5 failed)",
            RootCause.UNKNOWN: "unknown cause",
        }.get(self.root_cause, "unknown cause")

        curve_desc = ""
        if self.curvature_at_loss != 0.0 and abs(self.curvature_at_loss) < 9999.0:
            curve_desc = " on a left curve" if self.curvature_at_loss > 0 else " on a right curve"

        drift_desc = ""
        if self.drift_direction != "none":
            drift_desc = (
                f". During loss, CTE increased to {abs(self.cte_max_during_loss):.2f}m "
                f"(drifting {self.drift_direction})"
            )

        recovery_txt = {
            RecoveryMethod.CONF_RESTORED: "confidence rose above threshold",
            RecoveryMethod.PHASE_RESTORED: "perception phases restored",
            RecoveryMethod.MODE_CHANGE: "mode changed back",
            RecoveryMethod.NOT_RECOVERED: "lane was NOT recovered by end of run",
        }.get(self.recovery_method, "recovered")

        return (
            f"Lane lost at frame {self.frame_idx_start} (t={self.timestamp_start:.1f}s): "
            f"confidence dropped from {self.conf_curve[0]:.2f if self.conf_curve else 0.0:.2f} "
            f"to {self.conf_at_loss:.2f}, root cause was '{self.root_cause.value}' "
            f"({cause_txt}). Vehicle was traveling at {self.vehicle_speed_at_loss:.0f} km/h"
            f"{curve_desc}. "
            f"Lane recovered after {self.duration_frames} frames "
            f"({self.duration_seconds:.1f}s) when {recovery_txt}{drift_desc}. "
            f"Severity: {self.severity.value}."
        )


# ── LaneLossDetector ───────────────────────────────────────────────────────────

class LaneLossDetector:
    """
    ตรวจจับ lane loss transition และวิเคราะห์ root cause.

    เรียก update() ทุก frame. คืน LaneLossIncident เมื่อ lane กลับมา recover
    (หรือเมื่อ finalize() ถ้ายัง lost อยู่ตอนจบ run).

    Args:
        conf_loss_threshold: confidence ต่ำกว่านี้ = เริ่ม lost
        conf_recover_threshold: confidence ขึ้นเหนือนี้ = recovered
        conf_curve_len: จำนวน frame ก่อนหน้าที่เก็บใน conf_curve
        severe_cte_threshold: |CTE| เกินนี้ระหว่าง loss = severity สูง
        severe_duration_frames: duration เกินนี้ = severity สูง
    """

    # Default thresholds (align กับ BehaviorLogger)
    CONF_LOSS_THRESHOLD = 0.3
    CONF_RECOVER_THRESHOLD = 0.5
    CONF_CURVE_LEN = 30
    SEVERE_CTE_THRESHOLD = 1.0
    SEVERE_DURATION_FRAMES = 30
    CRITICAL_CTE_THRESHOLD = 1.75
    CRITICAL_DURATION_FRAMES = 60

    def __init__(
        self,
        conf_loss_threshold: float = CONF_LOSS_THRESHOLD,
        conf_recover_threshold: float = CONF_RECOVER_THRESHOLD,
        conf_curve_len: int = CONF_CURVE_LEN,
        severe_cte_threshold: float = SEVERE_CTE_THRESHOLD,
        severe_duration_frames: int = SEVERE_DURATION_FRAMES,
    ) -> None:
        self.conf_loss_threshold = conf_loss_threshold
        self.conf_recover_threshold = conf_recover_threshold
        self.conf_curve_len = conf_curve_len
        self.severe_cte_threshold = severe_cte_threshold
        self.severe_duration_frames = severe_duration_frames

        # Incidents
        self._incidents: List[LaneLossIncident] = []

        # Rolling confidence history (ก่อน loss เกิด)
        self._conf_history: deque = deque(maxlen=conf_curve_len)

        # State
        self._lane_lost = False
        self._start_time = time.time()
        self._total_frames = 0
        self._prev_mode = ""
        self._prev_geom_valid = True

        # Current incident being built
        self._current: Optional[LaneLossIncident] = None
        self._cte_during_loss: List[float] = []
        self._phases_failed_at_loss: Dict[str, bool] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        frame_state: Any,
        context: Dict[str, Any],
    ) -> Optional[LaneLossIncident]:
        """
        อัปเดต detector ทุก frame.

        Args:
            frame_state: FrameState (จาก src/state.py)
            context: dict ที่มีอย่างน้อย 'frame_idx'
                (optional: 'timestamp', 'speed_ms', 'loop_time_ms')

        Returns:
            LaneLossIncident เมื่อ lane กลับมา recover ใน frame นี้, ไม่งั้น None
        """
        frame_idx = int(context.get("frame_idx", self._total_frames))
        timestamp = float(context.get("timestamp", time.time() - self._start_time))
        self._total_frames += 1

        conf = float(getattr(frame_state, "lane_conf", 0.0))
        cte = float(getattr(frame_state, "cte_m", 0.0))
        heading = float(getattr(frame_state, "heading_rad", 0.0))
        curvature = float(getattr(frame_state, "curvature", 0.0))
        mode = str(getattr(frame_state, "mode", "unknown"))
        geom_valid = bool(getattr(frame_state, "geometry_valid", False))
        speed_kmh = float(getattr(frame_state, "speed_kmh", 0.0))
        steer = float(getattr(frame_state, "final_steer", 0.0))

        # Phase flags
        p1_ok = bool(getattr(frame_state, "phase_p1_ok", False))
        p2_ok = bool(getattr(frame_state, "phase_p2_ok", False))
        p3_ok = bool(getattr(frame_state, "phase_p3_ok", False))
        p4_ok = bool(getattr(frame_state, "phase_p4_ok", False))
        p5_ok = bool(getattr(frame_state, "phase_p5_ok", False))
        p2_case = str(getattr(frame_state, "phase_p2_case", "none"))
        p3_case = str(getattr(frame_state, "phase_p3_case", "none"))

        # ── Track confidence history (ก่อน loss) ────────────────────────────
        if not self._lane_lost:
            self._conf_history.append(conf)

        # ── Detect transition INTO lane lost ─────────────────────────────────
        is_lost = self._is_lane_lost(conf, geom_valid, p1_ok, p2_ok, p3_ok, p4_ok, p5_ok)

        if is_lost and not self._lane_lost:
            # ── Lane loss START ──────────────────────────────────────────────
            self._lane_lost = True
            self._cte_during_loss = [cte]
            self._phases_failed_at_loss = {
                "P1": not p1_ok, "P2": not p2_ok, "P3": not p3_ok,
                "P4": not p4_ok, "P5": not p5_ok,
            }

            root_cause, failed_phase, phase_case, detail = self._classify_root_cause(
                conf, geom_valid, p1_ok, p2_ok, p3_ok, p4_ok, p5_ok, p2_case, p3_case, mode,
            )

            # สร้าง incident (ยังไม่ปิด)
            self._current = LaneLossIncident(
                timestamp_start=timestamp,
                frame_idx_start=frame_idx,
                root_cause=root_cause,
                failed_phase=failed_phase,
                phase_case=phase_case,
                root_cause_detail=detail,
                conf_at_loss=conf,
                conf_curve=deque(self._conf_history, maxlen=self.conf_curve_len),
                cte_at_loss=cte,
                heading_at_loss=heading,
                curvature_at_loss=curvature,
                vehicle_speed_at_loss=speed_kmh,
                vehicle_pose_at_loss=(
                    float(getattr(frame_state, "vehicle_x", 0.0)),
                    float(getattr(frame_state, "vehicle_y", 0.0)),
                    float(getattr(frame_state, "vehicle_z", 0.0)),
                    float(getattr(frame_state, "vehicle_yaw", 0.0)),
                ),
                environmental_context={
                    "speed_kmh": round(speed_kmh, 2),
                    "steering": round(steer, 4),
                    "road_curvature": round(curvature, 6),
                    "heading_rad": round(heading, 4),
                },
                mode_before=self._prev_mode if self._prev_mode else mode,
                mode_during=mode,
            )
            logger.info(
                "LaneLossDetector: lane LOST at frame %d (conf=%.3f, cause=%s, phase=%s)",
                frame_idx, conf, root_cause.value, failed_phase,
            )

        elif is_lost and self._lane_lost and self._current is not None:
            # ── Still lost: track CTE ────────────────────────────────────────
            self._cte_during_loss.append(cte)
            # อัปเดต mode_during ถ้าเปลี่ยน
            if mode != self._current.mode_during:
                self._current.mode_during = mode

        elif not is_lost and self._lane_lost and self._current is not None:
            # ── Lane RECOVERED ───────────────────────────────────────────────
            self._lane_lost = False
            incident = self._current
            self._current = None

            # ปิด incident
            incident.timestamp_end = timestamp
            incident.frame_idx_end = frame_idx
            incident.duration_frames = frame_idx - incident.frame_idx_start
            incident.duration_seconds = timestamp - incident.timestamp_start
            incident.conf_at_recovery = conf
            incident.mode_after = mode

            # CTE สถิติระหว่าง loss
            if self._cte_during_loss:
                cte_arr = np.array(self._cte_during_loss)
                incident.cte_max_during_loss = float(np.max(np.abs(cte_arr)))
                incident.cte_mean_during_loss = float(np.mean(cte_arr))
                # drift direction: CTE สูงสุด (สัญลักษณ์)
                max_signed = float(cte_arr[np.argmax(np.abs(cte_arr))])
                if abs(max_signed) > 0.3:
                    incident.drift_direction = "left" if max_signed > 0 else "right"
                else:
                    incident.drift_direction = "none"

            # Recovery method
            incident.recovery_method, incident.recovery_detail = self._classify_recovery(
                conf, geom_valid, p1_ok, p2_ok, p3_ok, p4_ok, p5_ok,
                incident.mode_before, mode, self.conf_recover_threshold,
            )

            # Severity
            incident.severity = self._classify_severity(incident)

            self._incidents.append(incident)
            self._cte_during_loss = []
            self._phases_failed_at_loss = {}

            logger.info(
                "LaneLossDetector: lane RECOVERED at frame %d (duration=%d frames, "
                "recovery=%s, severity=%s)",
                frame_idx, incident.duration_frames,
                incident.recovery_method.value, incident.severity.value,
            )
            # เคลียร์ conf_history เริ่มใหม่
            self._conf_history.clear()
            self._conf_history.append(conf)
            self._prev_mode = mode
            self._prev_geom_valid = geom_valid
            return incident

        # ── Track mode change (สำหรับ mode_before ครั้งต่อไป) ─────────────────
        if not self._lane_lost:
            self._prev_mode = mode
            self._prev_geom_valid = geom_valid

        return None

    def finalize(self) -> Optional[LaneLossIncident]:
        """
        เรียกตอนจบ run. ถ้ายัง lost อยู่ ให้ปิด incident แบบ NOT_RECOVERED.

        Returns:
            LaneLossIncident ถ้ายัง lost อยู่, ไม่งั้น None
        """
        if self._lane_lost and self._current is not None:
            incident = self._current
            self._current = None
            self._lane_lost = False
            incident.timestamp_end = time.time() - self._start_time
            incident.frame_idx_end = self._total_frames - 1
            incident.duration_frames = incident.frame_idx_end - incident.frame_idx_start
            incident.duration_seconds = incident.timestamp_end - incident.timestamp_start
            incident.conf_at_recovery = 0.0
            incident.mode_after = incident.mode_during
            incident.recovery_method = RecoveryMethod.NOT_RECOVERED
            incident.recovery_detail = "Run ended while lane was still lost"

            if self._cte_during_loss:
                cte_arr = np.array(self._cte_during_loss)
                incident.cte_max_during_loss = float(np.max(np.abs(cte_arr)))
                incident.cte_mean_during_loss = float(np.mean(cte_arr))
                max_signed = float(cte_arr[np.argmax(np.abs(cte_arr))])
                if abs(max_signed) > 0.3:
                    incident.drift_direction = "left" if max_signed > 0 else "right"

            incident.severity = self._classify_severity(incident)
            self._incidents.append(incident)
            logger.warning(
                "LaneLossDetector: run ended with lane still LOST (frame %d, duration=%d)",
                incident.frame_idx_start, incident.duration_frames,
            )
            return incident
        return None

    def get_incidents(self) -> List[LaneLossIncident]:
        """คืน list ของ LaneLossIncident ทั้งหมด"""
        return list(self._incidents)

    def get_summary(self) -> Dict[str, Any]:
        """
        สรุปสถิติ lane loss ทั้ง run.

        Returns:
            dict ที่มี: total_losses, total_frames_lost, avg_duration, max_duration,
            root_cause_distribution, most_common_cause
        """
        incidents = self._incidents
        if not incidents:
            return {
                "total_losses": 0,
                "total_frames_lost": 0,
                "avg_duration_frames": 0.0,
                "avg_duration_seconds": 0.0,
                "max_duration_frames": 0,
                "max_duration_seconds": 0.0,
                "root_cause_distribution": {},
                "most_common_cause": "none",
                "severity_distribution": {},
                "recovery_method_distribution": {},
            }

        total_frames = sum(i.duration_frames for i in incidents)
        durations = [i.duration_frames for i in incidents]
        duration_secs = [i.duration_seconds for i in incidents]

        cause_dist: Dict[str, int] = {}
        severity_dist: Dict[str, int] = {}
        recovery_dist: Dict[str, int] = {}
        for inc in incidents:
            cause_dist[inc.root_cause.value] = cause_dist.get(inc.root_cause.value, 0) + 1
            severity_dist[inc.severity.value] = severity_dist.get(inc.severity.value, 0) + 1
            recovery_dist[inc.recovery_method.value] = (
                recovery_dist.get(inc.recovery_method.value, 0) + 1
            )

        most_common = max(cause_dist, key=cause_dist.get) if cause_dist else "none"

        return {
            "total_losses": len(incidents),
            "total_frames_lost": total_frames,
            "avg_duration_frames": float(np.mean(durations)),
            "avg_duration_seconds": float(np.mean(duration_secs)),
            "max_duration_frames": int(np.max(durations)),
            "max_duration_seconds": float(np.max(duration_secs)),
            "root_cause_distribution": cause_dist,
            "most_common_cause": most_common,
            "severity_distribution": severity_dist,
            "recovery_method_distribution": recovery_dist,
        }

    def to_ai_readable_text(self) -> str:
        """
        สรุปแบบ natural language สำหรับ AI อ่านและเข้าใจ root cause.

        รูปแบบ: ภาพรวม → ราย incident → คำแนะนำ
        """
        summary = self.get_summary()
        incidents = self._incidents

        if not incidents:
            return (
                "No lane loss events detected during this run. "
                "Lane detection remained stable throughout."
            )

        lines: List[str] = []
        lines.append(
            f"Lane was lost {summary['total_losses']} time(s) during this run, "
            f"totaling {summary['total_frames_lost']} frames lost "
            f"(avg {summary['avg_duration_frames']:.1f} frames / "
            f"{summary['avg_duration_seconds']:.2f}s per loss, "
            f"max {summary['max_duration_frames']} frames / "
            f"{summary['max_duration_seconds']:.2f}s)."
        )

        cause_dist = summary["root_cause_distribution"]
        cause_str = ", ".join(f"{k}={v}" for k, v in sorted(cause_dist.items(), key=lambda x: -x[1]))
        lines.append(f"Root cause distribution: {cause_str}.")
        lines.append(
            f"Most common cause: '{summary['most_common_cause']}'."
        )

        sev = summary["severity_distribution"]
        if sev:
            sev_str = ", ".join(f"{k}={v}" for k, v in sorted(sev.items(), key=lambda x: -x[1]))
            lines.append(f"Severity distribution: {sev_str}.")

        lines.append("")
        lines.append("Incident details:")

        for idx, inc in enumerate(incidents, 1):
            lines.append(f"  [{idx}] {inc.to_ai_text()}")

        # ── คำแนะนำเบื้องต้น ────────────────────────────────────────────────
        lines.append("")
        lines.append(self._generate_recommendation(summary))

        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_lane_lost(
        self,
        conf: float,
        geom_valid: bool,
        p1_ok: bool,
        p2_ok: bool,
        p3_ok: bool,
        p4_ok: bool,
        p5_ok: bool,
    ) -> bool:
        """
        กำหนดว่า frame นี้ "lane lost" หรือไม่.

        เกณฑ์: confidence ต่ำกว่า threshold **หรือ** geometry invalid
        **หรือ** phase สำคัญ (P1-P4) ล้มเหลว
        """
        if conf < self.conf_loss_threshold:
            return True
        if not geom_valid:
            return True
        # ถ้า P1-P4 ล้มเหลวทั้งคู่ = ไม่มี lane state ใช้ได้
        if not p1_ok and not p4_ok:
            return True
        return False

    def _classify_root_cause(
        self,
        conf: float,
        geom_valid: bool,
        p1_ok: bool,
        p2_ok: bool,
        p3_ok: bool,
        p4_ok: bool,
        p5_ok: bool,
        p2_case: str,
        p3_case: str,
        mode: str,
    ) -> Tuple[RootCause, str, str, str]:
        """
        จำแนก root cause จากสถานะ phase ณ จังหวะ lost.

        ลำดับความสำคัญ: P1 → P2 → P3 → P4 → P5 (phase แรกที่ล้มเหลว = root cause)
        คืน (RootCause, failed_phase, phase_case, detail)
        """
        # P1: confidence ต่ำ = low_confidence
        if not p1_ok or conf < self.conf_loss_threshold:
            detail = f"confidence={conf:.3f} below threshold {self.conf_loss_threshold}"
            return RootCause.LOW_CONFIDENCE, "P1", "", detail

        # P2: boundary lost
        if not p2_ok:
            case = p2_case if p2_case != "none" else "no_boundary"
            detail = f"left/right boundary not detected (case={case})"
            return RootCause.BOUNDARY_LOST, "P2", case, detail

        # P3: centerline failed
        if not p3_ok:
            case = p3_case if p3_case != "none" else "no_centerline"
            detail = f"centerline construction failed (case={case})"
            return RootCause.CENTERLINE_FAILED, "P3", case, detail

        # P4: geometry invalid
        if not p4_ok or not geom_valid:
            detail = f"lane geometry invalid (geom_valid={geom_valid})"
            return RootCause.GEOMETRY_INVALID, "P4", "", detail

        # P5: mode fallback
        if not p5_ok:
            detail = f"lane phase/mode fallback (mode={mode})"
            return RootCause.MODE_FALLBACK, "P5", "", detail

        # ไม่ตรงเงื่อนไขใดชัดเจน
        return RootCause.UNKNOWN, "", "", "no specific phase failure identified"

    def _classify_recovery(
        self,
        conf: float,
        geom_valid: bool,
        p1_ok: bool,
        p2_ok: bool,
        p3_ok: bool,
        p4_ok: bool,
        p5_ok: bool,
        mode_before: str,
        mode_now: str,
        recover_threshold: float,
    ) -> Tuple[RecoveryMethod, str]:
        """
        จำแนกวิธี recovery จากสิ่งที่กลับมาก่อน.

        ลำดับ: mode_change → conf_restored → phase_restored
        """
        if mode_before and mode_now != mode_before:
            return RecoveryMethod.MODE_CHANGE, f"mode {mode_before} → {mode_now}"

        if conf >= recover_threshold:
            return RecoveryMethod.CONF_RESTORED, f"confidence rose to {conf:.3f}"

        # confidence ยังไม่ถึง threshold แต่ phase กลับ ok
        phases_ok = all([p1_ok, p2_ok, p3_ok, p4_ok, p5_ok])
        if phases_ok and geom_valid:
            return RecoveryMethod.PHASE_RESTORED, "all phases P1-P5 restored"

        return RecoveryMethod.CONF_RESTORED, f"confidence={conf:.3f} (assumed recovered)"

    def _classify_severity(self, inc: LaneLossIncident) -> Severity:
        """
        จัดระดับความรุนแรงจาก duration และ CTE ระหว่าง loss.

        - CRITICAL: duration >= CRITICAL_DURATION หรือ CTE >= CRITICAL_CTE
        - HIGH: duration >= SEVERE_DURATION หรือ CTE >= SEVERE_CTE
        - MEDIUM: duration >= SEVERE_DURATION/2 หรือ CTE >= SEVERE_CTE/2
        - LOW: อื่นๆ
        """
        dur = inc.duration_frames
        cte_max = abs(inc.cte_max_during_loss)

        if dur >= self.CRITICAL_DURATION_FRAMES or cte_max >= self.CRITICAL_CTE_THRESHOLD:
            return Severity.CRITICAL
        if dur >= self.severe_duration_frames or cte_max >= self.severe_cte_threshold:
            return Severity.HIGH
        if dur >= self.severe_duration_frames // 2 or cte_max >= self.severe_cte_threshold / 2:
            return Severity.MEDIUM
        return Severity.LOW

    def _generate_recommendation(self, summary: Dict[str, Any]) -> str:
        """สร้างคำแนะนำเบื้องต้นจาก root cause ที่พบบ่อยที่สุด"""
        most_common = summary.get("most_common_cause", "none")
        total = summary.get("total_losses", 0)

        if total == 0:
            return "No recommendations: lane detection was stable."

        rec_map = {
            RootCause.LOW_CONFIDENCE.value: (
                "Recommendation: Most losses were caused by low confidence (P1). "
                "Consider improving lane mask segmentation, adjusting confidence "
                "threshold, or adding temporal smoothing for confidence."
            ),
            RootCause.BOUNDARY_LOST.value: (
                "Recommendation: Most losses were caused by boundary detection failure (P2). "
                "Check sliding-window parameters, boundary fitting, and mirror/completion logic."
            ),
            RootCause.CENTERLINE_FAILED.value: (
                "Recommendation: Most losses were caused by centerline construction failure (P3). "
                "Review centerline derivation from boundaries and completion fallback."
            ),
            RootCause.GEOMETRY_INVALID.value: (
                "Recommendation: Most losses were caused by invalid geometry (P4). "
                "Validate CTE/heading/curvature computation and geometry sanity checks."
            ),
            RootCause.MODE_FALLBACK.value: (
                "Recommendation: Most losses were caused by mode fallback (P5). "
                "Review lane phase/mode state machine and fallback triggers."
            ),
            RootCause.UNKNOWN.value: (
                "Recommendation: Root cause could not be clearly identified. "
                "Enable more detailed phase logging to diagnose future losses."
            ),
        }
        return rec_map.get(most_common, "Review lane detection pipeline for stability.")

    def reset(self) -> None:
        """รีเซ็ต detector สำหรับ run ใหม่"""
        self._incidents.clear()
        self._conf_history.clear()
        self._lane_lost = False
        self._current = None
        self._cte_during_loss = []
        self._phases_failed_at_loss = {}
        self._total_frames = 0
        self._prev_mode = ""
        self._prev_geom_valid = True
        self._start_time = time.time()
        logger.debug("LaneLossDetector: reset")
