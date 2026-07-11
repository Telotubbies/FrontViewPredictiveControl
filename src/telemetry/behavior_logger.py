"""
BehaviorLogger — บันทึกพฤติกรรมรถและการล้มเหลวของ lane detection

ติดตาม:
- Lane detection failures (confidence drop, phase failures, geometry invalid)
- Behavior patterns (drifting, oscillation, off-track, sudden braking)
- Event snapshots (บันทึก metadata ของ critical events เพื่อวิเคราะห์ภายหลัง)
- Behavior summary (สรุปพฤติกรรมทั้ง run ให้ AI อ่านและเข้าใจ)

ใช้คู่กับ RunLogger แต่เพิ่มระดับ abstraction:
- RunLogger = low-level per-frame CSV
- BehaviorLogger = high-level behavioral events + pattern detection
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BehaviorType(str, Enum):
    """ประเภทของ behavior event"""
    LANE_CONF_DROP = "lane_conf_drop"
    LANE_LOST = "lane_lost"
    LANE_RECOVERED = "lane_recovered"
    PHASE_FAILURE = "phase_failure"
    GEOMETRY_INVALID = "geometry_invalid"
    DRIFTING_LEFT = "drifting_left"
    DRIFTING_RIGHT = "drifting_right"
    OFF_TRACK = "off_track"
    RETURNED_TO_LANE = "returned_to_lane"
    OSCILLATION = "oscillation"
    SUDDEN_BRAKE = "sudden_brake"
    EXCESSIVE_STEERING = "excessive_steering"
    MPC_UNSTABLE = "mpc_unstable"
    PERCEPTION_MODE_CHANGE = "perception_mode_change"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class BehaviorEvent:
    """1 behavior event"""
    timestamp: float
    frame_idx: int
    event_type: BehaviorType
    severity: Severity
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "frame_idx": self.frame_idx,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class BehaviorSummary:
    """สรุปพฤติกรรมทั้ง run"""
    total_events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)

    # Lane detection health
    lane_conf_mean: float = 0.0
    lane_conf_min: float = 0.0
    lane_conf_drop_count: int = 0
    lane_lost_count: int = 0
    lane_lost_total_frames: int = 0
    lane_lost_max_duration_frames: int = 0
    geometry_invalid_count: int = 0
    phase_failure_counts: Dict[str, int] = field(default_factory=dict)

    # Behavior patterns
    drift_left_count: int = 0
    drift_right_count: int = 0
    off_track_count: int = 0
    off_track_total_frames: int = 0
    oscillation_count: int = 0
    sudden_brake_count: int = 0
    excessive_steering_count: int = 0
    mpc_unstable_count: int = 0
    mode_change_count: int = 0

    # Assessment
    overall_lane_keeping_quality: str = "unknown"  # excellent | good | fair | poor | critical
    dominant_failure_mode: str = "none"
    ai_readable_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "events_by_type": self.events_by_type,
            "events_by_severity": self.events_by_severity,
            "lane_conf_mean": round(self.lane_conf_mean, 4),
            "lane_conf_min": round(self.lane_conf_min, 4),
            "lane_conf_drop_count": self.lane_conf_drop_count,
            "lane_lost_count": self.lane_lost_count,
            "lane_lost_total_frames": self.lane_lost_total_frames,
            "lane_lost_max_duration_frames": self.lane_lost_max_duration_frames,
            "geometry_invalid_count": self.geometry_invalid_count,
            "phase_failure_counts": self.phase_failure_counts,
            "drift_left_count": self.drift_left_count,
            "drift_right_count": self.drift_right_count,
            "off_track_count": self.off_track_count,
            "off_track_total_frames": self.off_track_total_frames,
            "oscillation_count": self.oscillation_count,
            "sudden_brake_count": self.sudden_brake_count,
            "excessive_steering_count": self.excessive_steering_count,
            "mpc_unstable_count": self.mpc_unstable_count,
            "mode_change_count": self.mode_change_count,
            "overall_lane_keeping_quality": self.overall_lane_keeping_quality,
            "dominant_failure_mode": self.dominant_failure_mode,
            "ai_readable_summary": self.ai_readable_summary,
        }


class BehaviorLogger:
    """
    ติดตามและบันทึกพฤติกรรมรถแบบ real-time.

    เรียก update() ทุก frame พร้อม FrameState + context.
    ตรวจจับ patterns อัตโนมัติและสร้าง BehaviorEvent.
    เรียก get_summary() เมื่อจบ run เพื่อสรุปพฤติกรรม.
    """

    # Thresholds
    CONF_DROP_THRESHOLD = 0.3       # confidence ต่ำกว่านี้ = drop
    CONF_RECOVER_THRESHOLD = 0.5    # confidence ขึ้นเหนือนี้ = recovered
    DRIFT_CTE_THRESHOLD = 0.8       # |CTE| เกินนี้ = drifting
    OFF_TRACK_CTE_THRESHOLD = 1.75  # |CTE| เกินนี้ = off-track
    OSCILLATION_WINDOW = 30         # frames สำหรับตรวจ oscillation
    OSCILLATION_STEER_THRESHOLD = 0.15  # std ของ steer เกินนี้ = oscillation
    SUDDEN_BRAKE_THRESHOLD = 0.5    # brake เพิ่มขึ้นเกินนี้ใน 1 frame
    EXCESSIVE_STEER_THRESHOLD = 0.35  # |steer| เกินนี้
    MPC_UNSTABLE_CONSECUTIVE = 5    # fallback ติดต่อกันเกินนี้ = unstable

    def __init__(self, snapshot_dir: Optional[str] = None):
        """
        Args:
            snapshot_dir: directory สำหรับบันทึก snapshot metadata (ไม่เก็บรูป)
        """
        self.events: List[BehaviorEvent] = []
        self._start_time = time.time()
        self._snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self._snapshot_dir:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Rolling windows for pattern detection
        self._conf_history: deque = deque(maxlen=100)
        self._cte_history: deque = deque(maxlen=100)
        self._steer_history: deque = deque(maxlen=self.OSCILLATION_WINDOW)
        self._brake_history: deque = deque(maxlen=5)

        # State tracking
        self._prev_conf = 1.0
        self._prev_brake = 0.0
        self._prev_mode = ""
        self._lane_lost = False
        self._lane_lost_start_frame = 0
        self._off_track = False
        self._off_track_start_frame = 0
        self._fallback_consecutive = 0
        self._total_frames = 0

        # Cumulative stats
        self._conf_sum = 0.0
        self._conf_min = 1.0
        self._phase_failures: Dict[str, int] = {}
        self._lane_lost_total = 0
        self._lane_lost_max = 0
        self._off_track_total = 0

    def update(self, frame_state, context: Dict[str, Any]) -> None:
        """
        อัปเดต behavior tracking ทุก frame.

        Args:
            frame_state: FrameState
            context: dict with 'frame_idx', 'speed_ms', 'loop_time_ms', etc.
        """
        frame_idx = context.get("frame_idx", self._total_frames)
        timestamp = time.time() - self._start_time
        self._total_frames += 1

        conf = float(getattr(frame_state, "lane_conf", 0.0))
        cte = float(getattr(frame_state, "cte_m", 0.0))
        steer = float(getattr(frame_state, "final_steer", 0.0))
        brake = float(getattr(frame_state, "final_brake", 0.0))
        mode = str(getattr(frame_state, "mode", "unknown"))
        geom_valid = bool(getattr(frame_state, "geometry_valid", False))
        solver_status = str(getattr(frame_state, "solver_status", ""))

        # Update rolling windows
        self._conf_history.append(conf)
        self._cte_history.append(cte)
        self._steer_history.append(steer)
        self._brake_history.append(brake)

        # Update cumulative stats
        self._conf_sum += conf
        self._conf_min = min(self._conf_min, conf)

        # ── Detect: Lane confidence drop ──────────────────────────────
        if conf < self.CONF_DROP_THRESHOLD and self._prev_conf >= self.CONF_DROP_THRESHOLD:
            self._log_event(
                frame_idx, timestamp, BehaviorType.LANE_CONF_DROP, Severity.WARNING,
                f"Lane confidence dropped to {conf:.2f} (was {self._prev_conf:.2f})",
                {"conf": conf, "prev_conf": self._prev_conf, "cte_m": cte},
            )

        # ── Detect: Lane lost (consecutive low confidence) ────────────
        if conf < self.CONF_DROP_THRESHOLD:
            if not self._lane_lost:
                self._lane_lost = True
                self._lane_lost_start_frame = frame_idx
                self._log_event(
                    frame_idx, timestamp, BehaviorType.LANE_LOST, Severity.ERROR,
                    f"Lane detection lost (conf={conf:.2f})",
                    {"conf": conf, "cte_m": cte, "mode": mode},
                )
        else:
            if self._lane_lost:
                self._lane_lost = False
                duration = frame_idx - self._lane_lost_start_frame
                self._lane_lost_total += duration
                self._lane_lost_max = max(self._lane_lost_max, duration)
                self._log_event(
                    frame_idx, timestamp, BehaviorType.LANE_RECOVERED, Severity.INFO,
                    f"Lane detection recovered after {duration} frames",
                    {"conf": conf, "lost_duration_frames": duration},
                )

        # ── Detect: Phase failures ────────────────────────────────────
        for phase_name, attr in [("P1", "phase_p1_ok"), ("P2", "phase_p2_ok"),
                                  ("P3", "phase_p3_ok"), ("P4", "phase_p4_ok"),
                                  ("P5", "phase_p5_ok")]:
            phase_ok = bool(getattr(frame_state, attr, False))
            if not phase_ok:
                self._phase_failures[phase_name] = self._phase_failures.get(phase_name, 0) + 1

        # Log phase failure if all phases were previously ok and now some fail
        if conf < self.CONF_DROP_THRESHOLD and not geom_valid:
            failed_phases = [p for p, a in [("P1", "phase_p1_ok"), ("P2", "phase_p2_ok"),
                                             ("P3", "phase_p3_ok"), ("P4", "phase_p4_ok"),
                                             ("P5", "phase_p5_ok")]
                             if not bool(getattr(frame_state, a, False))]
            if failed_phases:
                self._log_event(
                    frame_idx, timestamp, BehaviorType.PHASE_FAILURE, Severity.WARNING,
                    f"Perception phases failed: {', '.join(failed_phases)}",
                    {"failed_phases": failed_phases, "conf": conf},
                )

        # ── Detect: Geometry invalid ──────────────────────────────────
        if not geom_valid and self._total_frames > 1:
            # Only log if geometry was valid before
            if self._total_frames == 2 or self._was_geom_valid:
                self._log_event(
                    frame_idx, timestamp, BehaviorType.GEOMETRY_INVALID, Severity.WARNING,
                    f"Lane geometry invalid (conf={conf:.2f})",
                    {"conf": conf, "cte_m": cte},
                )
        self._was_geom_valid = geom_valid

        # ── Detect: Drifting ──────────────────────────────────────────
        if abs(cte) > self.DRIFT_CTE_THRESHOLD and abs(cte) <= self.OFF_TRACK_CTE_THRESHOLD:
            drift_dir = "left" if cte > 0 else "right"
            # Only log on transition into drift zone
            if abs(self._cte_history[-2]) <= self.DRIFT_CTE_THRESHOLD if len(self._cte_history) > 1 else True:
                btype = BehaviorType.DRIFTING_LEFT if cte > 0 else BehaviorType.DRIFTING_RIGHT
                self._log_event(
                    frame_idx, timestamp, btype, Severity.WARNING,
                    f"Vehicle drifting {drift_dir} (CTE={cte:+.2f}m)",
                    {"cte_m": cte, "conf": conf},
                )

        # ── Detect: Off-track ─────────────────────────────────────────
        if abs(cte) > self.OFF_TRACK_CTE_THRESHOLD:
            if not self._off_track:
                self._off_track = True
                self._off_track_start_frame = frame_idx
                self._log_event(
                    frame_idx, timestamp, BehaviorType.OFF_TRACK, Severity.CRITICAL,
                    f"Vehicle OFF TRACK (CTE={cte:+.2f}m, conf={conf:.2f})",
                    {"cte_m": cte, "conf": conf, "mode": mode},
                )
        else:
            if self._off_track:
                self._off_track = False
                duration = frame_idx - self._off_track_start_frame
                self._off_track_total += duration
                self._log_event(
                    frame_idx, timestamp, BehaviorType.RETURNED_TO_LANE, Severity.INFO,
                    f"Vehicle returned to lane after {duration} frames off-track",
                    {"cte_m": cte, "off_track_duration_frames": duration},
                )

        # ── Detect: Oscillation ───────────────────────────────────────
        if len(self._steer_history) >= self.OSCILLATION_WINDOW:
            steer_arr = np.array(self._steer_history)
            steer_std = float(np.std(steer_arr))
            if steer_std > self.OSCILLATION_STEER_THRESHOLD:
                # Only log once per oscillation episode
                if not getattr(self, "_oscillating", False):
                    self._oscillating = True
                    self._log_event(
                        frame_idx, timestamp, BehaviorType.OSCILLATION, Severity.WARNING,
                        f"Steering oscillation detected (std={steer_std:.3f})",
                        {"steer_std": steer_std, "cte_m": cte},
                    )
            else:
                self._oscillating = False

        # ── Detect: Sudden brake ──────────────────────────────────────
        if len(self._brake_history) >= 2:
            brake_delta = brake - self._brake_history[-2]
            if brake_delta > self.SUDDEN_BRAKE_THRESHOLD and brake > 0.3:
                self._log_event(
                    frame_idx, timestamp, BehaviorType.SUDDEN_BRAKE, Severity.WARNING,
                    f"Sudden braking (brake={brake:.2f}, delta={brake_delta:+.2f})",
                    {"brake": brake, "brake_delta": brake_delta, "speed_ms": context.get("speed_ms", 0)},
                )

        # ── Detect: Excessive steering ────────────────────────────────
        if abs(steer) > self.EXCESSIVE_STEER_THRESHOLD:
            if not getattr(self, "_excessive_steer_logged", False):
                self._excessive_steer_logged = True
                self._log_event(
                    frame_idx, timestamp, BehaviorType.EXCESSIVE_STEERING, Severity.WARNING,
                    f"Excessive steering (steer={steer:+.3f})",
                    {"steer": steer, "cte_m": cte, "conf": conf},
                )
        else:
            self._excessive_steer_logged = False

        # ── Detect: MPC unstable (consecutive fallbacks) ──────────────
        if "Fallback" in solver_status:
            self._fallback_consecutive += 1
            if self._fallback_consecutive == self.MPC_UNSTABLE_CONSECUTIVE:
                self._log_event(
                    frame_idx, timestamp, BehaviorType.MPC_UNSTABLE, Severity.ERROR,
                    f"MPC unstable — {self._fallback_consecutive} consecutive fallbacks",
                    {"consecutive_fallbacks": self._fallback_consecutive, "cte_m": cte},
                )
        else:
            self._fallback_consecutive = 0

        # ── Detect: Perception mode change ────────────────────────────
        if self._prev_mode and mode != self._prev_mode:
            self._log_event(
                frame_idx, timestamp, BehaviorType.PERCEPTION_MODE_CHANGE, Severity.INFO,
                f"Perception mode changed: {self._prev_mode} → {mode}",
                {"prev_mode": self._prev_mode, "new_mode": mode, "conf": conf},
            )
        self._prev_mode = mode

        # Update prev values
        self._prev_conf = conf
        self._prev_brake = brake

    def _log_event(
        self,
        frame_idx: int,
        timestamp: float,
        event_type: BehaviorType,
        severity: Severity,
        message: str,
        context: Dict[str, Any],
    ) -> None:
        """Create and store a behavior event."""
        event = BehaviorEvent(
            timestamp=timestamp,
            frame_idx=frame_idx,
            event_type=event_type,
            severity=severity,
            message=message,
            context=context,
        )
        self.events.append(event)
        logger.info(
            "Behavior [%s] frame=%d %s: %s",
            severity.value, frame_idx, event_type.value, message,
        )

    def get_summary(self) -> BehaviorSummary:
        """สรุปพฤติกรรมทั้ง run"""
        summary = BehaviorSummary()
        summary.total_events = len(self.events)

        # Events by type/severity
        for ev in self.events:
            t = ev.event_type.value
            summary.events_by_type[t] = summary.events_by_type.get(t, 0) + 1
            s = ev.severity.value
            summary.events_by_severity[s] = summary.events_by_severity.get(s, 0) + 1

        # Lane detection health
        if self._total_frames > 0:
            summary.lane_conf_mean = self._conf_sum / self._total_frames
        summary.lane_conf_min = self._conf_min
        summary.lane_conf_drop_count = summary.events_by_type.get("lane_conf_drop", 0)
        summary.lane_lost_count = summary.events_by_type.get("lane_lost", 0)
        summary.lane_lost_total_frames = self._lane_lost_total
        summary.lane_lost_max_duration_frames = self._lane_lost_max
        summary.geometry_invalid_count = summary.events_by_type.get("geometry_invalid", 0)
        summary.phase_failure_counts = dict(self._phase_failures)

        # Behavior patterns
        summary.drift_left_count = summary.events_by_type.get("drifting_left", 0)
        summary.drift_right_count = summary.events_by_type.get("drifting_right", 0)
        summary.off_track_count = summary.events_by_type.get("off_track", 0)
        summary.off_track_total_frames = self._off_track_total
        summary.oscillation_count = summary.events_by_type.get("oscillation", 0)
        summary.sudden_brake_count = summary.events_by_type.get("sudden_brake", 0)
        summary.excessive_steering_count = summary.events_by_type.get("excessive_steering", 0)
        summary.mpc_unstable_count = summary.events_by_type.get("mpc_unstable", 0)
        summary.mode_change_count = summary.events_by_type.get("perception_mode_change", 0)

        # Assessment
        summary.overall_lane_keeping_quality = self._assess_quality(summary)
        summary.dominant_failure_mode = self._identify_dominant_failure(summary)
        summary.ai_readable_summary = self._generate_ai_summary(summary)

        return summary

    def _assess_quality(self, s: BehaviorSummary) -> str:
        """ประเมินคุณภาพ lane keeping โดยรวม"""
        if s.off_track_count > 3:
            return "critical"
        if s.off_track_count > 0 or s.lane_lost_count > 5:
            return "poor"
        if s.lane_lost_count > 2 or s.lane_conf_mean < 0.5:
            return "fair"
        if s.lane_conf_mean > 0.7 and s.off_track_count == 0:
            return "excellent"
        return "good"

    def _identify_dominant_failure(self, s: BehaviorSummary) -> str:
        """ระบุ failure mode ที่เด่นที่สุด"""
        candidates = {
            "lane_conf_drop": s.lane_conf_drop_count,
            "lane_lost": s.lane_lost_count,
            "geometry_invalid": s.geometry_invalid_count,
            "off_track": s.off_track_count,
            "oscillation": s.oscillation_count,
            "mpc_unstable": s.mpc_unstable_count,
            "drifting": s.drift_left_count + s.drift_right_count,
        }
        dominant = max(candidates, key=candidates.get)
        if candidates[dominant] > 0:
            return dominant
        return "none"

    def _generate_ai_summary(self, s: BehaviorSummary) -> str:
        """สร้างสรุปแบบ readable สำหรับ AI วิเคราะห์"""
        lines = []
        lines.append(f"Run quality: {s.overall_lane_keeping_quality}")
        lines.append(f"Total frames: {self._total_frames}, events: {s.total_events}")
        lines.append(f"Lane confidence: mean={s.lane_conf_mean:.2f}, min={s.lane_conf_min:.2f}")

        if s.lane_lost_count > 0:
            lines.append(
                f"Lane detection lost {s.lane_lost_count} times, "
                f"total {s.lane_lost_total_frames} frames, "
                f"max duration {s.lane_lost_max_duration_frames} frames"
            )
        if s.off_track_count > 0:
            lines.append(
                f"Vehicle went off-track {s.off_track_count} times, "
                f"total {s.off_track_total_frames} frames"
            )
        if s.drift_left_count or s.drift_right_count:
            lines.append(
                f"Drifting: left={s.drift_left_count}, right={s.drift_right_count}"
            )
        if s.oscillation_count:
            lines.append(f"Steering oscillation: {s.oscillation_count} episodes")
        if s.mpc_unstable_count:
            lines.append(f"MPC unstable: {s.mpc_unstable_count} episodes")
        if s.sudden_brake_count:
            lines.append(f"Sudden braking: {s.sudden_brake_count} events")
        if s.phase_failure_counts:
            top_phase = max(s.phase_failure_counts, key=s.phase_failure_counts.get)
            lines.append(
                f"Phase failures: {s.phase_failure_counts} "
                f"(most failed: {top_phase})"
            )
        if s.dominant_failure_mode != "none":
            lines.append(f"Dominant failure mode: {s.dominant_failure_mode}")

        if s.overall_lane_keeping_quality == "excellent":
            lines.append("Assessment: Vehicle maintained lane consistently with high confidence.")
        elif s.overall_lane_keeping_quality == "good":
            lines.append("Assessment: Vehicle kept lane well with minor confidence drops.")
        elif s.overall_lane_keeping_quality == "fair":
            lines.append("Assessment: Vehicle struggled with lane detection but mostly recovered.")
        elif s.overall_lane_keeping_quality == "poor":
            lines.append("Assessment: Vehicle frequently lost lane detection or drifted significantly.")
        else:
            lines.append("Assessment: Vehicle went off-track multiple times — lane keeping failed critically.")

        return " | ".join(lines)

    def save(self, run_dir: Optional[str] = None) -> Dict[str, str]:
        """
        บันทึก behavior events + summary เป็น JSON.

        Returns:
            {"events": path, "summary": path}
        """
        import json

        out_dir = Path(run_dir) if run_dir else (self._snapshot_dir or Path("."))
        out_dir.mkdir(parents=True, exist_ok=True)

        events_path = out_dir / "behavior_events.json"
        summary_path = out_dir / "behavior_summary.json"

        # Save events
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump([ev.to_dict() for ev in self.events], f, indent=2, default=str)
        logger.info("Behavior events saved: %s (%d events)", events_path, len(self.events))

        # Save summary
        summary = self.get_summary()
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        logger.info("Behavior summary saved: %s", summary_path)

        return {"events": str(events_path), "summary": str(summary_path)}

    def reset(self) -> None:
        """ล้างข้อมูลทั้งหมด"""
        self.events.clear()
        self._conf_history.clear()
        self._cte_history.clear()
        self._steer_history.clear()
        self._brake_history.clear()
        self._prev_conf = 1.0
        self._prev_brake = 0.0
        self._prev_mode = ""
        self._lane_lost = False
        self._off_track = False
        self._fallback_consecutive = 0
        self._total_frames = 0
        self._conf_sum = 0.0
        self._conf_min = 1.0
        self._phase_failures.clear()
        self._lane_lost_total = 0
        self._lane_lost_max = 0
        self._off_track_total = 0
        self._start_time = time.time()
