"""
BehaviorContextLogger — Black box recorder สำหรับ behavior events

ทำหน้าที่เหมือน flight data recorder: เก็บ context รอบๆ behavior event
เพื่อให้ AI สามารถ reconstruct สิ่งที่เกิดขึ้นก่อน/ระหว่าง/หลัง event ได้

โครงสร้าง:
- รักษา rolling window ของ FrameState ย้อนหลัง N frames (default 60)
- เมื่อมี behavior event เกิดขึ้น จะ capture:
  * 30 frames ก่อน event (pre-event context)
  * frame ณ จุดเกิด event
  * 15 frames หลัง event (post-event context)
- สร้าง BehaviorContextSnapshot ที่มีข้อมูลครบสำหรับวิเคราะห์
- สร้าง AI-readable narrative ของแต่ละ event
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from telemetry.behavior_logger import BehaviorEvent, BehaviorType, Severity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — แปลง FrameState เป็น compact dict
# ─────────────────────────────────────────────────────────────────────────────

_COMPACT_FIELDS = (
    "frame_idx",
    "speed_kmh",
    "cte_m",
    "lane_conf",
    "steer",
    "throttle",
    "brake",
    "mode",
    "geometry_valid",
    "phase_p1_ok",
    "phase_p2_ok",
    "phase_p3_ok",
    "phase_p4_ok",
    "phase_p5_ok",
    "final_steer",
    "final_throttle",
    "final_brake",
    "vehicle_x",
    "vehicle_y",
    "vehicle_yaw",
    "heading_rad",
    "curvature",
    "aeb_active",
    "acc_active",
    "ldw_state",
    "lka_pro_assist",
    "mpc_solve_time_ms",
    "solver_status",
)


def _frame_to_compact_dict(frame_state: Any, frame_idx: int) -> Dict[str, Any]:
    """แปลง FrameState เป็น compact dict เก็บเฉพาะฟิลด์สำคัญ"""
    d: Dict[str, Any] = {"frame_idx": frame_idx}
    for f in _COMPACT_FIELDS:
        if f == "frame_idx":
            continue
        val = getattr(frame_state, f, None)
        # numpy scalar → python native
        if isinstance(val, (np.floating, np.integer)):
            val = val.item()
        elif isinstance(val, np.bool_):
            val = bool(val)
        d[f] = val
    return d


# ─────────────────────────────────────────────────────────────────────────────
# BehaviorContextSnapshot — ข้อมูล context ของหนึ่ง behavior event
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BehaviorContextSnapshot:
    """
    Snapshot ของ behavior event พร้อม surrounding context.

    เก็บ pre/post frames, peak metrics, trajectory, และ recovery info
    เพื่อให้ AI สามารถเข้าใจเหตุการณ์ได้ครบถ้วน
    """
    event: BehaviorEvent
    pre_frames: List[Dict[str, Any]] = field(default_factory=list)
    post_frames: List[Dict[str, Any]] = field(default_factory=list)

    # Peak/extreme metrics ระหว่าง event
    peak_cte_during_event: float = 0.0
    min_conf_during_event: float = 1.0
    max_steer_during_event: float = 0.0
    max_brake_during_event: float = 0.0

    # Spatial trajectory สำหรับ visualization
    vehicle_trajectory_during_event: List[Tuple[float, float]] = field(default_factory=list)

    # Duration
    duration_frames: int = 0

    # Recovery info
    recovered: bool = False
    recovery_frame_idx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "pre_frames": self.pre_frames,
            "post_frames": self.post_frames,
            "peak_cte_during_event": round(self.peak_cte_during_event, 4),
            "min_conf_during_event": round(self.min_conf_during_event, 4),
            "max_steer_during_event": round(self.max_steer_during_event, 4),
            "max_brake_during_event": round(self.max_brake_during_event, 4),
            "vehicle_trajectory_during_event": [
                [round(x, 3), round(y, 3)] for x, y in self.vehicle_trajectory_during_event
            ],
            "duration_frames": self.duration_frames,
            "recovered": self.recovered,
            "recovery_frame_idx": self.recovery_frame_idx,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PendingSnapshot — tracking snapshot ที่กำลังเก็บ post-event frames
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _PendingSnapshot:
    """Internal: snapshot ที่กำลังรอเก็บ post-event frames"""
    snapshot: BehaviorContextSnapshot
    frames_remaining: int
    event_frame_idx: int
    # ใช้สำหรับตรวจ recovery
    event_type: BehaviorType
    # threshold สำหรับตรวจ recovery ตามประเภท event
    recovery_check: Optional[str] = None  # "conf" | "cte" | "brake" | "steer" | "solver"


# ─────────────────────────────────────────────────────────────────────────────
# BehaviorContextLogger — Black box recorder
# ─────────────────────────────────────────────────────────────────────────────

class BehaviorContextLogger:
    """
    Black box recorder สำหรับ behavior events.

    ใช้คู่กับ BehaviorLogger:
    - BehaviorLogger: ตรวจจับ pattern และสร้าง events
    - BehaviorContextLogger: เก็บ surrounding context ของแต่ละ event

    Usage:
        ctx_logger = BehaviorContextLogger()

        # ทุก frame
        ctx_logger.update(frame_state, frame_idx)

        # เมื่อ BehaviorLogger สร้าง event (หรือเรียกเอง)
        snapshot = ctx_logger.record_event(
            event_type=BehaviorType.LANE_LOST,
            severity=Severity.ERROR,
            message="Lane detection lost",
            frame_state=frame_state,
        )

        # สร้าง AI-readable text
        narrative = ctx_logger.to_ai_readable_text()
    """

    def __init__(
        self,
        rolling_window: int = 60,
        pre_event_frames: int = 30,
        post_event_frames: int = 15,
        estimated_fps: float = 15.0,
    ):
        """
        Args:
            rolling_window: จำนวน frames ล่าสุดที่เก็บใน buffer (default 60 = ~3-6s)
            pre_event_frames: จำนวน frames ก่อน event ที่จะ capture (default 30)
            post_event_frames: จำนวน frames หลัง event ที่จะ capture (default 15)
            estimated_fps: ประมาณ fps สำหรับคำนวณ duration ในหน่วยวินาที
        """
        self.rolling_window = rolling_window
        self.pre_event_frames = pre_event_frames
        self.post_event_frames = post_event_frames
        self.estimated_fps = estimated_fps

        # Rolling buffer ของ (frame_idx, compact_dict, frame_state)
        self._buffer: deque = deque(maxlen=rolling_window)

        # Snapshots ที่เสร็จแล้ว
        self._snapshots: List[BehaviorContextSnapshot] = []

        # Snapshots ที่กำลังรอ post-event frames
        self._pending: List[_PendingSnapshot] = []

        # Start time สำหรับคำนวณ timestamp
        self._start_time = time.time()

        # Frame counter
        self._total_frames = 0

        logger.debug(
            "BehaviorContextLogger initialized (window=%d, pre=%d, post=%d, fps=%.1f)",
            rolling_window, pre_event_frames, post_event_frames, estimated_fps,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, frame_state: Any, frame_idx: Optional[int] = None) -> None:
        """
        อัปเดต rolling buffer ทุก frame — เรียกทุก frame.

        Args:
            frame_state: FrameState ปัจจุบัน
            frame_idx: frame index (ถ้าไม่ใส่จะใช้ counter ภายใน)
        """
        if frame_idx is None:
            frame_idx = self._total_frames
        self._total_frames += 1

        compact = _frame_to_compact_dict(frame_state, frame_idx)
        self._buffer.append((frame_idx, compact, frame_state))

        # อัปเดต pending snapshots (เก็บ post-event frames)
        self._update_pending(frame_idx, frame_state)

    def record_event(
        self,
        event_type: BehaviorType,
        severity: Severity,
        message: str,
        frame_state: Any,
        extra_context: Optional[Dict[str, Any]] = None,
        frame_idx: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> BehaviorContextSnapshot:
        """
        บันทึก behavior event พร้อม surrounding context.

        Args:
            event_type: ประเภทของ event
            severity: ระดับความรุนแรง
            message: ข้อความอธิบาย event
            frame_state: FrameState ณ จุดเกิด event
            extra_context: context เพิ่มเติม (optional)
            frame_idx: frame index (ถ้าไม่ใส่จะ infer จาก buffer)
            timestamp: timestamp (ถ้าไม่ใส่จะใช้ elapsed time)

        Returns:
            BehaviorContextSnapshot ที่สร้างขึ้น (post_frames จะยังไม่ครบ
            จนกว่าจะเรียก update() ครบ post_event_frames ครั้ง)
        """
        # หา frame_idx
        if frame_idx is None:
            if self._buffer:
                frame_idx = self._buffer[-1][0]
            else:
                frame_idx = self._total_frames

        if timestamp is None:
            timestamp = time.time() - self._start_time

        # สร้าง BehaviorEvent
        context: Dict[str, Any] = {}
        if extra_context:
            context.update(extra_context)

        event = BehaviorEvent(
            timestamp=timestamp,
            frame_idx=frame_idx,
            event_type=event_type,
            severity=severity,
            message=message,
            context=context,
        )

        # Capture pre-event frames จาก rolling buffer
        pre_frames = self._extract_pre_frames(frame_idx)

        # Frame ณ จุด event
        event_compact = _frame_to_compact_dict(frame_state, frame_idx)

        # คำนวณ peak metrics จาก pre + event frame
        all_so_far = pre_frames + [event_compact]
        peak_cte = max(abs(float(f.get("cte_m", 0) or 0)) for f in all_so_far)
        min_conf = min(
            (float(f.get("lane_conf", 1.0) or 1.0) for f in all_so_far),
            default=1.0,
        )
        max_steer = max(abs(float(f.get("final_steer", 0) or 0)) for f in all_so_far)
        max_brake = max(abs(float(f.get("final_brake", 0) or 0)) for f in all_so_far)

        # Trajectory จาก pre + event
        trajectory = self._extract_trajectory(pre_frames + [event_compact])

        # สร้าง snapshot
        snapshot = BehaviorContextSnapshot(
            event=event,
            pre_frames=pre_frames,
            post_frames=[],
            peak_cte_during_event=peak_cte,
            min_conf_during_event=min_conf,
            max_steer_during_event=max_steer,
            max_brake_during_event=max_brake,
            vehicle_trajectory_during_event=trajectory,
            duration_frames=1,
            recovered=False,
            recovery_frame_idx=None,
        )

        # กำหนด recovery check ตามประเภท event
        recovery_check = self._get_recovery_check(event_type)

        # สร้าง pending เพื่อเก็บ post-event frames
        pending = _PendingSnapshot(
            snapshot=snapshot,
            frames_remaining=self.post_event_frames,
            event_frame_idx=frame_idx,
            event_type=event_type,
            recovery_check=recovery_check,
        )
        self._pending.append(pending)

        logger.info(
            "Context snapshot created for [%s] frame=%d (pre=%d, pending post=%d)",
            event_type.value, frame_idx, len(pre_frames), self.post_event_frames,
        )

        return snapshot

    def get_snapshots(self) -> List[BehaviorContextSnapshot]:
        """คืน snapshots ทั้งหมดที่เสร็จสมบูรณ์แล้ว"""
        return list(self._snapshots)

    def get_snapshot_summary(self, snapshot: BehaviorContextSnapshot) -> str:
        """
        สร้างคำอธิบายแบบ human/AI readable ของหนึ่ง event.

        Args:
            snapshot: BehaviorContextSnapshot ที่ต้องการอธิบาย

        Returns:
            str: คำอธิบาย narrative ของ event พร้อม context
        """
        return self._format_snapshot_narrative(snapshot)

    def to_ai_readable_text(self) -> str:
        """
        สร้าง narrative ทั้งหมดของ behavior events ทั้งหมดพร้อม context.

        Returns:
            str: full narrative สำหรับ AI วิเคราะห์
        """
        snapshots = self.get_snapshots()
        if not snapshots:
            return "No behavior events recorded."

        lines = []
        lines.append("=" * 80)
        lines.append("BEHAVIOR CONTEXT REPORT — Black Box Recording")
        lines.append(f"Total events: {len(snapshots)} | Frames recorded: {self._total_frames}")
        lines.append("=" * 80)
        lines.append("")

        for i, snap in enumerate(snapshots, 1):
            lines.append(self._format_snapshot_narrative(snap, index=i))
            lines.append("")

        # Summary statistics
        lines.append("-" * 80)
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(self._generate_summary_stats(snapshots))

        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _extract_pre_frames(self, event_frame_idx: int) -> List[Dict[str, Any]]:
        """ดึง pre-event frames จาก rolling buffer"""
        pre = []
        # buffer เก็บ (frame_idx, compact, frame_state)
        for fidx, compact, _ in self._buffer:
            if fidx >= event_frame_idx:
                break
            pre.append(compact)
        # เก็บเฉพาะ pre_event_frames สุดท้าย
        if len(pre) > self.pre_event_frames:
            pre = pre[-self.pre_event_frames:]
        return pre

    def _extract_trajectory(self, frames: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """ดึง vehicle (x, y) trajectory จาก frames"""
        trajectory = []
        for f in frames:
            x = float(f.get("vehicle_x", 0) or 0)
            y = float(f.get("vehicle_y", 0) or 0)
            trajectory.append((x, y))
        return trajectory

    def _update_pending(self, current_frame_idx: int, frame_state: Any) -> None:
        """อัปเดต pending snapshots — เก็บ post-event frames และตรวจ recovery"""
        if not self._pending:
            return

        compact = _frame_to_compact_dict(frame_state, current_frame_idx)
        still_pending = []

        for p in self._pending:
            # ไม่เก็บ frame ที่เป็น event frame เอง
            if current_frame_idx <= p.event_frame_idx:
                still_pending.append(p)
                continue

            # เพิ่ม post frame
            p.snapshot.post_frames.append(compact)

            # อัปเดต peak metrics
            cte = abs(float(compact.get("cte_m", 0) or 0))
            conf = float(compact.get("lane_conf", 1.0) or 1.0)
            steer = abs(float(compact.get("final_steer", 0) or 0))
            brake = abs(float(compact.get("final_brake", 0) or 0))

            p.snapshot.peak_cte_during_event = max(p.snapshot.peak_cte_during_event, cte)
            p.snapshot.min_conf_during_event = min(p.snapshot.min_conf_during_event, conf)
            p.snapshot.max_steer_during_event = max(p.snapshot.max_steer_during_event, steer)
            p.snapshot.max_brake_during_event = max(p.snapshot.max_brake_during_event, brake)

            # เพิ่ม trajectory point
            x = float(compact.get("vehicle_x", 0) or 0)
            y = float(compact.get("vehicle_y", 0) or 0)
            p.snapshot.vehicle_trajectory_during_event.append((x, y))

            # อัปเดต duration
            p.snapshot.duration_frames = current_frame_idx - p.event_frame_idx

            # ตรวจ recovery
            if not p.snapshot.recovered and p.recovery_check:
                if self._check_recovery(p, compact):
                    p.snapshot.recovered = True
                    p.snapshot.recovery_frame_idx = current_frame_idx

            p.frames_remaining -= 1
            if p.frames_remaining <= 0:
                # Snapshot เสร็จสมบูรณ์
                self._snapshots.append(p.snapshot)
                logger.debug(
                    "Snapshot completed [%s] duration=%d recovered=%s",
                    p.event_type.value, p.snapshot.duration_frames, p.snapshot.recovered,
                )
            else:
                still_pending.append(p)

        self._pending = still_pending

    def _get_recovery_check(self, event_type: BehaviorType) -> Optional[str]:
        """กำหนดวิธีตรวจ recovery ตามประเภท event"""
        recovery_map = {
            BehaviorType.LANE_LOST: "conf",
            BehaviorType.LANE_CONF_DROP: "conf",
            BehaviorType.GEOMETRY_INVALID: "geom",
            BehaviorType.OFF_TRACK: "cte",
            BehaviorType.DRIFTING_LEFT: "cte",
            BehaviorType.DRIFTING_RIGHT: "cte",
            BehaviorType.OSCILLATION: "steer",
            BehaviorType.SUDDEN_BRAKE: "brake",
            BehaviorType.MPC_UNSTABLE: "solver",
            BehaviorType.EXCESSIVE_STEERING: "steer",
        }
        return recovery_map.get(event_type)

    def _check_recovery(self, pending: _PendingSnapshot, compact: Dict[str, Any]) -> bool:
        """ตรวจว่า event นี้ recover แล้วหรือยัง"""
        check = pending.recovery_check
        if check == "conf":
            return float(compact.get("lane_conf", 0) or 0) > 0.5
        elif check == "cte":
            return abs(float(compact.get("cte_m", 0) or 0)) < 0.8
        elif check == "steer":
            return abs(float(compact.get("final_steer", 0) or 0)) < 0.15
        elif check == "brake":
            return float(compact.get("final_brake", 0) or 0) < 0.2
        elif check == "solver":
            status = str(compact.get("solver_status", ""))
            return "Fallback" not in status
        elif check == "geom":
            return bool(compact.get("geometry_valid", False))
        return False

    def _format_snapshot_narrative(
        self, snap: BehaviorContextSnapshot, index: Optional[int] = None
    ) -> str:
        """สร้าง narrative text ของหนึ่ง snapshot"""
        ev = snap.event
        prefix = f"EVENT {index}" if index else "EVENT"
        header = (
            f"{prefix} [frame {ev.frame_idx}, t={ev.timestamp:.1f}s] "
            f"{ev.event_type.value.upper()} ({ev.severity.value.upper()}): {ev.message}"
        )

        lines = [header]

        # Pre-event analysis
        pre_desc = self._describe_pre_event(snap.pre_frames)
        lines.append(f"  Pre-event ({len(snap.pre_frames)} frames): {pre_desc}")

        # At event
        at_desc = self._describe_at_event(snap)
        lines.append(f"  At event: {at_desc}")

        # Post-event
        post_desc = self._describe_post_event(snap.post_frames, snap)
        lines.append(f"  Post-event ({len(snap.post_frames)} frames): {post_desc}")

        # Duration + peaks
        duration_s = snap.duration_frames / self.estimated_fps if self.estimated_fps > 0 else 0
        lines.append(
            f"  Duration: {snap.duration_frames} frames ({duration_s:.1f}s). "
            f"Peak CTE: {snap.peak_cte_during_event:.2f}m. "
            f"Min conf: {snap.min_conf_during_event:.2f}. "
            f"Max steer: {snap.max_steer_during_event:.3f}. "
            f"Max brake: {snap.max_brake_during_event:.2f}."
        )

        # Recovery
        if snap.recovered:
            rec_s = (
                (snap.recovery_frame_idx - ev.frame_idx) / self.estimated_fps
                if snap.recovery_frame_idx and self.estimated_fps > 0
                else 0
            )
            lines.append(
                f"  Recovery: yes (frame {snap.recovery_frame_idx}, "
                f"{rec_s:.1f}s after event)."
            )
        else:
            lines.append("  Recovery: not detected within post-event window.")

        return "\n".join(lines)

    def _describe_pre_event(self, pre_frames: List[Dict[str, Any]]) -> str:
        """อธิบายสภาพก่อน event"""
        if not pre_frames:
            return "No pre-event data available."

        # หาค่าเฉลี่ย/ตัวอย่างจาก pre frames
        ctes = [abs(float(f.get("cte_m", 0) or 0)) for f in pre_frames]
        confs = [float(f.get("lane_conf", 0) or 0) for f in pre_frames]
        speeds = [float(f.get("speed_kmh", 0) or 0) for f in pre_frames]
        steers = [abs(float(f.get("final_steer", 0) or 0)) for f in pre_frames]

        avg_cte = sum(ctes) / len(ctes) if ctes else 0
        avg_conf = sum(confs) / len(confs) if confs else 0
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        avg_steer = sum(steers) / len(steers) if steers else 0

        # ประเมินสภาพถนน
        last = pre_frames[-1]
        mode = str(last.get("mode", "unknown"))
        geom = bool(last.get("geometry_valid", False))

        parts = []
        if avg_conf > 0.7 and avg_cte < 0.3:
            parts.append("Vehicle was tracking well")
        elif avg_conf > 0.5:
            parts.append("Vehicle was tracking with moderate confidence")
        else:
            parts.append("Vehicle had low confidence tracking")

        parts.append(f"CTE={avg_cte:.2f}m, conf={avg_conf:.2f}, speed={avg_speed:.0f}km/h")

        # สภาพถนน
        curvatures = [abs(float(f.get("curvature", 0) or 0)) for f in pre_frames]
        avg_curv = sum(curvatures) / len(curvatures) if curvatures else 0
        if avg_curv < 0.005:
            parts.append("straight road")
        elif avg_curv < 0.02:
            parts.append("gentle curve")
        else:
            parts.append("sharp curve")

        if avg_steer > 0.2:
            parts.append(f"active steering (avg={avg_steer:.2f})")

        if not geom:
            parts.append("geometry already invalid")

        parts.append(f"mode='{mode}'")

        return ", ".join(parts) + "."

    def _describe_at_event(self, snap: BehaviorContextSnapshot) -> str:
        """อธิบายสภาพ ณ จุด event"""
        ev = snap.event
        ctx = ev.context
        parts = []

        # ใช้ context จาก event ถ้ามี
        if "conf" in ctx:
            parts.append(f"confidence={ctx['conf']:.2f}")
        if "cte_m" in ctx:
            parts.append(f"CTE={ctx['cte_m']:+.2f}m")
        if "mode" in ctx:
            parts.append(f"mode='{ctx['mode']}'")
        if "steer" in ctx:
            parts.append(f"steer={ctx['steer']:+.3f}")
        if "brake" in ctx:
            parts.append(f"brake={ctx['brake']:.2f}")
        if "prev_mode" in ctx and "new_mode" in ctx:
            parts.append(f"mode changed '{ctx['prev_mode']}' → '{ctx['new_mode']}'")
        if "consecutive_fallbacks" in ctx:
            parts.append(f"{ctx['consecutive_fallbacks']} consecutive fallbacks")
        if "steer_std" in ctx:
            parts.append(f"steer std={ctx['steer_std']:.3f}")
        if "brake_delta" in ctx:
            parts.append(f"brake delta={ctx['brake_delta']:+.2f}")

        if not parts:
            parts.append(ev.message)

        return ", ".join(parts) + "."

    def _describe_post_event(
        self, post_frames: List[Dict[str, Any]], snap: BehaviorContextSnapshot
    ) -> str:
        """อธิบายสภาพหลัง event"""
        if not post_frames:
            return "No post-event data yet (still recording)."

        ctes = [float(f.get("cte_m", 0) or 0) for f in post_frames]
        confs = [float(f.get("lane_conf", 0) or 0) for f in post_frames]
        speeds = [float(f.get("speed_kmh", 0) or 0) for f in post_frames]
        steers = [float(f.get("final_steer", 0) or 0) for f in post_frames]
        brakes = [float(f.get("final_brake", 0) or 0) for f in post_frames]

        parts = []

        # CTE trend
        if len(ctes) >= 2:
            cte_start = ctes[0]
            cte_end = ctes[-1]
            if abs(cte_end) > abs(cte_start):
                direction = "right" if cte_end < cte_start else "left"
                parts.append(f"CTE rose to {abs(cte_end):.2f}m (drifting {direction})")
            else:
                parts.append(f"CTE settled to {abs(cte_end):.2f}m")

        # Confidence trend
        if len(confs) >= 2:
            conf_end = confs[-1]
            if conf_end > 0.5:
                parts.append(f"confidence restored to {conf_end:.2f}")
            elif conf_end > 0.3:
                parts.append(f"confidence partial {conf_end:.2f}")
            else:
                parts.append(f"confidence still low ({conf_end:.2f})")

        # Speed
        if speeds:
            parts.append(f"speed={speeds[-1]:.0f}km/h")

        # Steering
        max_post_steer = max(abs(s) for s in steers) if steers else 0
        if max_post_steer > 0.2:
            parts.append(f"steering increased to {max_post_steer:.2f}")

        # Brake
        max_post_brake = max(b for b in brakes) if brakes else 0
        if max_post_brake > 0.3:
            parts.append(f"braking active ({max_post_brake:.2f})")

        # Mode
        last = post_frames[-1]
        mode = str(last.get("mode", "unknown"))
        parts.append(f"mode='{mode}'")

        return ", ".join(parts) + "."

    def _generate_summary_stats(self, snapshots: List[BehaviorContextSnapshot]) -> str:
        """สร้างสถิติสรุปของ snapshots ทั้งหมด"""
        if not snapshots:
            return "No events to summarize."

        lines = []
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        recovered_count = 0
        total_duration = 0
        peak_cte_overall = 0.0
        min_conf_overall = 1.0

        for snap in snapshots:
            t = snap.event.event_type.value
            by_type[t] = by_type.get(t, 0) + 1
            s = snap.event.severity.value
            by_severity[s] = by_severity.get(s, 0) + 1
            if snap.recovered:
                recovered_count += 1
            total_duration += snap.duration_frames
            peak_cte_overall = max(peak_cte_overall, snap.peak_cte_during_event)
            min_conf_overall = min(min_conf_overall, snap.min_conf_during_event)

        lines.append(f"Events by type: {by_type}")
        lines.append(f"Events by severity: {by_severity}")
        lines.append(
            f"Recovery rate: {recovered_count}/{len(snapshots)} "
            f"({100*recovered_count/len(snapshots):.0f}%)"
        )
        lines.append(f"Total event duration: {total_duration} frames")
        lines.append(f"Peak CTE across all events: {peak_cte_overall:.2f}m")
        lines.append(f"Min confidence across all events: {min_conf_overall:.2f}")

        return "\n".join(lines)
