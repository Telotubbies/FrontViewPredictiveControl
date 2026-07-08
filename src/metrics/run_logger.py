"""
RunLogger — บันทึกทุก frame ลง CSV + events ลง log file แบบ real-time

ทำงานคู่กับ control loop:
- log_frame(): บันทึก 1 row ลง CSV (non-blocking buffer)
- log_event(): บันทึกเหตุการณ์สำคัญ (AEB, stuck, fallback, departure)
- flush(): เขียน buffer ลงไฟล์
- ใช้ background thread เพื่อไม่ block control loop
"""
from __future__ import annotations

import csv
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class LogEvent:
    """เหตุการณ์สำคัญใน 1 frame ที่สมควรบันทึกแยก"""
    timestamp: float
    frame_idx: int
    event_type: str  # aeb | stuck | fallback | departure | safety | adas | info
    severity: str    # info | warning | error | critical
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class RunLogger:
    """
    Real-time run logger — บันทึกทุก frame + events ลงไฟล์.

    ใช้ background thread เพื่อไม่ block control loop:
    - log_frame() ใส่ลง queue (non-blocking, drop ถ้าเต็ม)
    - log_event() เขียนลง events.log ทันที (sync, แต่เร็ว)
    - flush() รอ background thread เขียนหมด
    """

    # CSV columns — ครบทุก field สำหรับวิเคราะห์ภายหลัง
    CSV_COLUMNS = [
        "frame_idx", "timestamp", "sim_time",
        # Vehicle
        "vehicle_x", "vehicle_y", "vehicle_z", "vehicle_yaw",
        "speed_ms", "speed_kmh", "accel_ms2",
        # Control (final sent to CARLA)
        "steer", "throttle", "brake", "reverse",
        # MPC
        "mpc_steer_rad", "mpc_accel", "mpc_solver_status", "mpc_solve_time_ms",
        # Perception
        "cte_m", "heading_err_rad", "heading_err_deg", "curvature",
        "lane_conf", "geometry_valid", "perception_mode",
        # Safety / Overrides
        "safety_override_active", "stuck_recovery_active", "stuck_recovery_phase",
        "adas_override_active", "winning_source",
        # ADAS
        "aeb_active", "aeb_ttc", "aeb_warning_level",
        "acc_active", "acc_target_speed_ms", "acc_distance_m",
        "ldw_state", "ldw_warning_active", "ldw_side",
        "lka_pro_assist",
        "bsw_left_alert", "bsw_right_alert",
        "tsr_speed_limit_kmh", "tsr_traffic_light", "tsr_action",
        "tja_state", "tja_active", "stop_and_go_stopped",
        # Performance
        "fps", "loop_time_ms", "perception_time_ms", "mpc_time_ms",
    ]

    def __init__(
        self,
        run_dir: str,
        map_name: str = "unknown",
        vehicle_type: str = "unknown",
        target_speed_kmh: float = 0.0,
        buffer_size: int = 5000,
    ):
        """
        Args:
            run_dir: directory สำหรับเก็บไฟล์ของ run นี้
            map_name: ชื่อแผนที่
            vehicle_type: ชื่อรถ
            target_speed_kmh: ความเร็วเป้าหมาย
            buffer_size: ขนาด queue (frame ที่รอเขียน)
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)

        self.map_name = map_name
        self.vehicle_type = vehicle_type
        self.target_speed_kmh = target_speed_kmh

        self._csv_path = self.run_dir / "frame_log.csv"
        self._events_path = self.run_dir / "events.log"
        self._meta_path = self.run_dir / "run_meta.json"

        # Frame queue (producer=control loop, consumer=writer thread)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._event_queue: queue.Queue = queue.Queue(maxsize=buffer_size)

        # Writer thread
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False
        self._frames_written = 0
        self._events_written = 0
        self._dropped_frames = 0
        self._start_time = time.time()

        # Previous state for event detection
        self._prev_aeb = False
        self._prev_stuck = False
        self._prev_fallback = False
        self._prev_departure = False
        self._prev_safety = False

    def start(self) -> None:
        """เริ่ม background writer thread"""
        self._running = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        self._write_meta()
        logger.info(f"RunLogger started → {self.run_dir}")

    def stop(self) -> None:
        """หยุด writer thread และ flush ข้อมูลที่เหลือ"""
        self._running = False
        # sentinel
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._event_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
        self._update_meta()
        logger.info(
            f"RunLogger stopped: {self._frames_written} frames, "
            f"{self._events_written} events, {self._dropped_frames} dropped"
        )

    def log_frame(self, frame_state, context: Dict[str, Any]) -> None:
        """
        บันทึก 1 frame ลง queue (non-blocking).

        Args:
            frame_state: FrameState object
            context: dict with keys like 'mpc_steer_rad', 'mpc_accel',
                     'loop_time_ms', 'perception_time_ms', 'sim_time'
        """
        row = self._frame_state_to_row(frame_state, context)
        try:
            self._frame_queue.put_nowait(row)
        except queue.Full:
            self._dropped_frames += 1
            if self._dropped_frames % 100 == 1:
                logger.warning(f"RunLogger queue full, dropped {self._dropped_frames} frames")

        # Detect events (transitions)
        self._detect_events(frame_state, context)

    def log_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        frame_idx: int = 0,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """บันทึก event สำคัญ"""
        event = LogEvent(
            timestamp=time.time() - self._start_time,
            frame_idx=frame_idx,
            event_type=event_type,
            severity=severity,
            message=message,
            data=data or {},
        )
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            logger.warning("Event queue full, dropping event")

    def flush(self) -> None:
        """รอ writer thread เขียนข้อมูลที่เหลือทั้งหมด"""
        # Put sentinel and wait
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass

    def _frame_state_to_row(self, fs, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """แปลง FrameState + context → dict row สำหรับ CSV"""
        import math
        row = {}
        for col in self.CSV_COLUMNS:
            if col == "frame_idx":
                row[col] = ctx.get("frame_idx", 0)
            elif col == "timestamp":
                row[col] = time.time() - self._start_time
            elif col == "sim_time":
                row[col] = ctx.get("sim_time", 0.0)
            elif col == "speed_ms":
                row[col] = ctx.get("speed_ms", 0.0)
            elif col == "speed_kmh":
                row[col] = ctx.get("speed_ms", 0.0) * 3.6
            elif col == "accel_ms2":
                row[col] = ctx.get("accel_ms2", 0.0)
            elif col == "mpc_steer_rad":
                row[col] = ctx.get("mpc_steer_rad", 0.0)
            elif col == "mpc_accel":
                row[col] = ctx.get("mpc_accel", 0.0)
            elif col == "mpc_time_ms":
                row[col] = getattr(fs, "mpc_solve_time_ms", 0.0)
            elif col == "perception_time_ms":
                row[col] = ctx.get("perception_time_ms", 0.0)
            elif col == "loop_time_ms":
                row[col] = ctx.get("loop_time_ms", 0.0)
            elif col == "fps":
                row[col] = ctx.get("fps", 0.0)
            elif col == "heading_err_deg":
                row[col] = math.degrees(getattr(fs, "heading_rad", 0.0))
            elif col == "perception_mode":
                row[col] = getattr(fs, "mode", "unknown")
            elif col == "winning_source":
                row[col] = getattr(fs, "winning_source", "mpc")
            elif col == "reverse":
                row[col] = ctx.get("reverse", False)
            elif col == "vehicle_yaw":
                row[col] = getattr(fs, "vehicle_yaw", 0.0)
            else:
                # Direct attribute from FrameState
                row[col] = getattr(fs, col, None)

        return row

    def _detect_events(self, fs, ctx: Dict[str, Any]) -> None:
        """ตรวจจับ transitions ที่สำคัญแล้ว log event"""
        frame_idx = ctx.get("frame_idx", 0)

        # AEB activation
        aeb_now = bool(getattr(fs, "aeb_active", False))
        if aeb_now and not self._prev_aeb:
            self.log_event(
                "aeb", "critical",
                f"AEB activated (TTC={getattr(fs, 'aeb_ttc', -1):.2f}s)",
                frame_idx,
                {"ttc": getattr(fs, "aeb_ttc", -1), "speed_ms": ctx.get("speed_ms", 0)},
            )
        self._prev_aeb = aeb_now

        # Stuck recovery
        stuck_now = bool(getattr(fs, "stuck_recovery_active", False))
        if stuck_now and not self._prev_stuck:
            self.log_event(
                "stuck", "warning",
                f"Stuck recovery started (phase={getattr(fs, 'stuck_recovery_phase', 'unknown')})",
                frame_idx,
                {"phase": getattr(fs, "stuck_recovery_phase", "unknown")},
            )
        self._prev_stuck = stuck_now

        # MPC fallback
        solver = str(getattr(fs, "solver_status", ""))
        fallback_now = "Fallback" in solver
        if fallback_now and not self._prev_fallback:
            self.log_event(
                "fallback", "warning",
                "MPC solver fallback to Pure Pursuit",
                frame_idx,
                {"cte_m": getattr(fs, "cte_m", 0), "speed_ms": ctx.get("speed_ms", 0)},
            )
        self._prev_fallback = fallback_now

        # Safety override
        safety_now = bool(getattr(fs, "safety_override_active", False))
        if safety_now and not self._prev_safety:
            self.log_event(
                "safety", "warning",
                f"Safety override activated (CTE={getattr(fs, 'cte_m', 0):.2f}m)",
                frame_idx,
                {"cte_m": getattr(fs, "cte_m", 0)},
            )
        self._prev_safety = safety_now

        # Lane departure
        ldw_state = str(getattr(fs, "ldw_state", "in_lane"))
        departed_now = ldw_state == "departed"
        if departed_now and not self._prev_departure:
            self.log_event(
                "departure", "error",
                f"Lane departure detected (side={getattr(fs, 'ldw_side', 'none')})",
                frame_idx,
                {"cte_m": getattr(fs, "cte_m", 0), "side": getattr(fs, "ldw_side", "none")},
            )
        self._prev_departure = departed_now

    def _writer_loop(self) -> None:
        """Background thread: อ่านจาก queue เขียนลง CSV + events.log"""
        csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=self.CSV_COLUMNS)
        csv_writer.writeheader()

        events_file = open(self._events_path, "w", encoding="utf-8")

        try:
            while self._running:
                # Write frames
                try:
                    row = self._frame_queue.get(timeout=0.1)
                    if row is None:
                        break
                    csv_writer.writerow(row)
                    self._frames_written += 1
                    # Flush every 50 frames for real-time access
                    if self._frames_written % 50 == 0:
                        csv_file.flush()
                except queue.Empty:
                    pass

                # Write events
                try:
                    event = self._event_queue.get_nowait()
                    if event is None:
                        continue
                    events_file.write(
                        f"[{event.timestamp:.3f}] frame={event.frame_idx} "
                        f"{event.severity.upper()} {event.event_type}: {event.message}\n"
                    )
                    if event.data:
                        events_file.write(f"  data: {event.data}\n")
                    events_file.flush()
                    self._events_written += 1
                except queue.Empty:
                    pass

        finally:
            # Drain remaining frames
            while True:
                try:
                    row = self._frame_queue.get_nowait()
                    if row is None:
                        break
                    csv_writer.writerow(row)
                    self._frames_written += 1
                except queue.Empty:
                    break

            # Drain remaining events
            while True:
                try:
                    event = self._event_queue.get_nowait()
                    if event is None:
                        break
                    events_file.write(
                        f"[{event.timestamp:.3f}] frame={event.frame_idx} "
                        f"{event.severity.upper()} {event.event_type}: {event.message}\n"
                    )
                    self._events_written += 1
                except queue.Empty:
                    break

            csv_file.flush()
            events_file.flush()
            csv_file.close()
            events_file.close()

    def _write_meta(self) -> None:
        """เขียน run metadata"""
        import json
        meta = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "map_name": self.map_name,
            "vehicle_type": self.vehicle_type,
            "target_speed_kmh": self.target_speed_kmh,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        }
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _update_meta(self) -> None:
        """อัปเดต metadata เมื่อจบ run"""
        import json
        meta = {}
        if self._meta_path.exists():
            with open(self._meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        meta.update({
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(time.time() - self._start_time, 2),
            "frames_written": self._frames_written,
            "events_written": self._events_written,
            "dropped_frames": self._dropped_frames,
        })
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    @property
    def events_path(self) -> Path:
        return self._events_path
