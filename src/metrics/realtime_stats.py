"""
RealTimeStats — คำนวณและแสดงสถิติแบบ real-time ระหว่างรัน

แสดงทุก 50 frames ใน console:
- CTE: current, mean, max, RMSE
- Speed: current, mean, target
- MPC: solve time, fallback rate
- Safety: AEB/stuck/safety counts
- Performance: FPS, loop time
- Lane: in-lane %, departure count
"""
from __future__ import annotations

import logging
import sys
import time
from collections import deque
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class RealTimeStats:
    """
    Real-time statistics calculator — แสดง live metrics ใน console.

    ใช้ rolling window (deque) สำหรับ recent stats
    และ cumulative counters สำหรับ totals.
    """

    def __init__(
        self,
        window_size: int = 200,
        print_interval: int = 50,
        target_speed_kmh: float = 0.0,
    ):
        """
        Args:
            window_size: จำนวน frames ล่าสุดสำหรับ rolling stats
            print_interval: แสดง summary ทุก N frames
            target_speed_kmh: ความเร็วเป้าหมาย
        """
        self.window_size = window_size
        self.print_interval = print_interval
        self.target_speed_kmh = target_speed_kmh

        # Rolling window (recent frames)
        self._cte_window: deque = deque(maxlen=window_size)
        self._speed_window: deque = deque(maxlen=window_size)
        self._heading_window: deque = deque(maxlen=window_size)
        self._mpc_time_window: deque = deque(maxlen=window_size)
        self._loop_time_window: deque = deque(maxlen=window_size)
        self._steer_window: deque = deque(maxlen=window_size)

        # Cumulative counters (entire run)
        self._total_frames = 0
        self._aeb_count = 0
        self._stuck_count = 0
        self._safety_count = 0
        self._fallback_count = 0
        self._departure_count = 0
        self._in_lane_frames = 0

        # All-time CTE for RMSE
        self._all_cte: deque = deque(maxlen=100000)

        self._start_time = time.time()
        self._last_print_time = 0.0

    def update(self, frame_state, context: Dict[str, Any]) -> None:
        """
        อัปเดต stats ด้วย frame ปัจจุบัน.

        Args:
            frame_state: FrameState
            context: dict with 'frame_idx', 'loop_time_ms', 'speed_ms', etc.
        """
        cte = float(getattr(frame_state, "cte_m", 0.0))
        speed_ms = float(context.get("speed_ms", 0.0))
        heading = float(getattr(frame_state, "heading_rad", 0.0))
        mpc_time = float(getattr(frame_state, "mpc_solve_time_ms", 0.0))
        loop_time = float(context.get("loop_time_ms", 0.0))
        steer = float(getattr(frame_state, "final_steer", 0.0))

        self._cte_window.append(cte)
        self._speed_window.append(speed_ms)
        self._heading_window.append(heading)
        self._mpc_time_window.append(mpc_time)
        self._loop_time_window.append(loop_time)
        self._steer_window.append(steer)
        self._all_cte.append(cte)

        self._total_frames += 1

        # Counters
        if bool(getattr(frame_state, "aeb_active", False)):
            self._aeb_count += 1
        if bool(getattr(frame_state, "stuck_recovery_active", False)):
            self._stuck_count += 1
        if bool(getattr(frame_state, "safety_override_active", False)):
            self._safety_count += 1
        if "Fallback" in str(getattr(frame_state, "solver_status", "")):
            self._fallback_count += 1
        ldw_state = str(getattr(frame_state, "ldw_state", "in_lane"))
        if ldw_state == "departed":
            self._departure_count += 1
        else:
            self._in_lane_frames += 1

        # Print summary at interval
        frame_idx = context.get("frame_idx", self._total_frames)
        if frame_idx > 0 and frame_idx % self.print_interval == 0:
            self._print_summary(frame_idx)

    def _print_summary(self, frame_idx: int) -> None:
        """พิมพ์สถิติลง console (stderr เพื่อไม่รบกวน stdout)"""
        if not self._cte_window:
            return

        cte_arr = np.array(self._cte_window)
        speed_arr = np.array(self._speed_window)
        mpc_arr = np.array(self._mpc_time_window)
        loop_arr = np.array(self._loop_time_window)
        steer_arr = np.array(self._steer_window)

        all_cte_arr = np.array(self._all_cte)

        elapsed = time.time() - self._start_time
        fps = self._total_frames / elapsed if elapsed > 0 else 0

        in_lane_pct = (self._in_lane_frames / max(1, self._total_frames)) * 100
        fallback_rate = (self._fallback_count / max(1, self._total_frames)) * 100

        # Recent (window) stats
        cte_cur = cte_arr[-1]
        cte_mean = np.mean(cte_arr)
        cte_max = np.max(np.abs(cte_arr))

        # All-time RMSE
        cte_rmse = np.sqrt(np.mean(all_cte_arr ** 2)) if len(all_cte_arr) > 0 else 0

        speed_cur = speed_arr[-1] * 3.6  # km/h
        speed_mean = np.mean(speed_arr) * 3.6

        mpc_mean = np.mean(mpc_arr)
        loop_mean = np.mean(loop_arr)

        steer_cur = steer_arr[-1]
        steer_max = np.max(np.abs(steer_arr))

        # Format output
        lines = [
            "",
            f"┌─ Frame {frame_idx:>6d} │ {elapsed:>7.1f}s │ {fps:>5.1f} FPS ─────────────────────┐",
            f"│ CTE:  cur={cte_cur:+.2f}m  mean={cte_mean:+.2f}m  max={cte_max:.2f}m  RMSE={cte_rmse:.2f}m        │",
            f"│ Speed: cur={speed_cur:>5.1f}km/h  mean={speed_mean:>5.1f}km/h  target={self.target_speed_kmh:.0f}km/h       │",
            f"│ Steer: cur={steer_cur:+.3f}  max={steer_max:.3f}                              │",
            f"│ MPC:  solve={mpc_mean:>5.1f}ms  fallback={fallback_rate:.1f}%  loop={loop_mean:.1f}ms          │",
            f"│ Lane: in-lane={in_lane_pct:.1f}%  departures={self._departure_count}                        │",
            f"│ Events: AEB={self._aeb_count}  Stuck={self._stuck_count}  Safety={self._safety_count}  Fallback={self._fallback_count}  │",
            f"└──────────────────────────────────────────────────────────┘",
        ]

        output = "\n".join(lines)
        print(output, file=sys.stderr, flush=True)

        self._last_print_time = time.time()

    def get_current_stats(self) -> Dict[str, Any]:
        """คืนสถิติปัจจุบันเป็น dict (สำหรับ dashboard หรือ API)"""
        if not self._cte_window:
            return {}

        cte_arr = np.array(self._cte_window)
        speed_arr = np.array(self._speed_window)
        all_cte_arr = np.array(self._all_cte)

        return {
            "frame_count": self._total_frames,
            "elapsed_s": time.time() - self._start_time,
            "cte_current": float(cte_arr[-1]),
            "cte_mean": float(np.mean(cte_arr)),
            "cte_rmse": float(np.sqrt(np.mean(all_cte_arr ** 2))) if len(all_cte_arr) > 0 else 0,
            "cte_max_abs": float(np.max(np.abs(all_cte_arr))) if len(all_cte_arr) > 0 else 0,
            "speed_current_kmh": float(speed_arr[-1] * 3.6),
            "speed_mean_kmh": float(np.mean(speed_arr) * 3.6),
            "in_lane_pct": (self._in_lane_frames / max(1, self._total_frames)) * 100,
            "aeb_count": self._aeb_count,
            "stuck_count": self._stuck_count,
            "safety_count": self._safety_count,
            "fallback_count": self._fallback_count,
            "departure_count": self._departure_count,
        }

    def reset(self) -> None:
        """รีเซ็ตทุก counter"""
        self._cte_window.clear()
        self._speed_window.clear()
        self._heading_window.clear()
        self._mpc_time_window.clear()
        self._loop_time_window.clear()
        self._steer_window.clear()
        self._all_cte.clear()
        self._total_frames = 0
        self._aeb_count = 0
        self._stuck_count = 0
        self._safety_count = 0
        self._fallback_count = 0
        self._departure_count = 0
        self._in_lane_frames = 0
        self._start_time = time.time()
