"""
Lane Departure Warning (LDW) + Lane Keeping Assist Pro (LKA Pro).

LDW: ตรวจจับการออกนอกเลนจากค่า CTE (cross-track error)
     → เตือนผู้ขับ (visual + steering nudge) เมื่อ |cte| เกิน threshold
LKA Pro: เพิ่ม corrective steering assist แบบ proactive (additive กับ MPC output)

State machine:
    IN_LANE  ──(|cte| > threshold)──▶  DEPARTING
    DEPARTING ──(|cte| > threshold + hysteresis)──▶  DEPARTED
    DEPARTED/DEPARTING ──(|cte| < threshold - hysteresis)──▶  IN_LANE

หมายเหตุ: ไม่ทำงานเมื่อความเร็วต่ำกว่า 5 m/s (parking speed)
"""
import logging
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)

# ความเร็วขั้นต่ำที่จะเปิดใช้งาน LDW/LKA (m/s)
LDW_MIN_SPEED_MS = 5.0


@dataclass
class LDWConfig:
    """Configuration for Lane Departure Warning + LKA Pro."""

    departure_threshold_m: float = 0.3
    """CTE threshold ที่เริ่มพิจารณาว่ากำลังออกนอกเลน (m)."""

    departure_hysteresis_m: float = 0.1
    """Hysteresis เพื่อป้องกันการกระพริบของสถานะ (m)."""

    warning_duration_s: float = 2.0
    """ระยะเวลาที่ warning ยังคง active หลังเข้าสู่ DEPARTED (s)."""

    lka_pro_assist_gain: float = 0.5
    """Steering assist gain เมื่อกำลังออกนอกเลน."""

    blink_rate_hz: float = 1.5
    """Blink rate สำหรับ visual warning (Hz)."""

    def __post_init__(self):
        if self.departure_threshold_m <= 0:
            raise ValueError("departure_threshold_m must be > 0")
        if self.departure_hysteresis_m >= self.departure_threshold_m:
            raise ValueError("departure_hysteresis_m must be < departure_threshold_m")
        if self.blink_rate_hz < 0:
            raise ValueError("blink_rate_hz must be >= 0")


class LDWState(IntEnum):
    """State ของ Lane Departure Warning."""

    IN_LANE = 0
    DEPARTING = 1
    DEPARTED = 2


class LaneDepartureWarning:
    """Lane Departure Warning with corrective steering nudge."""

    def __init__(self, config: Optional[LDWConfig] = None) -> None:
        self.config = config if config is not None else LDWConfig()
        self._state: LDWState = LDWState.IN_LANE
        self._side: str = "none"
        self._warning_active: bool = False
        self._warning_timer_s: float = 0.0
        self._last_timestamp: float = float("nan")
        self._blink_on: bool = False
        self._blink_phase_s: float = 0.0
        self._steering_assist: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        cte: float,
        heading_err: float,
        speed_ms: float,
        timestamp: float,
    ) -> dict:
        """Update LDW state machine.

        Args:
            cte: Cross-track error (m). บวก = อยู่ซ้ายของ centerline,
                 ลบ = อยู่ขวา (ตาม convention ของระบบ)
            heading_err: Heading error (rad) — สำรองไว้ใช้ในอนาคต
            speed_ms: ความเร็วยานพาหนะ (m/s)
            timestamp: เวลาปัจจุบัน (s)

        Returns:
            dict ที่มี state, warning_active, side, steering_assist, blink_on
        """
        cfg = self.config
        abs_cte = abs(cte)

        # คำนวณ dt
        if not math.isnan(self._last_timestamp):
            dt = max(timestamp - self._last_timestamp, 0.0)
        else:
            dt = 0.0
        self._last_timestamp = timestamp

        # ไม่ทำงานเมื่อความเร็วต่ำ → reset สู่ IN_LANE
        if speed_ms < LDW_MIN_SPEED_MS:
            self._state = LDWState.IN_LANE
            self._warning_active = False
            self._side = "none"
            self._steering_assist = 0.0
            self._blink_on = False
            self._warning_timer_s = 0.0
            self._blink_phase_s = 0.0
            return self._build_output()

        # ── State transitions ──────────────────────────────────────────────
        if self._state == LDWState.IN_LANE:
            if abs_cte > cfg.departure_threshold_m:
                self._state = LDWState.DEPARTING
        elif self._state == LDWState.DEPARTING:
            if abs_cte > cfg.departure_threshold_m + cfg.departure_hysteresis_m:
                self._state = LDWState.DEPARTED
                self._warning_timer_s = cfg.warning_duration_s
            elif abs_cte < cfg.departure_threshold_m - cfg.departure_hysteresis_m:
                self._state = LDWState.IN_LANE
        elif self._state == LDWState.DEPARTED:
            if abs_cte < cfg.departure_threshold_m - cfg.departure_hysteresis_m:
                self._state = LDWState.IN_LANE

        # ── Side detection ──────────────────────────────────────────────────
        if self._state == LDWState.IN_LANE:
            self._side = "none"
        else:
            if cte > 0.0:
                self._side = "left"
            elif cte < 0.0:
                self._side = "right"
            else:
                self._side = "none"

        # ── Warning active logic ────────────────────────────────────────────
        # warning_active เมื่ออยู่ใน DEPARTING หรือ DEPARTED (และ timer ยังไม่หมด)
        if self._state == LDWState.DEPARTING:
            self._warning_active = True
            self._warning_timer_s = cfg.warning_duration_s
        elif self._state == LDWState.DEPARTED:
            self._warning_timer_s = max(self._warning_timer_s - dt, 0.0)
            self._warning_active = self._warning_timer_s > 0.0
        else:  # IN_LANE
            self._warning_timer_s = 0.0
            self._warning_active = False

        # ── Steering assist (corrective nudge) ──────────────────────────────
        # เมื่อ DEPARTING ให้ steer กลับเข้าเลน: -sign(cte) * gain * min(|cte|/thr, 1)
        if self._state == LDWState.DEPARTING:
            ratio = min(abs_cte / cfg.departure_threshold_m, 1.0)
            self._steering_assist = (
                -math.copysign(cfg.lka_pro_assist_gain * ratio, cte)
            )
        elif self._state == LDWState.DEPARTED:
            # ยังคง assist แต่ limit ที่ 1.0
            ratio = min(abs_cte / cfg.departure_threshold_m, 1.0)
            self._steering_assist = (
                -math.copysign(cfg.lka_pro_assist_gain * ratio, cte)
            )
        else:
            self._steering_assist = 0.0

        # ── Blink toggle ────────────────────────────────────────────────────
        if self._warning_active:
            self._blink_phase_s += dt
            blink_period = 1.0 / max(cfg.blink_rate_hz, 1e-6)
            # toggle เมื่อ phase เกินครึ่ง period
            if self._blink_phase_s >= blink_period / 2.0:
                self._blink_on = not self._blink_on
                self._blink_phase_s = 0.0
        else:
            self._blink_on = False
            self._blink_phase_s = 0.0

        return self._build_output()

    def get_status(self) -> dict:
        """Return current LDW status snapshot."""
        return {
            "state": self._state.name if isinstance(self._state, LDWState) else self._state,
            "side": self._side,
            "warning_active": self._warning_active,
            "steering_assist": self._steering_assist,
        }

    def reset(self) -> None:
        """Reset state machine to IN_LANE."""
        self._state = LDWState.IN_LANE
        self._side = "none"
        self._warning_active = False
        self._warning_timer_s = 0.0
        self._blink_on = False
        self._blink_phase_s = 0.0
        self._steering_assist = 0.0
        self._last_timestamp = float("nan")

    # ── Internal ────────────────────────────────────────────────────────────

    def _build_output(self) -> dict:
        return {
            "state": self._state.name if isinstance(self._state, LDWState) else self._state,
            "warning_active": self._warning_active,
            "side": self._side,
            "steering_assist": self._steering_assist,
            "blink_on": self._blink_on,
        }


class LKAPro:
    """Enhanced Lane Keeping Assist — proactive corrective steering.

    ใช้ LDW state machine เป็น core แล้วเพิ่ม EMA-smoothed assist
    ที่สามารถ scale ได้ถึง 2x gain เมื่อ CTE ใหญ่มาก
    """

    EMA_ALPHA = 0.3

    def __init__(self, config: Optional[LDWConfig] = None) -> None:
        self.config = config if config is not None else LDWConfig()
        self._ldw = LaneDepartureWarning(self.config)
        self._last_assist: float = 0.0
        self._last_cte: float = 0.0
        self._last_state: LDWState = LDWState.IN_LANE

    def compute_assist(
        self,
        cte: float,
        heading_err: float,
        speed_ms: float,
        timestamp: float,
    ) -> float:
        """Compute proactive steering assist (rad).

        Returns additive steering assist ที่จะนำไปบวกกับ MPC output.
        คืน 0 เมื่อความเร็วต่ำกว่า 5 m/s
        """
        cfg = self.config

        # อัปเดต LDW state machine (ใช้สำหรับ debug/status)
        self._ldw.update(cte, heading_err, speed_ms, timestamp)
        self._last_state = self._ldw._state
        self._last_cte = cte

        # ไม่ assist เมื่อความเร็วต่ำ
        if speed_ms < LDW_MIN_SPEED_MS:
            raw_assist = 0.0
        elif abs(cte) > cfg.departure_threshold_m:
            ratio = min(abs(cte) / cfg.departure_threshold_m, 2.0)
            raw_assist = -math.copysign(cfg.lka_pro_assist_gain * ratio, cte)
        else:
            raw_assist = 0.0

        # EMA smoothing
        self._last_assist = (
            self.EMA_ALPHA * raw_assist + (1.0 - self.EMA_ALPHA) * self._last_assist
        )
        return self._last_assist

    def get_debug_info(self) -> dict:
        """Return debug info for monitoring."""
        return {
            "last_cte": self._last_cte,
            "last_assist": self._last_assist,
            "last_state": (
                self._last_state.name
                if isinstance(self._last_state, LDWState)
                else self._last_state
            ),
        }
