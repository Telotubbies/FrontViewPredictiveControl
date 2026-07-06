"""
Traffic Jam Assist (TJA) + Stop & Go.

TJA: รักษาระยะตามรถข้างหน้าในจราจรติดขัด (ความเร็วต่ำ)
Stop & Go: หยุดและเริ่มขับต่ออย่างนุ่มนวลเมื่อรถข้างหน้าหยุด/เคลื่อนที่

ทำงานเฉพาะความเร็วต่ำ (activate < 30 km/h) มี hysteresis ป้องกันการกระพริบ
"""
import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)

INF = math.inf  # convenience alias for "no lead"


@dataclass
class TJAConfig:
    """Configuration for Traffic Jam Assist and Stop & Go."""

    activate_speed_ms: float = 8.33       # 30 km/h — active เฉพาะต่ำกว่านี้
    deactivate_speed_ms: float = 12.0     # hysteresis: deactivate เมื่อเร็วกว่า 43 km/h
    min_follow_distance_m: float = 3.0    # ระยะขั้นต่ำจากรถข้างหน้า (m)
    time_gap_s: float = 1.2               # time gap สั้นกว่า ACC (ตามใกล้ในจราจร)
    stop_hold_time_s: float = 10.0        # หยุดรอกี่วินาทีก่อนปล่อย
    creep_speed_ms: float = 1.0           # ความเร็ว creep เมื่อเริ่มเคลื่อนที่
    stop_brake: float = 0.3               # brake สำหรับ holding stop
    creep_throttle: float = 0.15          # throttle สำหรับ creeping
    lead_detect_max_m: float = field(
        default=50.0,
        metadata={"desc": "ระยะสูงสุดที่ถือว่าตรวจจับรถข้างหน้าได้ (m)"},
    )


class TJAState(IntEnum):
    """State machine states for Traffic Jam Assist."""

    INACTIVE = 0   # ความเร็วสูงเกินไป
    ACTIVE = 1     # ตามรถข้างหน้า
    STOPPED = 2    # หยุดนิ่ง กด brake ค้าง
    CREEPING = 3   # เคลื่อนที่ช้าๆ
    NO_LEAD = 4    # ไม่พบรถข้างหน้า คงความเร็วต่ำ


class TrafficJamAssist:
    """
    Traffic Jam Assist — ติดตามรถข้างหน้าในจราจรติดขัด

    State machine:
      INACTIVE ↔ ACTIVE ↔ STOPPED ↔ CREEPING
                 ↕
               NO_LEAD
    """

    def __init__(self, config: Optional[TJAConfig] = None) -> None:
        self.config = config if config is not None else TJAConfig()
        self._state: TJAState = TJAState.INACTIVE
        self._state_enter_time: float = 0.0
        self._last_timestamp: float = 0.0
        self._stop_start_time: float = 0.0
        self._last_target_speed: float = 0.0
        self._last_throttle: float = 0.0
        self._last_brake: float = 0.0

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _lead_detected(self, lead_distance_m: float) -> bool:
        """ตรวจว่ามีรถข้างหน้าหรือไม่ (0 <= ระยะ < lead_detect_max_m และไม่ใช่ inf)."""
        return 0.0 <= lead_distance_m < self.config.lead_detect_max_m

    def _time_in_state(self, timestamp: float) -> float:
        """เวลาที่อยู่ใน state ปัจจุบัน (s)."""
        return max(0.0, timestamp - self._state_enter_time)

    def _enter_state(self, new_state: TJAState, timestamp: float) -> None:
        """เปลี่ยน state และบันทึกเวลาเข้า."""
        if new_state != self._state:
            logger.debug(
                "TJA %s -> %s @ %.2f", self._state.name, new_state.name, timestamp
            )
        self._state = new_state
        self._state_enter_time = timestamp
        if new_state == TJAState.STOPPED:
            self._stop_start_time = timestamp

    # ── Main update ────────────────────────────────────────────────────────────

    def update(
        self,
        lead_distance_m: float,
        lead_speed_ms: float,
        ego_speed_ms: float,
        timestamp: float,
    ) -> dict:
        """
        อัปเดต state machine และคืนคำสั่งควบคุม

        Returns:
            dict: state, throttle, brake, target_speed_ms, active, reason
        """
        cfg = self.config
        lead_detected = self._lead_detected(lead_distance_m)
        time_in_stop = (
            max(0.0, timestamp - self._stop_start_time)
            if self._state == TJAState.STOPPED
            else 0.0
        )

        # ── State transitions ──────────────────────────────────────────────────
        if self._state == TJAState.INACTIVE:
            if ego_speed_ms < cfg.activate_speed_ms and lead_detected:
                self._enter_state(TJAState.ACTIVE, timestamp)

        elif self._state == TJAState.ACTIVE:
            if ego_speed_ms > cfg.deactivate_speed_ms:
                self._enter_state(TJAState.INACTIVE, timestamp)
            elif not lead_detected:
                self._enter_state(TJAState.NO_LEAD, timestamp)
            elif (
                lead_distance_m < cfg.min_follow_distance_m
                and lead_speed_ms < 0.5
            ):
                self._enter_state(TJAState.STOPPED, timestamp)

        elif self._state == TJAState.STOPPED:
            if ego_speed_ms > cfg.deactivate_speed_ms:
                self._enter_state(TJAState.INACTIVE, timestamp)
            elif not lead_detected:
                self._enter_state(TJAState.NO_LEAD, timestamp)
            elif lead_speed_ms > 1.0 or time_in_stop > cfg.stop_hold_time_s:
                self._enter_state(TJAState.CREEPING, timestamp)

        elif self._state == TJAState.CREEPING:
            if ego_speed_ms > cfg.deactivate_speed_ms:
                self._enter_state(TJAState.INACTIVE, timestamp)
            elif not lead_detected:
                self._enter_state(TJAState.NO_LEAD, timestamp)
            elif (
                lead_distance_m < cfg.min_follow_distance_m
                and lead_speed_ms < 0.5
            ):
                self._enter_state(TJAState.STOPPED, timestamp)
            elif ego_speed_ms > 2.0:
                self._enter_state(TJAState.ACTIVE, timestamp)

        elif self._state == TJAState.NO_LEAD:
            if lead_detected:
                self._enter_state(TJAState.ACTIVE, timestamp)
            elif ego_speed_ms > cfg.deactivate_speed_ms:
                self._enter_state(TJAState.INACTIVE, timestamp)

        # ── Output per state ───────────────────────────────────────────────────
        if self._state == TJAState.INACTIVE:
            throttle, brake = -1.0, -1.0
            target_speed = -1.0
            active = False
            reason = "speed_too_high"

        elif self._state == TJAState.ACTIVE:
            # ตามรถข้างหน้า: target = lead_speed, รักษา time_gap
            desired_distance = max(
                cfg.min_follow_distance_m,
                ego_speed_ms * cfg.time_gap_s,
            )
            target_speed = lead_speed_ms
            if lead_distance_m < desired_distance:
                # ใกล้เกินไป → ลด speed
                throttle = 0.0
                brake = min(1.0, 0.2 + 0.1 * (desired_distance - lead_distance_m))
            else:
                # ระยะปลอดภัย → ตาม speed
                throttle = 0.2 if ego_speed_ms < target_speed else 0.0
                brake = 0.0
            active = True
            reason = "following_lead"

        elif self._state == TJAState.STOPPED:
            throttle = 0.0
            brake = cfg.stop_brake
            target_speed = 0.0
            active = True
            reason = "stopped_holding_brake"

        elif self._state == TJAState.CREEPING:
            throttle = cfg.creep_throttle
            brake = 0.0
            target_speed = cfg.creep_speed_ms
            active = True
            reason = "creeping_forward"

        else:  # NO_LEAD
            throttle = 0.1
            brake = 0.0
            target_speed = cfg.creep_speed_ms
            active = True
            reason = "no_lead_slow_forward"

        self._last_target_speed = target_speed
        self._last_throttle = throttle
        self._last_brake = brake
        self._last_timestamp = timestamp

        return {
            "state": self._state,
            "throttle": throttle,
            "brake": brake,
            "target_speed_ms": target_speed,
            "active": active,
            "reason": reason,
        }

    # ── Status / reset ─────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """คืนสถานะปัจจุบัน."""
        return {
            "state": self._state,
            "time_in_state": (
                self._last_timestamp - self._state_enter_time
                if self._last_timestamp > 0
                else 0.0
            ),
            "last_target_speed": self._last_target_speed,
        }

    def reset(self) -> None:
        """รีเซ็ตเป็น INACTIVE."""
        self._state = TJAState.INACTIVE
        self._state_enter_time = 0.0
        self._last_timestamp = 0.0
        self._stop_start_time = 0.0
        self._last_target_speed = 0.0
        self._last_throttle = 0.0
        self._last_brake = 0.0


class StopAndGo:
    """
    Stop & Go — เน้นการหยุดและเริ่มขับต่ออย่างนุ่มนวล

    ใช้ TJAConfig เดียวกับ TJA แต่ state machine ง่ายกว่า:
      moving → stopped → restarting (throttle ramp) → moving
    """

    def __init__(self, config: Optional[TJAConfig] = None) -> None:
        self.config = config if config is not None else TJAConfig()
        self._stopped: bool = False
        self._stop_start_time: float = 0.0
        self._restart_start_time: float = 0.0
        self._restart_ramp: float = 0.0
        self._last_throttle: float = 0.0
        self._last_timestamp: float = 0.0

    # ── Main update ────────────────────────────────────────────────────────────

    def update(
        self,
        lead_distance_m: float,
        lead_speed_ms: float,
        ego_speed_ms: float,
        timestamp: float,
    ) -> dict:
        """
        อัปเดต Stop & Go และคืนคำสั่งควบคุม

        Returns:
            dict: throttle, brake, target_speed_ms, stopped, restart_ramp
        """
        cfg = self.config

        # ── Decide stopped ─────────────────────────────────────────────────────
        if (
            lead_distance_m < cfg.min_follow_distance_m
            and lead_speed_ms < 0.5
        ):
            if not self._stopped:
                self._stopped = True
                self._stop_start_time = timestamp
                self._restart_ramp = 0.0
        else:
            # รถข้างหน้าเคลื่อนที่ → restart
            if self._stopped and lead_speed_ms > 1.0:
                self._stopped = False
                self._restart_start_time = timestamp
                self._restart_ramp = 0.0

        # ── Output ─────────────────────────────────────────────────────────────
        if self._stopped:
            throttle = 0.0
            brake = cfg.stop_brake
            target_speed = 0.0
            restart_ramp = 0.0
        else:
            # กำลัง restart → ramp throttle 0.0 → 0.2 ใน 1 วินาที
            if self._restart_start_time > 0:
                elapsed = timestamp - self._restart_start_time
                restart_ramp = min(1.0, elapsed / 1.0)
                throttle = 0.2 * restart_ramp
            else:
                restart_ramp = 1.0
                throttle = 0.2
            brake = 0.0
            target_speed = lead_speed_ms if lead_speed_ms > 0 else cfg.creep_speed_ms

        self._restart_ramp = restart_ramp
        self._last_throttle = throttle
        self._last_timestamp = timestamp

        return {
            "throttle": throttle,
            "brake": brake,
            "target_speed_ms": target_speed,
            "stopped": self._stopped,
            "restart_ramp": restart_ramp,
        }

    # ── Debug ──────────────────────────────────────────────────────────────────

    def get_debug_info(self) -> dict:
        """คืนข้อมูล debug."""
        stopped_duration = (
            self._last_timestamp - self._stop_start_time
            if self._stopped
            else 0.0
        )
        return {
            "stopped_duration": stopped_duration,
            "restart_progress": self._restart_ramp,
            "last_throttle": self._last_throttle,
        }
