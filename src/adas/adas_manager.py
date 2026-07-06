"""
ADAS Manager — รวมทุกฟีเจอร์ ADAS ในที่เดียว

Priority (สูง → ต่ำ):
1. AEB (emergency brake) — override brake ได้โดยตรง
2. Traffic Light (red/yellow) — brake หรือ slow
3. TJA + Stop & Go — ปรับ target speed ในรถติด
4. ACC — ปรับ target speed ตามรถข้างหน้า
5. LKA Pro — steering assist additive
6. LDW — warning only
7. BSW + LCA — warning only

ทุกฟีเจอร์ทำงานอิสระ ไม่ขัดกัน:
- AEB + Traffic Light ใช้ brake override
- TJA + ACC ใช้ target speed reduction
- LKA Pro ใช้ steering assist (additive)
- LDW + BSW + LCA เป็น warning เท่านั้น
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from safety.emergency_braking_adaptive_cruise_control import AEBACC
from safety.lane_departure_warning import LaneDepartureWarning, LKAPro, LDWConfig
from safety.blind_spot_warning_lane_change_assist import BlindSpotWarning, LaneChangeAssist, BSWConfig
from safety.traffic_jam_assist_stop_go import TrafficJamAssist, StopAndGo, TJAConfig
from perception.traffic_sign_recognition import (
    TrafficSignRecognizer,
    TrafficLightController,
)

logger = logging.getLogger(__name__)


@dataclass
class ADASControlOutput:
    """ผลจาก ADAS Manager ต่อ frame"""
    # Override values (-1 = no override, ใช้ค่าจาก MPC)
    steer_override: float = -1.0
    throttle_override: float = -1.0
    brake_override: float = -1.0
    target_speed_override: float = -1.0

    # Additive values (เพิ่มเติมจาก MPC output)
    steer_assist: float = 0.0

    # Status สำหรับ dashboard
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
    bsw_safe_left: bool = True
    bsw_safe_right: bool = True
    tsr_speed_limit_kmh: Optional[float] = None
    tsr_traffic_light: str = "unknown"
    tsr_traffic_light_distance: float = -1.0
    tsr_action: str = "none"
    tja_state: str = "inactive"
    tja_active: bool = False
    stop_and_go_stopped: bool = False
    active_features: List[str] = field(default_factory=list)


class ADASManager:
    """
    รวมทุกฟีเจอร์ ADAS — เรียก update() ต่อ frame
    รับ: obstacles, ego state, lane info, CARLA world
    คืน: ADASControlOutput (override + assist + status)
    """

    def __init__(
        self,
        aeb_acc: Optional[AEBACC] = None,
        ldw_config: Optional[LDWConfig] = None,
        bsw_config: Optional[BSWConfig] = None,
        tja_config: Optional[TJAConfig] = None,
        enable_tsr: bool = True,
    ):
        self.aeb_acc = aeb_acc or AEBACC()
        self.ldw = LaneDepartureWarning(ldw_config)
        self.lka_pro = LKAPro(ldw_config)
        self.bsw = BlindSpotWarning(bsw_config)
        self.lca = LaneChangeAssist(bsw_config)
        self.tja = TrafficJamAssist(tja_config)
        self.stop_and_go = StopAndGo(tja_config)
        self.tsr = TrafficSignRecognizer() if enable_tsr else None
        self.traffic_light_ctrl = TrafficLightController()

        self._last_timestamp = 0.0
        self._last_speed_limit = None
        logger.info("ADASManager initialized — full ADAS suite active")

    def update(
        self,
        obstacles: List[Any],
        ego_x: float,
        ego_y: float,
        ego_heading: float,
        ego_speed: float,
        cte: float,
        heading_err: float,
        curvature: float,
        timestamp: float,
        carla_world: Optional[Any] = None,
        vehicle_transform: Optional[Any] = None,
        nominal_target_speed_ms: float = 8.33,
    ) -> ADASControlOutput:
        """
        รันทุกฟีเจอร์ ADAS ต่อ frame
        คืน ADASControlOutput ที่มี override + assist + status
        """
        out = ADASControlOutput()
        active = []

        # ── 1. AEB (highest priority) ─────────────────────────────────────
        aeb_result = self.aeb_acc.check_aeb(
            obstacles, ego_x, ego_y, ego_heading, ego_speed
        )
        out.aeb_active = aeb_result.active
        out.aeb_ttc = aeb_result.ttc
        out.aeb_warning_level = self.aeb_acc.get_collision_warning_level()
        if aeb_result.active:
            out.brake_override = aeb_result.brake_override
            out.throttle_override = 0.0
            active.append("AEB")

        # ── 2. ACC (target speed adjustment) ──────────────────────────────
        acc_result = self.aeb_acc.check_acc(
            obstacles, ego_x, ego_y, ego_heading, ego_speed,
            nominal_target_speed_ms
        )
        out.acc_active = acc_result.active
        out.acc_distance_m = acc_result.distance_m
        if acc_result.active and acc_result.target_speed_ms < nominal_target_speed_ms:
            out.target_speed_override = acc_result.target_speed_ms
            out.acc_target_speed_ms = acc_result.target_speed_ms
            active.append("ACC")
        else:
            out.acc_target_speed_ms = nominal_target_speed_ms

        # ── 3. TSR + Traffic Light ────────────────────────────────────────
        if self.tsr and carla_world is not None and vehicle_transform is not None:
            try:
                signs_lights = self.tsr.detect_from_carla(
                    carla_world, vehicle_transform
                )
                signs = [s for s in signs_lights if hasattr(s, 'sign_type')]
                lights = [s for s in signs_lights if hasattr(s, 'state')]

                speed_limit = self.tsr.detect_speed_limit(signs)
                if speed_limit is not None:
                    out.tsr_speed_limit_kmh = speed_limit
                    speed_limit_ms = speed_limit / 3.6
                    # ลด target speed ถ้าเห็นป้ายจำกัดความเร็ว
                    if speed_limit_ms < out.target_speed_override or out.target_speed_override < 0:
                        if speed_limit_ms < nominal_target_speed_ms:
                            out.target_speed_override = speed_limit_ms
                            active.append("TSR")

                traffic_light = self.tsr.detect_traffic_light(lights)
                if traffic_light is not None:
                    out.tsr_traffic_light = traffic_light.state.name.lower()
                    out.tsr_traffic_light_distance = traffic_light.distance_m

                    stop_line_dist = self.tsr.get_stop_line_distance(
                        carla_world, vehicle_transform
                    )
                    tl_ctrl = self.traffic_light_ctrl.compute_control(
                        traffic_light, ego_speed, stop_line_dist
                    )
                    out.tsr_action = tl_ctrl.get("reason", "none")

                    if tl_ctrl["active"]:
                        # Traffic light override มี priority สูง แต่ต่ำกว่า AEB
                        if aeb_result.active is False:
                            if tl_ctrl["brake"] > out.brake_override:
                                out.brake_override = tl_ctrl["brake"]
                            if tl_ctrl["throttle"] >= 0:
                                out.throttle_override = tl_ctrl["throttle"]
                            if tl_ctrl["target_speed_ms"] >= 0:
                                out.target_speed_override = tl_ctrl["target_speed_ms"]
                        active.append("TrafficLight")
            except Exception as e:
                logger.debug(f"TSR update skipped: {e}")

        # ── 4. TJA + Stop & Go ────────────────────────────────────────────
        # ใช้ ACC distance เป็น lead distance ถ้ามี
        lead_distance = acc_result.distance_m if acc_result.active else float('inf')
        lead_speed = ego_speed - acc_result.closing_rate_ms if acc_result.active else 0.0

        tja_result = self.tja.update(
            lead_distance, lead_speed, ego_speed, timestamp
        )
        out.tja_state = tja_result["state"]
        out.tja_active = tja_result["active"]
        if tja_result["active"]:
            if tja_result["target_speed_ms"] >= 0:
                tja_target = tja_result["target_speed_ms"]
                if out.target_speed_override < 0:
                    out.target_speed_override = tja_target
                else:
                    out.target_speed_override = min(
                        out.target_speed_override, tja_target
                    )
            if tja_result["throttle"] >= 0 and not aeb_result.active:
                if out.throttle_override < 0:
                    out.throttle_override = tja_result["throttle"]
            if tja_result["brake"] >= 0 and not aeb_result.active:
                if tja_result["brake"] > out.brake_override:
                    out.brake_override = tja_result["brake"]
            active.append("TJA")

        # Stop & Go
        sg_result = self.stop_and_go.update(
            lead_distance, lead_speed, ego_speed, timestamp
        )
        out.stop_and_go_stopped = sg_result["stopped"]
        if sg_result["stopped"] and not aeb_result.active:
            if sg_result["brake"] > out.brake_override:
                out.brake_override = sg_result["brake"]
            active.append("StopAndGo")

        # ── 5. LKA Pro (steering assist, additive) ────────────────────────
        steer_assist = self.lka_pro.compute_assist(
            cte, heading_err, ego_speed, timestamp
        )
        out.lka_pro_assist = steer_assist
        out.steer_assist = steer_assist
        if abs(steer_assist) > 0.01:
            active.append("LKAPro")

        # ── 6. LDW (warning only) ─────────────────────────────────────────
        ldw_result = self.ldw.update(cte, heading_err, ego_speed, timestamp)
        out.ldw_state = ldw_result["state"]
        out.ldw_warning_active = ldw_result["warning_active"]
        out.ldw_side = ldw_result["side"]
        if ldw_result["warning_active"]:
            active.append("LDW")

        # ── 7. BSW + LCA (warning only) ───────────────────────────────────
        bsw_result = self.bsw.update(
            obstacles, ego_x, ego_y, ego_heading, ego_speed, timestamp
        )
        out.bsw_left_alert = bsw_result["left_alert"]
        out.bsw_right_alert = bsw_result["right_alert"]
        out.bsw_safe_left = bsw_result["safe_to_change_left"]
        out.bsw_safe_right = bsw_result["safe_to_change_right"]
        if bsw_result["left_alert"] != "clear" or bsw_result["right_alert"] != "clear":
            active.append("BSW")

        out.active_features = active
        self._last_timestamp = timestamp
        return out

    def apply_to_control(
        self,
        steer: float,
        throttle: float,
        brake: float,
        adas_out: ADASControlOutput,
    ) -> Tuple[float, float, float, float]:
        """
        ประยุกต์ ADAS output ลงบน MPC control output
        คืน (steer, throttle, brake, target_speed)
        """
        # AEB / Traffic Light brake override
        if adas_out.brake_override > brake:
            brake = adas_out.brake_override
            throttle = 0.0

        if adas_out.throttle_override >= 0:
            throttle = adas_out.throttle_override

        # LKA Pro steering assist (additive)
        steer += adas_out.steer_assist
        steer = float(np.clip(steer, -1.0, 1.0))

        # Target speed (ส่งกลับเพื่อใช้ใน frame ถัดไป)
        target_speed = adas_out.target_speed_override

        return steer, throttle, brake, target_speed

    def get_status(self) -> Dict[str, Any]:
        """สถานะ ADAS ทั้งหมดสำหรับ dashboard"""
        return {
            "aeb": self.aeb_acc.get_status(),
            "ldw": self.ldw.get_status(),
            "lka_pro": self.lka_pro.get_debug_info(),
            "bsw": self.bsw.get_status(),
            "tja": self.tja.get_status(),
            "stop_and_go": self.stop_and_go.get_debug_info(),
            "tsr_active": self.tsr is not None,
        }

    def reset(self):
        """Reset ทุกฟีเจอร์"""
        self.aeb_acc.reset()
        self.ldw.reset()
        self.bsw.reset()
        self.tja.reset()
        self._last_timestamp = 0.0
