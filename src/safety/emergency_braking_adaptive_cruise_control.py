"""
AEB (Automatic Emergency Braking) + ACC (Adaptive Cruise Control).

AEB: ตรวจจับสิ่งกีดขวางข้างหน้าในเลนเดียวกัน → เบรกฉุกเมื่อ TTC < threshold
ACC: ปรับ target speed ตากรถข้างหน้า → รักษาระยะปลอดภัย (time gap)

ใช้ bridge/obstacles.py สำหรับดึง obstacle list จาก CARLA
คำนวณใน vehicle frame (x=forward, y=left)
"""
import logging
import math
from dataclasses import dataclass
from typing import List, Tuple


from bridge.obstacles import Obstacle, ObstacleType

logger = logging.getLogger(__name__)

# ── AEB parameters ──────────────────────────────────────────────────────────
AEB_TTC_THRESHOLD_S = 2.5       # TTC < นี่ → AEB active
AEB_TTC_CRITICAL_S = 1.0        # TTC < นี่ → full brake
AEB_BRAKE_PROPORTIONAL = 0.6    # brake gain for proportional AEB
AEB_FULL_BRAKE = 1.0            # full brake
AEB_LATERAL_THRESHOLD_M = 1.8   # ครึ่งหนึ่งของ lane width (~3.5m) + margin

# ── ACC parameters ──────────────────────────────────────────────────────────
ACC_TIME_GAP_S = 1.8            # time gap ที่ต้องการ (s) — ระยะ = v * time_gap
ACC_MIN_DISTANCE_M = 5.0        # ระยะขั้นต่ำ (m) แม้ตัวรถหยุดนิ่ง
ACC_DECEL_LIMIT_MS2 = 3.0       # จำกัด deceleration ของ ACC (นุ่มกว่า AEB)
ACC_SPEED_REDUCTION_FACTOR = 0.9  # ลด speed 10% เมื่อ ACC active


@dataclass
class AEBResult:
    """ผลการตรวจจับ AEB"""
    active: bool = False                # AEB เปิดใช้งานหรือไม่
    brake_override: float = 0.0         # ค่า brake ที่ AEB กำหนด [0, 1]
    ttc: float = float('inf')           # time-to-collision (s)
    obstacle_id: int = -1               # id ของสิ่งกีดขวางที่ trigger
    reason: str = ""                    # เหตุผลที่ trigger


@dataclass
class ACCResult:
    """ผลการตรวจจับ ACC"""
    active: bool = False                # ACC เปิดใช้งานหรือไม่
    target_speed_ms: float = 0.0        # target speed ที่ ACC กำหนด
    distance_m: float = float('inf')    # ระยะรถข้างหน้า (m)
    closing_rate_ms: float = 0.0        # อัตราเข้าหา (m/s) — บวก = เข้าใกล้
    obstacle_id: int = -1


class AEBACC:
    """
    AEB + ACC — ทำงานอิสระจาก MPC
    รับ obstacle list + ego state → ส่ง brake override + target speed
    """

    def __init__(
        self,
        ttc_threshold: float = AEB_TTC_THRESHOLD_S,
        ttc_critical: float = AEB_TTC_CRITICAL_S,
        time_gap: float = ACC_TIME_GAP_S,
        min_distance: float = ACC_MIN_DISTANCE_M,
    ):
        self.ttc_threshold = ttc_threshold
        self.ttc_critical = ttc_critical
        self.time_gap = time_gap
        self.min_distance = min_distance
        self._last_aeb = AEBResult()
        self._last_acc = ACCResult()

    def _obstacle_in_lane(
        self,
        obs: Obstacle,
        ego_x: float,
        ego_y: float,
        ego_heading: float,
    ) -> Tuple[bool, float, float]:
        """
        ตรวจว่า obstacle อยู่ในเลนเดียวกันข้างหน้าหรือไม่
        คืน (in_lane, distance_forward, lateral_offset)
        """
        dx = obs.position[0] - ego_x
        dy = obs.position[1] - ego_y
        cos_h = math.cos(ego_heading)
        sin_h = math.sin(ego_heading)
        forward = dx * cos_h + dy * sin_h
        lateral = -dx * sin_h + dy * cos_h
        in_lane = (
            forward > 0.0
            and abs(lateral) < AEB_LATERAL_THRESHOLD_M
        )
        return in_lane, forward, lateral

    def _compute_ttc(
        self,
        distance: float,
        closing_rate: float,
    ) -> float:
        """Time-to-collision: คืน inf ถ้าไม่เข้าใกล้"""
        if closing_rate <= 0.01:
            return float('inf')
        return distance / closing_rate

    def check_aeb(
        self,
        obstacles: List[Obstacle],
        ego_x: float,
        ego_y: float,
        ego_heading: float,
        ego_speed: float,
    ) -> AEBResult:
        """
        ตรวจจับ AEB — คืน brake override ถ้าสิ่งกีดขวางใกล้เกินไป
        """
        result = AEBResult()
        closest_ttc = float('inf')
        closest_obs_id = -1

        for obs in obstacles:
            if obs.type not in (ObstacleType.VEHICLE, ObstacleType.BICYCLE,
                                ObstacleType.PEDESTRIAN, ObstacleType.UNKNOWN_MOVABLE):
                continue

            in_lane, forward, _ = self._obstacle_in_lane(
                obs, ego_x, ego_y, ego_heading
            )
            if not in_lane:
                continue

            # closing rate = ego speed - obstacle forward speed
            obs_forward_speed = (
                obs.velocity[0] * math.cos(ego_heading)
                + obs.velocity[1] * math.sin(ego_heading)
            )
            closing_rate = ego_speed - obs_forward_speed
            ttc = self._compute_ttc(forward, closing_rate)

            if ttc < closest_ttc:
                closest_ttc = ttc
                closest_obs_id = obs.id

        # AEB decision
        if closest_ttc < self.ttc_threshold:
            result.active = True
            result.ttc = closest_ttc
            result.obstacle_id = closest_obs_id

            if closest_ttc < self.ttc_critical:
                result.brake_override = AEB_FULL_BRAKE
                result.reason = f"AEB CRITICAL: TTC={closest_ttc:.2f}s < {self.ttc_critical}s"
                logger.warning(result.reason)
            else:
                # Proportional brake: ยิ่งใกล้ ยิ่งเบรกแรง
                ratio = (self.ttc_threshold - closest_ttc) / (
                    self.ttc_threshold - self.ttc_critical
                )
                result.brake_override = min(
                    AEB_BRAKE_PROPORTIONAL * ratio, 0.8
                )
                result.reason = f"AEB: TTC={closest_ttc:.2f}s, brake={result.brake_override:.2f}"
                logger.warning(result.reason)

        self._last_aeb = result
        return result

    def check_acc(
        self,
        obstacles: List[Obstacle],
        ego_x: float,
        ego_y: float,
        ego_heading: float,
        ego_speed: float,
        nominal_target_ms: float,
    ) -> ACCResult:
        """
        ตรวจจับ ACC — ปรับ target speed ตามรถข้างหน้า
        คืน target_speed_ms ที่ควรใช้ (อาจต่ำกว่า nominal)
        """
        result = ACCResult(target_speed_ms=nominal_target_ms)
        closest_dist = float('inf')
        closest_obs_id = -1
        closest_closing = 0.0

        for obs in obstacles:
            if obs.type not in (ObstacleType.VEHICLE, ObstacleType.BICYCLE):
                continue

            in_lane, forward, _ = self._obstacle_in_lane(
                obs, ego_x, ego_y, ego_heading
            )
            if not in_lane:
                continue

            if forward < closest_dist:
                obs_forward_speed = (
                    obs.velocity[0] * math.cos(ego_heading)
                    + obs.velocity[1] * math.sin(ego_heading)
                )
                closest_dist = forward
                closest_obs_id = obs.id
                closest_closing = ego_speed - obs_forward_speed

        if closest_dist < float('inf'):
            result.active = True
            result.distance_m = closest_dist
            result.closing_rate_ms = closest_closing
            result.obstacle_id = closest_obs_id

            # ระยะปลอดภัย = max(min_distance, speed * time_gap)
            safe_distance = max(
                self.min_distance,
                ego_speed * self.time_gap,
            )

            if closest_dist < safe_distance:
                # ลด target speed ตามระยะ
                lead_speed = ego_speed - closest_closing
                # ถ้ารถข้างหน้าช้ากว่า → ลด target ลง
                if lead_speed < nominal_target_ms:
                    result.target_speed_ms = max(
                        0.0,
                        lead_speed * ACC_SPEED_REDUCTION_FACTOR,
                    )
                logger.info(
                    "ACC: dist=%.1fm, safe=%.1fm, target=%.1f m/s (nominal=%.1f)",
                    closest_dist, safe_distance,
                    result.target_speed_ms, nominal_target_ms,
                )

        self._last_acc = result
        return result

    def get_status(self) -> dict:
        """สถานะ AEB + ACC สำหรับ dashboard"""
        return {
            'aeb_active': self._last_aeb.active,
            'aeb_ttc': self._last_aeb.ttc,
            'aeb_brake': self._last_aeb.brake_override,
            'acc_active': self._last_acc.active,
            'acc_distance': self._last_acc.distance_m,
            'acc_target_speed': self._last_acc.target_speed_ms,
        }

    def get_collision_warning_level(self) -> str:
        """ระดับคำเนตยนต์การชน (none | caution | warning | critical) สำหรับ ADASManager."""
        ttc = self._last_aeb.ttc
        if self._last_aeb.active and ttc < self.ttc_critical:
            return "critical"
        if self._last_aeb.active:
            return "warning"
        if ttc < self.ttc_threshold * 1.5:
            return "caution"
        return "none"

    def reset(self):
        self._last_aeb = AEBResult()
        self._last_acc = ACCResult()
