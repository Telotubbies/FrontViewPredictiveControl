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
from typing import Dict, List, Optional, Tuple


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

# ── Pedestrian / bicycle parameters ──────────────────────────────────────────
PEDESTRIAN_TTC_FACTOR = 0.5       # ลด TTC threshold 50% สำหรับคนเดิน
BICYCLE_WARNING_LATERAL_MIN_M = 1.0  # ขอบล่าง lateral สำหรับ bicycle warning
BICYCLE_WARNING_LATERAL_MAX_M = 1.8  # ขอบบน lateral สำหรับ bicycle warning

# ── Warning level TTC ranges (s) ─────────────────────────────────────────────
WARNING_CAUTION_MIN_S = 2.5
WARNING_CAUTION_MAX_S = 4.0
WARNING_CRITICAL_S = 1.0

# ── Consolidated config dict ─────────────────────────────────────────────────
AEB_ACC_CONFIG: Dict[str, float] = {
    # AEB
    'aeb_ttc_threshold_s': AEB_TTC_THRESHOLD_S,
    'aeb_ttc_critical_s': AEB_TTC_CRITICAL_S,
    'aeb_brake_proportional': AEB_BRAKE_PROPORTIONAL,
    'aeb_full_brake': AEB_FULL_BRAKE,
    'aeb_lateral_threshold_m': AEB_LATERAL_THRESHOLD_M,
    # ACC
    'acc_time_gap_s': ACC_TIME_GAP_S,
    'acc_min_distance_m': ACC_MIN_DISTANCE_M,
    'acc_decel_limit_ms2': ACC_DECEL_LIMIT_MS2,
    'acc_speed_reduction_factor': ACC_SPEED_REDUCTION_FACTOR,
    # Pedestrian priority
    'pedestrian_ttc_factor': PEDESTRIAN_TTC_FACTOR,
    # Bicycle filter
    'bicycle_warning_lateral_min_m': BICYCLE_WARNING_LATERAL_MIN_M,
    'bicycle_warning_lateral_max_m': BICYCLE_WARNING_LATERAL_MAX_M,
    # Warning level TTC ranges
    'warning_caution_min_s': WARNING_CAUTION_MIN_S,
    'warning_caution_max_s': WARNING_CAUTION_MAX_S,
    'warning_critical_s': WARNING_CRITICAL_S,
}


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
        config: Optional[Dict[str, float]] = None,
        ttc_threshold: float = AEB_TTC_THRESHOLD_S,
        ttc_critical: float = AEB_TTC_CRITICAL_S,
        time_gap: float = ACC_TIME_GAP_S,
        min_distance: float = ACC_MIN_DISTANCE_M,
    ):
        # Start from defaults, then apply explicit kwargs, then optional config
        cfg = dict(AEB_ACC_CONFIG)
        cfg['aeb_ttc_threshold_s'] = ttc_threshold
        cfg['aeb_ttc_critical_s'] = ttc_critical
        cfg['acc_time_gap_s'] = time_gap
        cfg['acc_min_distance_m'] = min_distance
        if config is not None:
            cfg.update(config)

        self.ttc_threshold = cfg['aeb_ttc_threshold_s']
        self.ttc_critical = cfg['aeb_ttc_critical_s']
        self.time_gap = cfg['acc_time_gap_s']
        self.min_distance = cfg['acc_min_distance_m']
        self._brake_proportional = cfg['aeb_brake_proportional']
        self._full_brake = cfg['aeb_full_brake']
        self._lateral_threshold = cfg['aeb_lateral_threshold_m']
        self._speed_reduction_factor = cfg['acc_speed_reduction_factor']
        self._pedestrian_ttc_factor = cfg['pedestrian_ttc_factor']
        self._bicycle_warning_lateral_min = cfg['bicycle_warning_lateral_min_m']
        self._bicycle_warning_lateral_max = cfg['bicycle_warning_lateral_max_m']
        self._warning_caution_min = cfg['warning_caution_min_s']
        self._warning_caution_max = cfg['warning_caution_max_s']
        self._warning_critical = cfg['warning_critical_s']
        self._last_aeb = AEBResult()
        self._last_acc = ACCResult()
        self._warning_level = "none"
        self._bicycle_warning = False

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
            and abs(lateral) < self._lateral_threshold
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

        # Track most urgent AEB trigger (highest brake override)
        best_brake = 0.0
        best_ttc = float('inf')
        best_obs_id = -1
        best_reason = ""

        self._bicycle_warning = False

        for obs in obstacles:
            if obs.type not in (ObstacleType.VEHICLE, ObstacleType.BICYCLE,
                                ObstacleType.PEDESTRIAN, ObstacleType.UNKNOWN_MOVABLE):
                continue

            in_lane, forward, lateral = self._obstacle_in_lane(
                obs, ego_x, ego_y, ego_heading
            )
            if not in_lane:
                continue

            # Bicycle filter: at edge of lane → warning flag only, no AEB
            if obs.type == ObstacleType.BICYCLE and (
                self._bicycle_warning_lateral_min < abs(lateral) < self._bicycle_warning_lateral_max
            ):
                self._bicycle_warning = True
                logger.info(
                    "AEB: bicycle at lane edge (lateral=%.2fm) — warning only, no brake",
                    abs(lateral),
                )
                continue

            # closing rate = ego speed - obstacle forward speed
            obs_forward_speed = (
                obs.velocity[0] * math.cos(ego_heading)
                + obs.velocity[1] * math.sin(ego_heading)
            )
            closing_rate = ego_speed - obs_forward_speed
            ttc = self._compute_ttc(forward, closing_rate)

            # Track closest TTC overall (for warning level)
            if ttc < closest_ttc:
                closest_ttc = ttc

            # Pedestrian priority: reduce TTC thresholds by 50%
            # (pedestrians are unpredictable, brake earlier)
            if obs.type == ObstacleType.PEDESTRIAN:
                eff_threshold = self.ttc_threshold * self._pedestrian_ttc_factor
                eff_critical = self.ttc_critical * self._pedestrian_ttc_factor
            else:
                eff_threshold = self.ttc_threshold
                eff_critical = self.ttc_critical

            if ttc < eff_threshold:
                if ttc < eff_critical:
                    brake = self._full_brake
                    reason = (
                        f"AEB CRITICAL: TTC={ttc:.2f}s < {eff_critical:.2f}s"
                        f" (obs={obs.id}, type={obs.type.name})"
                    )
                else:
                    # Proportional brake: ยิ่งใกล้ ยิ่งเบรกแรง
                    ratio = (eff_threshold - ttc) / (eff_threshold - eff_critical)
                    brake = min(self._brake_proportional * ratio, 0.8)
                    reason = (
                        f"AEB: TTC={ttc:.2f}s, brake={brake:.2f}"
                        f" (obs={obs.id}, type={obs.type.name})"
                    )

                if brake > best_brake:
                    best_brake = brake
                    best_ttc = ttc
                    best_obs_id = obs.id
                    best_reason = reason

        # AEB decision — pick the most urgent trigger
        if best_brake > 0.0:
            result.active = True
            result.ttc = best_ttc
            result.obstacle_id = best_obs_id
            result.brake_override = best_brake
            result.reason = best_reason
            logger.warning(best_reason)

        # Update warning level based on closest TTC
        self._warning_level = self._compute_warning_level(closest_ttc)

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
                        lead_speed * self._speed_reduction_factor,
                    )
                logger.info(
                    "ACC: dist=%.1fm, safe=%.1fm, target=%.1f m/s (nominal=%.1f)",
                    closest_dist, safe_distance,
                    result.target_speed_ms, nominal_target_ms,
                )

        self._last_acc = result
        return result

    def _compute_warning_level(self, ttc: float) -> str:
        """
        คำนวณ warning level จาก TTC:
          - critical: TTC < 1.0s
          - warning:  1.0s ≤ TTC < 2.5s
          - caution:  2.5s ≤ TTC < 4.0s
          - none:     TTC ≥ 4.0s
        """
        if ttc < self._warning_critical:
            return "critical"
        if ttc < self._warning_caution_min:
            return "warning"
        if ttc < self._warning_caution_max:
            return "caution"
        return "none"

    def get_collision_warning_level(self) -> str:
        """คืน warning level ปัจจุบัน: 'none', 'caution', 'warning', 'critical'"""
        return self._warning_level

    def get_status(self) -> dict:
        """สถานะ AEB + ACC สำหรับ dashboard"""
        return {
            'aeb_active': self._last_aeb.active,
            'aeb_ttc': self._last_aeb.ttc,
            'aeb_brake': self._last_aeb.brake_override,
            'acc_active': self._last_acc.active,
            'acc_distance': self._last_acc.distance_m,
            'acc_target_speed': self._last_acc.target_speed_ms,
            'warning_level': self._warning_level,
            'bicycle_warning': self._bicycle_warning,
        }

    def reset(self):
        self._last_aeb = AEBResult()
        self._last_acc = ACCResult()
        self._warning_level = "none"
        self._bicycle_warning = False
