"""
CarlaDataSampler — ตัวอย่างข้อมูลดิบจาก CARLA ทุก frame เพื่อส่งให้ AI วิเคราะห์

ความแตกต่างจาก MetricsCollector:
- MetricsCollector = ข้อมูลที่ผ่าน pipeline แล้ว (cte, heading, curvature, mpc_solve_time)
- CarlaDataSampler = ข้อมูลดิบจาก CARLA API โดยตรง (velocity 3D, acceleration 3D,
  angular velocity, waypoint, road_id, lane_id, weather, time of day)

หน้าที่หลัก:
1. เก็บ raw CARLA vehicle/world data ทุก frame
2. คำนวณ derived values (speed_from_velocity, heading_from_yaw,
   lateral_acceleration, slip_angle estimate)
3. เก็บใน rolling buffer (deque maxlen=500) เพื่อเข้าถึงข้อมูลล่าสุด
4. แปลงเป็น dict / compact dict เพื่อ serialize ให้ AI วิเคราะห์
5. จัดการกรณี CARLA ไม่พร้อมใช้งานอย่างปลอดภัย (try/except, return None)
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

# CARLA Python API — import แบบ optional เพื่อให้ module นี้ import-safe
# แม้ CARLA ไม่ได้ติดตั้งก็ไม่ crash ตอน import (จะ crash ตอนเรียกใช้จริงเท่านั้น)
try:
    import carla
except ImportError:
    carla = None  # type: ignore[assignment]

if TYPE_CHECKING:
    # เพื่อ type hint โดยไม่บังคับให้มี carla ตอน runtime
    import carla  # noqa: F811

logger = logging.getLogger(__name__)

# ค่า default ขนาด rolling buffer
_DEFAULT_BUFFER_SIZE = 500


# ─────────────────────────────────────────────────────────────────────────────
#  Derived-value helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe_vec3_to_tuple(vec: Any) -> tuple:
    """แปลง carla.Vector3D เป็น tuple (x, y, z) อย่างปลอดภัย"""
    try:
        return (float(vec.x), float(vec.y), float(vec.z))
    except (AttributeError, TypeError):
        return (0.0, 0.0, 0.0)


def _speed_from_velocity(vx: float, vy: float, vz: float) -> float:
    """คำนวณ speed (m/s) จาก 3D velocity vector"""
    return float(math.sqrt(vx * vx + vy * vy + vz * vz))


def _heading_from_yaw(yaw_deg: float) -> float:
    """แปลง yaw (degrees) เป็น heading (radians) — normalize ให้อยู่ใน [-pi, pi]"""
    rad = math.radians(yaw_deg)
    # normalize ให้อยู่ใน [-pi, pi]
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return float(rad)


def _lateral_acceleration(accel_x: float, accel_y: float, yaw_rad: float) -> float:
    """
    คำนวณ lateral acceleration (m/s²) ในกรอบพิกัดของรถ
    ฉาย acceleration vector ลงบนแกน lateral (ตั้งฉากกับ heading)
    """
    # แกน lateral ใน world frame: (-sin(yaw), cos(yaw))
    lat_x = -math.sin(yaw_rad)
    lat_y = math.cos(yaw_rad)
    return float(accel_x * lat_x + accel_y * lat_y)


def _slip_angle_estimate(
    vx: float,
    vy: float,
    yaw_rate: float,
    speed: float,
    wheelbase: float = 2.9,
) -> float:
    """
    ประมาณ slip angle (rad) แบบง่าย ๆ

    slip_angle ≈ atan2(vy, vx) - (yaw_rate * wheelbase / speed)
    ถ้า speed ต่ำมาก (< 0.5 m/s) ให้คืน 0 เพื่อหาระดมศูนย์
    """
    if speed < 0.5:
        return 0.0
    body_slip = math.atan2(vy, vx)
    yaw_correction = (yaw_rate * wheelbase) / speed
    return float(body_slip - yaw_correction)


# ─────────────────────────────────────────────────────────────────────────────
#  CarlaSample dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CarlaSample:
    """
    ข้อมูลดิบ 1 frame จาก CARLA — รวม vehicle kinematics, pose, road context,
    environment, และ derived values ที่คำนวณเพิ่ม

    ใช้ to_dict() เพื่อ serialize แบบเต็ม หรือ to_compact_dict() เพื่อเอา
    เฉพาะ key fields สำหรับส่งให้ AI วิเคราะห์
    """
    # ── Metadata ────────────────────────────────────────────────────────────
    frame_idx: int = -1
    timestamp: float = 0.0

    # ── Vehicle kinematics (raw from CARLA) ─────────────────────────────────
    velocity_x: float = 0.0          # m/s
    velocity_y: float = 0.0          # m/s
    velocity_z: float = 0.0          # m/s
    acceleration_x: float = 0.0      # m/s²
    acceleration_y: float = 0.0      # m/s²
    acceleration_z: float = 0.0      # m/s²
    angular_velocity_x: float = 0.0  # rad/s (roll rate)
    angular_velocity_y: float = 0.0  # rad/s (pitch rate)
    angular_velocity_z: float = 0.0  # rad/s (yaw rate)
    speed_kmh: float = 0.0           # km/h (จาก vehicle.get_speed())

    # ── Vehicle pose (raw from CARLA) ───────────────────────────────────────
    location_x: float = 0.0          # m
    location_y: float = 0.0          # m
    location_z: float = 0.0          # m
    rotation_pitch: float = 0.0      # deg
    rotation_yaw: float = 0.0        # deg
    rotation_roll: float = 0.0       # deg

    # ── Road context (from waypoint) ────────────────────────────────────────
    road_id: Optional[int] = None
    lane_id: Optional[int] = None
    lane_width: float = 0.0          # m
    lane_type: str = "unknown"
    is_junction: bool = False
    speed_limit: float = 0.0         # km/h (จาก waypoint หรือ vehicle)
    waypoint_x: float = 0.0          # m
    waypoint_y: float = 0.0          # m
    waypoint_z: float = 0.0          # m

    # ── Environment ─────────────────────────────────────────────────────────
    weather_precipitation: float = 0.0
    weather_cloudiness: float = 0.0
    weather_fog_density: float = 0.0
    weather_sun_altitude: float = 0.0
    time_of_day: str = "unknown"     # e.g. "12:00:00"

    # ── Camera metadata ─────────────────────────────────────────────────────
    cam_frame_width: int = 0
    cam_frame_height: int = 0

    # ── Derived values (computed by sampler) ────────────────────────────────
    speed_from_velocity_ms: float = 0.0   # m/s (จาก sqrt(vx²+vy²+vz²))
    speed_from_velocity_kmh: float = 0.0  # km/h
    heading_rad: float = 0.0              # rad (จาก yaw, normalized)
    lateral_acceleration: float = 0.0     # m/s²
    slip_angle_rad: float = 0.0           # rad (estimate)

    # ── Link to FrameState (optional, for cross-reference) ──────────────────
    frame_state_speed_kmh: float = 0.0
    frame_state_cte_m: float = 0.0
    frame_state_heading_rad: float = 0.0
    frame_state_mode: str = "unknown"

    # ── Sampling status ─────────────────────────────────────────────────────
    valid: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dict แบบเต็ม (ทุก field) เพื่อ serialize"""
        return asdict(self)

    def to_compact_dict(self) -> Dict[str, Any]:
        """
        แปลงเป็น dict แบบกระชับ — เก็บเฉพาะ key fields ที่ AI ต้องการ
        วิเคราะห์พฤติกรรมรถและสภาพถนน
        """
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 4),
            "speed_kmh": round(self.speed_kmh, 2),
            "speed_from_velocity_kmh": round(self.speed_from_velocity_kmh, 2),
            "heading_rad": round(self.heading_rad, 4),
            "location_x": round(self.location_x, 2),
            "location_y": round(self.location_y, 2),
            "location_z": round(self.location_z, 2),
            "rotation_yaw": round(self.rotation_yaw, 2),
            "angular_velocity_z": round(self.angular_velocity_z, 4),
            "lateral_acceleration": round(self.lateral_acceleration, 3),
            "slip_angle_rad": round(self.slip_angle_rad, 4),
            "road_id": self.road_id,
            "lane_id": self.lane_id,
            "lane_width": round(self.lane_width, 2),
            "is_junction": self.is_junction,
            "speed_limit": round(self.speed_limit, 2),
            "time_of_day": self.time_of_day,
            "weather_fog_density": round(self.weather_fog_density, 2),
            "weather_precipitation": round(self.weather_precipitation, 2),
            "frame_state_cte_m": round(self.frame_state_cte_m, 3),
            "frame_state_mode": self.frame_state_mode,
            "valid": self.valid,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  CarlaDataSampler
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CarlaDataSampler:
    """
    ตัว sample ข้อมูลดิบจาก CARLA ทุก frame และเก็บใน rolling buffer

    การใช้งาน:
        sampler = CarlaDataSampler(buffer_size=500)
        sample = sampler.sample(vehicle, world, frame_idx=42, frame_state=fs)
        recent = sampler.get_recent_samples(n=10)
        data = sampler.to_dict_list()

    ถ้า CARLA ไม่พร้อมใช้งาน (carla is None หรือ vehicle/world เป็น None)
    sample() จะคืน CarlaSample ที่ valid=False และ error บอกสาเหตุ
    แทนที่จะ raise exception
    """
    buffer_size: int = _DEFAULT_BUFFER_SIZE
    wheelbase: float = 2.9  # m — ใช้ใน slip_angle estimate

    _buffer: deque = field(init=False)

    def __post_init__(self) -> None:
        self._buffer = deque(maxlen=self.buffer_size)
        logger.debug(
            "CarlaDataSampler initialized: buffer_size=%d, wheelbase=%.2f",
            self.buffer_size,
            self.wheelbase,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def sample(
        self,
        vehicle: Any,
        world: Any,
        frame_idx: int,
        frame_state: Any = None,
    ) -> CarlaSample:
        """
        Sample ข้อมูลดิบจาก CARLA สำหรับ 1 frame

        Args:
            vehicle: carla.Vehicle object (หรือ None ถ้ายังไม่ spawn)
            world: carla.World object (หรือ None ถ้ายังไม่ connect)
            frame_idx: index ของ frame ปัจจุบัน
            frame_state: FrameState object (optional) สำหรับ cross-reference

        Returns:
            CarlaSample — ถ้า CARLA ไม่พร้อมจะคืน sample ที่ valid=False
        """
        ts = time.time()

        # ── Graceful degradation: CARLA ไม่พร้อม ──────────────────────────
        if carla is None:
            logger.warning("CARLA API not available — returning invalid sample")
            s = CarlaSample(
                frame_idx=frame_idx,
                timestamp=ts,
                valid=False,
                error="carla module not imported",
            )
            self._buffer.append(s)
            return s

        if vehicle is None or world is None:
            logger.debug(
                "vehicle/world is None (frame %d) — returning invalid sample",
                frame_idx,
            )
            s = CarlaSample(
                frame_idx=frame_idx,
                timestamp=ts,
                valid=False,
                error="vehicle or world is None",
            )
            self._buffer.append(s)
            return s

        try:
            sample = self._collect_sample(vehicle, world, frame_idx, ts, frame_state)
            self._buffer.append(sample)
            return sample
        except Exception as e:
            logger.error(
                "Failed to sample CARLA data (frame %d): %s", frame_idx, e,
                exc_info=True,
            )
            s = CarlaSample(
                frame_idx=frame_idx,
                timestamp=ts,
                valid=False,
                error=f"exception: {e}",
            )
            self._buffer.append(s)
            return s

    def get_recent_samples(self, n: int = 10) -> List[CarlaSample]:
        """
        คืน sample ล่าสุด n ตัว (เรียงจากเก่า → ใหม่)

        ถ้า n > จำนวน sample ใน buffer จะคืนทั้งหมดที่มี
        """
        if n <= 0:
            return []
        items = list(self._buffer)
        return items[-n:] if n < len(items) else items

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """แปลงทุก sample ใน buffer เป็น list ของ dict (เต็ม field)"""
        return [s.to_dict() for s in self._buffer]

    def to_compact_dict_list(self) -> List[Dict[str, Any]]:
        """แปลงทุก sample ใน buffer เป็น list ของ compact dict"""
        return [s.to_compact_dict() for s in self._buffer]

    def clear(self) -> None:
        """ล้าง buffer ทั้งหมด"""
        self._buffer.clear()
        logger.debug("CarlaDataSampler buffer cleared")

    def __len__(self) -> int:
        return len(self._buffer)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _collect_sample(
        self,
        vehicle: Any,
        world: Any,
        frame_idx: int,
        ts: float,
        frame_state: Any,
    ) -> CarlaSample:
        """รวบรวมข้อมูลทั้งหมดจาก CARLA objects แล้วคำนวณ derived values"""
        s = CarlaSample(frame_idx=frame_idx, timestamp=ts)

        # ── Vehicle kinematics ─────────────────────────────────────────────
        try:
            vel = vehicle.get_velocity()
            vx, vy, vz = _safe_vec3_to_tuple(vel)
            s.velocity_x, s.velocity_y, s.velocity_z = vx, vy, vz
        except Exception as e:
            logger.debug("get_velocity failed: %s", e)

        try:
            accel = vehicle.get_acceleration()
            ax, ay, az = _safe_vec3_to_tuple(accel)
            s.acceleration_x, s.acceleration_y, s.acceleration_z = ax, ay, az
        except Exception as e:
            logger.debug("get_acceleration failed: %s", e)

        try:
            ang_vel = vehicle.get_angular_velocity()
            avx, avy, avz = _safe_vec3_to_tuple(ang_vel)
            s.angular_velocity_x = avx
            s.angular_velocity_y = avy
            s.angular_velocity_z = avz
        except Exception as e:
            logger.debug("get_angular_velocity failed: %s", e)

        try:
            # vehicle.get_speed() คืน speed ในหน่วย m/s ใน CARLA รุ่นใหม่
            speed_ms = float(vehicle.get_speed())
            s.speed_kmh = speed_ms * 3.6
        except Exception as e:
            logger.debug("get_speed failed: %s", e)

        # ── Vehicle pose ───────────────────────────────────────────────────
        transform = None
        try:
            transform = vehicle.get_transform()
            if transform is not None:
                loc = transform.location
                rot = transform.rotation
                s.location_x = float(loc.x)
                s.location_y = float(loc.y)
                s.location_z = float(loc.z)
                s.rotation_pitch = float(rot.pitch)
                s.rotation_yaw = float(rot.yaw)
                s.rotation_roll = float(rot.roll)
        except Exception as e:
            logger.debug("get_transform failed: %s", e)

        # ── Road context (waypoint) ────────────────────────────────────────
        try:
            map_obj = world.get_map()
            if transform is not None:
                wp = map_obj.get_waypoint(transform.location)
            else:
                # fallback: ใช้ location จาก transform ที่อ่านได้
                wp = None

            if wp is not None:
                s.road_id = int(wp.road_id)
                s.lane_id = int(wp.lane_id)
                s.lane_width = float(wp.lane_width)
                s.lane_type = str(wp.lane_type)
                s.is_junction = bool(wp.is_junction)
                wp_loc = wp.transform.location
                s.waypoint_x = float(wp_loc.x)
                s.waypoint_y = float(wp_loc.y)
                s.waypoint_z = float(wp_loc.z)
        except Exception as e:
            logger.debug("waypoint lookup failed: %s", e)

        # ── Speed limit ────────────────────────────────────────────────────
        try:
            # ลองอ่านจาก vehicle ก่อน (บางเวอร์ชันมี get_speed_limit)
            if hasattr(vehicle, "get_speed_limit"):
                s.speed_limit = float(vehicle.get_speed_limit())
        except Exception as e:
            logger.debug("get_speed_limit failed: %s", e)

        # ── Environment: weather + time of day ─────────────────────────────
        try:
            weather = world.get_weather()
            s.weather_precipitation = float(getattr(weather, "precipitation", 0.0))
            s.weather_cloudiness = float(getattr(weather, "cloudiness", 0.0))
            s.weather_fog_density = float(getattr(weather, "fog_density", 0.0))
            s.weather_sun_altitude = float(getattr(weather, "sun_altitude_angle", 0.0))
        except Exception as e:
            logger.debug("get_weather failed: %s", e)

        try:
            s.time_of_day = str(world.get_snapshot().timestamp)
        except Exception:
            # fallback: ลอง environmental weather time
            try:
                tod = world.get_environment_weather()  # type: ignore[attr-defined]
                s.time_of_day = str(tod)
            except Exception:
                s.time_of_day = "unknown"

        # ── Camera metadata ────────────────────────────────────────────────
        if frame_state is not None:
            try:
                rgb = getattr(frame_state, "rgb", None)
                if rgb is not None and hasattr(rgb, "shape"):
                    s.cam_frame_height = int(rgb.shape[0])
                    s.cam_frame_width = int(rgb.shape[1]) if rgb.ndim >= 2 else 0
            except Exception:
                pass

        # ── Derived values ─────────────────────────────────────────────────
        s.speed_from_velocity_ms = _speed_from_velocity(
            s.velocity_x, s.velocity_y, s.velocity_z
        )
        s.speed_from_velocity_kmh = s.speed_from_velocity_ms * 3.6
        s.heading_rad = _heading_from_yaw(s.rotation_yaw)
        s.lateral_acceleration = _lateral_acceleration(
            s.acceleration_x, s.acceleration_y, s.heading_rad
        )
        s.slip_angle_rad = _slip_angle_estimate(
            s.velocity_x,
            s.velocity_y,
            s.angular_velocity_z,
            s.speed_from_velocity_ms,
            wheelbase=self.wheelbase,
        )

        # ── Cross-reference with FrameState ────────────────────────────────
        if frame_state is not None:
            try:
                s.frame_state_speed_kmh = float(getattr(frame_state, "speed_kmh", 0.0))
                s.frame_state_cte_m = float(getattr(frame_state, "cte_m", 0.0))
                s.frame_state_heading_rad = float(getattr(frame_state, "heading_rad", 0.0))
                s.frame_state_mode = str(getattr(frame_state, "mode", "unknown"))
            except Exception:
                pass

        s.valid = True
        return s
