#!/usr/bin/env python3
"""
CARLA Interface Module - จัดการการเชื่อมต่อกับ CARLA simulator

รวบรวมฟังก์ชันที่เกี่ยวข้องกับ:
- CARLA client connection
- Vehicle spawning และ setup
- Camera setup
- Waypoint management
- Cleanup
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING
import sys

from config import CAM_FOV_DEG

if TYPE_CHECKING:
    import carla

# CARLA Python API is imported lazily via setup_carla_paths() to allow a
# helpful ImportError message when carla is not installed.
# If carla is not available, this stays None and setup_carla_paths() raises.
try:
    import carla
except ImportError:
    carla = None  # type: ignore[assignment]

import numpy as np

logger = logging.getLogger(__name__)


class CarlaInterface:
    """CARLA simulator interface สำหรับ lane keeping system"""

    def __init__(self, host: str = "localhost", port: int = 2000, timeout: float = 10.0):
        if carla is None:
            raise ImportError(
                "CARLA Python API not installed. "
                "Install with: pip install carla-0.10.0-cp311-cp311-win_amd64.whl "
                "from https://github.com/carla-simulator/carla/releases"
            )
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[carla.Sensor] = None
        self.blueprint_library: Optional[carla.BlueprintLibrary] = None

    def connect(self) -> bool:
        """เชื่อมต่อกับ CARLA server"""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            self._sync_mode = False

            logger.info(f"Connected to CARLA server at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False

    def spawn_vehicle(self,
                     vehicle_model: str = "vehicle.lincoln.mkz",
                     spawn_point: Optional[carla.Transform] = None) -> bool:
        """Spawn vehicle ใน CARLA world"""
        if not self.world or not self.blueprint_library:
            logger.error("CARLA not connected")
            return False

        try:
            # Get vehicle blueprint — try exact match first, then fallback to any lincoln.mkz
            try:
                blueprint = self.blueprint_library.filter(vehicle_model)[0]
            except IndexError:
                # Fallback: try lincoln.mkz variants (mkz_2017, mkz_2020) then any 4-wheel vehicle
                fallbacks = ["vehicle.lincoln.mkz_2017", "vehicle.lincoln.mkz_2020",
                             "vehicle.tesla.model3", "vehicle.toyota.prius",
                             "vehicle.audi.a2", "vehicle.nissan.micra"]
                blueprint = None
                for fb in fallbacks:
                    matches = self.blueprint_library.filter(fb)
                    if matches:
                        blueprint = matches[0]
                        vehicle_model = fb
                        logger.info(f"Vehicle '{vehicle_model}' not found, using fallback: {fb}")
                        break
                if blueprint is None:
                    # Last resort: any car (not bike/motorcycle)
                    for bp in self.blueprint_library.filter('vehicle.*'):
                        bp_id = bp.id
                        if not any(x in bp_id for x in ['bike', 'motorcycle', 'cyclist',
                                                         'scooter', 'omafiets', 'century',
                                                         'ninja', 'low_rider', 'zx125']):
                            blueprint = bp
                            vehicle_model = bp_id
                            break
                if blueprint is None:
                    logger.error(f"No suitable vehicle blueprint found")
                    return False
            blueprint.set_attribute('role_name', 'ego_vehicle')

            # Get spawn point — try multiple points to find one without collision
            if spawn_point is None:
                spawn_points = self.world.get_map().get_spawn_points()
                if not spawn_points:
                    logger.error("No spawn points available")
                    return False
                # Try each spawn point until one succeeds (no collision)
                for i, sp in enumerate(spawn_points):
                    try:
                        self.vehicle = self.world.spawn_actor(blueprint, sp)
                        if self.vehicle:
                            logger.info(f"Spawned {vehicle_model} at spawn point #{i}")
                            # ── Initialize vehicle control: ปลด hand brake + ตั้ง gear ──
                            # CARLA 0.9.x: รถใหม่ spawn มี hand_brake=True และ
                            # manual_gear_shift=True บางครั้ง ทำให้รถไม่ขยับ
                            init_ctrl = self.vehicle.get_control()
                            init_ctrl.hand_brake = False
                            init_ctrl.brake = 0.0
                            init_ctrl.reverse = False
                            init_ctrl.manual_gear_shift = False
                            init_ctrl.gear = 1
                            init_ctrl.throttle = 0.0
                            init_ctrl.steer = 0.0
                            self.vehicle.apply_control(init_ctrl)
                            # ให้ world tick หนึ่งครั้งเพื่อให้ control มีผล
                            try:
                                self.world.tick()
                            except Exception:
                                pass
                            # อีกครั้งหลัง tick (manual_gear_shift อาจ reset)
                            init_ctrl.manual_gear_shift = False
                            init_ctrl.gear = 1
                            init_ctrl.hand_brake = False
                            self.vehicle.apply_control(init_ctrl)
                            logger.info("Vehicle control initialized (hand_brake off, gear=1)")
                            return True
                    except Exception:
                        continue
                logger.error("All spawn points occupied or blocked")
                return False

            # Spawn vehicle at specified spawn point
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
            if self.vehicle:
                logger.info(f"Spawned {vehicle_model} at spawn point")
                return True
            else:
                logger.error("Failed to spawn vehicle")
                return False

        except Exception as e:
            logger.error(f"Failed to spawn vehicle: {e}")
            return False

    def setup_camera(self,
                    cam_w: int = 640,
                    cam_h: int = 480,
                    cam_fov: float = CAM_FOV_DEG,
                    sensor_type: str = 'sensor.camera.rgb') -> bool:
        """Setup camera บน vehicle"""
        if not self.vehicle or not self.blueprint_library:
            logger.error("Vehicle not spawned or blueprint library not available")
            return False

        try:
            # Get camera blueprint
            camera_bp = self.blueprint_library.find(sensor_type)
            camera_bp.set_attribute('image_size_x', str(cam_w))
            camera_bp.set_attribute('image_size_y', str(cam_h))
            camera_bp.set_attribute('fov', str(cam_fov))

            # Create transform for camera (front mount)
            camera_transform = carla.Transform(
                carla.Location(x=1.5, z=2.0),  # 1.5m forward, 2.0m height
                carla.Rotation(pitch=-8)  # -8° pitch for lane detection
            )

            # Spawn camera
            self.camera = self.world.spawn_actor(
                camera_bp,
                camera_transform,
                attach_to=self.vehicle
            )

            if self.camera:
                logger.info(f"Camera setup: {cam_w}x{cam_h}, FOV={cam_fov}°")
                return True
            else:
                logger.error("Failed to spawn camera")
                return False

        except Exception as e:
            logger.error(f"Failed to setup camera: {e}")
            return False

    def get_vehicle_control(self) -> carla.VehicleControl:
        """Get vehicle control object"""
        if self.vehicle:
            return carla.VehicleControl()
        return carla.VehicleControl()

    def apply_control(self, control: carla.VehicleControl) -> bool:
        """Apply control to vehicle"""
        if self.vehicle:
            try:
                self.vehicle.apply_control(control)
                return True
            except Exception as e:
                logger.error(f"Failed to apply control: {e}")
                return False
        return False

    def get_vehicle_velocity(self) -> float:
        """Get current vehicle speed in m/s"""
        if self.vehicle:
            try:
                velocity = self.vehicle.get_velocity()
                return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            except Exception as e:
                logger.error(f"Failed to get velocity: {e}")
        return 0.0

    def get_vehicle_transform(self) -> Optional[carla.Transform]:
        """Get current vehicle transform"""
        if self.vehicle:
            try:
                return self.vehicle.get_transform()
            except Exception as e:
                logger.error(f"Failed to get transform: {e}")
        return None

    def get_waypoints(self,
                     transform: carla.Transform,
                     num_wp: int = 40,
                     wp_step: float = 2.0) -> List[carla.Waypoint]:
        """Get waypoints from current position"""
        if not self.world:
            return []

        try:
            map_ = self.world.get_map()
            waypoint = map_.get_waypoint(transform.location)
            waypoints = []

            for i in range(num_wp):
                if waypoint is None:
                    break
                waypoints.append(waypoint)
                waypoint = waypoint.next(wp_step)[0] if waypoint.next(wp_step) else None

            return waypoints
        except Exception as e:
            logger.error(f"Failed to get waypoints: {e}")
            return []

    def cleanup(self):
        """Cleanup CARLA actors and connection"""
        try:
            # Destroy camera
            if self.camera:
                self.camera.destroy()
                self.camera = None
                logger.info("Camera destroyed")

            # Destroy vehicle
            if self.vehicle:
                self.vehicle.destroy()
                self.vehicle = None
                logger.info("Vehicle destroyed")

            # คืนค่า async mode
            if getattr(self, '_sync_mode', False):
                try:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = None
                    self.world.apply_settings(settings)
                    logger.info("CARLA restored to asynchronous mode")
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry"""
        if self.connect():
            return self
        raise ConnectionError("Failed to connect to CARLA")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


def setup_carla_paths():
    """Setup project root on sys.path; CARLA Python API is pip-installed (carla>=0.10.0)."""
    global carla
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        import carla as _carla
        carla = _carla
    except ImportError:
        raise ImportError(
            "CARLA Python API not installed. "
            "Install with: pip install carla-0.10.0-cp311-cp311-win_amd64.whl "
            "from https://github.com/carla-simulator/carla/releases"
        )
