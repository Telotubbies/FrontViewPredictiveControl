# CARLA + pipeline worker (runs in QThread)
import math
import queue
import time
from typing import TYPE_CHECKING

import numpy as np

from .qt_compatibility import QObject, pyqtSignal
from .models import FrameData

if TYPE_CHECKING:
    pass


class CarlaWorker(QObject):
    """Runs CARLA connection and pipeline step loop in a separate thread."""
    frame_ready = pyqtSignal(object)  # FrameData
    fps_updated = pyqtSignal(float)
    log_message = pyqtSignal(str)
    connection_changed = pyqtSignal(bool, str)  # connected, town
    emergency_stop = pyqtSignal()

    def __init__(self, model_path: str, parent=None):
        super().__init__(parent)
        self._model_path = model_path
        self._running = False
        self._town = "Town04"
        self._fusion_enabled = True
        self._unet_weight = 0.5
        self._mpc_horizon = 10
        self._emergency_requested = False

    def set_town(self, town: str) -> None:
        self._town = town

    def set_fusion_enabled(self, enabled: bool) -> None:
        self._fusion_enabled = enabled

    def set_unet_weight(self, value: float) -> None:
        self._unet_weight = max(0.0, min(1.0, value))

    def set_mpc_horizon(self, value: int) -> None:
        self._mpc_horizon = max(5, min(20, value))

    def start_sim(self) -> None:
        self._running = True
        self._emergency_requested = False

    def stop_sim(self) -> None:
        self._running = False

    def request_emergency_brake(self) -> None:
        self._emergency_requested = True

    def run(self) -> None:
        """Thread entry: connect CARLA, spawn vehicle/camera, run pipeline loop."""
        try:
            import carla
        except ImportError:
            self.log_message.emit("ERROR: carla module not found. Add PythonAPI to path.")
            self.connection_changed.emit(False, "")
            return

        client = None
        world = None
        vehicle = None
        camera = None
        cam_queue = queue.Queue(maxsize=2)
        pipeline = None
        prev_steer, prev_throttle = 0.0, 0.0
        frame_count = 0
        t0 = time.time()

        try:
            client = carla.Client("localhost", 2000)
            client.set_timeout(10)
            world = client.get_world()
            current_map = world.get_map().name.split("/")[-1]
            if current_map != self._town:
                self.log_message.emit(f"Loading world: {self._town}")
                world = client.load_world(self._town)
            self.connection_changed.emit(True, self._town)
            self.log_message.emit(f"Connected. CARLA world: {self._town}")

            from config import CAM_W, CAM_H, CAM_FOV_DEG
            from pipeline import LKAPipeline
            from carla_input_output import get_waypoints, waypoints_to_cte_heading, cleanup
            from utils.device_utils import get_device
            device = get_device()
            pipeline = LKAPipeline(self._model_path, device, target_speed_kmh=25.0, use_trajectory_pipeline=True)
            pipeline.reset()

            bp = world.get_blueprint_library()
            cleanup(world)
            veh_bp = bp.find("vehicle.tesla.model3")
            spawn_points = world.get_map().get_spawn_points()
            for sp in spawn_points[:30]:
                vehicle = world.try_spawn_actor(veh_bp, sp)
                if vehicle is not None:
                    break
            if vehicle is None:
                self.log_message.emit("ERROR: No spawn point available")
                self.connection_changed.emit(False, "")
                return

            cam_bp = bp.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(CAM_W))
            cam_bp.set_attribute("image_size_y", str(CAM_H))
            cam_bp.set_attribute("fov", str(int(CAM_FOV_DEG)))
            camera = world.spawn_actor(
                cam_bp,
                carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15)),
                attach_to=vehicle,
            )
            camera.listen(cam_queue.put)
            cmap = world.get_map()

            self.log_message.emit("Vehicle and camera ready. Starting simulation.")
            last_img = None

            while self._running:
                if self._emergency_requested:
                    vehicle.apply_control(carla.VehicleControl(steer=0, throttle=0, brake=1.0))
                    self.log_message.emit("EMERGENCY BRAKE applied")
                    self._emergency_requested = False

                img = None
                try:
                    while True:
                        img = cam_queue.get_nowait()
                except queue.Empty:
                    pass
                if img is None:
                    if last_img is None:
                        time.sleep(0.01)
                        continue
                    img = last_img
                last_img = img

                raw = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))
                rgb = raw[:, :, 2::-1].copy()

                vel = vehicle.get_velocity()
                speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                wps = get_waypoints(vehicle, cmap)
                wp_state = waypoints_to_cte_heading(vehicle.get_transform(), wps) if wps else None

                if not self._fusion_enabled and wp_state is not None:
                    wp_state = None

                steer, throttle, brake, frame_state = pipeline.step(
                    rgb, speed_ms, wp_state, prev_steer, prev_throttle
                )
                prev_steer, prev_throttle = steer, throttle

                if self._emergency_requested:
                    vehicle.apply_control(carla.VehicleControl(steer=0, throttle=0, brake=1.0))
                else:
                    vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))

                fd = FrameData(
                    rgb=frame_state.rgb,
                    speed_kmh=frame_state.speed_kmh,
                    steer=frame_state.steer,
                    throttle=frame_state.throttle,
                    brake=frame_state.brake,
                    cte_m=frame_state.cte_m,
                    heading_rad=frame_state.heading_rad,
                    curvature=frame_state.curvature,
                    mode=frame_state.mode,
                    lane_conf=frame_state.lane_conf,
                    reference_path=frame_state.reference_path,
                    lane_overlay=frame_state.lane_overlay,
                    mask=frame_state.mask,
                )
                self.frame_ready.emit(fd)
                frame_count += 1
                elapsed = time.time() - t0
                if elapsed >= 0.5:
                    self.fps_updated.emit(frame_count / elapsed)
                    frame_count = 0
                    t0 = time.time()

        except Exception as e:
            self.log_message.emit(f"ERROR: {e}")
            self.connection_changed.emit(False, "")
            import traceback
            self.log_message.emit(traceback.format_exc())
        finally:
            if camera is not None:
                try:
                    camera.destroy()
                except Exception:
                    pass
            if vehicle is not None:
                try:
                    vehicle.destroy()
                except Exception:
                    pass
            self.connection_changed.emit(False, "")
            self.log_message.emit("Simulation stopped.")
