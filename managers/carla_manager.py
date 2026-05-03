#!/usr/bin/env python3
"""
Carla Manager - Simple CARLA interface management

Handles:
- CARLA connection
- Vehicle spawning and control
- Camera setup and callbacks
- Cleanup
"""

import logging
from typing import Optional, Callable
import queue
import numpy as np

# Setup CARLA paths
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    import carla
except ImportError:
    for _p in (_root.parent / "PythonAPI", _root / ".carla_py"):
        if _p.exists() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import carla

from utils.type_hints import CarlaTransform, CarlaVehicleControl, CameraFrame
from config_clean import get_config

logger = logging.getLogger(__name__)


class CarlaManager:
    """Simple CARLA management interface"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 2000,
                 timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        
        # CARLA objects
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[carla.Sensor] = None
        self.blueprint_library: Optional[carla.BlueprintLibrary] = None
        
        # Configuration
        self.config = get_config()
        
        # State
        self.connected = False
        self.camera_callback: Optional[Callable] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        
    def connect(self) -> bool:
        """Connect to CARLA server"""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            self.connected = True
            logger.info(f"Connected to CARLA at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False
    
    def spawn_vehicle(self, 
                     vehicle_model: str = "vehicle.tesla.model3",
                     spawn_point: Optional[carla.Transform] = None) -> bool:
        """Spawn vehicle"""
        if not self.connected:
            logger.error("Not connected to CARLA")
            return False
            
        try:
            blueprint = self.blueprint_library.filter(vehicle_model)[0]
            blueprint.set_attribute('role_name', 'ego_vehicle')
            
            if spawn_point is None:
                spawn_points = self.world.get_map().get_spawn_points()
                if not spawn_points:
                    logger.error("No spawn points available")
                    return False
                spawn_point = spawn_points[0]
            
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
            if self.vehicle:
                logger.info(f"Spawned {vehicle_model}")
                return True
            else:
                logger.error("Failed to spawn vehicle")
                return False
                
        except Exception as e:
            logger.error(f"Failed to spawn vehicle: {e}")
            return False
    
    def setup_camera(self) -> bool:
        """Setup camera with configuration"""
        if not self.vehicle:
            logger.error("Vehicle not spawned")
            return False
            
        try:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            
            # Use configuration values
            camera_bp.set_attribute('image_size_x', str(self.config.CAM_W))
            camera_bp.set_attribute('image_size_y', str(self.config.CAM_H))
            camera_bp.set_attribute('fov', str(self.config.CAM_FOV_DEG))
            
            # Camera transform - optimized for lane detection
            camera_transform = carla.Transform(
                carla.Location(x=1.5, z=2.0),  # 1.5m forward, 2.0m height
                carla.Rotation(pitch=-8)        # -8° pitch for balanced near/far
            )
            
            self.camera = self.world.spawn_actor(
                camera_bp, 
                camera_transform, 
                attach_to=self.vehicle
            )
            
            if self.camera:
                logger.info(f"Camera setup: {self.config.CAM_W}x{self.config.CAM_H}")
                return True
            else:
                logger.error("Failed to spawn camera")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup camera: {e}")
            return False
    
    def start_camera(self, callback: Optional[Callable] = None) -> bool:
        """Start camera with callback"""
        if not self.camera:
            logger.error("Camera not setup")
            return False
        
        def camera_callback(image):
            try:
                # Convert CARLA image to numpy array
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                array = array[:, :, :3]  # Remove alpha
                array = array[:, :, ::-1]  # BGR to RGB
                
                # Create CameraFrame
                frame = CameraFrame(
                    rgb=array,
                    timestamp=image.timestamp,
                    width=image.width,
                    height=image.height,
                    fov=float(self.config.CAM_FOV_DEG)
                )
                
                # Call user callback if provided
                if callback:
                    callback(frame)
                
                # Put in queue for internal use
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Camera callback error: {e}")
        
        self.camera.listen(camera_callback)
        self.camera_callback = callback
        logger.info("Camera started")
        return True
    
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Get latest camera frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def apply_control(self, control: CarlaVehicleControl) -> bool:
        """Apply control to vehicle"""
        if self.vehicle:
            try:
                self.vehicle.apply_control(control)
                return True
            except Exception as e:
                logger.error(f"Failed to apply control: {e}")
        return False
    
    def get_vehicle_state(self) -> Optional[dict]:
        """Get current vehicle state"""
        if not self.vehicle:
            return None
            
        try:
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            angular_velocity = self.vehicle.get_angular_velocity()
            
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            return {
                'transform': transform,
                'speed_ms': speed,
                'speed_kmh': speed * 3.6,
                'velocity': velocity,
                'angular_velocity': angular_velocity
            }
        except Exception as e:
            logger.error(f"Failed to get vehicle state: {e}")
            return None
    
    def stop_camera(self):
        """Stop camera"""
        if self.camera:
            self.camera.stop()
            logger.info("Camera stopped")
    
    def cleanup(self):
        """Cleanup all CARLA resources"""
        try:
            # Stop camera
            self.stop_camera()
            
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
                
            # Clear queues
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
                    
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        finally:
            self.connected = False
    
    def __enter__(self):
        """Context manager entry"""
        if self.connect():
            return self
        raise ConnectionError("Failed to connect to CARLA")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
