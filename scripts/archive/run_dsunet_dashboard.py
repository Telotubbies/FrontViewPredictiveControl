#!/usr/bin/env python3
"""
DSUNet Dashboard Runner - รัน DSUNet lane detection กับ dashboard ใหม่
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import carla
import pygame

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CAM_W, CAM_H, CAM_FOV_DEG,
    TARGET_SPEED_KMH, USE_PERCEPTION_THREAD,
    CONTROL_HZ, STALE_PERCEPTION_S
)
from gui.dsunet_dashboard import create_dsunet_dashboard
from perception.lane_detector import LaneDetector
from perception.lane_trajectory import LaneTrajectoryPipeline
from alg.step import LKAStep

logger = logging.getLogger(__name__)


class DSUNetDashboardRunner:
    """Main runner for DSUNet dashboard"""
    
    def __init__(self, model_path: str, town: str = "Town04", speed: float = 25.0):
        self.model_path = Path(model_path)
        self.town = town
        self.speed_kmh = speed
        self.target_speed_ms = speed / 3.6
        
        # Dashboard
        self.dashboard = None
        
        # CARLA components
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.camera_queue = None
        
        # Perception
        self.lka_step = None
        self.trajectory_pipeline = None
        
        # State
        self.running = False
        self.frame_count = 0
        self.last_perception_time = 0
        
    def setup_carla(self) -> bool:
        """Setup CARLA connection and vehicle"""
        try:
            # Connect to CARLA
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            
            # Load town if needed
            if self.world.get_map().name.split('/')[-1] != self.town:
                self.client.load_world(self.town)
                time.sleep(2.0)
                self.world = self.client.get_world()
            
            # Get vehicle blueprint
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.*')[0]
            
            # Spawn vehicle
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                logger.error("No spawn points available")
                return False
                
            spawn_point = spawn_points[0]
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            # Setup camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(CAM_W))
            camera_bp.set_attribute('image_size_y', str(CAM_H))
            camera_bp.set_attribute('fov', str(int(CAM_FOV_DEG)))
            
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=-15)
            )
            
            import queue
            self.camera_queue = queue.Queue()
            self.camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            self.camera.listen(self.camera_queue.put)
            
            logger.info(f"CARLA setup complete: {self.town}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup CARLA: {e}")
            return False
    
    def setup_perception(self) -> bool:
        """Setup DSUNet perception pipeline"""
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create LKA step with DSUNet
            self.lka_step = LKAStep(
                str(self.model_path), 
                device, 
                self.speed_kmh,
                use_trajectory_pipeline=True
            )
            
            # Get trajectory pipeline
            self.trajectory_pipeline = self.lka_step._trajectory
            
            logger.info(f"DSUNet perception setup complete: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup perception: {e}")
            return False
    
    def process_camera_frame(self) -> Optional[np.ndarray]:
        """Get and process camera frame"""
        try:
            # Get frame from queue
            frame = self.camera_queue.get(timeout=0.1)
            array = np.frombuffer(frame.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (frame.height, frame.width, 4))
            array = array[:, :, :3]  # Remove alpha
            
            return array
            
        except Exception as e:
            logger.warning(f"Failed to get camera frame: {e}")
            return None
    
    def run_dashboard_loop(self) -> None:
        """Main dashboard loop"""
        if not self.dashboard:
            logger.error("Dashboard not initialized")
            return
            
        clock = pygame.time.Clock()
        self.running = True
        
        # Vehicle control
        control = carla.VehicleControl()
        control.throttle = 0.3
        control.steer = 0.0
        
        logger.info("Starting DSUNet dashboard loop...")
        
        while self.running:
            try:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_SPACE:
                            # Toggle autopilot
                            control.throttle = 0.0 if control.throttle > 0 else 0.3
                
                # Get camera frame
                rgb_frame = self.process_camera_frame()
                
                if rgb_frame is not None and self.trajectory_pipeline:
                    # Process with DSUNet
                    start_time = time.time()
                    trajectory_result = self.trajectory_pipeline.process(rgb_frame)
                    perception_time = (time.time() - start_time) * 1000
                    
                    # Extract data
                    cte = trajectory_result.cte
                    confidence = trajectory_result.confidence
                    lane_overlay = trajectory_result.lane_overlay
                    bev_binary = trajectory_result.bev_binary
                    bev_window_vis = trajectory_result.bev_window_vis
                    
                    # Get curvature info
                    left_curvature = getattr(trajectory_result, 'left_curvature_m', 0.0)
                    right_curvature = getattr(trajectory_result, 'right_curvature_m', 0.0)
                    
                    # Simple control (could use MPC here)
                    if abs(cte) < 1.0:
                        control.steer = np.clip(cte * 0.5, -0.5, 0.5)
                    else:
                        control.steer = 0.0
                    
                    # Apply control
                    if self.vehicle:
                        self.vehicle.apply_control(control)
                    
                    # Calculate FPS
                    current_time = time.time()
                    fps = 1.0 / (current_time - self.last_perception_time) if self.last_perception_time > 0 else 0
                    self.last_perception_time = current_time
                    
                    # Get speed
                    speed_ms = self.vehicle.get_velocity().x if self.vehicle else 0
                    speed_kmh = abs(speed_ms) * 3.6
                    
                    # Get steering
                    steering = control.steer
                    
                    # Phase info
                    phase_info = f"Perception: {perception_time:.1f}ms"
                    
                    # Render dashboard
                    self.dashboard.clear_screen()
                    self.dashboard.render_header()
                    self.dashboard.render_main_camera(rgb_frame)
                    self.dashboard.render_dsunet_overlay(lane_overlay, confidence)
                    self.dashboard.render_bev_view(bev_binary, bev_window_vis)
                    self.dashboard.render_info_panel(
                        speed_kmh, steering, cte, confidence, fps, phase_info
                    )
                    self.dashboard.render_lane_markings_info(
                        left_curvature, right_curvature
                    )
                    self.dashboard.update_display()
                    
                    self.frame_count += 1
                
                # Control frame rate
                clock.tick(20)  # 20 FPS for dashboard
                
            except Exception as e:
                logger.error(f"Dashboard loop error: {e}")
                continue
    
    def run(self) -> None:
        """Main run method"""
        try:
            logger.info("Starting DSUNet Dashboard Runner...")
            
            # Initialize dashboard
            self.dashboard = create_dsunet_dashboard()
            
            # Setup CARLA
            if not self.setup_carla():
                logger.error("Failed to setup CARLA")
                return
            
            # Setup perception
            if not self.setup_perception():
                logger.error("Failed to setup perception")
                return
            
            # Warmup
            logger.info("Warming up...")
            time.sleep(2.0)
            
            # Clear camera queue
            import queue
            while not self.camera_queue.empty():
                try:
                    self.camera_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Run dashboard loop
            self.run_dashboard_loop()
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runner error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Cleaning up...")
        
        self.running = False
        
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        if self.dashboard:
            self.dashboard.cleanup()
        
        logger.info("Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DSUNet Dashboard Runner")
    parser.add_argument("--model", type=str, required=True, help="Path to DSUNet model")
    parser.add_argument("--town", type=str, default="Town04", help="CARLA town")
    parser.add_argument("--speed", type=float, default=25.0, help="Target speed km/h")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run
    runner = DSUNetDashboardRunner(args.model, args.town, args.speed)
    runner.run()


if __name__ == "__main__":
    main()
