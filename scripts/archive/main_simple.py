#!/usr/bin/env python3
"""
CARLA MPC System - Simplified Main Entry Point

Clean 4-manager architecture:
- CarlaManager (CARLA + vehicle + camera)
- PerceptionManager (UNet/Classical)  
- ControlManager (MPC + safety)
- DisplayManager (GUI + logging)

Usage:
    python main_simple.py --model model/lane_unet_final.pth --town Town04 --speed 30
    python main_simple.py --classical --town Town04 --speed 25
    python main_simple.py --no-gui --town Town04
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Setup CARLA paths
from core import setup_carla_paths
setup_carla_paths()

# Import managers
from managers import (
    CarlaManager, PerceptionManager, ControlManager, DisplayManager
)

# Import types and config
from utils.type_hints import DetectionMode, CameraFrame
from config_clean import get_config

logger = logging.getLogger(__name__)


class SimplifiedCARLAMPCSystem:
    """Simplified CARLA MPC system with 4 managers"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 target_speed_kmh: float = 25.0,
                 detection_mode: DetectionMode = DetectionMode.UNET,
                 enable_gui: bool = True,
                 enable_threading: bool = True):
        
        # Configuration
        self.config = get_config()
        self.target_speed_kmh = target_speed_kmh
        self.detection_mode = detection_mode
        self.enable_gui = enable_gui
        self.enable_threading = enable_threading
        
        # Managers
        self.carla_manager: Optional[CarlaManager] = None
        self.perception_manager: Optional[PerceptionManager] = None
        self.control_manager: Optional[ControlManager] = None
        self.display_manager: Optional[DisplayManager] = None
        
        # System state
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize managers
        self._initialize_managers(model_path)
    
    def _initialize_managers(self, model_path: Optional[str]):
        """Initialize all managers"""
        try:
            # Initialize display manager first (for early feedback)
            self.display_manager = DisplayManager(
                enable_gui=self.enable_gui,
                enable_logging=True,
                enable_recording=False
            )
            
            # Initialize perception manager
            self.perception_manager = PerceptionManager(
                model_path=model_path,
                detection_mode=self.detection_mode,
                use_threading=self.enable_threading
            )
            
            # Initialize control manager
            self.control_manager = ControlManager(
                target_speed_kmh=self.target_speed_kmh,
                enable_safety=True,
                enable_stuck_recovery=True
            )
            
            logger.info("All managers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize managers: {e}")
            raise
    
    def initialize_carla(self) -> bool:
        """Initialize CARLA connection and setup"""
        try:
            self.carla_manager = CarlaManager()
            
            # Connect to CARLA
            if not self.carla_manager.connect():
                return False
            
            # Spawn vehicle
            if not self.carla_manager.spawn_vehicle():
                return False
            
            # Setup camera
            if not self.carla_manager.setup_camera():
                return False
            
            # Start camera
            if not self.carla_manager.start_camera():
                return False
            
            logger.info("CARLA initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CARLA: {e}")
            return False
    
    def run(self) -> int:
        """Main system loop"""
        logger.info("Starting simplified CARLA MPC system...")
        
        # Initialize CARLA
        if not self.initialize_carla():
            return 1
        
        self.running = True
        
        try:
            # Start perception threading if enabled
            if self.enable_threading:
                self.perception_manager.start_threading()
            
            # Main control loop
            while self.running:
                loop_start_time = time.time()
                
                # Get latest camera frame
                frame = self.carla_manager.get_latest_frame()
                if frame is None:
                    continue
                
                # Process perception
                if self.enable_threading:
                    # Add frame to queue (threaded processing)
                    self.perception_manager.add_frame(frame)
                    perception = self.perception_manager.get_latest_result()
                else:
                    # Process directly
                    perception = self.perception_manager.process_frame(frame)
                
                # Get vehicle state
                vehicle_state = self.carla_manager.get_vehicle_state()
                if vehicle_state is None:
                    continue
                
                # Compute control
                control = self.control_manager.compute_control(perception, vehicle_state)
                
                # Apply control to vehicle
                carla_control = self.carla_manager.client.get_world().get_actor(
                    self.carla_manager.vehicle.id
                ).get_control()
                carla_control.steering = control.steering
                carla_control.throttle = control.throttle
                carla_control.brake = control.brake
                self.carla_manager.apply_control(carla_control)
                
                # Update display
                self.display_manager.update(
                    frame=frame,
                    perception=perception,
                    control=control,
                    vehicle_state=vehicle_state
                )
                
                # Check for GUI quit
                if self.enable_gui:
                    # Check if display wants to quit
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or \
                           (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            self.running = False
                            break
                
                # Update frame counter
                self.frame_count += 1
                
                # Maintain control frequency
                loop_time = time.time() - loop_start_time
                target_loop_time = 1.0 / self.config.CONTROL_HZ
                sleep_time = max(0, target_loop_time - loop_time - self.config.MAIN_LOOP_SLEEP_S)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Log performance periodically
                if self.frame_count % 100 == 0:
                    self._log_performance()
            
            return 0
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return 0
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            return 1
        finally:
            self.cleanup()
    
    def _log_performance(self):
        """Log system performance metrics"""
        try:
            # Get metrics from all managers
            carla_metrics = {}
            perception_metrics = self.perception_manager.get_metrics()
            control_metrics = self.control_manager.get_metrics()
            display_metrics = self.display_manager.get_metrics()
            
            # Calculate runtime
            runtime = time.time() - self.start_time
            
            logger.info(
                f"Performance - Frame: {self.frame_count}, "
                f"Runtime: {runtime:.1f}s, "
                f"Perception: {perception_metrics.get('last_processing_time', 0)*1000:.1f}ms, "
                f"Control: {control_metrics.get('last_control_time', 0)*1000:.1f}ms, "
                f"FPS: {display_metrics.get('fps_current', 0):.1f}"
            )
            
        except Exception as e:
            logger.error(f"Performance logging error: {e}")
    
    def cleanup(self):
        """Cleanup all system resources"""
        try:
            logger.info("Cleaning up simplified CARLA MPC system...")
            
            # Stop camera
            if self.carla_manager:
                self.carla_manager.stop_camera()
            
            # Cleanup managers
            if self.perception_manager:
                self.perception_manager.cleanup()
            
            if self.control_manager:
                self.control_manager.cleanup()
            
            if self.display_manager:
                self.display_manager.cleanup()
            
            if self.carla_manager:
                self.carla_manager.cleanup()
            
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="CARLA MPC Lane Keeping System (Simplified)")
    parser.add_argument("--model", type=str, default="model/lane_unet_final.pth",
                       help="Path to UNet model file")
    parser.add_argument("--town", type=str, default="Town04",
                       help="CARLA town name")
    parser.add_argument("--speed", type=float, default=25.0,
                       help="Target speed in km/h")
    parser.add_argument("--classical", action="store_true",
                       help="Use classical lane detection instead of UNet")
    parser.add_argument("--no-gui", action="store_true",
                       help="Disable GUI dashboard")
    parser.add_argument("--no-threading", action="store_true",
                       help="Disable perception threading")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    
    # Determine detection mode
    detection_mode = DetectionMode.CLASSICAL if args.classical else DetectionMode.UNET
    
    # Check model file for UNet mode
    if detection_mode == DetectionMode.UNET:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return 1
    else:
        model_path = None
    
    # Create and run system
    system = SimplifiedCARLAMPCSystem(
        model_path=model_path,
        target_speed_kmh=args.speed,
        detection_mode=detection_mode,
        enable_gui=not args.no_gui,
        enable_threading=not args.no_threading
    )
    
    return system.run()


if __name__ == "__main__":
    sys.exit(main())
