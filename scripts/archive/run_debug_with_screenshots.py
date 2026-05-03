#!/usr/bin/env python3
"""
Run ADAS with screenshot capture and detailed logging for debugging.

Captures:
- Screenshots every N frames
- Detailed perception logs
- Control command logs
- Lane detection quality metrics
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path(__file__).parent / "debug_logs" / timestamp
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging to file and console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("debug_runner")

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

def save_debug_frame(frame_data: dict, frame_num: int, output_dir: Path):
    """Save debug visualization frame."""
    try:
        # Save raw camera image
        if "rgb" in frame_data and frame_data["rgb"] is not None:
            rgb_path = output_dir / f"frame_{frame_num:04d}_camera.jpg"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(frame_data["rgb"], cv2.COLOR_RGB2BGR))
        
        # Save lane overlay
        if "lane_overlay" in frame_data and frame_data["lane_overlay"] is not None:
            overlay_path = output_dir / f"frame_{frame_num:04d}_overlay.jpg"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(frame_data["lane_overlay"], cv2.COLOR_RGB2BGR))
        
        # Save BEV visualization
        if "bev_window_vis" in frame_data and frame_data["bev_window_vis"] is not None:
            bev_path = output_dir / f"frame_{frame_num:04d}_bev.jpg"
            cv2.imwrite(str(bev_path), cv2.cvtColor(frame_data["bev_window_vis"], cv2.COLOR_RGB2BGR))
        
        # Save mask
        if "mask" in frame_data and frame_data["mask"] is not None:
            mask_path = output_dir / f"frame_{frame_num:04d}_mask.jpg"
            cv2.imwrite(str(mask_path), frame_data["mask"])
        
        logger.debug(f"Saved debug frame {frame_num}")
        
    except Exception as e:
        logger.error(f"Error saving debug frame {frame_num}: {e}")


def log_perception_state(state, frame_num: int):
    """Log detailed perception state."""
    logger.info(f"=== Frame {frame_num} Perception ===")
    logger.info(f"  Mode: {state.mode}")
    logger.info(f"  Lane Confidence: {state.lane_conf:.3f}")
    logger.info(f"  CTE: {state.cte_m:.3f}m")
    logger.info(f"  Heading: {np.rad2deg(state.heading_rad):.1f}°")
    logger.info(f"  Curvature: {state.curvature:.5f}")
    logger.info(f"  Geometry Valid: {state.geometry_valid}")
    logger.info(f"  Phases: P1={state.phase_p1_ok} P2={state.phase_p2_ok} P3={state.phase_p3_ok} P4={state.phase_p4_ok} P5={state.phase_p5_ok}")
    
    if hasattr(state, 'phase_p2_case'):
        logger.info(f"  P2 Case: {state.phase_p2_case}")
    if hasattr(state, 'phase_p3_case'):
        logger.info(f"  P3 Case: {state.phase_p3_case}")


def log_control_state(state, frame_num: int):
    """Log control commands."""
    logger.info(f"=== Frame {frame_num} Control ===")
    logger.info(f"  Speed: {state.speed_kmh:.1f} km/h")
    logger.info(f"  Steer: {state.steer:.3f}")
    logger.info(f"  Throttle: {state.throttle:.3f}")
    logger.info(f"  Brake: {state.brake:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Run ADAS with debug logging")
    parser.add_argument("--town", default="Town04", help="CARLA town")
    parser.add_argument("--speed", type=int, default=25, help="Target speed (km/h)")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to capture")
    parser.add_argument("--screenshot-interval", type=int, default=10, help="Save screenshot every N frames")
    args = parser.parse_args()
    
    logger.info(f"Starting debug run: {args.town}, {args.speed} km/h, {args.frames} frames")
    logger.info(f"Output directory: {log_dir}")
    
    # Create screenshot directory
    screenshot_dir = log_dir / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)
    
    # Import CARLA and ADAS components
    try:
        import carla
        from managers.carla_manager import CarlaManager
        from alg.step import LKAStep
        from managers.display_manager import DisplayManager
        from config_clean import get_config
        import torch
        
        logger.info("Imports successful")
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return 1
    
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize components
    carla_mgr = None
    display_mgr = None
    
    try:
        # Connect to CARLA
        carla_mgr = CarlaManager(host="localhost", port=2000, timeout=10.0)
        if not carla_mgr.connect():
            logger.error("Failed to connect to CARLA")
            return 1
        
        logger.info("Connected to CARLA")
        
        # Load map
        if not carla_mgr.load_map(args.town):
            logger.error(f"Failed to load map {args.town}")
            return 1
        
        logger.info(f"Loaded map: {args.town}")
        
        # Spawn vehicle
        if not carla_mgr.spawn_vehicle():
            logger.error("Failed to spawn vehicle")
            return 1
        
        logger.info("Vehicle spawned")
        
        # Setup camera
        if not carla_mgr.setup_camera():
            logger.error("Failed to setup camera")
            return 1
        
        logger.info("Camera setup complete")
        
        # Initialize LKA pipeline
        model_path = PROJECT_ROOT / "model" / "lane_unet_final.pth"
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return 1
        
        lka = LKAStep(
            model_path=str(model_path),
            device=device,
            target_speed_kmh=args.speed,
            use_trajectory_pipeline=True,
        )
        
        logger.info("LKA pipeline initialized")
        
        # Initialize display
        display_mgr = DisplayManager(enable_gui=False)  # No GUI for debug mode
        
        # Start camera
        frame_count = 0
        prev_steer = 0.0
        prev_throttle = 0.0
        
        def camera_callback(camera_frame):
            nonlocal frame_count, prev_steer, prev_throttle
            
            if frame_count >= args.frames:
                return
            
            try:
                # Get vehicle state
                vehicle = carla_mgr.vehicle
                velocity = vehicle.get_velocity()
                speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
                # Get waypoint state (if available)
                transform = vehicle.get_transform()
                location = transform.location
                
                try:
                    waypoint = carla_mgr.world.get_map().get_waypoint(location)
                    if waypoint:
                        # Compute CTE and heading error
                        from carla_io import waypoints_to_cte_heading
                        wp_list = [waypoint]
                        for _ in range(10):
                            wp_next = wp_list[-1].next(2.0)
                            if wp_next:
                                wp_list.append(wp_next[0])
                        
                        cte, heading_err, curv = waypoints_to_cte_heading(
                            wp_list, transform, speed_ms
                        )
                        wp_state = (cte, heading_err, curv)
                    else:
                        wp_state = None
                except Exception as e:
                    logger.warning(f"Waypoint error: {e}")
                    wp_state = None
                
                # Run LKA step
                steer, throttle, brake, state = lka.step(
                    rgb=camera_frame.rgb,
                    speed_ms=speed_ms,
                    wp_state=wp_state,
                    prev_steer=prev_steer,
                    prev_throttle=prev_throttle,
                )
                
                # Apply control
                control = carla.VehicleControl(
                    steer=float(steer),
                    throttle=float(throttle),
                    brake=float(brake),
                )
                vehicle.apply_control(control)
                
                # Log state
                if frame_count % 10 == 0:
                    log_perception_state(state, frame_count)
                    log_control_state(state, frame_count)
                
                # Save screenshots
                if frame_count % args.screenshot_interval == 0:
                    frame_data = {
                        "rgb": state.rgb,
                        "lane_overlay": state.lane_overlay,
                        "bev_window_vis": state.bev_window_vis,
                        "mask": state.mask,
                    }
                    save_debug_frame(frame_data, frame_count, screenshot_dir)
                
                prev_steer = steer
                prev_throttle = throttle
                frame_count += 1
                
                if frame_count % 50 == 0:
                    logger.info(f"Progress: {frame_count}/{args.frames} frames")
                
            except Exception as e:
                logger.error(f"Frame processing error: {e}", exc_info=True)
        
        # Start camera with callback
        carla_mgr.start_camera(camera_callback)
        
        logger.info("Camera started, collecting data...")
        
        # Wait for frames to be collected
        while frame_count < args.frames:
            time.sleep(0.1)
        
        logger.info(f"Data collection complete: {frame_count} frames")
        logger.info(f"Screenshots saved to: {screenshot_dir}")
        logger.info(f"Logs saved to: {log_dir / 'debug.log'}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        if carla_mgr:
            carla_mgr.cleanup()
        if display_mgr:
            display_mgr.cleanup()
        
        logger.info("Cleanup complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
