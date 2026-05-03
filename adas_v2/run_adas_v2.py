#!/usr/bin/env python3
"""
ADAS v2: Intelligent Mode Switching Lane Keeping System

Usage:
    python -m adas_v2.run_adas_v2 --town Town04 --speed 30
    
Features:
    - Intelligent mode switching (LANE_ONLY / WAYPOINT_ONLY / FUSION)
    - Curve anticipation braking (50m lookahead)
    - Smooth transitions between modes
    - Google Maps style: trust waypoints when turning
"""

import argparse
import logging
import math
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import carla
from adas_v2.controller import ADASController, ADASConfig
from adas_v2.mode_selector import ModeConfig
from adas_v2.recorder import ADASRecorder
from adas_v2.dashboard import ADASv2Dashboard, DashboardState
from adas_v2.lane_change import LaneChangeDetector
from adas_v2.metrics import PerformanceMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default model path
DEFAULT_MODEL = Path(__file__).parent.parent / "model" / "lane_unet_final.pth"

# Camera settings
CAM_W, CAM_H = 640, 480
CAM_FOV = 90


def get_waypoints(vehicle, carla_map, distance: float = 100.0, spacing: float = 2.0):
    """Get waypoints ahead of vehicle. Returns (waypoints, current_lane_id)."""
    loc = vehicle.get_location()
    wp = carla_map.get_waypoint(loc, project_to_road=True)
    if wp is None:
        return [], None
    
    current_lane_id = wp.lane_id  # CARLA lane_id for "cheating"
    
    waypoints = [wp]
    traveled = 0.0
    while traveled < distance:
        nexts = wp.next(spacing)
        if not nexts:
            break
        wp = nexts[0]
        waypoints.append(wp)
        traveled += spacing
    return waypoints, current_lane_id


def waypoints_to_state(vehicle_transform, waypoints):
    """Convert waypoints to CTE, heading, curvature."""
    if not waypoints or len(waypoints) < 2:
        return 0.0, 0.0, 0.0
    
    veh_loc = vehicle_transform.location
    veh_yaw = math.radians(vehicle_transform.rotation.yaw)
    
    # Find closest waypoint
    min_dist = float('inf')
    closest_idx = 0
    for i, wp in enumerate(waypoints):
        d = math.sqrt((wp.transform.location.x - veh_loc.x)**2 + 
                      (wp.transform.location.y - veh_loc.y)**2)
        if d < min_dist:
            min_dist = d
            closest_idx = i
    
    wp = waypoints[closest_idx]
    wp_loc = wp.transform.location
    wp_yaw = math.radians(wp.transform.rotation.yaw)
    
    # CTE: signed distance to waypoint
    dx = wp_loc.x - veh_loc.x
    dy = wp_loc.y - veh_loc.y
    cte = -dx * math.sin(veh_yaw) + dy * math.cos(veh_yaw)
    
    # Heading error
    heading = wp_yaw - veh_yaw
    heading = math.atan2(math.sin(heading), math.cos(heading))
    
    # Curvature from waypoints
    curv = 0.0
    if closest_idx < len(waypoints) - 2:
        wp1 = waypoints[closest_idx]
        wp2 = waypoints[closest_idx + 1]
        wp3 = waypoints[closest_idx + 2]
        
        yaw1 = math.radians(wp1.transform.rotation.yaw)
        yaw2 = math.radians(wp2.transform.rotation.yaw)
        
        d = math.sqrt((wp2.transform.location.x - wp1.transform.location.x)**2 +
                      (wp2.transform.location.y - wp1.transform.location.y)**2)
        if d > 0.1:
            dyaw = math.atan2(math.sin(yaw2 - yaw1), math.cos(yaw2 - yaw1))
            curv = dyaw / d
    
    return float(np.clip(cte, -5, 5)), float(np.clip(heading, -1, 1)), float(np.clip(curv, -0.1, 0.1))


def waypoint_lookahead_curvature(waypoints, lookahead_m: float = 50.0):
    """Compute max curvature in lookahead distance."""
    if not waypoints or len(waypoints) < 3:
        return 0.0
    
    max_curv = 0.0
    cum_dist = 0.0
    prev_loc = None
    
    for i, wp in enumerate(waypoints[:-1]):
        loc = wp.transform.location
        if prev_loc:
            cum_dist += math.sqrt((loc.x - prev_loc.x)**2 + (loc.y - prev_loc.y)**2)
        prev_loc = loc
        
        if cum_dist > lookahead_m:
            break
            
        # Compute curvature
        next_wp = waypoints[i + 1]
        yaw1 = math.radians(wp.transform.rotation.yaw)
        yaw2 = math.radians(next_wp.transform.rotation.yaw)
        
        d = math.sqrt((next_wp.transform.location.x - loc.x)**2 +
                      (next_wp.transform.location.y - loc.y)**2)
        if d > 0.1:
            dyaw = math.atan2(math.sin(yaw2 - yaw1), math.cos(yaw2 - yaw1))
            curv = abs(dyaw / d)
            max_curv = max(max_curv, curv)
    
    return float(min(max_curv, 0.5))


def waypoints_to_path(vehicle_transform, waypoints):
    """Convert waypoints to (s, lat) path in vehicle frame."""
    if not waypoints:
        return None
    
    veh_loc = vehicle_transform.location
    veh_yaw = math.radians(vehicle_transform.rotation.yaw)
    
    path = []
    for wp in waypoints:
        dx = wp.transform.location.x - veh_loc.x
        dy = wp.transform.location.y - veh_loc.y
        
        # Transform to vehicle frame
        s = dx * math.cos(veh_yaw) + dy * math.sin(veh_yaw)
        lat = -dx * math.sin(veh_yaw) + dy * math.cos(veh_yaw)
        
        if s > 0:  # Only ahead
            path.append((s, lat))
    
    return path if path else None


def main():
    parser = argparse.ArgumentParser(description="ADAS v2: Intelligent Mode Switching")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--town", type=str, default="Town04")
    parser.add_argument("--speed", type=float, default=30.0)
    parser.add_argument("--spawn", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--record", action="store_true", help="Enable recording")
    parser.add_argument("--dashboard", action="store_true", help="Show dashboard")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    args = parser.parse_args()
    
    logger.info(f"ADAS v2 starting: town={args.town}, speed={args.speed} km/h")
    
    # Connect to CARLA
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    
    world = client.load_world(args.town)
    time.sleep(2.0)
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    
    carla_map = world.get_map()
    bp_lib = world.get_blueprint_library()
    
    # Spawn vehicle
    spawn_points = carla_map.get_spawn_points()
    spawn_idx = min(args.spawn, len(spawn_points) - 1)
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[spawn_idx])
    logger.info(f"Vehicle spawned at spawn point {spawn_idx}")
    
    # Spawn camera
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(CAM_W))
    camera_bp.set_attribute("image_size_y", str(CAM_H))
    camera_bp.set_attribute("fov", str(CAM_FOV))
    camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    
    # Image queue
    image_queue = queue.Queue(maxsize=1)
    camera.listen(lambda img: image_queue.put(img) if image_queue.empty() else None)
    
    # Initialize ADAS controller
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = ADASConfig(
        target_speed_kmh=args.speed,
        mode_config=ModeConfig(
            conf_high=0.75,
            conf_low=0.40,
            curv_straight=0.008,
            curv_turning=0.025,
        ),
    )
    controller = ADASController(args.model, device, config)
    
    # Initialize optional components
    recorder = ADASRecorder() if args.record else None
    dashboard = ADASv2Dashboard(width=800, height=600) if args.dashboard else None
    metrics = PerformanceMetrics()
    lane_change_detector = LaneChangeDetector()
    
    logger.info("ADAS v2 controller initialized")
    if recorder:
        recorder.start()
        logger.info("Recording enabled")
    
    # Main loop
    start_time = time.time()
    frame = 0
    last_fps_time = time.time()
    fps = 0.0
    
    try:
        while True:
            world.tick()
            
            # Check duration
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                logger.info("Duration reached, stopping")
                break
            
            # Get image
            try:
                img = image_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            raw = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))
            rgb = raw[:, :, 2::-1].copy()  # BGRA → RGB
            
            # Get vehicle state
            vel = vehicle.get_velocity()
            speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # Check traffic light with distance-based braking
            traffic_light = vehicle.get_traffic_light()
            traffic_light_state = None
            if traffic_light is not None:
                traffic_light_state = traffic_light.get_state()
                
                # Get distance to traffic light
                tl_loc = traffic_light.get_location()
                veh_loc = vehicle.get_location()
                distance_to_light = math.sqrt(
                    (tl_loc.x - veh_loc.x)**2 + 
                    (tl_loc.y - veh_loc.y)**2
                )
                
                # Calculate required braking based on distance and speed
                # Stopping distance = v^2 / (2 * deceleration)
                # Comfortable deceleration = 3 m/s^2, max = 8 m/s^2
                stopping_distance_comfort = (speed_ms ** 2) / (2 * 3.0) + 5.0  # +5m safety margin
                stopping_distance_emergency = (speed_ms ** 2) / (2 * 8.0) + 2.0
                
                if traffic_light_state == carla.TrafficLightState.Red:
                    if distance_to_light < stopping_distance_emergency:
                        # Emergency brake - too close
                        brake_force = 1.0
                    elif distance_to_light < stopping_distance_comfort:
                        # Progressive braking based on distance
                        # More distance = less brake, closer = more brake
                        brake_ratio = 1.0 - (distance_to_light - stopping_distance_emergency) / (stopping_distance_comfort - stopping_distance_emergency + 0.1)
                        brake_force = max(0.3, min(1.0, brake_ratio))
                    elif distance_to_light < stopping_distance_comfort * 1.5:
                        # Light braking / coast
                        brake_force = 0.2
                    else:
                        # Far away, just reduce throttle
                        brake_force = 0.0
                    
                    if brake_force > 0 or distance_to_light < stopping_distance_comfort * 2:
                        ctrl = carla.VehicleControl(steer=0.0, throttle=0.0, brake=brake_force)
                        vehicle.apply_control(ctrl)
                        if frame % 20 == 0:
                            logger.info(f"[{frame}] RED LIGHT - dist={distance_to_light:.1f}m brk={brake_force:.2f}")
                        if brake_force > 0.5:
                            continue
                            
                elif traffic_light_state == carla.TrafficLightState.Yellow:
                    # Yellow: brake if we can't make it through
                    if distance_to_light > 10.0 and speed_ms > 5.0:
                        # Calculate if we can stop comfortably
                        if distance_to_light < stopping_distance_comfort:
                            brake_force = min(0.6, (stopping_distance_comfort - distance_to_light) / stopping_distance_comfort + 0.2)
                            ctrl = carla.VehicleControl(steer=0.0, throttle=0.0, brake=brake_force)
                            vehicle.apply_control(ctrl)
                            if frame % 20 == 0:
                                logger.info(f"[{frame}] YELLOW LIGHT - dist={distance_to_light:.1f}m brk={brake_force:.2f}")
                            continue
            
            # Get waypoints with lane_id for validation
            wps, current_lane_id = get_waypoints(vehicle, carla_map)
            wp_cte, wp_heading, wp_curv = waypoints_to_state(vehicle.get_transform(), wps)
            wp_lookahead_curv = waypoint_lookahead_curvature(wps, lookahead_m=50.0)
            wp_path = waypoints_to_path(vehicle.get_transform(), wps)
            
            # ADAS step with lane_id and waypoints for ego-lane filtering
            output = controller.step(
                rgb=rgb,
                speed_ms=speed_ms,
                waypoint_cte=wp_cte,
                waypoint_heading=wp_heading,
                waypoint_curv=wp_curv,
                wp_lookahead_curv=wp_lookahead_curv,
                waypoint_path=wp_path,
                lane_id=current_lane_id,
                waypoints=wps,
                vehicle_transform=vehicle.get_transform(),
            )
            
            # Apply control
            ctrl = carla.VehicleControl(
                steer=float(output.steer),
                throttle=float(output.throttle),
                brake=float(output.brake),
            )
            vehicle.apply_control(ctrl)
            
            # Debug: log control when speed is low
            if speed_ms < 1.0 and frame % 5 == 0:
                logger.debug(f"[{frame}] LOW SPEED: thr={output.throttle:.2f} brk={output.brake:.2f} spd={speed_ms:.2f}")
            
            # Update metrics
            metrics.update(
                cte=output.fused_cte,
                speed_ms=speed_ms,
                target_speed_ms=output.target_speed_ms,
                steer=output.steer,
                heading_rad=output.fused_heading,
                mode=output.mode,
            )
            
            # Lane change detection
            lc_event = lane_change_detector.update(output.fused_cte, time.time())
            
            # Recording
            if recorder:
                recorder.record_frame(
                    rgb=rgb,
                    speed_ms=speed_ms,
                    target_speed_ms=output.target_speed_ms,
                    steer=output.steer,
                    throttle=output.throttle,
                    brake=output.brake,
                    cte=output.fused_cte,
                    heading=output.fused_heading,
                    curvature=output.fused_curvature,
                    lane_confidence=output.lane_weight,
                    lane_weight=output.lane_weight,
                    mode=output.mode,
                    wp_lookahead_curv=wp_lookahead_curv,
                )
            
            # FPS calculation
            if frame % 10 == 0:
                now = time.time()
                fps = 10.0 / (now - last_fps_time) if now > last_fps_time else 0
                last_fps_time = now
            
            # Log
            if frame % 20 == 0:
                speed_kmh = speed_ms * 3.6
                target_kmh = output.target_speed_ms * 3.6
                lc_warn = " [!LDW]" if lc_event.warning else ""
                logger.info(
                    f"[{frame:4d}] {output.mode:12s} | "
                    f"spd={speed_kmh:4.1f}/{target_kmh:4.1f} km/h | "
                    f"cte={output.fused_cte:+5.2f}m | "
                    f"curv={wp_lookahead_curv:.3f}{lc_warn}"
                )
            
            # Display
            if not args.no_display:
                # Use Addison-style lane overlay
                vis_rgb = rgb.copy()
                h, w = vis_rgb.shape[:2]
                bev_thumb = None
                
                # Get lane detection with Addison-style visualization
                if hasattr(controller, '_detector') and controller._detector is not None:
                    try:
                        from perception.addison_lane import AddisonLaneDetector
                        global _addison_detector
                        if '_addison_detector' not in globals() or _addison_detector is None:
                            _addison_detector = AddisonLaneDetector(
                                use_unet=True,
                                model_path=str(Path(__file__).parent.parent / "model" / "lane_unet_final.pth")
                            )
                            logger.info(f"Addison detector loaded: {_addison_detector.model is not None}")
                        
                        result, curvature, offset, confidence, bev_thumb = _addison_detector.process(rgb)
                        # result is BGR from cv2, keep as RGB for dashboard (dashboard will convert)
                        vis_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        if frame == 20:
                            logger.info(f"Addison: conf={confidence:.2f}, curv={curvature:.1f}, offset={offset:.3f}")
                    except ImportError as e:
                        logger.warning(f"Addison import error: {e}")
                        # Fallback to test_unet_lane
                        try:
                            from test_unet_lane import process_frame_vis
                            vis_rgb, bev_thumb = process_frame_vis(
                                controller._detector.model, rgb, frame, output.mode
                            )
                        except:
                            pass
                    except Exception as e:
                        logger.warning(f"Addison error: {e}")
                
                # Add BEV thumbnail (top-right corner)
                if bev_thumb is not None:
                    thumb_h = h // 3
                    thumb_w = thumb_h
                    bev_small = cv2.resize(bev_thumb, (thumb_w, thumb_h))
                    vis_rgb[5:5 + thumb_h, w - 5 - thumb_w:w - 5] = bev_small
                
                # Draw info overlay
                mode_color = (0, 255, 0) if "LANE" in output.mode else (0, 200, 255) if "FUSION" in output.mode else (255, 150, 0)
                info_text = f"{output.mode} | UNet | L:{output.lane_weight*100:.0f}%"
                cv2.putText(vis_rgb, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(vis_rgb, f"Spd: {speed_ms*3.6:.1f}/{output.target_speed_ms*3.6:.1f} km/h", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(vis_rgb, f"CTE: {output.fused_cte:+.2f}m  Steer: {output.steer:+.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
                if dashboard:
                    # Use dashboard
                    dash_state = DashboardState(
                        rgb=vis_rgb,
                        speed_kmh=speed_ms * 3.6,
                        target_speed_kmh=output.target_speed_ms * 3.6,
                        steer=output.steer,
                        throttle=output.throttle,
                        brake=output.brake,
                        cte=output.fused_cte,
                        heading_deg=math.degrees(output.fused_heading),
                        curvature=output.fused_curvature,
                        lane_confidence=output.lane_weight,
                        lane_weight=output.lane_weight,
                        mode=output.mode,
                        wp_lookahead_curv=wp_lookahead_curv,
                        frame_id=frame,
                        fps=fps,
                    )
                    vis = dashboard.render(dash_state)
                else:
                    # Simple display with overlay
                    vis = vis_rgb.copy()
                    cv2.putText(vis, f"Mode: {output.mode}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis, f"Speed: {speed_ms*3.6:.1f} km/h", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis, f"CTE: {output.fused_cte:+.2f}m", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis, f"Thr: {output.throttle:.2f} Brk: {output.brake:.2f}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                
                cv2.imshow("ADAS v2", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame += 1
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Print metrics summary
        metrics.print_summary()
        
        # Stop recording
        if recorder:
            rec_metrics = recorder.stop()
            logger.info(f"Recording saved with metrics: CTE RMS={rec_metrics.get('cte_rms', 0):.3f}m")
        
        # Cleanup
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        cv2.destroyAllWindows()
        logger.info("ADAS v2 stopped")


if __name__ == "__main__":
    main()
