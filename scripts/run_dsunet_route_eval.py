#!/usr/bin/env python3
"""
DSUNet + Route Planner + Lane-Map Fusion — Integration test in CARLA.

This script:
1. Connects to CARLA
2. Spawns a vehicle
3. Plans a route from current position to a random spawn point using A*
4. Drives the route using DSUNet for lane detection
5. Uses lane+map fusion to select the correct lane at intersections
6. Measures DSUNet performance (IoU, CTE, lane departures) in real-time
7. Prints a summary report

Usage:
    cd /home/supawich/Desktop/FrontViewPredictiveControl
    source venv/bin/activate
    PYTHONPATH=src:. python3 scripts/run_dsunet_route_eval.py --duration 60
"""
import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (SRC, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DSUNet + Route + Lane-Map Fusion Eval")
    parser.add_argument("--model", type=str,
                        default="dsunet_carla_20260302_031021/best_model_iou.pth")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--speed", type=float, default=30.0, help="Target speed km/h")
    parser.add_argument("--town", type=str, default="Town04")
    parser.add_argument("--no-route", action="store_true",
                        help="Skip route planning, just drive straight")
    args = parser.parse_args()

    import carla

    # ── 1. Connect to CARLA ──────────────────────────────────────────────
    logger.info("Connecting to CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load town
    try:
        world = client.load_world(args.town)
    except Exception:
        world = client.get_world()
    world_map = world.get_map()

    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 Hz
    world.apply_settings(settings)

    # ── 2. Spawn vehicle ─────────────────────────────────────────────────
    spawn_points = world_map.get_spawn_points()
    if not spawn_points:
        logger.error("No spawn points available")
        return

    # Use first spawn point
    start_transform = spawn_points[0]
    blueprint = world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
    vehicle = world.spawn_actor(blueprint, start_transform)
    logger.info("Vehicle spawned at %s", start_transform.location)

    # ── 3. Setup camera ──────────────────────────────────────────────────
    bp_cam = world.get_blueprint_library().find('sensor.camera.rgb')
    bp_cam.set_attribute('image_size_x', '640')
    bp_cam.set_attribute('image_size_y', '480')
    bp_cam.set_attribute('fov', '90')
    cam_transform = carla.Transform(
        carla.Location(x=2.0, z=1.8),
        carla.Rotation(pitch=-8.0),
    )
    camera = world.spawn_actor(bp_cam, cam_transform, attach_to=vehicle)

    # Camera callback
    import queue
    image_queue = queue.Queue(maxsize=2)
    def camera_callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((480, 640, 4))
        array = array[:, :, :3]  # BGRA → BGR
        array = array[:, :, ::-1]  # BGR → RGB
        try:
            image_queue.put_nowait(array)
        except queue.Full:
            try:
                image_queue.get_nowait()
                image_queue.put_nowait(array)
            except queue.Empty:
                pass
    camera.listen(camera_callback)

    # ── 4. Setup route planner ───────────────────────────────────────────
    logger.info("Building route planner...")
    from planning.route_planner import GlobalRoutePlanner, LANE_KEEP, ARRIVE

    planner = GlobalRoutePlanner(world_map)

    # Pick a destination (far spawn point)
    goal_transform = spawn_points[len(spawn_points) // 2]
    logger.info("Planning route from %s to %s",
                start_transform.location, goal_transform.location)

    route = planner.plan_route(start_transform, goal_transform)
    logger.info("Route planned: %d entries", len(route))

    # Print route summary
    maneuvers = [e for e in route if e.action != LANE_KEEP]
    for m in maneuvers[:10]:
        logger.info("  %s at (%.1f, %.1f) dist=%.1fm",
                    m.action, m.waypoint.transform.location.x,
                    m.waypoint.transform.location.y,
                    m.distance_from_start_m)

    # ── 5. Setup lane-map fusion ─────────────────────────────────────────
    from planning.lane_map_fusion import LaneMapFusion
    fusion = LaneMapFusion(world_map, cam_w=640, cam_h=480)

    # ── 6. Setup DSUNet evaluator ────────────────────────────────────────
    from evaluation.dsunet_driving_eval import DSUNetDrivingEvaluator
    evaluator = DSUNetDrivingEvaluator(
        model_path=args.model,
        world=world, vehicle=vehicle,
        target_speed_kmh=args.speed,
    )

    # ── 7. Setup MPC pipeline for driving ────────────────────────────────
    from pipeline import LKAPipeline
    from utils.device_utils import get_device
    device = get_device()
    pipeline = LKAPipeline(
        model_path=args.model,
        device=device,
        target_speed_kmh=args.speed,
        use_trajectory_pipeline=True,
        model_type="dsunet",
    )

    # ── 8. Drive! ────────────────────────────────────────────────────────
    logger.info("Starting driving evaluation (%d seconds)...", args.duration)
    start_time = time.time()
    frame_idx = 0
    prev_steer = 0.0
    prev_throttle = 0.0

    try:
        while time.time() - start_time < args.duration:
            world.tick()  # Advance simulation

            # Get camera frame
            try:
                rgb = image_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Get vehicle state
            vt = vehicle.get_transform()
            vel = vehicle.get_velocity()
            speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            # Get route info
            lane_info = fusion.get_lane_selection_info(vt, route)
            next_maneuver = lane_info["next_maneuver"]
            dist_to_maneuver = lane_info["distance_to_maneuver"]

            # Run pipeline (DSUNet → mask → MPC)
            steer, throttle, brake, frame_state = pipeline.step(
                rgb=rgb, speed_ms=speed_ms,
                wp_state=None,  # Don't use waypoint fusion — pure DSUNet
                prev_steer=prev_steer,
                prev_throttle=prev_throttle,
                world=world, vehicle=vehicle,
            )

            # Evaluate this frame
            metrics = evaluator.evaluate_frame(
                rgb, frame_idx, steer, throttle, brake,
            )

            # Apply control
            control = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
            )
            vehicle.apply_control(control)

            prev_steer = steer
            prev_throttle = throttle

            # Log periodically
            if frame_idx % 50 == 0:
                logger.info(
                    "frame=%d speed=%.1fkm/h steer=%.3f cte=%.3f iou=%.3f "
                    "mode=%s maneuver=%s@%.0fm",
                    frame_idx, speed_ms * 3.6, steer, metrics.cte_m,
                    metrics.dsunet_iou, metrics.mode,
                    next_maneuver or "none", dist_to_maneuver,
                )

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    # ── 9. Generate report ───────────────────────────────────────────────
    evaluator._start_location = start_transform.location
    final_loc = vehicle.get_transform().location
    evaluator._prev_location = final_loc

    report = evaluator.generate_report()
    evaluator.print_summary(report)

    # Save report
    output_path = f"runs/dsunet_eval_{int(time.time())}.json"
    evaluator.save_report(report, output_path)

    # ── 10. Cleanup ──────────────────────────────────────────────────────
    logger.info("Cleaning up...")
    camera.stop()
    camera.destroy()
    vehicle.destroy()

    # Reset world settings
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    logger.info("Done. Report saved to %s", output_path)


if __name__ == "__main__":
    main()
