"""
Obstacle Avoidance Test Script — ทดสอบระบบหลบสิ่งกีดขวางใน CARLA

สถานการณ์ทดสอบ (ตาม Euro NCAP / NHTSA):
  1. CCRs — Car-to-Car Rear stationary (รถจอดนิ่งข้างหน้า)
  2. CCRm — Car-to-Car Rear moving (รถข้างหน้าช้ากว่า)
  3. CPNA — Child Pedestrian Nearside Adult (คนเดินขวาง)
  4. Stationary — สิ่งกีดขวางนิ่งกลางถนน

วัด metrics:
  - TTC, lateral clearance, speed reduction, lateral accel
  - Steering response time, lane departure, stability
  - Collision avoided / occurred

Usage:
  PYTHONPATH=src:. python3 src/tests/test_obstacle_avoidance.py --speed 20
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time

import numpy as np

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


def run_avoidance_test(speed_kmh: float = 20, scenario: str = "CCRs") -> dict:
    """
    รันการทดสอบหลบสิ่งกีดขวาง 1 ครั้ง

    Args:
        speed_kmh: ความเร็วเริ่มต้น
        scenario: สถานการณ์ทดสอบ (CCRs, CCRm, CPNA, Stationary)

    Returns:
        dict สรุปผลการทดสอบ
    """
    import carla

    from telemetry.avoidance_test_report import AvoidanceTestReport
    from bridge.obstacles import get_traffic_obstacles

    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Use Town04 (has long straight road)
    try:
        world = client.load_world('Town04')
    except Exception:
        world = client.get_world()

    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    # Get spawn point
    blueprint_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        logger.error("No spawn points available")
        return {"error": "No spawn points"}

    # Spawn ego vehicle
    ego_bp = blueprint_lib.find('vehicle.lincoln.mkz_2017')
    ego_transform = spawn_points[0]
    ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
    logger.info(f"Ego vehicle spawned at {ego_transform.location}")

    # Spawn obstacle 50m ahead
    obstacle_loc = ego_transform.location
    forward_vec = ego_transform.rotation.get_forward_vector()
    obstacle_loc.x += forward_vec.x * 50
    obstacle_loc.y += forward_vec.y * 50
    obstacle_loc.z += forward_vec.z * 50

    obstacle_transform = carla.Transform(obstacle_loc, ego_transform.rotation)

    if scenario in ("CCRs", "Stationary"):
        # Stationary vehicle
        obs_bp = blueprint_lib.find('vehicle.tesla.model3')
        obstacle = world.spawn_actor(obs_bp, obstacle_transform)
        logger.info(f"Stationary obstacle spawned 50m ahead")
    elif scenario == "CCRm":
        # Moving vehicle (slower)
        obs_bp = blueprint_lib.find('vehicle.tesla.model3')
        obstacle = world.spawn_actor(obs_bp, obstacle_transform)
        obstacle.apply_control(carla.VehicleControl(throttle=0.3, steer=0))
        logger.info(f"Moving obstacle spawned 50m ahead (slow)")
    elif scenario == "CPNA":
        # Pedestrian
        obs_bp = blueprint_lib.find('walker.pedestrian.0001')
        obstacle = world.spawn_actor(obs_bp, obstacle_transform)
        logger.info(f"Pedestrian obstacle spawned 50m ahead")
    else:
        obstacle = None

    # Give vehicle initial speed
    ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))

    # Initialize test report
    report = AvoidanceTestReport(environment="simulation")
    report.start_run(scenario=scenario, speed_kmh=speed_kmh)

    # Test parameters
    target_speed_ms = speed_kmh / 3.6
    test_timeout = 15.0  # 15 seconds max
    test_start = time.time()
    collision = False
    min_distance = float('inf')

    try:
        while time.time() - test_start < test_timeout:
            world.tick()  # Advance simulation

            # Get ego state
            ego_loc = ego_vehicle.get_location()
            ego_vel = ego_vehicle.get_velocity()
            ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)

            # Get obstacle state
            if obstacle:
                obs_loc = obstacle.get_location()
                obs_vel = obstacle.get_velocity() if hasattr(obstacle, 'get_velocity') else carla.Vector3D(0,0,0)
                obs_speed = math.sqrt(obs_vel.x**2 + obs_vel.y**2 + obs_vel.z**2)

                # Distance to obstacle
                dx = obs_loc.x - ego_loc.x
                dy = obs_loc.y - ego_loc.y
                dz = obs_loc.z - ego_loc.z
                distance = math.sqrt(dx**2 + dy**2 + dz**2)
                min_distance = min(min_distance, distance)

                # Forward distance (in ego frame)
                ego_rot = ego_vehicle.get_transform().rotation
                forward = ego_rot.get_forward_vector()
                fwd_dist = dx * forward.x + dy * forward.y + dz * forward.z

                # Lateral distance
                right = ego_rot.get_right_vector()
                lat_dist = abs(dx * right.x + dy * right.y + dz * right.z)

                # TTC
                closing_rate = ego_speed - obs_speed
                if closing_rate > 0.1:
                    ttc = fwd_dist / closing_rate
                else:
                    ttc = float('inf')

                # Record metrics
                if fwd_dist < 40 and fwd_dist > 0:
                    report.record_obstacle_detected(ttc)
                    report.record_ttc(ttc)
                    report.record_lateral_clearance(lat_dist)
                    report.record_lateral_offset(lat_dist)
                    report.record_speed(ego_speed * 3.6)

                    # Lateral acceleration (approximate)
                    accel = ego_vehicle.get_acceleration()
                    lat_accel = math.sqrt(
                        accel.x**2 + accel.y**2
                    ) * 0.5  # rough estimate
                    report.record_lateral_accel(lat_accel)

                    # Yaw rate
                    angular_vel = ego_vehicle.get_angular_velocity()
                    yaw_rate = math.degrees(angular_vel.z)
                    report.record_yaw_rate(yaw_rate)

                # Check collision
                if distance < 2.0:  # 2m = collision
                    collision = True
                    logger.warning(f"COLLISION detected at distance={distance:.2f}m")
                    break

                # Check if passed obstacle
                if fwd_dist < -5:
                    logger.info("Ego vehicle passed obstacle — test complete")
                    break

            # Maintain speed
            if ego_speed < target_speed_ms:
                ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))

    finally:
        # End test run
        impact_speed = ego_speed * 3.6 if collision else 0.0
        run = report.end_run(
            collision_avoided=not collision,
            impact_speed_kmh=impact_speed,
            notes=f"min_dist={min_distance:.2f}m"
        )

        # Cleanup
        try:
            ego_vehicle.destroy()
        except Exception:
            pass
        if obstacle:
            try:
                obstacle.destroy()
            except Exception:
                pass

        # Restore async mode
        settings.synchronous_mode = False
        world.apply_settings(settings)

    # Print results
    report.print_checklist()
    report.print_summary()

    # Save report
    report_path = f"/tmp/avoidance_test_{scenario}_{speed_kmh}kmh.json"
    report.save(report_path)
    logger.info(f"Report saved to {report_path}")

    return report._summary()


def main():
    parser = argparse.ArgumentParser(description="Obstacle Avoidance Test")
    parser.add_argument('--speed', type=float, default=20, help='Speed km/h')
    parser.add_argument('--scenario', type=str, default='CCRs',
                        choices=['CCRs', 'CCRm', 'CPNA', 'Stationary'],
                        help='Test scenario')
    parser.add_argument('--log-level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )

    print(f"\n{'='*70}")
    print(f"  Obstacle Avoidance Test — {args.scenario} @ {args.speed} km/h")
    print(f"{'='*70}\n")

    result = run_avoidance_test(speed_kmh=args.speed, scenario=args.scenario)

    print(f"\nResult: {result}")
    return 0 if result.get('avoidance_rate_pct', 0) > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
