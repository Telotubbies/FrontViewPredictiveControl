#!/usr/bin/env python3
"""Debug ego-lane mask generation."""

import sys
import cv2
import numpy as np
import queue
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import carla
from perception.ego_lane_mask import create_ego_lane_mask_from_waypoints

MODEL_PATH = Path(__file__).parent / "model" / "lane_unet_final.pth"


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    carla_map = world.get_map()
    
    spawn_points = carla_map.get_spawn_points()
    spawn_idx = 100
    spawn = spawn_points[spawn_idx]
    print(f"Spawning at point {spawn_idx}")
    
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn)
    
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')
    
    cam_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    
    img_queue = queue.Queue()
    camera.listen(lambda img: img_queue.put(img))
    
    try:
        for frame in range(3):
            world.tick()
            
            try:
                img = img_queue.get(timeout=2.0)
            except queue.Empty:
                continue
            
            raw = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))
            rgb = raw[:, :, 2::-1].copy()
            
            # Get waypoints
            loc = vehicle.get_location()
            wp = carla_map.get_waypoint(loc, project_to_road=True)
            
            waypoints = [wp]
            traveled = 0.0
            while traveled < 50.0:
                nexts = wp.next(2.0)
                if not nexts:
                    break
                wp = nexts[0]
                waypoints.append(wp)
                traveled += 2.0
            
            print(f"\nFrame {frame}: {len(waypoints)} waypoints")
            print(f"Vehicle location: {vehicle.get_location()}")
            print(f"First waypoint: {waypoints[0].transform.location}")
            
            # Create ego-lane mask
            ego_mask = create_ego_lane_mask_from_waypoints(
                image_shape=(480, 640),
                waypoints=waypoints,
                vehicle_transform=vehicle.get_transform(),
                fov=90.0,
                lane_half_width=2.0,
            )
            
            print(f"Ego mask nonzero: {np.count_nonzero(ego_mask)}")
            
            # Save images
            cv2.imwrite(f"debug_ego_mask_{frame}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"debug_ego_mask_{frame}_mask.png", ego_mask)
            
            # Overlay
            overlay = rgb.copy()
            overlay[ego_mask > 0] = overlay[ego_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
            cv2.imwrite(f"debug_ego_mask_{frame}_overlay.png", cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            print(f"Saved debug_ego_mask_{frame}_*.png")
        
        print("\nDebug complete!")
        
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()


if __name__ == "__main__":
    main()
