#!/usr/bin/env python3
"""
Test BEV lane detection with 20 images from center-lane positions
where both L and R are dash lines.
"""

import sys
import cv2
import numpy as np
import queue
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import carla
from perception.lane_detector import LaneDetector
from perception.bev_lane_pipeline import BEVLanePipeline

MODEL_PATH = Path(__file__).parent / "model" / "lane_unet_final.pth"
OUTPUT_DIR = Path(__file__).parent / "test_dash_results"


def find_dash_lane_spawns(carla_map, spawn_points):
    """Find spawn points where vehicle is in middle lane with dash lines on both sides."""
    dash_spawns = []
    
    for idx, sp in enumerate(spawn_points):
        wp = carla_map.get_waypoint(sp.location, project_to_road=True)
        if wp is None:
            continue
        
        # Check if both left and right lanes exist and are dashed
        left_wp = wp.get_left_lane()
        right_wp = wp.get_right_lane()
        
        # Get lane markings
        left_marking = wp.left_lane_marking
        right_marking = wp.right_lane_marking
        
        # Check for dashed markings (Broken, BrokenBroken, etc.)
        left_is_dash = left_marking and left_marking.type in [
            carla.LaneMarkingType.Broken,
            carla.LaneMarkingType.BrokenBroken,
            carla.LaneMarkingType.BrokenSolid,
        ]
        right_is_dash = right_marking and right_marking.type in [
            carla.LaneMarkingType.Broken,
            carla.LaneMarkingType.BrokenBroken,
            carla.LaneMarkingType.SolidBroken,
        ]
        
        if left_is_dash and right_is_dash:
            dash_spawns.append((idx, sp, wp))
            print(f"  Spawn {idx}: L={left_marking.type}, R={right_marking.type}")
    
    return dash_spawns


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    
    print(f"Map: {carla_map.name}")
    print(f"Total spawn points: {len(spawn_points)}")
    print("\nFinding spawn points with dash lines on both sides...")
    
    dash_spawns = find_dash_lane_spawns(carla_map, spawn_points)
    print(f"\nFound {len(dash_spawns)} spawn points with L/R dash lines")
    
    if len(dash_spawns) < 20:
        print(f"Warning: Only {len(dash_spawns)} dash-lane spawns found, will use all")
    
    # Use up to 20 spawn points
    test_spawns = dash_spawns[:20]
    
    # Initialize detector
    detector = LaneDetector(model_path=str(MODEL_PATH), model_type="unet")
    pipeline = BEVLanePipeline()
    
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')
    
    results = []
    
    for test_idx, (spawn_idx, spawn, wp) in enumerate(test_spawns):
        print(f"\n{'='*60}")
        print(f"Test {test_idx+1}/20: Spawn point {spawn_idx}")
        print(f"{'='*60}")
        
        # Spawn vehicle
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        
        # Attach camera
        cam_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        
        img_queue = queue.Queue()
        camera.listen(lambda img: img_queue.put(img))
        
        try:
            # Wait for stable image
            for _ in range(5):
                world.tick()
            
            try:
                img = img_queue.get(timeout=2.0)
            except queue.Empty:
                print("  No image received!")
                results.append({
                    'spawn_idx': spawn_idx,
                    'success': False,
                    'reason': 'no_image'
                })
                continue
            
            # Convert to RGB
            raw = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))
            rgb = raw[:, :, 2::-1].copy()
            
            # Run BEV detection
            result = detector.detect_lanes_bev(rgb, return_vis=True)
            left_c, right_c, cte, heading, curv, conf, bev_vis = result
            
            # Analyze result
            left_detected = left_c is not None
            right_detected = right_c is not None
            both_detected = left_detected and right_detected
            
            print(f"  Left detected: {left_detected}")
            print(f"  Right detected: {right_detected}")
            print(f"  CTE: {cte:.3f}m")
            print(f"  Confidence: {conf:.2f}")
            
            # Save images
            cv2.imwrite(str(OUTPUT_DIR / f"test_{test_idx:02d}_sp{spawn_idx}_rgb.png"), 
                       cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            if bev_vis is not None:
                cv2.imwrite(str(OUTPUT_DIR / f"test_{test_idx:02d}_sp{spawn_idx}_bev.png"), 
                           cv2.cvtColor(bev_vis, cv2.COLOR_RGB2BGR))
            
            results.append({
                'spawn_idx': spawn_idx,
                'success': both_detected,
                'left_detected': left_detected,
                'right_detected': right_detected,
                'cte': cte,
                'confidence': conf,
            })
            
        finally:
            camera.stop()
            camera.destroy()
            vehicle.destroy()
            time.sleep(0.2)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total = len(results)
    both_ok = sum(1 for r in results if r.get('success', False))
    left_ok = sum(1 for r in results if r.get('left_detected', False))
    right_ok = sum(1 for r in results if r.get('right_detected', False))
    
    print(f"Total tests: {total}")
    print(f"Both L+R detected: {both_ok}/{total} ({100*both_ok/total:.1f}%)")
    print(f"Left detected: {left_ok}/{total} ({100*left_ok/total:.1f}%)")
    print(f"Right detected: {right_ok}/{total} ({100*right_ok/total:.1f}%)")
    
    if both_ok < total:
        print(f"\nFailed tests:")
        for r in results:
            if not r.get('success', False):
                print(f"  Spawn {r['spawn_idx']}: L={r.get('left_detected')}, R={r.get('right_detected')}")
    
    avg_cte = np.mean([r['cte'] for r in results if 'cte' in r])
    avg_conf = np.mean([r['confidence'] for r in results if 'confidence' in r])
    print(f"\nAverage CTE: {avg_cte:.3f}m")
    print(f"Average confidence: {avg_conf:.2f}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
