#!/usr/bin/env python3
"""
Debug script for dash line detection.
Visualizes BEV pipeline steps to diagnose detection issues.
"""

import sys
import cv2
import numpy as np
import queue
import math
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import carla
from perception.lane_detector import LaneDetector
from perception.bev_lane_pipeline import (
    BEVLanePipeline, to_bev, from_bev,
    BEV_SRC, BEV_DST, BEV_W, BEV_H, PROB_THRESHOLD,
    find_lane_starts_bev, validate_bev_pair
)

# Model path
MODEL_PATH = Path(__file__).parent / "model" / "lane_unet_final.pth"


def debug_bev_pipeline(prob_map: np.ndarray, save_prefix: str = "debug"):
    """Debug BEV pipeline step by step."""
    
    print(f"\n{'='*60}")
    print("BEV Pipeline Debug")
    print(f"{'='*60}")
    print(f"Input prob_map shape: {prob_map.shape}")
    print(f"Input prob_map range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    
    # Step 1: Warp to BEV
    bev_prob = to_bev(prob_map.astype(np.float32))
    print(f"\n1. BEV warp: {bev_prob.shape}")
    print(f"   BEV prob range: [{bev_prob.min():.3f}, {bev_prob.max():.3f}]")
    
    # Save BEV prob
    bev_prob_vis = (bev_prob * 255).astype(np.uint8)
    cv2.imwrite(f"{save_prefix}_1_bev_prob.png", bev_prob_vis)
    
    # Step 2: Threshold
    bev_binary = (bev_prob > PROB_THRESHOLD).astype(np.uint8)
    nonzero = np.count_nonzero(bev_binary)
    print(f"\n2. Threshold ({PROB_THRESHOLD}): {nonzero} nonzero pixels")
    cv2.imwrite(f"{save_prefix}_2_threshold.png", bev_binary * 255)
    
    # Step 3: Initial dilation
    dilate_kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 50))
    bev_dilated_1 = cv2.dilate(bev_binary, dilate_kernel_1)
    nonzero = np.count_nonzero(bev_dilated_1)
    print(f"\n3. Initial dilation (11x50): {nonzero} nonzero pixels")
    cv2.imwrite(f"{save_prefix}_3_dilate1.png", bev_dilated_1 * 255)
    
    # Step 4: Vertical closing
    close_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 300))
    bev_closed = cv2.morphologyEx(bev_dilated_1, cv2.MORPH_CLOSE, close_kernel_v)
    nonzero = np.count_nonzero(bev_closed)
    print(f"\n4. Vertical closing (11x300): {nonzero} nonzero pixels")
    cv2.imwrite(f"{save_prefix}_4_closed.png", bev_closed * 255)
    
    # Step 5: Final dilation
    kernel = np.ones((80, 8), np.uint8)
    bev_dilated = cv2.dilate(bev_closed, kernel)
    nonzero = np.count_nonzero(bev_dilated)
    print(f"\n5. Final dilation (80x8): {nonzero} nonzero pixels")
    cv2.imwrite(f"{save_prefix}_5_final.png", bev_dilated * 255)
    
    # Step 6: Find lane starts
    (bot_lx, bot_rx), (top_lx, top_rx), hist_info = find_lane_starts_bev(bev_dilated)
    print(f"\n6. Lane starts:")
    print(f"   Bottom: L={bot_lx}, R={bot_rx}")
    print(f"   Top: L={top_lx}, R={top_rx}")
    
    # Validate
    bot_lx_v, bot_rx_v = validate_bev_pair(bot_lx, bot_rx, BEV_W)
    print(f"   Validated: L={bot_lx_v}, R={bot_rx_v}")
    
    if bot_lx_v is not None and bot_rx_v is not None:
        width = bot_rx_v - bot_lx_v
        print(f"   Lane width: {width} pixels")
    
    # Visualize histogram
    if hist_info is not None and len(hist_info) == 2:
        bot_hist, top_hist = hist_info
        hist_vis = np.zeros((200, BEV_W, 3), dtype=np.uint8)
        
        # Normalize and draw
        if bot_hist.max() > 0:
            bot_norm = (bot_hist / bot_hist.max() * 150).astype(np.int32)
            for x, h in enumerate(bot_norm):
                cv2.line(hist_vis, (x, 199), (x, 199-h), (0, 255, 0), 1)
        
        # Mark detected peaks
        if bot_lx is not None:
            cv2.line(hist_vis, (bot_lx, 0), (bot_lx, 199), (255, 0, 0), 2)
        if bot_rx is not None:
            cv2.line(hist_vis, (bot_rx, 0), (bot_rx, 199), (0, 0, 255), 2)
        
        cv2.imwrite(f"{save_prefix}_6_histogram.png", hist_vis)
    
    # Final visualization
    final_vis = cv2.cvtColor(bev_dilated * 255, cv2.COLOR_GRAY2BGR)
    if bot_lx_v is not None:
        cv2.circle(final_vis, (bot_lx_v, BEV_H - 10), 10, (255, 0, 0), -1)
    if bot_rx_v is not None:
        cv2.circle(final_vis, (bot_rx_v, BEV_H - 10), 10, (0, 0, 255), -1)
    cv2.imwrite(f"{save_prefix}_7_final_vis.png", final_vis)
    
    print(f"\nSaved debug images with prefix: {save_prefix}_*.png")
    
    return bev_dilated, (bot_lx_v, bot_rx_v)


def main():
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    carla_map = world.get_map()
    
    # Get spawn point with dash lines (Town04 highway)
    spawn_points = carla_map.get_spawn_points()
    spawn_idx = 100  # Highway with dash lines
    
    if spawn_idx >= len(spawn_points):
        spawn_idx = 0
    
    spawn = spawn_points[spawn_idx]
    print(f"Spawning at point {spawn_idx}: {spawn.location}")
    
    # Spawn vehicle
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn)
    
    # Spawn camera
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')
    
    cam_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    
    # Image queue
    img_queue = queue.Queue()
    camera.listen(lambda img: img_queue.put(img))
    
    # Initialize detector
    detector = LaneDetector(str(MODEL_PATH))
    pipeline = BEVLanePipeline()
    
    try:
        print("\nWaiting for camera images...")
        
        # Capture a few frames
        for frame_idx in range(5):
            # Wait for image
            try:
                img = img_queue.get(timeout=2.0)
            except queue.Empty:
                print("No image received")
                continue
            
            # Convert to RGB
            raw = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))
            rgb = raw[:, :, 2::-1].copy()
            
            # Save raw image
            cv2.imwrite(f"debug_frame{frame_idx}_0_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # Get probability map from UNet directly
            import torch
            with torch.no_grad():
                img_tensor = torch.FloatTensor(rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
                img_tensor = img_tensor.to(detector.model.device)
                output = detector.model(img_tensor)
                probs = torch.softmax(output, dim=1)
                prob_map = probs[0, 1].cpu().numpy()
            
            # Save prob map
            prob_vis = (prob_map * 255).astype(np.uint8)
            cv2.imwrite(f"debug_frame{frame_idx}_0_prob.png", prob_vis)
            
            print(f"\n{'#'*60}")
            print(f"Frame {frame_idx}")
            print(f"{'#'*60}")
            
            # Debug BEV pipeline
            debug_bev_pipeline(prob_map, f"debug_frame{frame_idx}")
            
            # Small delay
            world.tick()
        
        print("\n" + "="*60)
        print("Debug complete! Check debug_frame*_*.png files")
        print("="*60)
        
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()


if __name__ == "__main__":
    main()
