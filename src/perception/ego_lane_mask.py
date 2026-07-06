#!/usr/bin/env python3
"""
Ego-Lane Mask Generator using CARLA Waypoints.

Creates a mask that filters UNet output to only show the ego-lane,
preventing detection of adjacent lanes.
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple


def create_ego_lane_mask_from_waypoints(
    image_shape: Tuple[int, int],  # (H, W)
    waypoints: List,  # List of CARLA waypoints
    vehicle_transform,  # CARLA Transform
    camera_transform=None,  # CARLA Transform (relative to vehicle)
    fov: float = 90.0,
    lane_half_width: float = 2.0,  # meters
) -> np.ndarray:
    """
    Create a binary mask showing only the ego-lane area.
    
    This mask can be applied to UNet probability map to filter out
    adjacent lanes and only keep the ego-lane.
    
    Args:
        image_shape: (H, W) of the image
        waypoints: List of CARLA waypoints for ego-lane
        vehicle_transform: Vehicle's world transform
        camera_transform: Camera transform relative to vehicle
        fov: Camera field of view in degrees
        lane_half_width: Half-width of lane in meters
        
    Returns:
        Binary mask (H, W) with 255 for ego-lane area
    """
    try:
        import carla
    except ImportError:
        return np.ones(image_shape, dtype=np.uint8) * 255
    
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if not waypoints or len(waypoints) < 2:
        return np.ones((h, w), dtype=np.uint8) * 255
    
    # Camera intrinsic
    f = w / (2.0 * np.tan(np.radians(fov / 2.0)))
    cx, cy = w / 2.0, h / 2.0
    
    # Camera world transform
    if camera_transform is None:
        camera_transform = carla.Transform(
            carla.Location(x=2.0, z=1.4),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
    
    # Compose transforms
    try:
        cam_world_tf = vehicle_transform * camera_transform
    except TypeError:
        vl = vehicle_transform.location
        vr = vehicle_transform.rotation
        cl = camera_transform.location
        cr = camera_transform.rotation
        cam_world_tf = carla.Transform(
            carla.Location(x=vl.x + cl.x, y=vl.y + cl.y, z=vl.z + cl.z),
            carla.Rotation(
                pitch=vr.pitch + cr.pitch,
                yaw=vr.yaw + cr.yaw,
                roll=vr.roll + cr.roll,
            ),
        )
    
    # World → Camera matrix
    W2C = np.array(cam_world_tf.get_inverse_matrix(), dtype=np.float64)
    
    def project(wx: float, wy: float, wz: float) -> Optional[Tuple[int, int]]:
        """Project world point to image coordinates."""
        pw = np.array([wx, wy, wz, 1.0], dtype=np.float64)
        pc = W2C @ pw
        
        # CARLA: X=forward, Y=right, Z=up → OpenCV: X=right, Y=down, Z=forward
        cam_x, cam_y, cam_z = pc[1], -pc[2], pc[0]
        
        if cam_z <= 0.1:
            return None
        
        u = int(f * cam_x / cam_z + cx)
        v = int(f * cam_y / cam_z + cy)
        
        if 0 <= u < w and 0 <= v < h:
            return (u, v)
        return None
    
    def get_lane_edge_points(wp) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Get left and right edge points of lane at waypoint."""
        loc = wp.transform.location
        
        # Use CARLA's built-in right vector
        right_vec = wp.transform.get_right_vector()
        
        # Left and right edge points (use actual lane width from waypoint if available)
        actual_half_width = wp.lane_width / 2.0 if hasattr(wp, 'lane_width') else lane_half_width
        
        left_pt = project(
            loc.x - right_vec.x * actual_half_width,
            loc.y - right_vec.y * actual_half_width,
            loc.z
        )
        right_pt = project(
            loc.x + right_vec.x * actual_half_width,
            loc.y + right_vec.y * actual_half_width,
            loc.z
        )
        
        return left_pt, right_pt
    
    # Collect polygon points
    left_points = []
    right_points = []
    
    for wp in waypoints[:30]:  # Limit to first 30 waypoints
        left_pt, right_pt = get_lane_edge_points(wp)
        if left_pt is not None:
            left_points.append(left_pt)
        if right_pt is not None:
            right_points.append(right_pt)
    
    if len(left_points) < 2 or len(right_points) < 2:
        return np.ones((h, w), dtype=np.uint8) * 255
    
    # Add bottom edge points to close the polygon at image bottom
    # Find the leftmost and rightmost points at the bottom (highest v value)
    left_bottom = max(left_points, key=lambda p: p[1])  # highest v (closest to bottom)
    right_bottom = max(right_points, key=lambda p: p[1])
    
    # Extend to image bottom edge - use the actual lane edge positions
    # Add margin for lane markings
    margin = 30
    bottom_left = (max(0, left_bottom[0] - margin), h - 1)
    bottom_right = (min(w - 1, right_bottom[0] + margin), h - 1)
    
    # Create polygon: left points → bottom corners → right points (reversed)
    polygon_points = left_points + [bottom_left, bottom_right] + right_points[::-1]
    polygon = np.array(polygon_points, dtype=np.int32)
    
    # Fill polygon
    cv2.fillPoly(mask, [polygon], 255)
    
    # Dilate to add margin for lane markings (moderate size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    mask = cv2.dilate(mask, kernel)
    
    return mask


def filter_prob_map_with_ego_mask(
    prob_map: np.ndarray,
    ego_mask: np.ndarray,
    blend_factor: float = 0.0,  # 0 = completely suppress non-ego areas
) -> np.ndarray:
    """
    Filter UNet probability map to only show ego-lane.
    
    Args:
        prob_map: UNet probability map (H, W) in [0, 1]
        ego_mask: Binary mask (H, W) with 255 for ego-lane
        blend_factor: How much to keep non-ego areas (0=suppress all, 1=keep all)
        
    Returns:
        Filtered probability map
    """
    # Normalize mask to [0, 1]
    mask_norm = ego_mask.astype(np.float32) / 255.0
    
    # Apply mask: inside ego-lane = full prob, outside = suppressed
    # blend_factor=0 means completely zero out non-ego areas
    filtered = prob_map * (mask_norm + (1 - mask_norm) * blend_factor)
    
    return filtered
