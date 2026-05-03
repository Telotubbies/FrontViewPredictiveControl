#!/usr/bin/env python3
"""
Optimized Reference Path Generation
Eliminate redundant polynomial sampling by passing coefficients directly
"""

import numpy as np
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


def generate_reference_path_optimized(
    waypoints: Optional[List],
    left_coeffs: Optional[np.ndarray],
    right_coeffs: Optional[np.ndarray],
    wp_weight: float = 0.8,
    lane_weight: float = 0.2,
    lookahead_m: float = 35.0,
    n_points: int = 70
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate optimized reference path using coefficients directly
    
    Args:
        waypoints: CARLA waypoints list
        left_coeffs: Left lane polynomial coefficients [a, b, c]
        right_coeffs: Right lane polynomial coefficients [a, b, c]
        wp_weight: Weight for waypoint fusion
        lane_weight: Weight for lane fusion
        lookahead_m: Lookahead distance in meters
        n_points: Number of reference points
        
    Returns:
        (x_ref, y_ref, yaw_ref, coeffs) where coeffs are centerline polynomial coefficients
    """
    
    # 1. Process waypoints (if available)
    wp_x, wp_y, wp_yaw = None, None, None
    if waypoints is not None and len(waypoints) > 0:
        # Extract waypoint coordinates
        wp_x = np.array([wp.transform.location.x for wp in waypoints])
        wp_y = np.array([wp.transform.location.y for wp in waypoints])
        wp_yaw = np.array([wp.transform.rotation.yaw for wp in waypoints])
        
        # Convert to vehicle frame (simplified - assumes proper transformation)
        # In practice, this should use proper world→vehicle transformation
        wp_x = wp_x - wp_x[0]  # Relative to first waypoint
        wp_y = wp_y - wp_y[0]
        wp_yaw = wp_yaw - wp_yaw[0]
    
    # 2. Process lane coefficients (if available)
    lane_coeffs = None
    if left_coeffs is not None and right_coeffs is not None:
        # Centerline = average of left and right lanes
        lane_coeffs = (left_coeffs + right_coeffs) / 2.0
    
    # 3. Fusion logic
    if wp_x is not None and lane_coeffs is not None:
        # Both waypoints and lane available - fuse them
        x_sample = np.linspace(0, lookahead_m, n_points)
        
        # Sample lane at reference points
        y_lane = np.polyval(lane_coeffs, x_sample)
        
        # Interpolate waypoints to same x positions
        y_waypoint = np.interp(x_sample, wp_x, wp_y)
        yaw_waypoint = np.interp(x_sample, wp_x, wp_yaw)
        
        # Weighted fusion
        y_fused = wp_weight * y_waypoint + lane_weight * y_lane
        yaw_fused = wp_weight * yaw_waypoint + lane_weight * yaw_waypoint
        
        # Fit new polynomial to fused path (for MPC)
        fused_coeffs = np.polyfit(x_sample, y_fused, deg=2)
        
        return x_sample, y_fused, yaw_fused, fused_coeffs
        
    elif wp_x is not None:
        # Waypoint-only path
        return wp_x, wp_y, wp_yaw, None
        
    elif lane_coeffs is not None:
        # Lane-only path
        x_sample = np.linspace(0, lookahead_m, n_points)
        y_lane = np.polyval(lane_coeffs, x_sample)
        
        # Calculate heading from polynomial derivative
        dy_dx = np.gradient(y_lane, x_sample)
        yaw_lane = np.arctan2(dy_dx, 1.0)
        
        return x_sample, y_lane, yaw_lane, lane_coeffs
    
    else:
        # No reference available
        return None, None, None, None


def evaluate_reference_at_horizon(
    coeffs: np.ndarray,
    x_positions: np.ndarray
) -> np.ndarray:
    """
    Evaluate reference path at specific positions (optimized for MPC)
    
    Args:
        coeffs: Polynomial coefficients [a, b, c]
        x_positions: X positions to evaluate at
        
    Returns:
        Y positions at specified x positions
    """
    return np.polyval(coeffs, x_positions)


def evaluate_reference_with_heading(
    coeffs: np.ndarray,
    x_positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate reference path with heading information
    
    Args:
        coeffs: Polynomial coefficients [a, b, c]
        x_positions: X positions to evaluate at
        
    Returns:
        (y_positions, yaw_angles)
    """
    y_positions = np.polyval(coeffs, x_positions)
    
    # Calculate heading from derivative
    # dy/dx = 2ax + b
    dy_dx = 2 * coeffs[0] * x_positions + coeffs[1]
    yaw_angles = np.arctan2(dy_dx, 1.0)
    
    return y_positions, yaw_angles


def test_optimized_reference():
    """Test optimized reference path generation"""
    try:
        logger.info("Testing optimized reference path generation")
        
        # Test data
        left_coeffs = np.array([0.001, 0.0, -1.5])  # y = 0.001x² + 0x - 1.5
        right_coeffs = np.array([0.001, 0.0, 1.5])   # y = 0.001x² + 0x + 1.5
        
        # Generate reference path
        x_ref, y_ref, yaw_ref, coeffs = generate_reference_path_optimized(
            waypoints=None,
            left_coeffs=left_coeffs,
            right_coeffs=right_coeffs,
            lookahead_m=30.0,
            n_points=50
        )
        
        logger.info(f"Generated {len(x_ref)} reference points")
        logger.info(f"Centerline coeffs: {coeffs}")
        
        # Test evaluation at MPC horizon
        mpc_horizon = np.arange(1, 11) * 2.0  # 2m steps, 10 points
        y_mpc = evaluate_reference_at_horizon(coeffs, mpc_horizon)
        y_mpc, yaw_mpc = evaluate_reference_with_heading(coeffs, mpc_horizon)
        
        logger.info(f"MPC horizon evaluation: {len(y_mpc)} points")
        logger.info(f"Sample y positions: {y_mpc[:3]}")
        logger.info(f"Sample yaw angles: {np.degrees(yaw_mpc[:3])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Optimized reference test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_optimized_reference()
