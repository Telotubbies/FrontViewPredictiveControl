"""
Reference path: dynamic lookahead + path points in vehicle frame.
"""
from typing import List, Tuple

import numpy as np

from config import (
    REF_PATH_LOOKAHEAD_M,
    REF_PATH_NUM_PTS,
    LOOKAHEAD_MIN_M,
    LOOKAHEAD_TIME_S,
)


def dynamic_lookahead_m(speed_mps: float) -> float:
    """lookahead_distance = max(MIN_LA, speed_mps × LOOKAHEAD_TIME)."""
    return max(LOOKAHEAD_MIN_M, speed_mps * LOOKAHEAD_TIME_S)


def smooth_path_lat(
    path: List[Tuple[float, float]],
    alpha: float = 0.6,
) -> List[Tuple[float, float]]:
    """
    EMA smoothing on lateral (lat) along path to reduce jitter for overlay/MPC.
    prev is updated along s; both perception and WP path in vehicle frame (s, lat).
    """
    if not path or len(path) < 2:
        return path
    smoothed: List[Tuple[float, float]] = []
    prev_lat = path[0][1]
    for s, lat in path:
        lat_s = alpha * prev_lat + (1.0 - alpha) * lat
        smoothed.append((s, lat_s))
        prev_lat = lat_s
    return smoothed


def resample_path(
    path: List[Tuple[float, float]],
    lookahead_m: float,
    num_pts: int,
) -> List[Tuple[float, float]]:
    """
    Resample waypoint path (s, lat) to uniform s in [0, lookahead_m] with num_pts+1 points.
    Path from waypoints; use in turn when |curv| high so reference follows map.
    """
    if not path or len(path) < 2:
        return []
    s_vals = np.array([p[0] for p in path], dtype=np.float64)
    lat_vals = np.array([p[1] for p in path], dtype=np.float64)
    # keep only s in [0, lookahead_m]
    valid = (s_vals >= 0) & (s_vals <= lookahead_m)
    if not np.any(valid):
        return []
    s_vals = s_vals[valid]
    lat_vals = lat_vals[valid]
    # sort by s
    order = np.argsort(s_vals)
    s_vals = s_vals[order]
    lat_vals = lat_vals[order]
    # unique s for interpolation
    s_unq, idx = np.unique(s_vals, return_index=True)
    lat_unq = lat_vals[idx]
    if len(s_unq) < 2:
        return [(float(s_unq[0]), float(lat_unq[0]))]
    s_tgt = np.linspace(0, min(lookahead_m, float(s_unq[-1])), num_pts + 1)
    lat_tgt = np.interp(s_tgt, s_unq, lat_unq)
    return [(float(s_tgt[i]), float(lat_tgt[i])) for i in range(len(s_tgt))]


def get_quintic_coefficients(
    y0: float, dy0: float, ddy0: float,
    yT: float, dyT: float, ddyT: float,
    T: float
) -> np.ndarray:
    """
    Solve quintic polynomial coefficients for boundary conditions.
    
    Quintic: y(s) = a5*s^5 + a4*s^4 + a3*s^3 + a2*s^2 + a1*s + a0
    
    Boundary conditions:
    - y(0) = y0, y'(0) = dy0, y''(0) = ddy0
    - y(T) = yT, y'(T) = dyT, y''(T) = ddyT
    
    Returns: [a5, a4, a3, a2, a1, a0]
    """
    # Build matrix equation: A * coeffs = b
    A = np.array([
        [0, 0, 0, 0, 0, 1],           # y(0) = a0
        [0, 0, 0, 0, 1, 0],           # y'(0) = a1
        [0, 0, 0, 2, 0, 0],           # y''(0) = 2*a2
        [T**5, T**4, T**3, T**2, T, 1],  # y(T)
        [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],  # y'(T)
        [20*T**3, 12*T**2, 6*T, 2, 0, 0],     # y''(T)
    ])
    
    b = np.array([y0, dy0, ddy0, yT, dyT, ddyT])
    
    coeffs = np.linalg.solve(A, b)
    return coeffs


def get_reference_path_quintic(
    cte_m: float,
    head_rad: float,
    curv: float,
    lookahead_m: float = REF_PATH_LOOKAHEAD_M,
    num_pts: int = REF_PATH_NUM_PTS,
) -> List[Tuple[float, float]]:
    """
    Generate reference path using quintic polynomial for smooth trajectories.
    
    Quintic polynomial (5th order) provides:
    - Continuous position, velocity, acceleration
    - Smoother curvature
    - Better passenger comfort
    
    Boundary conditions:
    - Initial: y(0) = cte_m, y'(0) = tan(head_rad) ≈ head_rad, y''(0) = curv
    - Final: y(T) = 0 (lane center), y'(T) = 0 (aligned), y''(T) = 0 (straight)
    """
    # Initial conditions
    y0 = cte_m
    dy0 = np.tan(head_rad) if abs(head_rad) < 0.5 else head_rad  # Small angle approximation
    ddy0 = curv
    
    # Final conditions (converge to lane center)
    yT = 0.0
    dyT = 0.0
    ddyT = 0.0
    
    T = lookahead_m
    
    # Solve quintic polynomial
    coeffs = get_quintic_coefficients(y0, dy0, ddy0, yT, dyT, ddyT, T)
    
    # Generate path points
    path = []
    for i in range(num_pts + 1):
        s = lookahead_m * i / num_pts
        # Evaluate quintic: y = a5*s^5 + a4*s^4 + a3*s^3 + a2*s^2 + a1*s + a0
        y = np.polyval(coeffs, s)
        path.append((s, float(y)))
    
    return path


def get_reference_path(
    cte_m: float,
    head_rad: float,
    curv: float,
    lookahead_m: float = REF_PATH_LOOKAHEAD_M,
    num_pts: int = REF_PATH_NUM_PTS,
    use_quintic: bool = True,
) -> List[Tuple[float, float]]:
    """
    Generate reference path in vehicle frame from current state.
    
    Args:
        cte_m: Cross-track error (m)
        head_rad: Heading error (rad)
        curv: Curvature (1/m)
        lookahead_m: Lookahead distance (m)
        num_pts: Number of points
        use_quintic: Use quintic polynomial (smoother) vs parabolic (faster)
    
    Returns:
        List of (s_m, lateral_m) path points
    """
    if use_quintic:
        return get_reference_path_quintic(cte_m, head_rad, curv, lookahead_m, num_pts)
    else:
        # Original parabolic path (2nd order)
        return [
            (
                lookahead_m * i / num_pts,
                cte_m
                + head_rad * (lookahead_m * i / num_pts)
                + 0.5 * curv * (lookahead_m * i / num_pts) ** 2,
            )
            for i in range(num_pts + 1)
        ]
