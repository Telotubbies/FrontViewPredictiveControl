"""
Reference path: dynamic lookahead + path points in vehicle frame.
"""
from typing import List, Tuple

import numpy as np

from config import (
    REF_PATH_LOOKAHEAD_M,
    REF_PATH_NUM_PTS,
)


def dynamic_lookahead_m(speed_ms: float, curvature: float = 0.0) -> float:
    """
    Compute a speed- and curvature-aware lookahead distance.

    Physics:
        A longer lookahead gives the controller more preview time to react,
        which is beneficial at high speed where vehicle dynamics are faster
        and braking distance grows quadratically with speed.  Conversely, in
        tight curves a shorter lookahead keeps the reference point close to
        the vehicle so the controller does not "cut" the corner.

    Formula:
        base = 20 m
        speed_gain = +0.3 * speed_ms   (capped at +10 m)
        curve_pen  = -15 * |curvature| (capped at -10 m)
        lookahead  = clamp(base + speed_gain + curve_pen, 8, 30) m

    Backward compatibility:
        ``curvature`` defaults to 0.0 (straight road) so existing
        single-argument callers (``dynamic_lookahead_m(speed_ms)``) continue
        to work without modification, now using the speed-scaled formula with
        no curve penalty.

    Args:
        speed_ms:  Vehicle longitudinal speed (m/s).
        curvature: Path curvature at the current location (1/m). Defaults to
            0.0 (straight road).

    Returns:
        Lookahead distance in meters, clamped to [8, 30].
    """
    base = 20.0
    speed_gain = min(0.3 * speed_ms, 10.0)
    curve_pen = max(-15.0 * abs(curvature), -10.0)
    lookahead = base + speed_gain + curve_pen
    return float(np.clip(lookahead, 8.0, 30.0))


def compute_curvature_profile(
    path: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Compute the curvature profile of a path given in (s, lat) coordinates.

    Physics:
        In the vehicle frame the path is described by a lateral offset
        ``lat(s)`` as a function of arc length ``s``.  For small angles the
        curvature ``κ`` is well approximated by the second derivative of the
        lateral displacement with respect to arc length:

            κ(s) ≈ d²lat / ds²

        This is the standard small-angle approximation used in path-tracking
        controllers (e.g., pure pursuit / MPC bicycle models) where the
        heading deviation is small enough that the full curvature formula
        ``κ = |y''| / (1 + y'²)^(3/2)`` reduces to ``|y''|``.

    The second derivative is estimated with a central finite difference on
    the interior points and a one-sided (forward/backward) difference at the
    boundaries so that the output has the same length as the input.

    Args:
        path: List of (s, lat) tuples ordered by increasing arc length ``s``.

    Returns:
        np.ndarray of curvature values (1/m) with the same length as ``path``.
        Returns an array of zeros if the path has fewer than 3 points or if
        the arc-length spacing is degenerate.
    """
    n = len(path)
    if n < 3:
        return np.zeros(n, dtype=np.float64)

    s = np.array([p[0] for p in path], dtype=np.float64)
    lat = np.array([p[1] for p in path], dtype=np.float64)

    # Sort by arc length to ensure monotonic spacing.
    order = np.argsort(s)
    s = s[order]
    lat = lat[order]

    # Guard against zero / duplicate spacing which would divide by zero.
    ds = np.diff(s)
    if np.any(ds <= 0):
        # Fall back to a uniform spacing assumption.
        ds = np.full(n - 1, 1.0)

    # Second derivative via finite differences.
    # Interior:  central difference  d²lat/ds² ≈ (lat[i+1] - 2*lat[i] + lat[i-1]) / (ds[i]*ds[i-1])
    # Boundary:  one-sided difference using the adjacent three points.
    curvature = np.zeros(n, dtype=np.float64)

    # Interior points (central difference).
    for i in range(1, n - 1):
        h_prev = ds[i - 1]
        h_next = ds[i]
        curvature[i] = (
            2.0
            * (
                lat[i - 1] / (h_prev * (h_prev + h_next))
                - lat[i] / (h_prev * h_next)
                + lat[i + 1] / (h_next * (h_prev + h_next))
            )
        )

    # Boundary points: use a one-sided second difference.
    # Forward difference at the start.
    h0, h1 = ds[0], ds[1]
    curvature[0] = (
        2.0
        * (
            lat[0] / (h0 * (h0 + h1))
            - lat[1] / (h0 * h1)
            + lat[2] / (h1 * (h0 + h1))
        )
    )
    # Backward difference at the end.
    hm1, hm2 = ds[n - 2], ds[n - 3]
    curvature[n - 1] = (
        2.0
        * (
            lat[n - 3] / (hm2 * (hm1 + hm2))
            - lat[n - 2] / (hm1 * hm2)
            + lat[n - 1] / (hm1 * (hm1 + hm2))
        )
    )

    return curvature


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
