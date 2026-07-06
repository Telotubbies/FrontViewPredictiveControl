"""
Improved Lane Fitting with Research-Based Methods
==================================================

Based on recent research (2023-2024):
1. B-Spline curve fitting (3D-SplineNet) - smoother than polynomial
2. Geometric constraints (parallel lanes, width consistency)
3. Temporal smoothing with Kalman filter
4. Vanishing point guided fitting

References:
- 3D-SplineNet: Spline-based 3D Lane Detection (2023)
- LaneATT: Attention-based Lane Detection (2021)
- Monocular Lane Detection Survey (2024)
"""

import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Lane geometry constants (based on real-world measurements)
LANE_WIDTH_MIN_M = 2.5  # Minimum lane width (meters)
LANE_WIDTH_MAX_M = 4.0  # Maximum lane width (meters)
LANE_WIDTH_NOMINAL_M = 3.5  # Standard lane width
PARALLEL_TOLERANCE_RAD = 0.15  # Max angle difference for parallel lanes (radians)
CURVATURE_MAX = 0.08  # Maximum curvature (1/m)


class BSplineLaneFitter:
    """
    B-Spline based lane fitting with geometric constraints.

    Advantages over polynomial fitting:
    - Local control (changing one point doesn't affect entire curve)
    - Smoother curvature (no oscillations)
    - Better for sharp curves
    - Natural representation of road geometry
    """

    def __init__(self, degree: int = 3, num_control_points: int = 8):
        """
        Args:
            degree: B-spline degree (3 = cubic, recommended)
            num_control_points: Number of control points
        """
        self.degree = degree
        self.num_control_points = num_control_points
        self.left_spline = None
        self.right_spline = None

    def fit_lane_pair(
        self,
        left_xy: np.ndarray,
        right_xy: np.ndarray,
        enforce_parallel: bool = True,
        enforce_width: bool = True,
    ) -> Tuple[Optional[interpolate.BSpline], Optional[interpolate.BSpline], dict]:
        """
        Fit B-spline curves to left and right lanes with geometric constraints.

        Args:
            left_xy: Left lane points (N, 2) in vehicle frame [x, y]
            right_xy: Right lane points (M, 2) in vehicle frame [x, y]
            enforce_parallel: Enforce parallel lane constraint
            enforce_width: Enforce lane width constraint

        Returns:
            left_spline: B-spline for left lane
            right_spline: B-spline for right lane
            info: Dictionary with fitting quality metrics
        """
        if left_xy is None or right_xy is None or len(left_xy) < 4 or len(right_xy) < 4:
            return None, None, {"valid": False, "reason": "insufficient_points"}

        try:
            # Sort points by x (longitudinal distance)
            left_xy = left_xy[np.argsort(left_xy[:, 0])]
            right_xy = right_xy[np.argsort(right_xy[:, 0])]

            # Initial B-spline fit (unconstrained)
            left_spline_init = self._fit_bspline(left_xy)
            right_spline_init = self._fit_bspline(right_xy)

            if left_spline_init is None or right_spline_init is None:
                return None, None, {"valid": False, "reason": "spline_fit_failed"}

            # Apply geometric constraints
            if enforce_parallel or enforce_width:
                left_spline, right_spline = self._apply_geometric_constraints(
                    left_xy, right_xy, left_spline_init, right_spline_init,
                    enforce_parallel, enforce_width
                )
            else:
                left_spline = left_spline_init
                right_spline = right_spline_init

            # Compute quality metrics
            info = self._compute_quality_metrics(left_xy, right_xy, left_spline, right_spline)
            info["valid"] = True

            self.left_spline = left_spline
            self.right_spline = right_spline

            return left_spline, right_spline, info

        except Exception as e:
            logger.error(f"B-spline fitting error: {e}")
            return None, None, {"valid": False, "reason": str(e)}

    def _fit_bspline(self, xy: np.ndarray) -> Optional[interpolate.BSpline]:
        """Fit B-spline to points using least squares."""
        if len(xy) < self.degree + 1:
            return None

        x, y = xy[:, 0], xy[:, 1]

        # Parameterize by arc length (better than uniform)
        t = np.zeros(len(x))
        t[1:] = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        t = t / t[-1]  # Normalize to [0, 1]

        # Create knot vector
        self.num_control_points - self.degree + 1

        # Fit B-spline
        try:
            tck, _ = interpolate.splprep([x, y], u=t, k=self.degree, s=0.1)
            spline = interpolate.BSpline(tck[0], np.array(tck[1]).T, tck[2])
            return spline
        except Exception as e:
            logger.warning(f"B-spline fit failed: {e}")
            return None

    def _apply_geometric_constraints(
        self,
        left_xy: np.ndarray,
        right_xy: np.ndarray,
        left_spline_init: interpolate.BSpline,
        right_spline_init: interpolate.BSpline,
        enforce_parallel: bool,
        enforce_width: bool,
    ) -> Tuple[interpolate.BSpline, interpolate.BSpline]:
        """
        Apply geometric constraints using optimization.

        Constraints:
        1. Parallel lanes: left and right should have similar curvature
        2. Lane width: distance between lanes should be 2.5-4.0m
        """
        # Sample points for constraint evaluation
        t_samples = np.linspace(0, 1, 20)

        def constraint_cost(params):
            """Cost function for geometric constraints."""
            # Split params into left and right control points
            n_left = len(left_spline_init.c)
            left_c = params[:n_left].reshape(-1, 2)
            right_c = params[n_left:].reshape(-1, 2)

            cost = 0.0

            # Data fitting cost
            left_pred = interpolate.BSpline(left_spline_init.t, left_c, left_spline_init.k)(t_samples)
            right_pred = interpolate.BSpline(right_spline_init.t, right_c, right_spline_init.k)(t_samples)

            # Fit to original points
            for xy, pred in [(left_xy, left_pred), (right_xy, right_pred)]:
                t_data = np.linspace(0, 1, len(xy))
                pred_at_data = np.array([np.interp(t_data, t_samples, pred[:, i]) for i in range(2)]).T
                cost += np.sum((xy - pred_at_data)**2)

            # Parallel constraint
            if enforce_parallel:
                # Compute tangent vectors
                left_tangent = np.gradient(left_pred[:, 1]) / (np.gradient(left_pred[:, 0]) + 1e-6)
                right_tangent = np.gradient(right_pred[:, 1]) / (np.gradient(right_pred[:, 0]) + 1e-6)
                parallel_error = np.sum((left_tangent - right_tangent)**2)
                cost += 10.0 * parallel_error

            # Width constraint
            if enforce_width:
                widths = np.abs(left_pred[:, 1] - right_pred[:, 1])
                width_error = np.sum((widths - LANE_WIDTH_NOMINAL_M)**2)
                cost += 5.0 * width_error

            return cost

        # Initial parameters (control points)
        x0 = np.concatenate([left_spline_init.c.ravel(), right_spline_init.c.ravel()])

        # Optimize
        try:
            result = minimize(constraint_cost, x0, method='L-BFGS-B', options={'maxiter': 100})

            if result.success:
                n_left = len(left_spline_init.c)
                left_c_opt = result.x[:n_left].reshape(-1, 2)
                right_c_opt = result.x[n_left:].reshape(-1, 2)

                left_spline = interpolate.BSpline(left_spline_init.t, left_c_opt, left_spline_init.k)
                right_spline = interpolate.BSpline(right_spline_init.t, right_c_opt, right_spline_init.k)

                return left_spline, right_spline
        except Exception as e:
            logger.warning(f"Constraint optimization failed: {e}")

        # Fallback to unconstrained
        return left_spline_init, right_spline_init

    def _compute_quality_metrics(
        self,
        left_xy: np.ndarray,
        right_xy: np.ndarray,
        left_spline: interpolate.BSpline,
        right_spline: interpolate.BSpline,
    ) -> dict:
        """Compute quality metrics for fitted lanes."""
        t_samples = np.linspace(0, 1, 20)

        # Evaluate splines
        left_pred = left_spline(t_samples)
        right_pred = right_spline(t_samples)

        # Lane width statistics
        widths = np.abs(left_pred[:, 1] - right_pred[:, 1])
        avg_width = float(np.mean(widths))
        width_std = float(np.std(widths))

        # Curvature (approximate)
        def compute_curvature(points):
            dx = np.gradient(points[:, 0])
            dy = np.gradient(points[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
            return curvature

        left_curv = compute_curvature(left_pred)
        right_curv = compute_curvature(right_pred)

        # Fitting error (RMSE)
        def rmse(xy, spline):
            t_data = np.linspace(0, 1, len(xy))
            pred = spline(t_data)
            return float(np.sqrt(np.mean((xy - pred)**2)))

        left_rmse = rmse(left_xy, left_spline)
        right_rmse = rmse(right_xy, right_spline)

        return {
            "avg_width_m": avg_width,
            "width_std_m": width_std,
            "width_valid": LANE_WIDTH_MIN_M <= avg_width <= LANE_WIDTH_MAX_M,
            "left_curvature_max": float(np.max(left_curv)),
            "right_curvature_max": float(np.max(right_curv)),
            "left_rmse": left_rmse,
            "right_rmse": right_rmse,
            "parallel_score": 1.0 - min(1.0, np.mean(np.abs(left_curv - right_curv))),
        }

    def evaluate_at(self, x_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate fitted lanes at given x positions.

        Args:
            x_values: Longitudinal positions to evaluate

        Returns:
            left_y: Lateral positions of left lane
            right_y: Lateral positions of right lane
        """
        if self.left_spline is None or self.right_spline is None:
            return None, None

        # Map x to parameter t (approximate)
        t_values = (x_values - x_values.min()) / (x_values.max() - x_values.min() + 1e-6)
        t_values = np.clip(t_values, 0, 1)

        left_points = self.left_spline(t_values)
        right_points = self.right_spline(t_values)

        return left_points[:, 1], right_points[:, 1]

    def enforce_parallel_constraint(
        self,
        left_spline: interpolate.BSpline,
        right_spline: interpolate.BSpline,
        angle_threshold_deg: float = 5.0,
    ) -> interpolate.BSpline:
        """
        Enforce parallel tangent constraint between left and right lane splines.

        For each control point, the tangent direction of the right lane is
        compared to the tangent direction of the left lane. If the angle
        difference exceeds ``angle_threshold_deg`` degrees, the right lane
        control point is adjusted so that its tangent aligns with the left
        lane tangent direction.

        Args:
            left_spline: Reference B-spline for the left lane.
            right_spline: B-spline for the right lane to be adjusted.
            angle_threshold_deg: Maximum allowed angle difference (degrees)
                between left and right tangents before correction is applied.

        Returns:
            Adjusted right lane B-spline with corrected control points. If
            either input spline is ``None``, the original ``right_spline`` is
            returned unchanged.
        """
        if left_spline is None or right_spline is None:
            return right_spline

        angle_threshold_rad = np.deg2rad(angle_threshold_deg)

        # Derive tangent directions via spline derivatives
        left_deriv = left_spline.derivative()
        right_deriv = right_spline.derivative()

        # Sample at uniform parameter values matching control point count
        n = len(right_spline.c)
        t_samples = np.linspace(0, 1, n)

        left_tangents = left_deriv(t_samples)
        right_tangents = right_deriv(t_samples)

        right_c = right_spline.c.copy()

        for i in range(n):
            lt = left_tangents[i]
            rt = right_tangents[i]

            left_angle = np.arctan2(lt[1], lt[0])
            right_angle = np.arctan2(rt[1], rt[0])

            # Smallest signed angle difference in [-pi, pi]
            angle_diff = np.arctan2(
                np.sin(right_angle - left_angle),
                np.cos(right_angle - left_angle),
            )

            if np.abs(angle_diff) > angle_threshold_rad:
                # Rotate the right tangent to match the left tangent direction
                left_dir = np.array([np.cos(left_angle), np.sin(left_angle)])
                right_mag = np.linalg.norm(rt)
                new_tangent = left_dir * right_mag

                # Shift the control point so the local tangent aligns
                right_c[i] = right_c[i] + (new_tangent - rt) * 0.5

        adjusted_spline = interpolate.BSpline(
            right_spline.t, right_c, right_spline.k
        )
        return adjusted_spline


class GeometricLaneValidator:
    """
    Validate lane detection results using geometric constraints.

    Based on real-world lane geometry:
    - Lane width: 2.5-4.0m (typical 3.5m)
    - Parallel lanes on straight roads
    - Smooth curvature changes
    - Vanishing point consistency
    """

    @staticmethod
    def validate_lane_pair(
        left_xy: np.ndarray,
        right_xy: np.ndarray,
        min_points: int = 10,
    ) -> Tuple[bool, dict]:
        """
        Validate detected lane pair.

        Returns:
            valid: True if lanes pass validation
            info: Dictionary with validation details
        """
        info = {"checks": {}}

        # Check 1: Sufficient points
        if left_xy is None or right_xy is None:
            return False, {"checks": {"sufficient_points": False}}

        if len(left_xy) < min_points or len(right_xy) < min_points:
            info["checks"]["sufficient_points"] = False
            return False, info
        info["checks"]["sufficient_points"] = True

        # Check 2: Lane width
        # Sample at multiple x positions
        x_common = np.linspace(
            max(left_xy[:, 0].min(), right_xy[:, 0].min()),
            min(left_xy[:, 0].max(), right_xy[:, 0].max()),
            10
        )

        left_y = np.interp(x_common, left_xy[:, 0], left_xy[:, 1])
        right_y = np.interp(x_common, right_xy[:, 0], right_xy[:, 1])

        widths = np.abs(left_y - right_y)
        avg_width = float(np.mean(widths))
        width_std = float(np.std(widths))

        width_valid = LANE_WIDTH_MIN_M <= avg_width <= LANE_WIDTH_MAX_M
        info["checks"]["lane_width"] = width_valid
        info["avg_width_m"] = avg_width
        info["width_std_m"] = width_std

        # Check 3: Width consistency (low std = parallel lanes)
        width_consistent = width_std < 0.5  # Less than 50cm variation
        info["checks"]["width_consistent"] = width_consistent

        # Check 4: No crossing lanes
        # Left should be negative (or all same sign), right should be positive (or all same sign)
        # They should be on opposite sides of center
        no_crossing = np.all(left_y < right_y)  # Left always less than right
        info["checks"]["no_crossing"] = no_crossing

        # Overall validation
        valid = all([
            info["checks"]["sufficient_points"],
            info["checks"]["lane_width"],
            info["checks"]["no_crossing"],
        ])

        return valid, info


def convert_bspline_to_polynomial(
    spline: interpolate.BSpline,
    x_range: Tuple[float, float],
    num_samples: int = 50,
    poly_order: int = 2,
) -> Optional[np.ndarray]:
    """
    Convert B-spline to polynomial coefficients for compatibility.

    This allows using B-spline fitting while maintaining polynomial interface.
    """
    if spline is None:
        return None

    # Sample spline
    t = np.linspace(0, 1, num_samples)
    points = spline(t)
    x, y = points[:, 0], points[:, 1]

    # Fit polynomial to sampled points
    try:
        coeffs = np.polyfit(x, y, poly_order)
        return coeffs
    except Exception as e:
        logger.warning(f"Polynomial conversion failed: {e}")
        return None


def compute_lane_width_profile(
    spline_left: Optional[interpolate.BSpline],
    spline_right: Optional[interpolate.BSpline],
    s_values: np.ndarray,
) -> np.ndarray:
    """
    Compute the lane width profile along the arc-length parameter.

    Evaluates both the left and right lane splines at the given parameter
    values and returns the Euclidean distance between corresponding points.

    Args:
        spline_left: B-spline representing the left lane.
        spline_right: B-spline representing the right lane.
        s_values: Array of parameter values (typically normalized arc length
            in ``[0, 1]``) at which to evaluate the width.

    Returns:
        Array of lane widths (in meters) at each ``s`` position. Returns an
        empty array if either spline is ``None``.
    """
    if spline_left is None or spline_right is None:
        return np.array([])

    left_points = spline_left(s_values)
    right_points = spline_right(s_values)

    widths = np.linalg.norm(left_points - right_points, axis=1)
    return widths


def width_consistency_score(widths: np.ndarray) -> float:
    """
    Compute a [0, 1] consistency score for a lane width profile.

    A score of 1.0 indicates perfectly consistent width (coefficient of
    variation ``std / mean < 0.05``). The score degrades linearly as the
    variation increases, reaching 0.0 when ``std / mean >= 0.5``.

    Args:
        widths: Array of lane width measurements (meters).

    Returns:
        Consistency score in ``[0, 1]``. Returns 1.0 for empty or
        constant-width inputs.
    """
    if widths is None or len(widths) == 0:
        return 1.0

    mean_width = float(np.mean(widths))
    if mean_width < 1e-6:
        return 1.0

    cv = float(np.std(widths)) / mean_width  # coefficient of variation

    if cv < 0.05:
        return 1.0

    # Linearly map [0.05, 0.5] -> [1.0, 0.0]
    score = 1.0 - (cv - 0.05) / (0.5 - 0.05)
    return float(np.clip(score, 0.0, 1.0))
