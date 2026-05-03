"""
LaneValidator: reject degenerate polynomial geometry before feeding MPC.

Failure modes addressed:
- Poly collapse: centerline becomes vertical strip (low lateral variance).
- Crossing lines: left edge crosses right edge (y_left > y_right at some x).
- Invalid width: lane width < 0.2m or > 5m, or inconsistent across x.
- Curvature explosion: sudden spike vs previous frame.
- Temporal jump: delta coeffs too large → reject and use Kalman predict.
- Consecutive invalid: 3+ frames invalid → waypoint-only fail-safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from config import MIN_LANE_WIDTH_M, MAX_LANE_WIDTH_M
except ImportError:
    MIN_LANE_WIDTH_M = 2.0
    MAX_LANE_WIDTH_M = 5.5
# Centerline lateral variance over lookahead: if too low = vertical strip collapse.
MIN_CENTERLINE_LATERAL_VAR = 0.02
# Max width std across x (consistency).
MAX_WIDTH_STD_M = 1.2
# Curvature: absolute limit and max change vs previous.
MAX_CURVATURE = 0.08
MAX_CURVATURE_DELTA = 0.04
# Center position jump vs previous (m).
MAX_CENTER_JUMP_M = 0.8
# Max change in poly coeffs (c2, c1, c0) between frames.
MAX_DELTA_C2 = 0.02
MAX_DELTA_C1 = 0.15
MAX_DELTA_C0 = 0.5
# Consecutive invalid frames before forcing waypoint-only.
CONSECUTIVE_INVALID_THRESHOLD = 3
# Lookahead range for sampling (m).
LOOKAHEAD_SAMPLE_M = 15.0
N_SAMPLE_X = 9


@dataclass
class LaneValidationResult:
    """Result of lane geometry validation."""
    valid: bool
    reason: str  # "" if valid, else failure reason
    waypoint_only_failsafe: bool = False  # True if 3+ consecutive invalid


def _check_left_right_cross(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    x_sample: np.ndarray,
) -> Tuple[bool, str]:
    """
    Reject if left and right edges cross: at any x we must have y_left < y_right
    (vehicle frame: y left negative, y right positive). Crossing = invalid poly.
    """
    if left_xy is None or right_xy is None or len(left_xy) < 2 or len(right_xy) < 2:
        return True, ""
    x_l, y_l = left_xy[:, 0].copy(), left_xy[:, 1].copy()
    x_r, y_r = right_xy[:, 0].copy(), right_xy[:, 1].copy()
    idx_l = np.argsort(x_l)
    x_l, y_l = x_l[idx_l], y_l[idx_l]
    idx_r = np.argsort(x_r)
    x_r, y_r = x_r[idx_r], y_r[idx_r]
    x_min = max(x_l.min(), x_r.min())
    x_max = min(x_l.max(), x_r.max())
    if x_max - x_min < 0.5:
        return False, "left_right_x_range_tiny"
    for x in x_sample:
        if x < x_min or x > x_max:
            continue
        y_left = np.interp(x, x_l, y_l)
        y_right = np.interp(x, x_r, y_r)
        # Left edge must be left of right edge (y_left <= y_right in vehicle frame).
        if y_left > y_right + 0.05:
            return False, "left_right_cross"
    return True, ""


def _check_lane_width(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    x_sample: np.ndarray,
    min_m: float = MIN_LANE_WIDTH_M,
    max_m: float = MAX_LANE_WIDTH_M,
) -> Tuple[bool, str]:
    """Width at multiple x must be in [min_m, max_m]. Reject if inconsistent (high std)."""
    if left_xy is None or right_xy is None or len(left_xy) < 2 or len(right_xy) < 2:
        return True, ""
    x_l, y_l = left_xy[:, 0].copy(), left_xy[:, 1].copy()
    x_r, y_r = right_xy[:, 0].copy(), right_xy[:, 1].copy()
    idx_l = np.argsort(x_l)
    x_l, y_l = x_l[idx_l], y_l[idx_l]
    idx_r = np.argsort(x_r)
    x_r, y_r = x_r[idx_r], y_r[idx_r]
    x_min = max(x_l.min(), x_r.min())
    x_max = min(x_l.max(), x_r.max())
    if x_max - x_min < 0.5:
        return False, "width_x_range_small"
    widths = []
    for x in x_sample:
        if x < x_min or x > x_max:
            continue
        w = np.interp(x, x_r, y_r) - np.interp(x, x_l, y_l)
        widths.append(w)
    if not widths:
        return False, "width_no_sample"
    widths = np.array(widths)
    if np.any(widths < min_m) or np.any(widths > max_m):
        return False, f"width_range_{widths.min():.2f}_{widths.max():.2f}m"
    if np.std(widths) > MAX_WIDTH_STD_M:
        return False, f"width_inconsistent_std_{np.std(widths):.2f}"
    return True, ""


def _check_centerline_variance(center_coeffs: np.ndarray) -> Tuple[bool, str]:
    """
    Vertical strip collapse: centerline has almost no lateral spread.
    Sample y = poly(x) over x in [0, LOOKAHEAD]; variance must be >= MIN_CENTERLINE_LATERAL_VAR.
    """
    if center_coeffs is None or len(center_coeffs) < 3:
        return True, ""
    x_sample = np.linspace(0, LOOKAHEAD_SAMPLE_M, N_SAMPLE_X)
    y_center = np.polyval(center_coeffs, x_sample)
    var = float(np.var(y_center))
    if var < MIN_CENTERLINE_LATERAL_VAR:
        return False, f"centerline_collapse_var_{var:.4f}"
    return True, ""


def _check_curvature_explosion(
    center_coeffs: np.ndarray,
    prev_center_coeffs: Optional[np.ndarray],
) -> Tuple[bool, str]:
    """Curvature at x=0 must be within limit and not jump suddenly vs previous."""
    if center_coeffs is None or len(center_coeffs) < 3:
        return True, ""
    c2, c1 = center_coeffs[0], center_coeffs[1]
    curv = 2.0 * c2 / ((1.0 + c1 * c1) ** 1.5 + 1e-9)
    if abs(curv) > MAX_CURVATURE:
        return False, f"curvature_explosion_{curv:.4f}"
    if prev_center_coeffs is not None and len(prev_center_coeffs) >= 3:
        p2, p1 = prev_center_coeffs[0], prev_center_coeffs[1]
        prev_curv = 2.0 * p2 / ((1.0 + p1 * p1) ** 1.5 + 1e-9)
        if abs(curv - prev_curv) > MAX_CURVATURE_DELTA:
            return False, f"curvature_jump_{abs(curv - prev_curv):.4f}"
    return True, ""


def _check_slope_sign_consistency(
    left_coeffs: np.ndarray,
    right_coeffs: np.ndarray,
    x_sample: np.ndarray,
) -> Tuple[bool, str]:
    """
    Left and right should not have opposite slope direction in a way that indicates
    crossing or degenerate fit. At each x, dy_left/dx and dy_right/dx: for a normal
    lane both sides curve same way (same sign of second derivative).
    """
    if left_coeffs is None or right_coeffs is None or len(left_coeffs) < 3 or len(right_coeffs) < 3:
        return True, ""
    # Second derivative: 2*c2. Same sign = both curve same way.
    d2_left = 2.0 * left_coeffs[0]
    d2_right = 2.0 * right_coeffs[0]
    if d2_left * d2_right < -0.01:
        return False, "slope_sign_opposite"
    return True, ""


def _check_temporal_coeffs(
    prev_center_coeffs: Optional[np.ndarray],
    new_center_coeffs: np.ndarray,
) -> Tuple[bool, str]:
    """Reject if delta in [c2, c1, c0] too large → use Kalman predict instead."""
    if prev_center_coeffs is None or len(prev_center_coeffs) != len(new_center_coeffs):
        return True, ""
    delta = np.abs(np.array(new_center_coeffs) - np.array(prev_center_coeffs))
    if delta[0] > MAX_DELTA_C2:
        return False, f"delta_c2_{delta[0]:.4f}"
    if delta[1] > MAX_DELTA_C1:
        return False, f"delta_c1_{delta[1]:.4f}"
    if delta[2] > MAX_DELTA_C0:
        return False, f"delta_c0_{delta[2]:.4f}"
    return True, ""


def _check_center_jump(
    prev_center_coeffs: Optional[np.ndarray],
    new_center_coeffs: np.ndarray,
) -> Tuple[bool, str]:
    """Lateral position at x=0 must not jump more than MAX_CENTER_JUMP_M."""
    if prev_center_coeffs is None:
        return True, ""
    y_prev = float(np.polyval(prev_center_coeffs, 0.0))
    y_new = float(np.polyval(new_center_coeffs, 0.0))
    if abs(y_new - y_prev) > MAX_CENTER_JUMP_M:
        return False, f"center_jump_{abs(y_new - y_prev):.2f}m"
    return True, ""


class LaneValidator:
    """
    Modular lane geometry validator. Run after poly fit, before accepting frame.

    Clear separation:
    - Detection: done elsewhere (UNet → BEV → sliding window → poly fit).
    - Validation: this class (degenerate, cross, width, variance, temporal).
    - Tracking: Kalman in pipeline (predict when validation fails).

    Never accept collapsed vertical lane or crossing lines.
    """

    def __init__(
        self,
        consecutive_invalid_threshold: int = CONSECUTIVE_INVALID_THRESHOLD,
        min_lane_width_m: float = MIN_LANE_WIDTH_M,
        max_lane_width_m: float = MAX_LANE_WIDTH_M,
    ):
        self.consecutive_invalid_threshold = consecutive_invalid_threshold
        self.min_lane_width_m = min_lane_width_m
        self.max_lane_width_m = max_lane_width_m
        self._consecutive_invalid = 0
        self._prev_center_coeffs: Optional[np.ndarray] = None

    def validate(
        self,
        left_xy: Optional[np.ndarray],
        right_xy: Optional[np.ndarray],
        left_coeffs: Optional[np.ndarray],
        right_coeffs: Optional[np.ndarray],
        center_coeffs: Optional[np.ndarray],
    ) -> LaneValidationResult:
        """
        Run all geometry sanity checks. If any fails → valid=False, reason set.
        If 3+ consecutive invalid → waypoint_only_failsafe=True.
        """
        x_sample = np.linspace(0, LOOKAHEAD_SAMPLE_M, N_SAMPLE_X)
        reason = ""

        # 1. Need at least center coeffs (or we have no poly)
        if center_coeffs is None or len(center_coeffs) < 3:
            self._consecutive_invalid += 1
            waypoint_only = self._consecutive_invalid >= self.consecutive_invalid_threshold
            return LaneValidationResult(
                valid=False,
                reason="no_center_coeffs",
                waypoint_only_failsafe=waypoint_only,
            )

        # 2. Left/right cross
        if left_xy is not None and right_xy is not None:
            ok, r = _check_left_right_cross(left_xy, right_xy, x_sample)
            if not ok:
                self._consecutive_invalid += 1
                return LaneValidationResult(
                    valid=False,
                    reason=r,
                    waypoint_only_failsafe=(self._consecutive_invalid >= self.consecutive_invalid_threshold),
                )

        # 3. Lane width range and consistency
        if left_xy is not None and right_xy is not None:
            ok, r = _check_lane_width(
                left_xy, right_xy, x_sample,
                min_m=self.min_lane_width_m,
                max_m=self.max_lane_width_m,
            )
            if not ok:
                self._consecutive_invalid += 1
                return LaneValidationResult(
                    valid=False,
                    reason=r,
                    waypoint_only_failsafe=(self._consecutive_invalid >= self.consecutive_invalid_threshold),
                )

        # 4. Centerline variance (vertical strip collapse)
        ok, r = _check_centerline_variance(center_coeffs)
        if not ok:
            self._consecutive_invalid += 1
            return LaneValidationResult(
                valid=False,
                reason=r,
                waypoint_only_failsafe=(self._consecutive_invalid >= self.consecutive_invalid_threshold),
            )

        # 5. Curvature explosion / jump
        ok, r = _check_curvature_explosion(center_coeffs, self._prev_center_coeffs)
        if not ok:
            self._consecutive_invalid += 1
            return LaneValidationResult(
                valid=False,
                reason=r,
                waypoint_only_failsafe=(self._consecutive_invalid >= self.consecutive_invalid_threshold),
            )

        # 6. Slope sign (left/right curve same way)
        if left_coeffs is not None and right_coeffs is not None:
            ok, r = _check_slope_sign_consistency(left_coeffs, right_coeffs, x_sample)
            if not ok:
                self._consecutive_invalid += 1
                return LaneValidationResult(
                    valid=False,
                    reason=r,
                    waypoint_only_failsafe=(self._consecutive_invalid >= self.consecutive_invalid_threshold),
                )

        # 7. Temporal coeff delta
        ok, r = _check_temporal_coeffs(self._prev_center_coeffs, center_coeffs)
        if not ok:
            self._consecutive_invalid += 1
            return LaneValidationResult(
                valid=False,
                reason=r,
                waypoint_only_failsafe=(self._consecutive_invalid >= self.consecutive_invalid_threshold),
            )

        # 8. Center jump
        ok, r = _check_center_jump(self._prev_center_coeffs, center_coeffs)
        if not ok:
            self._consecutive_invalid += 1
            return LaneValidationResult(
                valid=False,
                reason=r,
                waypoint_only_failsafe=(self._consecutive_invalid >= self.consecutive_invalid_threshold),
            )

        # Passed all checks
        self._prev_center_coeffs = np.array(center_coeffs, dtype=np.float64)
        self._consecutive_invalid = 0
        return LaneValidationResult(valid=True, reason="", waypoint_only_failsafe=False)

    def reset(self) -> None:
        """Clear temporal state (e.g. after recovery or map change)."""
        self._consecutive_invalid = 0
        self._prev_center_coeffs = None
