"""
Geometric validation for lane perception: reject non-road planar regions and enforce
ground-plane consistency. Robust against wall/vertical surface false detections.

Failure cases addressed:
- Walls/barriers: vertical continuity near image sides → reject
- Sky/ceiling: mask in upper region without road trapezoid → reject
- Invalid lane geometry: width out of range, center jump, curvature limit → reject
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Road trapezoid (ground-plane constraint): [left_bottom, right_bottom, right_top, left_top]
# Only accept pixels inside this ROI. Top narrower = perspective.
TRAPEZOID_TOP_RATIO = 0.40   # top edge width = 40% of bottom
TRAPEZOID_BOTTOM_MARGIN = 0.05  # leave 5% margin at bottom
TRAPEZOID_TOP_Y_RATIO = 0.35   # top edge at 35% from top of image

# Vertical continuity: walls have long vertical runs. Reject if mask has strong
# vertical structure in side regions (left/right 25% of image).
SIDE_MARGIN_RATIO = 0.25
VERTICAL_RUN_MIN_PX = 45   # min consecutive vertical pixels to count as "wall-like"
VERTICAL_DENSITY_THRESH = 0.35  # if > 35% of side columns have long vertical runs → reject

# Sanity limits (vehicle frame, meters)
MIN_LANE_WIDTH_M = 2.0
MAX_LANE_WIDTH_M = 5.5
MAX_CENTER_JUMP_M = 0.8
MAX_CURVATURE_PHYSICAL = 0.08


@dataclass
class GeometricValidationResult:
    """Result of geometric validation: mask (possibly filtered), pass/fail, reason."""
    mask: np.ndarray
    passed: bool
    reason: str  # "" if passed, else failure reason
    vertical_rejected: bool = False
    trapezoid_applied: bool = True


def get_road_trapezoid_mask(cam_w: int, cam_h: int) -> np.ndarray:
    """
    Binary mask of expected road region (trapezoid in image).
    Bottom = full width (with small margin); top = narrower (perspective).
    Enforces ground-plane consistency: only road-like region accepted.
    """
    bottom_y = int(cam_h * (1.0 - TRAPEZOID_BOTTOM_MARGIN))
    top_y = int(cam_h * TRAPEZOID_TOP_Y_RATIO)
    w_half = cam_w / 2.0
    top_half = w_half * TRAPEZOID_TOP_RATIO
    pts = np.array([
        [w_half - top_half, top_y],
        [w_half + top_half, top_y],
        [cam_w - 2, bottom_y],
        [2, bottom_y],
    ], dtype=np.int32)
    out = np.zeros((cam_h, cam_w), dtype=np.uint8)
    cv2.fillPoly(out, [pts], 255)
    return out


def _vertical_run_lengths(mask_bin: np.ndarray, col_start: int, col_end: int) -> np.ndarray:
    """For columns in [col_start, col_end), return max vertical run length per column."""
    h, w = mask_bin.shape
    runs = np.zeros(col_end - col_start, dtype=np.int32)
    for c in range(col_start, min(col_end, w)):
        col = mask_bin[:, c]
        max_run = 0
        curr = 0
        for r in range(h):
            if col[r] > 0:
                curr += 1
                max_run = max(max_run, curr)
            else:
                curr = 0
        runs[c - col_start] = max_run
    return runs


def reject_vertical_planes(mask: np.ndarray, cam_w: int, cam_h: int) -> Tuple[np.ndarray, bool]:
    """
    Reject masks that look like vertical surfaces (walls) near image sides.
    Walls produce long vertical runs in left/right regions; road is more horizontal.
    Returns (mask_with_sides_cleared, rejected).
    If rejected=True, the mask had high vertical continuity in side regions → cleared to zeros.
    """
    mask_bin = (mask > 0).astype(np.uint8)
    margin = int(cam_w * SIDE_MARGIN_RATIO)
    # Left side
    left_runs = _vertical_run_lengths(mask_bin, 0, margin)
    # Right side
    right_runs = _vertical_run_lengths(mask_bin, cam_w - margin, cam_w)
    long_run_left = np.sum(left_runs >= VERTICAL_RUN_MIN_PX) / max(len(left_runs), 1)
    long_run_right = np.sum(right_runs >= VERTICAL_RUN_MIN_PX) / max(len(right_runs), 1)
    if long_run_left >= VERTICAL_DENSITY_THRESH or long_run_right >= VERTICAL_DENSITY_THRESH:
        # Wall-like: clear side regions to remove vertical structure, or reject entire mask
        out = mask_bin.copy()
        out[:, :margin] = 0
        out[:, cam_w - margin:] = 0
        return out, True  # rejected as dominant vertical (we kept center only)
    return mask_bin, False


def apply_road_trapezoid(mask: np.ndarray, cam_w: int, cam_h: int) -> np.ndarray:
    """Only keep pixels inside the expected road trapezoid (ground-plane constraint)."""
    trapezoid = get_road_trapezoid_mask(cam_w, cam_h)
    if mask.shape[:2] != (cam_h, cam_w):
        mask = cv2.resize(mask, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask > 0).astype(np.uint8)
    out = cv2.bitwise_and(mask_bin, (trapezoid > 0).astype(np.uint8))
    return out


def validate_geometry(
    mask: np.ndarray,
    cam_w: int,
    cam_h: int,
    reject_vertical: bool = True,
    apply_trapezoid: bool = True,
) -> GeometricValidationResult:
    """
    Full geometric validation: vertical rejection + road trapezoid.
    Returns filtered mask and whether the result is acceptable for poly fitting.
    """
    reason = ""
    vertical_rejected = False
    out = (mask > 0).astype(np.uint8).copy()
    if out.shape[:2] != (cam_h, cam_w):
        out = cv2.resize(out, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
        if len(out.shape) == 3:
            out = (out.max(axis=2) > 0).astype(np.uint8)

    if reject_vertical:
        out, vertical_rejected = reject_vertical_planes(out, cam_w, cam_h)
    if apply_trapezoid:
        trapezoid = get_road_trapezoid_mask(cam_w, cam_h)
        out = cv2.bitwise_and(out, (trapezoid > 0).astype(np.uint8))

    # After filtering, require minimum road pixels (avoid empty mask)
    road_ratio = np.sum(out > 0) / max(cam_w * cam_h, 1)
    if road_ratio < 0.01:
        reason = "road_ratio_low"
        return GeometricValidationResult(mask=out, passed=False, reason=reason, vertical_rejected=vertical_rejected)

    return GeometricValidationResult(
        mask=out,
        passed=True,
        reason=reason,
        vertical_rejected=vertical_rejected,
        trapezoid_applied=apply_trapezoid,
    )


def sanity_check_lane_width(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    min_m: float = MIN_LANE_WIDTH_M,
    max_m: float = MAX_LANE_WIDTH_M,
) -> Tuple[bool, str]:
    """
    Left/right in vehicle frame (x forward, y lateral). Width = |y_right - y_left| at same x.
    Reject if width outside [min_m, max_m].
    """
    if left_xy is None or right_xy is None or len(left_xy) < 2 or len(right_xy) < 2:
        return False, "insufficient_points"
    x_l, y_l = left_xy[:, 0], left_xy[:, 1]
    x_r, y_r = right_xy[:, 0], right_xy[:, 1]
    # Sort by x for interp
    idx_l = np.argsort(x_l)
    x_l, y_l = x_l[idx_l], y_l[idx_l]
    idx_r = np.argsort(x_r)
    x_r, y_r = x_r[idx_r], y_r[idx_r]
    # Sample at a few x positions and check width
    x_min = max(x_l.min(), x_r.min())
    x_max = min(x_l.max(), x_r.max())
    if x_max - x_min < 1.0:
        return False, "width_x_range_small"
    x_sample = np.linspace(x_min, x_max, 5)
    for x in x_sample:
        y_left_at_x = np.interp(x, x_l, y_l)
        y_right_at_x = np.interp(x, x_r, y_r)
        w = abs(y_right_at_x - y_left_at_x)
        if w < min_m or w > max_m:
            return False, f"width_{w:.2f}m_out_of_range"
    return True, ""


def sanity_check_center_jump(
    prev_center_coeffs: Optional[np.ndarray],
    new_center_coeffs: np.ndarray,
    max_jump_m: float = MAX_CENTER_JUMP_M,
) -> Tuple[bool, str]:
    """Reject if lateral position at x=0 (ego) jumps more than max_jump_m."""
    if prev_center_coeffs is None:
        return True, ""
    y_prev = float(np.polyval(prev_center_coeffs, 0.0))
    y_new = float(np.polyval(new_center_coeffs, 0.0))
    if abs(y_new - y_prev) > max_jump_m:
        return False, f"center_jump_{abs(y_new - y_prev):.2f}m"
    return True, ""


def sanity_check_curvature(center_coeffs: np.ndarray, max_curv: float = MAX_CURVATURE_PHYSICAL) -> Tuple[bool, str]:
    """Curvature at x=0: for y = c2*x^2 + c1*x + c0, curvature = 2*c2 / (1 + c1^2)^1.5."""
    if center_coeffs is None or len(center_coeffs) < 3:
        return True, ""
    c0, c1, c2 = center_coeffs[2], center_coeffs[1], center_coeffs[0]
    curv = 2.0 * c2 / ((1.0 + c1 * c1) ** 1.5 + 1e-9)
    if abs(curv) > max_curv:
        return False, f"curvature_{curv:.4f}_exceeds_limit"
    return True, ""


def run_sanity_checks(
    left_xy: Optional[np.ndarray],
    right_xy: Optional[np.ndarray],
    center_coeffs: Optional[np.ndarray],
    prev_center_coeffs: Optional[np.ndarray],
) -> Tuple[bool, str]:
    """
    Run all sanity checks before accepting poly fit.
    Returns (passed, reason). reason empty if passed.
    """
    if left_xy is not None and right_xy is not None:
        ok, r = sanity_check_lane_width(left_xy, right_xy)
        if not ok:
            return False, r
    if center_coeffs is not None:
        ok, r = sanity_check_center_jump(prev_center_coeffs, center_coeffs)
        if not ok:
            return False, r
        ok, r = sanity_check_curvature(center_coeffs)
        if not ok:
            return False, r
    return True, ""
