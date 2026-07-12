"""
Lane trajectory pipeline — การจับเส้นแบบ phase (refactor).

Architecture: P1 → P2 → P3 → P4 → P5 → P6

  P1 Lane mask:      RGB → binary mask + confidence
  P2 BEV + quality:  mask → BEV warp, fill gaps, mask_quality
  P3 Boundaries:     BEV → left/right/center coeffs (sliding window or ego tracker)
  P4 Smooth center:  center_raw + Kalman → center_coeffs, geometry_valid
  P5 Trajectory:     center_coeffs → (x_ref, y_ref, cte, heading, curvature, v_ref)
  P6 Visualization:  BEV window vis + lane overlay

Visualization:
  - Warped Frame (BEV binary mask) — full resolution
  - Warped Frame With Search Window — sliding window colored (yellow left / blue right / green search area)
  - Original Frame With Lane Overlay — green filled lane on perspective view
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

import cv2
import numpy as np

# Import B-spline fitter for improved lane fitting
try:
    from .spline_lane_fitting import BSplineLaneFitter  # noqa: F401
    BSPLINE_AVAILABLE = True
except ImportError:
    BSPLINE_AVAILABLE = False
    logging.warning("B-spline fitter not available, using polynomial fitting")

from .kalman_lane_tracker import KalmanLaneTracker

try:
    from config import (
        MIN_LANE_WIDTH_M,
        MAX_LANE_WIDTH_M,
        ROAD_TOP_RATIO,
        BEV_LANE_SEARCH_TOP_RATIO,
        LANE_CONF_ROI_TOP_RATIO,
        BEV_HISTOGRAM_WEIGHT_FAR,
        BEV_HISTOGRAM_WEIGHT_NEAR,
        BEV_WARP_INTERPOLATION,
        BEV_RIGHTMOST_PEAK,
        BEV_RIGHT_PEAK_MIN_RATIO,
        BEV_ALONG_ROAD_DILATE_KERNEL,
        LANE_CONF_CONTINUITY_ALPHA,
        LANE_CONTINUITY_MIN_AREA_PX,
        MASK_DILATE_VERTICAL,
        MASK_CLOSE_KERNEL_LARGE,
        LANE_OVERLAY_DISPLAY_MARGIN_M,
        LANE_OVERLAY_MIN_DISPLAY_WIDTH_M,
        LANE_OVERLAY_MAX_WIDTH_DRAW_M,
        LANE_OVERLAY_ROAD_FORM,
        LANE_OVERLAY_STRIP_HALF_WIDTH_M,
        LANE_OVERLAY_STRAIGHT_NEAR_M,
        LANE_FRAME_WIDTH_MIN_M,
        LANE_FRAME_WIDTH_MAX_M,
        LANE_FRAME_WIDTH_NOMINAL_M,
        LANE_FRAME_WIDTH_EMA_ALPHA,
        LANE_OVERLAY_USE_FRAME_WIDTH,
        LANE_OVERLAY_TRAPEZOID,
        BEV_H,
        BEV_W,
    )
except ImportError as e:
    logger.warning(f"Config import failed: {e}, using defaults")
    # Default fallback values
    MIN_LANE_WIDTH_M = 2.0
    MAX_LANE_WIDTH_M = 5.5
    ROAD_TOP_RATIO = 0.0
    LANE_OVERLAY_DISPLAY_MARGIN_M = 0.2
    LANE_OVERLAY_MIN_DISPLAY_WIDTH_M = 2.0
    LANE_OVERLAY_MAX_WIDTH_DRAW_M = 6.0
    LANE_OVERLAY_ROAD_FORM = False
    LANE_OVERLAY_STRIP_HALF_WIDTH_M = 0.9
    LANE_OVERLAY_STRAIGHT_NEAR_M = 6.0
    LANE_FRAME_WIDTH_MIN_M = 2.0
    LANE_FRAME_WIDTH_MAX_M = 5.5
    LANE_FRAME_WIDTH_NOMINAL_M = 3.5
    LANE_FRAME_WIDTH_EMA_ALPHA = 0.6
    LANE_OVERLAY_USE_FRAME_WIDTH = True
    LANE_OVERLAY_TRAPEZOID = False
    BEV_LANE_SEARCH_TOP_RATIO = 0.25
    LANE_CONF_ROI_TOP_RATIO = 0.30
    BEV_HISTOGRAM_WEIGHT_FAR = 1.5
    BEV_HISTOGRAM_WEIGHT_NEAR = 0.5
    BEV_WARP_INTERPOLATION = "nearest"
    BEV_RIGHTMOST_PEAK = True
    BEV_RIGHT_PEAK_MIN_RATIO = 0.25
    BEV_ALONG_ROAD_DILATE_KERNEL = (3, 5)
    BEV_H = 320
    BEV_W = 450
    MASK_DILATE_VERTICAL = (3, 14)
    MASK_CLOSE_KERNEL_LARGE = (15, 8)
    LANE_CONF_CONTINUITY_ALPHA = 0.4
    LANE_CONTINUITY_MIN_AREA_PX = 50

# ── Pipeline constants ──────────────────────────────────────────────────────
LOOKAHEAD_M         = 30.0   # ระยะมองเลนข้างหน้า (m) — 30m เหมาะกับความเร็ว 25 km/h
HALF_WIDTH_M        = 8.0    # ความกว้าง BEV ด้านข้าง (m) — 16m total width
DS_RESAMPLE_M       = 0.5
POLY_ORDER          = 2
POLY_DIM            = POLY_ORDER + 1          # [c2, c1, c0]
KALMAN_Q            = 0.08    # เพิ่มจาก 0.01 → 0.08 (8x faster response, adaptive)
KALMAN_R            = 0.005   # ลดจาก 0.02 → 0.005 (เชื่อ measurement มากขึ้น)
MAX_CURVATURE       = 0.08
CURVATURE_RATE_LIMIT = 0.02
OUTLIER_RESIDUAL_THRESH = 2.0
# Adjacent lanes in BEV (optional หลายเส้น): draw candidates with |lateral_at_ego_m| in [MIN, MAX]
ADJACENT_LANE_MIN_M = 2.0
ADJACENT_LANE_MAX_M = 7.0
# Curvature temporal smoothing — ลด spike จาก mask noise, จับ lane นิ่งขึ้น
CURVATURE_EMA_ALPHA = 0.50       # สูงขึ้น = curvature เปลี่ยนช้าลง (นิ่งขึ้น)
CURVATURE_RATE_LIMIT_PER_STEP = 0.008   # จำกัดการเปลี่ยน curvature ต่อ frame
MIN_POINTS_POLY     = 8
N_WINDOWS           = 12                      # เพิ่ม = แบ่งแถวถี่ขึ้น → ติดตามเลนไปไกลขึ้น
MARGIN_WINDOW       = 60                      # wider margin for highway (was 50)
MIN_PIX_WINDOW      = 14                      # ลด = รับจุดไกลที่เบาบางได้ (เดิม 20)

# ── Phase 1: raw UNet mask → full line (ลดเส้นปะ/จุดต่อจุด) ─────────────────
MASK_DILATE_THIN        = (3, 3)               # ขยายเส้นเล็กให้หนาขึ้นก่อน close (เส้นเล็กๆ ก็จับ)
MASK_CLOSE_KERNEL_SMALL = (7, 7)             # initial close (existing)
# MASK_DILATE_VERTICAL, MASK_CLOSE_KERNEL_LARGE จาก config (ลดความหนา raw mask)
MASK_EMA_ALPHA          = 0.2                # ลดจาก 0.4 → 0.2 (80% current, 20% prev)
MASK_EMA_SEAM_CHANGE_RATIO = 0.25           # ถ้า mask เปลี่ยนเกินอัตรานี้ถือว่ารอยต่อ → ใช้ alpha_seam
MASK_EMA_ALPHA_SEAM     = 0.2               # ที่รอยต่อ: 20% prev, 80% current (ไม่ให้ mask ค้าง)

# ── Peter Moran's Robust Lane Tracking ─────────────────────────────────────
WINDOW_MIN_PIXEL_COVERAGE = 0.60  # 60% pixel coverage threshold (Peter Moran)
MAX_FROZEN_FRAMES = 5              # Drop window after frozen this many frames

# ── Phase 4: รอยต่อไม่ค้าง/ไม่เพี้ยน ─────────────────────────────────────────
MAX_FRAMES_STALE_CENTER = 3                 # เกินนี้ให้ reset center+Kalman; กลับมาแล้ว re-init ไม่ผสมเก่า

# ── BEV display: เส้นไม่กระตุก/ไม่หมุน ─────────────────────────────────────
BEV_DISPLAY_EMA_ALPHA = 0.3                 # ลดจาก 0.5 → 0.3 (30% prev, 70% current)

# ── BEV / lane overlay line thickness (ลดเส้นหนาเกิน) ──────────────────────
BEV_VIS_CENTER_THICKNESS = 3
BEV_VIS_BOUNDARY_THICKNESS = 2
LANE_OVERLAY_CENTER_THICKNESS = 5
LANE_OVERLAY_BOUNDARY_THICKNESS = 2

# ── BEV dimensions: match camera frame size (set in pipeline __init__) ───────
# These are module-level defaults; pipeline overrides them per instance.
_BEV_H = 400
_BEV_W = 600


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryOutput:
    x_ref:          np.ndarray
    y_ref:          np.ndarray
    yaw_ref:        np.ndarray
    v_ref:          np.ndarray
    cte:            float
    heading_err:    float
    curvature:      float
    v_ref_at_ego:   float
    confidence:     float
    mask_vis:       np.ndarray          # binary mask (H, W, 1) uint8 255
    # Visualization frames
    bev_binary:     Optional[np.ndarray] = None   # BEV warped mask (H_bev, W_bev) uint8
    bev_window_vis: Optional[np.ndarray] = None   # BEV + colored sliding windows (H_bev, W_bev, 3)
    lane_overlay:   Optional[np.ndarray] = None   # original RGB + green lane fill (H, W, 3)
    # NEW: Separate left/right masks and raw camera for perception display
    left_mask_vis:  Optional[np.ndarray] = None   # left lane mask only (H, W) uint8
    right_mask_vis: Optional[np.ndarray] = None   # right lane mask only (H, W) uint8
    raw_camera:     Optional[np.ndarray] = None   # original camera image (H, W, 3)
    # NEW: Per-lane curvature for Peter Moran visualization
    left_curvature_m:  float = 9999.0   # radius of curvature of left lane boundary
    right_curvature_m: float = 9999.0   # radius of curvature of right lane boundary
    waypoint_only:  bool = False
    geometry_valid: bool = False
    tracked:        bool = False   # ego-lane matched to previous frame
    used_completion: bool = False  # no detection, used previous frame
    # Image-space boundary points for segment display (row, col); None if not available
    left_px_img:   Optional[np.ndarray] = None  # (N, 2) row,col in camera image
    right_px_img:  Optional[np.ndarray] = None
    # Phase case labels for P2/P3 (for validation and reporting)
    phase_p2_case: str = "none"   # both | mirror_left | mirror_right | completion | none
    phase_p3_case: str = "none"   # from_both | completion | none
    # Option A: path (s_m, lat_m) from perception center_coeffs for alignment with lane overlay
    reference_path_from_perception: Optional[List[Tuple[float, float]]] = None


# ═══════════════════════════════════════════════════════════════════════════
# Kalman filter on polynomial coefficients
# ═══════════════════════════════════════════════════════════════════════════

class KalmanPoly:
    """Dropout-tolerant Kalman on [c2, c1, c0]."""

    def __init__(self, q: float = KALMAN_Q, r: float = KALMAN_R, dim: int = POLY_DIM):
        self.dim = dim
        self.x = np.zeros(dim, dtype=np.float64)
        self.P = np.eye(dim) * 0.1
        self.F = np.eye(dim)
        self.H = np.eye(dim)
        self.Q = np.eye(dim) * q
        self.R = np.eye(dim) * r
        self._initialized = False

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).ravel()
        if not self._initialized:
            self.x = z.copy()
            self._initialized = True
            return self.x.copy()
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x.copy()
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim) - K @ self.H) @ self.P
        return self.x.copy()

    def get_state(self) -> Optional[np.ndarray]:
        return self.x.copy() if self._initialized else None

    def reset(self) -> None:
        self._initialized = False
        self.x = np.zeros(self.dim, dtype=np.float64)
        self.P = np.eye(self.dim) * 0.1


class CurvatureSmoother:
    """EMA + rate limit on curvature to reduce mask-noise spikes."""

    def __init__(self, alpha: float = CURVATURE_EMA_ALPHA, rate_limit: float = CURVATURE_RATE_LIMIT_PER_STEP):
        self.alpha = alpha
        self.rate_limit = rate_limit
        self._prev = 0.0

    def update(self, curvature: float) -> float:
        curv = float(np.clip(curvature, -MAX_CURVATURE, MAX_CURVATURE))
        delta = np.clip(curv - self._prev, -self.rate_limit, self.rate_limit)
        smoothed = self._prev + delta
        out = self.alpha * self._prev + (1.0 - self.alpha) * smoothed
        self._prev = out
        return float(np.clip(out, -MAX_CURVATURE, MAX_CURVATURE))

    def reset(self) -> None:
        self._prev = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Mask separation for left/right lane visualization
# ═══════════════════════════════════════════════════════════════════════════

def create_left_lane_mask(
    full_mask: np.ndarray,
    left_points: Optional[np.ndarray],
    cam_w: int,
    cam_h: int,
) -> np.ndarray:
    """
    Extract left lane region from full UNet mask.

    Args:
        full_mask: Full lane mask from UNet (H x W)
        left_points: Left lane points in image coordinates (N x 2) as (row, col)
        cam_w: Camera width
        cam_h: Camera height

    Returns:
        Left lane mask only (H x W)
    """
    if left_points is None or len(left_points) == 0:
        # No left lane detected, return empty mask
        return np.zeros_like(full_mask)

    # Create mask for left half of image
    left_mask = np.zeros_like(full_mask)
    mid_col = cam_w // 2

    # Copy only left half of the mask
    left_mask[:, :mid_col] = full_mask[:, :mid_col]

    return left_mask


def create_right_lane_mask(
    full_mask: np.ndarray,
    right_points: Optional[np.ndarray],
    cam_w: int,
    cam_h: int,
) -> np.ndarray:
    """
    Extract right lane region from full UNet mask.

    Args:
        full_mask: Full lane mask from UNet (H x W)
        right_points: Right lane points in image coordinates (N x 2) as (row, col)
        cam_w: Camera width
        cam_h: Camera height

    Returns:
        Right lane mask only (H x W)
    """
    if right_points is None or len(right_points) == 0:
        # No right lane detected, return empty mask
        return np.zeros_like(full_mask)

    # Create mask for right half of image
    right_mask = np.zeros_like(full_mask)
    mid_col = cam_w // 2

    # Copy only right half of the mask
    right_mask[:, mid_col:] = full_mask[:, mid_col:]

    return right_mask


# ═══════════════════════════════════════════════════════════════════════════
# BEV transform
# ═══════════════════════════════════════════════════════════════════════════

def _get_bev_transform(
    cam_w: int, cam_h: int,
    src_top_ratio: float = 0.50,   # include far-road detections
    src_bot_ratio: float = 0.98,   # almost full bottom
    src_top_margin: float = 0.20,  # wide at top to capture side lanes
    src_bot_margin: float = 0.0,   # full width at bottom
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (M, M_inv, src_pts, dst_pts).

    src trapezoid: wide to capture all lane markings visible in FOV=110° camera.
    """
    top_y = int(cam_h * src_top_ratio)
    bot_y = int(cam_h * src_bot_ratio)
    src = np.float32([
        [int(cam_w * src_bot_margin),       bot_y],   # bottom-left
        [int(cam_w * (1 - src_bot_margin)), bot_y],   # bottom-right
        [int(cam_w * (1 - src_top_margin)), top_y],   # top-right
        [int(cam_w * src_top_margin),       top_y],   # top-left
    ])
    dst = np.float32([
        [0,           cam_h - 1],
        [cam_w - 1,   cam_h - 1],
        [cam_w - 1,   0],
        [0,           0],
    ])
    M     = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv, src, dst


def bev_transform(mask: np.ndarray, cam_w: int, cam_h: int,
                  M: Optional[np.ndarray] = None,
                  use_gpu: bool = False) -> np.ndarray:
    """
    Transform mask to bird's eye view.
    Output size = (cam_w, cam_h) to match downstream pipeline expectations.
    """
    if M is None:
        M, _, _, _ = _get_bev_transform(cam_w, cam_h)
    interp = cv2.INTER_NEAREST if (BEV_WARP_INTERPOLATION or "").lower() == "nearest" else cv2.INTER_LINEAR
    bev = cv2.warpPerspective(
        (mask > 0).astype(np.uint8) * 255,
        M, (cam_w, cam_h),
        flags=interp,
    )
    return bev


# ═══════════════════════════════════════════════════════════════════════════
# BEV pixel ↔ vehicle-frame XY
# ═══════════════════════════════════════════════════════════════════════════

def _bev_px_to_vehicle_xy(
    row: np.ndarray, col: np.ndarray,
    bev_h: int, bev_w: int,
    lookahead_m: float = LOOKAHEAD_M,
    half_width_m: float = HALF_WIDTH_M,
) -> Tuple[np.ndarray, np.ndarray]:
    x = (bev_h - 1 - row) * (lookahead_m / max(bev_h - 1, 1))
    y = (col - (bev_w - 1) / 2.0) * (half_width_m * 2.0 / max(bev_w - 1, 1))
    return x, y


def _vehicle_xy_to_bev_px(
    x: np.ndarray, y: np.ndarray,
    bev_h: int, bev_w: int,
    lookahead_m: float = LOOKAHEAD_M,
    half_width_m: float = HALF_WIDTH_M,
) -> Tuple[np.ndarray, np.ndarray]:
    row = (bev_h - 1) - x * (bev_h - 1) / max(lookahead_m, 1e-6)
    col = y * (bev_w - 1) / max(half_width_m * 2.0, 1e-6) + (bev_w - 1) / 2.0
    return row.astype(int), col.astype(int)


def _bev_coeffs_to_image_points(
    left_coeffs: Optional[np.ndarray],
    right_coeffs: Optional[np.ndarray],
    M_inv: np.ndarray,
    bev_h: int, bev_w: int,
    cam_h: int, cam_w: int,
    lookahead_m: float = LOOKAHEAD_M,
    half_width_m: float = HALF_WIDTH_M,
    max_x_m: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert BEV left/right poly coeffs to image-space (row, col) for segment display. max_x_m = ตัดยอด."""
    draw_to = min(lookahead_m, max_x_m) if max_x_m is not None else lookahead_m
    n_pts = max(bev_h * 2, 80)
    x_m = np.linspace(0, draw_to, n_pts)
    left_px_img, right_px_img = None, None

    def coeffs_to_img(coeffs: np.ndarray) -> Optional[np.ndarray]:
        y_m = np.polyval(coeffs, x_m)
        rows, cols = _vehicle_xy_to_bev_px(x_m, y_m, bev_h, bev_w, lookahead_m, half_width_m)
        mask = (rows >= 0) & (rows < bev_h) & (cols >= 0) & (cols < bev_w)
        if not np.any(mask):
            return None
        pts_bev = np.column_stack((cols[mask], rows[mask])).astype(np.float32).reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(pts_bev, M_inv)
        return np.column_stack((img_pts[:, 0, 1], img_pts[:, 0, 0]))  # row, col

    if left_coeffs is not None:
        left_px_img = coeffs_to_img(left_coeffs)
    if right_coeffs is not None:
        right_px_img = coeffs_to_img(right_coeffs)
    return left_px_img, right_px_img


# ═══════════════════════════════════════════════════════════════════════════
# Mask quality: 2-stage check (pixel count + spatial continuity)
# ═══════════════════════════════════════════════════════════════════════════

MIN_PX_TOTAL = 300
N_STRIPS = 5
MIN_PX_PER_STRIP = 20
CONTINUOUS_STRIPS_INVALID = 3   # < 3 strips with enough px → invalid
CONTINUOUS_STRIPS_DEGRADED = 5   # >= 5 → valid; 3–4 → degraded


def check_mask_quality(
    bev_binary: np.ndarray,
    min_px_total: int = MIN_PX_TOTAL,
    n_strips: int = N_STRIPS,
    min_px_per_strip: int = MIN_PX_PER_STRIP,
) -> str:
    """
    2-stage lane mask validation to reduce false LANE INVALID.

    Stage 1: total white pixels < min_px_total → invalid.
    Stage 2: divide ROI into n_strips horizontal strips; count strips with
             pixel count >= min_px_per_strip.
      - continuous_strips < 3 → invalid (broken line)
      - 3 <= continuous_strips < 5 → degraded (use lower section / still show valid)
      - continuous_strips >= 5 → valid

    Returns: "valid" | "degraded" | "invalid"
    """
    total_white = int(np.sum(bev_binary > 0))
    if total_white < min_px_total:
        return "invalid"

    h, w = bev_binary.shape
    strip_h = max(1, h // n_strips)
    continuous_count = 0
    for i in range(n_strips):
        y_lo = h - (i + 1) * strip_h
        y_hi = h - i * strip_h
        if y_lo < 0:
            break
        strip = bev_binary[y_lo:y_hi, :]
        px_in_strip = int(np.sum(strip > 0))
        if px_in_strip >= min_px_per_strip:
            continuous_count += 1

    if continuous_count < CONTINUOUS_STRIPS_INVALID:
        return "invalid"
    if continuous_count >= CONTINUOUS_STRIPS_DEGRADED:
        return "valid"
    return "degraded"


# ═══════════════════════════════════════════════════════════════════════════
# Sliding window search  (returns pixel coords + vehicle XY)
# ═══════════════════════════════════════════════════════════════════════════


def is_window_outlier(
    window_pixels: int,
    search_area_pixels: int,
    min_coverage: float = WINDOW_MIN_PIXEL_COVERAGE,
) -> Tuple[bool, str]:
    """
    Detect if window measurement is an outlier.
    Based on Peter Moran's robust lane tracking methodology.

    Two-stage outlier detection:
    1. Signal-to-noise ratio: pixel coverage must be >= min_coverage
    2. (Future) Kalman log likelihood check

    Args:
        window_pixels: Number of lane pixels found in window
        search_area_pixels: Total pixels in search area
        min_coverage: Minimum pixel coverage ratio (default 0.60 = 60%)

    Returns:
        (is_outlier, reason): True if outlier, with reason string
    """
    if window_pixels == 0:
        return True, "no_pixels"

    coverage = window_pixels / max(search_area_pixels, 1)
    if coverage < min_coverage:
        return True, f"low_coverage_{coverage:.2f}"

    return False, "valid"


def _rightmost_peak(hist: np.ndarray, min_ratio: float = 0.25) -> int:
    """
    คืน index ที่อยู่ขวาสุดใน hist ที่มีค่า >= max*min_ratio (ให้เส้นขวารับขอบถนน).
    hist = histogram ของครึ่งขวา BEV (index 0 = กลางภาพ).
    """
    if len(hist) == 0:
        return 0
    mx = float(np.max(hist))
    if mx < 1e-6:
        return int(np.argmax(hist))
    thresh = mx * min_ratio
    candidates = np.where(hist >= thresh)[0]
    if len(candidates) > 0:
        return int(np.max(candidates))
    return int(np.argmax(hist))


def sliding_window_search(
    bev_binary: np.ndarray,
    n_windows: int = N_WINDOWS,
    margin: int = MARGIN_WINDOW,
    min_pix: int = MIN_PIX_WINDOW,
    lookahead_m: float = LOOKAHEAD_M,
    half_width_m: float = HALF_WIDTH_M,
) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray],  # vehicle XY left/right
    Optional[np.ndarray], Optional[np.ndarray],  # pixel (row, col) left/right
    List,                                         # raw window rects left (all)
    List,                                         # raw window rects right (all)
    List,                                         # filtered window rects left (accepted)
    List,                                         # filtered window rects right (accepted)
    str,                                          # p2_case: both | mirror_left | mirror_right | none
]:
    """
    Independent left/right search: histogram on left half and right half separately
    so dominant lane does not suppress the other. If right has low confidence,
    use mirror of left at expected lane width.

    Returns:
      left_xy, right_xy                   — vehicle-frame points (N, 2)
      left_px, right_px                   — pixel points (N, 2) = (row, col) for visualization
      raw_left_windows, raw_right_windows — ALL windows (for red visualization)
      filt_left_windows, filt_right_windows — ACCEPTED windows (for green visualization)
      p2_case                             — detection case
    """
    h, w = bev_binary.shape
    mid_col = w // 2

    # Distance-weighted histograms: แถวบน (ไกล) น้ำหนักมาก — boost lane ไกล
    bottom = bev_binary[int(h * BEV_LANE_SEARCH_TOP_RATIO):, :]
    n_rows = bottom.shape[0]
    weights = np.linspace(BEV_HISTOGRAM_WEIGHT_FAR, BEV_HISTOGRAM_WEIGHT_NEAR, n_rows)[:, np.newaxis]
    left_bottom  = bottom[:, :mid_col]
    right_bottom = bottom[:, mid_col:]
    hist_l = np.sum((left_bottom > 0).astype(np.float32) * weights, axis=0)
    hist_r = np.sum((right_bottom > 0).astype(np.float32) * weights, axis=0)
    if len(hist_l) < 10 or len(hist_r) < 10:
        return None, None, None, None, [], [], [], [], "none"
    hist_l_sm = cv2.GaussianBlur(hist_l.reshape(1, -1), (1, 31), 0).ravel()
    hist_r_sm = cv2.GaussianBlur(hist_r.reshape(1, -1), (1, 31), 0).ravel()

    left_peak = int(np.argmax(hist_l_sm))
    if BEV_RIGHTMOST_PEAK:
        right_peak = _rightmost_peak(hist_r_sm, BEV_RIGHT_PEAK_MIN_RATIO) + mid_col
    else:
        right_peak = int(np.argmax(hist_r_sm)) + mid_col

    max_left = float(np.max(hist_l_sm))
    max_right = float(np.max(hist_r_sm))
    if max_left < 5 and max_right < 5:
        return None, None, None, None, [], [], [], [], "none"

    if max_left < 5:
        left_peak = max(0, mid_col - margin - 20)
    if max_right < 5:
        right_peak = min(w - 1, mid_col + margin + 20)

    window_h = max(1, h // n_windows)
    left_pts_px, right_pts_px = [], []
    # Peter Moran: separate raw (all) vs filtered (accepted) windows
    raw_left_wins, raw_right_wins = [], []      # Red: all windows
    filt_left_wins, filt_right_wins = [], []    # Green: accepted only
    margin_min = 15  # clamp เพื่อไม่ให้ window แคบเกินไป

    for win_i in range(n_windows):
        y_lo = h - (win_i + 1) * window_h
        y_hi = h - win_i * window_h
        if y_lo < 0:
            break
        y_mid = (y_lo + y_hi) / 2.0
        y_norm = y_mid / max(h, 1)
        margin_dyn = max(margin_min, int(margin * (1 - y_norm * 0.5)))
        # ระยะไกล (แถวบน BEV): ใช้ min_pix ต่ำลง เพื่อให้จับเลนที่เบาบางได้
        min_pix_win = min_pix if y_norm >= 0.5 else max(6, min_pix - 6)

        # Left window (independent) with Peter Moran's outlier detection
        xl1 = max(0, left_peak - margin_dyn)
        xl2 = min(w, left_peak + margin_dyn)
        raw_left_wins.append((xl1, y_lo, xl2, y_hi))  # Always add to raw (red)
        ys_l, xs_l = np.where(bev_binary[y_lo:y_hi, xl1:xl2] > 0)

        # Outlier detection: check pixel coverage
        search_area_l = (xl2 - xl1) * (y_hi - y_lo)
        is_outlier_l, reason_l = is_window_outlier(len(ys_l), search_area_l)

        if len(ys_l) >= min_pix_win and not is_outlier_l:
            filt_left_wins.append((xl1, y_lo, xl2, y_hi))  # Add to filtered (green)
            left_pts_px.append(np.column_stack((ys_l + y_lo, xs_l + xl1)))
            left_peak = int(np.mean(xs_l + xl1))
        # else: window rejected as outlier or insufficient pixels

        # Right window (independent) with Peter Moran's outlier detection
        xr1 = max(0, right_peak - margin_dyn)
        xr2 = min(w, right_peak + margin_dyn)
        raw_right_wins.append((xr1, y_lo, xr2, y_hi))  # Always add to raw (red)
        ys_r, xs_r = np.where(bev_binary[y_lo:y_hi, xr1:xr2] > 0)

        # Outlier detection: check pixel coverage
        search_area_r = (xr2 - xr1) * (y_hi - y_lo)
        is_outlier_r, reason_r = is_window_outlier(len(ys_r), search_area_r)

        if len(ys_r) >= min_pix_win and not is_outlier_r:
            filt_right_wins.append((xr1, y_lo, xr2, y_hi))  # Add to filtered (green)
            right_pts_px.append(np.column_stack((ys_r + y_lo, xs_r + xr1)))
            right_peak = int(np.mean(xs_r + xr1))
        # else: window rejected as outlier or insufficient pixels

    # Mirror strategy: if we have left but no right, reflect left across center (y_right = -y_left)
    if left_pts_px and not right_pts_px:
        left_px = np.vstack(left_pts_px)
        xl, yl = _bev_px_to_vehicle_xy(left_px[:, 0], left_px[:, 1], h, w, lookahead_m, half_width_m)
        left_xy = np.column_stack((xl, yl))
        # right_xy = mirror in vehicle frame: (x, -y) so lane symmetric about center
        right_xy = np.column_stack((xl.copy(), -np.asarray(yl)))
        # right_px for vis: approximate by reflecting col about mid_col
        right_px = np.column_stack((left_px[:, 0], 2 * mid_col - left_px[:, 1]))
        return left_xy, right_xy, left_px, right_px, raw_left_wins, raw_right_wins, filt_left_wins, filt_right_wins, "mirror_left"

    if right_pts_px and not left_pts_px:
        right_px = np.vstack(right_pts_px)
        xr, yr = _bev_px_to_vehicle_xy(right_px[:, 0], right_px[:, 1], h, w, lookahead_m, half_width_m)
        right_xy = np.column_stack((xr, yr))
        left_xy = np.column_stack((xr.copy(), -np.asarray(yr)))
        left_px = np.column_stack((right_px[:, 0], 2 * mid_col - right_px[:, 1]))
        return left_xy, right_xy, left_px, right_px, raw_left_wins, raw_right_wins, filt_left_wins, filt_right_wins, "mirror_right"

    if not left_pts_px or not right_pts_px:
        return None, None, None, None, raw_left_wins, raw_right_wins, filt_left_wins, filt_right_wins, "none"

    left_px  = np.vstack(left_pts_px)
    right_px = np.vstack(right_pts_px)

    xl, yl = _bev_px_to_vehicle_xy(left_px[:, 0],  left_px[:, 1],  h, w, lookahead_m, half_width_m)
    xr, yr = _bev_px_to_vehicle_xy(right_px[:, 0], right_px[:, 1], h, w, lookahead_m, half_width_m)

    return (
        np.column_stack((xl, yl)), np.column_stack((xr, yr)),
        left_px, right_px,
        raw_left_wins, raw_right_wins,
        filt_left_wins, filt_right_wins,
        "both",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Polynomial fit (robust IRLS-style)
# ═══════════════════════════════════════════════════════════════════════════

def _fit_poly_robust(x: np.ndarray, y: np.ndarray, order: int = POLY_ORDER) -> Optional[np.ndarray]:
    if len(x) < MIN_POINTS_POLY:
        return None
    coeffs = np.polyfit(x, y, order)
    for _ in range(3):
        pred   = np.polyval(coeffs, x)
        res    = np.abs(y - pred)
        std    = np.std(res)
        thresh = std * OUTLIER_RESIDUAL_THRESH if std > 1e-6 else np.max(res) + 1
        mask   = res <= thresh
        if np.sum(mask) < MIN_POINTS_POLY:
            return None
        x, y   = x[mask], y[mask]
        coeffs = np.polyfit(x, y, order)
    return coeffs


def _mirror_coeffs(coeffs: np.ndarray) -> np.ndarray:
    """Mirror lane coeffs across center (y → -y) for full line when one side is missing."""
    return -np.asarray(coeffs, dtype=np.float64)


def _densify_xy(xy: np.ndarray, n_pts: int = 40) -> np.ndarray:
    """Sort by x and interpolate to get dense points → poly fit เนียนขึ้น (full line)."""
    if xy is None or len(xy) < 2:
        return xy
    x, y = xy[:, 0], xy[:, 1]
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_min, x_max = float(x.min()), float(x.max())
    x_dense = np.linspace(x_min, x_max, n_pts)
    y_dense = np.interp(x_dense, x, y)
    return np.column_stack((x_dense, y_dense))


def polynomial_fit(
    left_xy: Optional[np.ndarray],
    right_xy: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if left_xy is None or right_xy is None:
        return None
    # Densify points ก่อน fit → เส้นต่อเนื่อง full line (ลด dashed/จุดต่อจุด)
    left_xy_d = _densify_xy(left_xy)
    right_xy_d = _densify_xy(right_xy)
    if left_xy_d is None or right_xy_d is None:
        return None
    if len(left_xy_d) < MIN_POINTS_POLY or len(right_xy_d) < MIN_POINTS_POLY:
        return None
    c_l = _fit_poly_robust(left_xy_d[:, 0], left_xy_d[:, 1])
    c_r = _fit_poly_robust(right_xy_d[:, 0], right_xy_d[:, 1])
    # ถ้าฝั่งใดฝั่งหนึ่ง fit ไม่ได้ ใช้ mirror จากอีกฝั่ง → ได้ full line เสมอ
    if c_l is None and c_r is not None:
        c_l = _mirror_coeffs(c_r)
    if c_r is None and c_l is not None:
        c_r = _mirror_coeffs(c_l)
    if c_l is None or c_r is None:
        return None
    return c_l, c_r, (c_l + c_r) / 2.0


def compute_rich_confidence(
    base_confidence: float,
    left_coeffs: Optional[np.ndarray],
    right_coeffs: Optional[np.ndarray],
    min_lane_width_m: float = 2.0,
    max_lane_width_m: float = 5.5,
) -> float:
    """
    Refine confidence using lane width at ego and geometry sanity.
    Penalizes when width at x=0 is outside [min_lane_width_m, max_lane_width_m].
    """
    conf = float(np.clip(base_confidence, 0.0, 1.0))
    if left_coeffs is None or right_coeffs is None:
        return conf
    try:
        left_coeffs = np.asarray(left_coeffs, dtype=np.float64).ravel()[:3]
        right_coeffs = np.asarray(right_coeffs, dtype=np.float64).ravel()[:3]
        y_left_0 = float(np.polyval(left_coeffs, 0.0))
        y_right_0 = float(np.polyval(right_coeffs, 0.0))
        width_ego = abs(y_right_0 - y_left_0)
        if width_ego < min_lane_width_m or width_ego > max_lane_width_m:
            penalty = 0.5 + 0.5 * min(
                1.0,
                min(abs(width_ego - min_lane_width_m), abs(width_ego - max_lane_width_m)) / 2.0,
            )
            conf = conf * penalty
    except (ValueError, TypeError):
        pass
    return float(np.clip(conf, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_trajectory(
    center_coeffs: np.ndarray,
    ds_m: float = DS_RESAMPLE_M,
    lookahead_m: float = LOOKAHEAD_M,
    curvature_prev: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_dense = np.linspace(0, lookahead_m, max(2, int(lookahead_m / 0.2)))
    y_dense = np.polyval(center_coeffs, x_dense)
    try:
        from scipy.interpolate import CubicSpline
        cs    = CubicSpline(x_dense, y_dense)
        s     = np.zeros(len(x_dense))
        s[1:] = np.cumsum(np.sqrt(np.diff(x_dense)**2 + np.diff(y_dense)**2))
        s_tgt = np.arange(0, min(s[-1], lookahead_m), ds_m)
        if len(s_tgt) < 2:
            s_tgt = np.array([0.0, 0.5])
        x_ref = np.interp(s_tgt, s, x_dense)
        y_ref = cs(x_ref)
        dy    = cs.derivative()(x_ref)
        d2y   = cs.derivative(2)(x_ref)
    except ImportError:
        x_ref = x_dense
        y_ref = y_dense
        dy    = np.gradient(y_dense, x_dense)
        d2y   = np.gradient(dy, x_dense)

    yaw_ref       = np.arctan(dy)
    curv          = d2y / (1 + dy**2)**1.5
    curv          = np.clip(curv, -MAX_CURVATURE, MAX_CURVATURE)
    d_curv        = np.clip(np.diff(np.concatenate(([curvature_prev], curv))),
                            -CURVATURE_RATE_LIMIT, CURVATURE_RATE_LIMIT)
    curv          = np.cumsum(d_curv)
    if len(curv) == 0:
        curv = np.array([curvature_prev])
    return x_ref, y_ref, yaw_ref, curv


def curvature_to_speed_factor(curvature: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - 4.0 * np.abs(curvature), 0.3, 1.0)


def mpc_reference_output(
    x_ref: np.ndarray, y_ref: np.ndarray,
    yaw_ref: np.ndarray, v_ref: np.ndarray,
    v_nominal_ms: float,
) -> Tuple[float, float, float, float]:
    if len(x_ref) < 2:
        return 0.0, 0.0, 0.0, v_nominal_ms
    idx = int(np.argmin(np.abs(x_ref)))
    cte = float(y_ref[idx])
    heading_err = float(yaw_ref[idx])
    v_at_ego = float(v_ref[idx]) if idx < len(v_ref) else v_nominal_ms
    curv_val = 0.0
    if idx < len(x_ref) - 1:
        dx = x_ref[idx + 1] - x_ref[idx]
        if dx > 1e-6:
            curv_val = float((yaw_ref[idx + 1] - yaw_ref[idx]) / dx)
    return cte, heading_err, np.clip(curv_val, -MAX_CURVATURE, MAX_CURVATURE), v_at_ego


# ═══════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════════════

def _draw_sliding_window_vis(
    bev_binary: np.ndarray,
    left_px:    Optional[np.ndarray],
    right_px:   Optional[np.ndarray],
    left_wins:  List,
    right_wins: List,
    left_coeffs:  Optional[np.ndarray] = None,
    right_coeffs: Optional[np.ndarray] = None,
    center_coeffs: Optional[np.ndarray] = None,
    lookahead_m: float = LOOKAHEAD_M,
    half_width_m: float = HALF_WIDTH_M,
    max_x_m: Optional[float] = None,
) -> np.ndarray:
    """
    BEV visualization: raw BEV เป็นพื้นหลังจาง + search windows, lane pixels, left/right/center curves.
    - Centerline: thick green line (แนวยาว) — classical style
    - Left/right: yellow/blue curves
    """
    h, w = bev_binary.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    # Option A: blend raw BEV (ที่ UNet ให้มา) เป็นพื้นหลังจาง — เห็นแถบ raw และสองเส้นที่ pipeline สร้าง
    bev_bg = np.stack([bev_binary, bev_binary, bev_binary], axis=-1).astype(np.float32) * 0.35
    vis = (np.clip(bev_bg, 0, 255)).astype(np.uint8)

    # Green search windows
    GREEN_WIN = (0, 100, 0)
    for (x1, y1, x2, y2) in left_wins:
        cv2.rectangle(vis, (x1, y1), (x2, y2), GREEN_WIN, -1)
    for (x1, y1, x2, y2) in right_wins:
        cv2.rectangle(vis, (x1, y1), (x2, y2), GREEN_WIN, -1)

    # Lane pixels
    if left_px is not None and len(left_px):
        for r, c in left_px:
            if 0 <= r < h and 0 <= c < w:
                vis[r, c] = (0, 255, 255)   # yellow (BGR)
    if right_px is not None and len(right_px):
        for r, c in right_px:
            if 0 <= r < h and 0 <= c < w:
                vis[r, c] = (255, 0, 0)     # blue (BGR)

    # Centerline: thick green (แนวยาว) — primary
    center = center_coeffs
    if center is None and left_coeffs is not None and right_coeffs is not None:
        center = (np.asarray(left_coeffs) + np.asarray(right_coeffs)) / 2.0
    if center is not None:
        _draw_poly_curve_bev(vis, center, h, w, lookahead_m, half_width_m,
                             color=(0, 255, 0), thickness=BEV_VIS_CENTER_THICKNESS, max_x_m=max_x_m)

    # Left/right curves
    if left_coeffs is not None:
        _draw_poly_curve_bev(vis, left_coeffs, h, w, lookahead_m, half_width_m,
                             color=(0, 255, 255), thickness=BEV_VIS_BOUNDARY_THICKNESS, max_x_m=max_x_m)
    if right_coeffs is not None:
        _draw_poly_curve_bev(vis, right_coeffs, h, w, lookahead_m, half_width_m,
                             color=(255, 100, 0), thickness=BEV_VIS_BOUNDARY_THICKNESS, max_x_m=max_x_m)

    # Optional: หลายเส้น — draw adjacent lane candidates (grey) in BEV
    # (lane_clustering module removed during cleanup; adjacent lane viz disabled)
    # try:
    #     candidates = cluster_lane_candidates(...)
    #     ...
    # except Exception:
    #     pass

    return vis


def _draw_poly_curve_bev(
    canvas: np.ndarray,
    coeffs: np.ndarray,
    bev_h: int, bev_w: int,
    lookahead_m: float, half_width_m: float,
    color: Tuple, thickness: int = 2,
    max_x_m: Optional[float] = None,
) -> None:
    """Draw polynomial curve onto BEV canvas (in-place). max_x_m = ตัดยอดที่ปลายที่จับได้."""
    draw_to = min(lookahead_m, max_x_m) if max_x_m is not None else lookahead_m
    x_m = np.linspace(0, draw_to, bev_h * 2)
    y_m = np.polyval(coeffs, x_m)
    rows, cols = _vehicle_xy_to_bev_px(x_m, y_m, bev_h, bev_w, lookahead_m, half_width_m)
    pts = []
    for r, c in zip(rows, cols):
        if 0 <= r < bev_h and 0 <= c < bev_w:
            pts.append([c, r])
    if len(pts) > 1:
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_arr], False, color, thickness, cv2.LINE_AA)


def _center_to_strip_boundaries(
    center_coeffs: np.ndarray,
    x_m: np.ndarray,
    half_width_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    จาก centerline (vehicle frame: x ไปข้างหน้า, y ไปข้างขวา) คำนวณขอบซ้าย/ขวา
    เป็น strip ความกว้างคงที่ในแนวตั้งฉาก — ตรงเมื่อถนนตรง โค้งตามทางโค้ง.

    Returns:
        left_xy: (N, 2) array [x, y] vehicle frame สำหรับขอบซ้าย
        right_xy: (N, 2) array [x, y] vehicle frame สำหรับขอบขวา
    """
    if center_coeffs is None or len(center_coeffs) < 1 or len(x_m) < 2:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    y_c = np.polyval(center_coeffs, x_m)
    slope = np.polyval(np.polyder(center_coeffs), x_m)
    norm = np.sqrt(1.0 + slope.astype(np.float64) ** 2)
    nx = (-slope / norm).astype(np.float64)
    ny = (1.0 / norm).astype(np.float64)
    hw = float(half_width_m)
    left_x = x_m.astype(np.float64) - hw * nx
    left_y = y_c.astype(np.float64) - hw * ny
    right_x = x_m.astype(np.float64) + hw * nx
    right_y = y_c.astype(np.float64) + hw * ny
    left_xy = np.column_stack((left_x, left_y))
    right_xy = np.column_stack((right_x, right_y))
    return left_xy, right_xy


def _draw_lane_overlay(
    rgb: np.ndarray,
    left_coeffs:  Optional[np.ndarray],
    right_coeffs: Optional[np.ndarray],
    M_inv: np.ndarray,
    bev_h: int, bev_w: int,
    cam_h: int, cam_w: int,
    lookahead_m: float = LOOKAHEAD_M,
    half_width_m: float = HALF_WIDTH_M,
    fill_color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
    center_coeffs: Optional[np.ndarray] = None,
    draw_fill: bool = True,
    max_x_m: Optional[float] = None,
    frame_width_m: Optional[float] = None,
) -> np.ndarray:
    """
    Lane overlay on original perspective: green longitudinal lines (classical style) + light fill.
    - max_x_m: ตัดยอดเส้นที่ระยะนี้ (ปลายที่จับได้); None = ใช้ lookahead_m เต็ม
    """
    overlay_bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    fill_bgr = (fill_color[2], fill_color[1], fill_color[0])  # BGR for cv2
    green_bgr = (0, 255, 0)  # cv2 green

    draw_to = min(lookahead_m, max_x_m) if max_x_m is not None else lookahead_m
    n_pts = max(bev_h * 2, 80)
    x_m = np.linspace(0, draw_to, n_pts)
    straight_near_m = LANE_OVERLAY_STRAIGHT_NEAR_M
    use_straight_near = (
        straight_near_m > 0.0
        and straight_near_m < draw_to
        and (center_coeffs is not None or (left_coeffs is not None and right_coeffs is not None))
    )
    if use_straight_near:
        straight_near_m = min(straight_near_m, draw_to - 0.5)

    def to_bev_pts(coeffs: np.ndarray):
        y_m = np.polyval(coeffs, x_m)
        rows, cols = _vehicle_xy_to_bev_px(x_m, y_m, bev_h, bev_w, lookahead_m, half_width_m)
        mask = (rows >= 0) & (rows < bev_h) & (cols >= 0) & (cols < bev_w)
        r, c = rows[mask], cols[mask]
        if len(r) < 2:
            return None
        return np.column_stack((c, r)).astype(np.int32).reshape(-1, 1, 2)

    def to_bev_pts_xy(x_vals: np.ndarray, y_vals: np.ndarray):
        """Draw polyline from (x_vals, y_vals) in vehicle frame."""
        rows, cols = _vehicle_xy_to_bev_px(x_vals, y_vals, bev_h, bev_w, lookahead_m, half_width_m)
        mask = (rows >= 0) & (rows < bev_h) & (cols >= 0) & (cols < bev_w)
        r, c = rows[mask], cols[mask]
        if len(r) < 2:
            return None
        return np.column_stack((c, r)).astype(np.int32).reshape(-1, 1, 2)

    def xy_to_bev_poly(left_xy: np.ndarray, right_xy: np.ndarray):
        """Convert (N,2) left/right vehicle (x,y) to BEV pixel polygon for fillPoly."""
        if left_xy.size == 0 or right_xy.size == 0:
            return None
        rows_l, cols_l = _vehicle_xy_to_bev_px(
            left_xy[:, 0], left_xy[:, 1], bev_h, bev_w, lookahead_m, half_width_m
        )
        rows_r, cols_r = _vehicle_xy_to_bev_px(
            right_xy[:, 0], right_xy[:, 1], bev_h, bev_w, lookahead_m, half_width_m
        )
        mask_l = (rows_l >= 0) & (rows_l < bev_h) & (cols_l >= 0) & (cols_l < bev_w)
        mask_r = (rows_r >= 0) & (rows_r < bev_h) & (cols_r >= 0) & (cols_r < bev_w)
        if not (np.any(mask_l) and np.any(mask_r)):
            return None
        left_pts = np.column_stack((cols_l[mask_l], rows_l[mask_l]))
        right_pts = np.column_stack((cols_r[mask_r], rows_r[mask_r]))
        return np.vstack([left_pts, right_pts[::-1]]).astype(np.int32)

    bev_canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    use_road_form = LANE_OVERLAY_ROAD_FORM
    strip_half_width = LANE_OVERLAY_STRIP_HALF_WIDTH_M
    use_frame_width = bool(
        LANE_OVERLAY_USE_FRAME_WIDTH and frame_width_m is not None and frame_width_m > 0
    )
    effective_strip_half = (float(frame_width_m) / 2.0) if use_frame_width else strip_half_width

    center = center_coeffs
    if center is None and left_coeffs is not None and right_coeffs is not None:
        center = (np.asarray(left_coeffs) + np.asarray(right_coeffs)) / 2.0

    # ── โหมดสี่เหลี่ยมคางหมู: วัดความกว้างต้น–ปลาย แล้ววาดเป็น trapezoid (ความกว้างจากคำนวณ) ──
    use_trapezoid = bool(LANE_OVERLAY_TRAPEZOID)
    if use_trapezoid and draw_fill and left_coeffs is not None and right_coeffs is not None:
        w_min = LANE_FRAME_WIDTH_MIN_M
        w_max = LANE_FRAME_WIDTH_MAX_M
        y_l0 = float(np.polyval(left_coeffs, 0.0))
        y_r0 = float(np.polyval(right_coeffs, 0.0))
        y_l_far = float(np.polyval(left_coeffs, draw_to))
        y_r_far = float(np.polyval(right_coeffs, draw_to))
        width_near = abs(y_r0 - y_l0)
        width_far = abs(y_r_far - y_l_far)
        width_near = float(np.clip(width_near, w_min, w_max))
        width_far = float(np.clip(width_far, w_min, w_max))
        center_0 = (y_l0 + y_r0) / 2.0
        center_far = (y_l_far + y_r_far) / 2.0
        # สี่เหลี่ยมคางหมู: 4 จุด (ใกล้ซ้าย, ใกล้ขวา, ปลายขวา, ปลายซ้าย)
        left_xy_trap = np.array([[0.0, center_0 - width_near / 2.0], [draw_to, center_far - width_far / 2.0]])
        right_xy_trap = np.array([[0.0, center_0 + width_near / 2.0], [draw_to, center_far + width_far / 2.0]])
        poly_pts = xy_to_bev_poly(left_xy_trap, right_xy_trap)
        if poly_pts is not None:
            cv2.fillPoly(bev_canvas, [poly_pts], fill_bgr)
        # เส้นขอบซ้าย/ขวา (เส้นตรงต้น–ปลาย) + เส้นกลางสี่เหลี่ยมคางหมู
        for xy_side in (left_xy_trap, right_xy_trap):
            pts = to_bev_pts_xy(xy_side[:, 0], xy_side[:, 1])
            if pts is not None:
                cv2.polylines(bev_canvas, [pts], False, green_bgr, LANE_OVERLAY_BOUNDARY_THICKNESS, cv2.LINE_AA)
        center_xy_trap = np.array([[0.0, center_0], [draw_to, center_far]])
        pts_c = to_bev_pts_xy(center_xy_trap[:, 0], center_xy_trap[:, 1])
        if pts_c is not None:
            cv2.polylines(bev_canvas, [pts_c], False, green_bgr, LANE_OVERLAY_CENTER_THICKNESS, cv2.LINE_AA)

    # Optional: build straight near + curved far segments for fill and center
    if use_straight_near:
        n_near = max(8, int(n_pts * straight_near_m / max(draw_to, 0.1)))
        n_far = n_pts
        x_near = np.linspace(0.0, straight_near_m, n_near)
        x_far = np.linspace(straight_near_m, draw_to, n_far)
        hw = effective_strip_half
        # Straight near: center y=0, left = -hw, right = +hw
        left_near_xy = np.column_stack((x_near, np.full_like(x_near, -hw)))
        right_near_xy = np.column_stack((x_near, np.full_like(x_near, hw)))

    # 1) Fill: road-form / frame-width (center + strip) หรือ แนบขอบ (left/right coeffs) — ข้ามถ้าใช้ trapezoid
    use_center_strip = (use_road_form or use_frame_width) and center is not None and len(center) >= 1
    if use_center_strip and draw_fill and not use_trapezoid:
        if use_straight_near:
            left_far_xy, right_far_xy = _center_to_strip_boundaries(center, x_far, effective_strip_half)
            left_xy = np.vstack([left_near_xy, left_far_xy])
            right_xy = np.vstack([right_near_xy, right_far_xy])
        else:
            left_xy, right_xy = _center_to_strip_boundaries(center, x_m, effective_strip_half)
        poly_pts = xy_to_bev_poly(left_xy, right_xy)
        if poly_pts is not None:
            cv2.fillPoly(bev_canvas, [poly_pts], fill_bgr)
    elif draw_fill and left_coeffs is not None and right_coeffs is not None and not use_frame_width and not use_trapezoid:
        margin_m = LANE_OVERLAY_DISPLAY_MARGIN_M
        min_w_m = LANE_OVERLAY_MIN_DISPLAY_WIDTH_M
        max_w_m = LANE_OVERLAY_MAX_WIDTH_DRAW_M
        y_l = np.polyval(left_coeffs, x_m)
        y_r = np.polyval(right_coeffs, x_m)
        width_at_ego = abs(float(y_r[0]) - float(y_l[0]))
        if width_at_ego <= max_w_m:
            if width_at_ego < min_w_m:
                yc = (y_l + y_r) / 2.0
                half = min_w_m / 2.0
                y_l_draw = yc - half
                y_r_draw = yc + half
            else:
                y_l_draw = y_l - margin_m
                y_r_draw = y_r + margin_m
            if use_straight_near:
                y_l_near = np.full_like(x_near, -strip_half_width)
                y_r_near = np.full_like(x_near, strip_half_width)
                y_l_far = np.polyval(left_coeffs, x_far) - margin_m
                y_r_far = np.polyval(right_coeffs, x_far) + margin_m
                y_l_draw = np.concatenate([y_l_near, y_l_far])
                y_r_draw = np.concatenate([y_r_near, y_r_far])
                x_fill = np.concatenate([x_near, x_far])
                rows_l, cols_l = _vehicle_xy_to_bev_px(x_fill, y_l_draw, bev_h, bev_w, lookahead_m, half_width_m)
                rows_r, cols_r = _vehicle_xy_to_bev_px(x_fill, y_r_draw, bev_h, bev_w, lookahead_m, half_width_m)
            else:
                rows_l, cols_l = _vehicle_xy_to_bev_px(x_m, y_l_draw, bev_h, bev_w, lookahead_m, half_width_m)
                rows_r, cols_r = _vehicle_xy_to_bev_px(x_m, y_r_draw, bev_h, bev_w, lookahead_m, half_width_m)
            mask_l = (rows_l >= 0) & (rows_l < bev_h) & (cols_l >= 0) & (cols_l < bev_w)
            mask_r = (rows_r >= 0) & (rows_r < bev_h) & (cols_r >= 0) & (cols_r < bev_w)
            if np.any(mask_l) and np.any(mask_r):
                left_pts = np.column_stack((cols_l[mask_l], rows_l[mask_l]))
                right_pts = np.column_stack((cols_r[mask_r], rows_r[mask_r]))
                poly_pts = np.vstack([left_pts, right_pts[::-1]]).astype(np.int32)
                cv2.fillPoly(bev_canvas, [poly_pts], fill_bgr)

    # 2) Centerline: straight near แล้วต่อโค้ง (ข้ามถ้าใช้ trapezoid — มีเส้นกลางในบล็อก trapezoid แล้ว)
    if center is not None and not use_trapezoid:
        if use_straight_near:
            y_center_near = np.zeros_like(x_near)
            y_center_far = np.polyval(center, x_far)
            x_center = np.concatenate([x_near, x_far])
            y_center = np.concatenate([y_center_near, y_center_far])
            pts_c = to_bev_pts_xy(x_center, y_center)
        else:
            pts_c = to_bev_pts(center)
        if pts_c is not None:
            cv2.polylines(bev_canvas, [pts_c], False, green_bgr, LANE_OVERLAY_CENTER_THICKNESS, cv2.LINE_AA)

    # 3) Left/right boundary lines: straight near + curved far หรือจาก coeffs
    if use_center_strip:
        if use_straight_near:
            left_far_xy, right_far_xy = _center_to_strip_boundaries(center, x_far, effective_strip_half)
            left_xy = np.vstack([left_near_xy, left_far_xy])
            right_xy = np.vstack([right_near_xy, right_far_xy])
        else:
            left_xy, right_xy = _center_to_strip_boundaries(center, x_m, effective_strip_half)
        if left_xy.size >= 4:
            rows_l, cols_l = _vehicle_xy_to_bev_px(
                left_xy[:, 0], left_xy[:, 1], bev_h, bev_w, lookahead_m, half_width_m
            )
            mask_l = (rows_l >= 0) & (rows_l < bev_h) & (cols_l >= 0) & (cols_l < bev_w)
            if np.sum(mask_l) >= 2:
                pts_l = np.column_stack((cols_l[mask_l], rows_l[mask_l])).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(bev_canvas, [pts_l], False, green_bgr, LANE_OVERLAY_BOUNDARY_THICKNESS, cv2.LINE_AA)
        if right_xy.size >= 4:
            rows_r, cols_r = _vehicle_xy_to_bev_px(
                right_xy[:, 0], right_xy[:, 1], bev_h, bev_w, lookahead_m, half_width_m
            )
            mask_r = (rows_r >= 0) & (rows_r < bev_h) & (cols_r >= 0) & (cols_r < bev_w)
            if np.sum(mask_r) >= 2:
                pts_r = np.column_stack((cols_r[mask_r], rows_r[mask_r])).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(bev_canvas, [pts_r], False, green_bgr, LANE_OVERLAY_BOUNDARY_THICKNESS, cv2.LINE_AA)
    else:
        if left_coeffs is not None:
            if use_straight_near:
                y_l_near = np.full_like(x_near, -strip_half_width)
                y_l_far = np.polyval(left_coeffs, x_far)
                x_l = np.concatenate([x_near, x_far])
                y_l = np.concatenate([y_l_near, y_l_far])
                pts_l = to_bev_pts_xy(x_l, y_l)
            else:
                pts_l = to_bev_pts(left_coeffs)
            if pts_l is not None:
                cv2.polylines(bev_canvas, [pts_l], False, green_bgr, LANE_OVERLAY_BOUNDARY_THICKNESS, cv2.LINE_AA)
        if right_coeffs is not None:
            if use_straight_near:
                y_r_near = np.full_like(x_near, strip_half_width)
                y_r_far = np.polyval(right_coeffs, x_far)
                x_r = np.concatenate([x_near, x_far])
                y_r = np.concatenate([y_r_near, y_r_far])
                pts_r = to_bev_pts_xy(x_r, y_r)
            else:
                pts_r = to_bev_pts(right_coeffs)
            if pts_r is not None:
                cv2.polylines(bev_canvas, [pts_r], False, green_bgr, LANE_OVERLAY_BOUNDARY_THICKNESS, cv2.LINE_AA)

    # Warp BEV back to perspective and blend
    fill_persp = cv2.warpPerspective(bev_canvas, M_inv, (cam_w, cam_h))
    fill_mask = (fill_persp.sum(axis=2) > 0).astype(np.uint8)
    blended = overlay_bgr.copy().astype(np.float32)
    blended[fill_mask > 0] = (
        alpha * fill_persp[fill_mask > 0].astype(np.float32)
        + (1 - alpha) * overlay_bgr[fill_mask > 0].astype(np.float32)
    )
    return cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGR2RGB)


def _draw_simple_lane_markings(
    rgb: np.ndarray,
    left_coeffs: Optional[np.ndarray],
    right_coeffs: Optional[np.ndarray],
    M_inv: np.ndarray,
    bev_h: int,
    bev_w: int,
    cam_h: int,
    cam_w: int,
    lookahead_m: float,
    half_width_m: float,
) -> np.ndarray:
    """
    Deterministic lane marking overlay from polynomial coefficients.
    Renders white lines directly on road surface.
    """
    if left_coeffs is None and right_coeffs is None:
        return rgb

    img = rgb.copy()

    def draw_lane_from_poly(coeffs, color=(255, 255, 255), thickness=3):
        if coeffs is None:
            return

        # Sample points along polynomial in vehicle frame
        x_m = np.linspace(0, lookahead_m, 100)
        y_m = np.polyval(coeffs, x_m)

        # Convert to BEV pixels
        rows = bev_h - 1 - (x_m / lookahead_m * (bev_h - 1))
        cols = (bev_w - 1) / 2.0 + (y_m / (2.0 * half_width_m) * (bev_w - 1))

        # BEV to camera transform
        pts_bev = np.stack([cols, rows, np.ones_like(cols)], axis=1)
        pts_cam = (M_inv @ pts_bev.T).T
        pts_cam = pts_cam[:, :2] / (pts_cam[:, 2:3] + 1e-8)

        # Filter valid points
        valid = (
            (pts_cam[:, 0] >= 0) & (pts_cam[:, 0] < cam_w) &
            (pts_cam[:, 1] >= 0) & (pts_cam[:, 1] < cam_h)
        )

        if not np.any(valid):
            return

        pts = pts_cam[valid].astype(np.int32)
        if len(pts) < 2:
            return

        # Draw white line
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)

    # Draw left and right lane boundaries
    draw_lane_from_poly(left_coeffs)
    draw_lane_from_poly(right_coeffs)

    return img


def _draw_lane_overlay_direct(
    rgb: np.ndarray,
    left_coeffs: Optional[np.ndarray],
    right_coeffs: Optional[np.ndarray],
    M_inv: np.ndarray,
    bev_h: int, bev_w: int,
    cam_h: int, cam_w: int,
    lookahead_m: float = LOOKAHEAD_M,
    half_width_m: float = HALF_WIDTH_M,
    fill_color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
    center_coeffs: Optional[np.ndarray] = None,
    max_x_m: Optional[float] = None,
    bev_binary: Optional[np.ndarray] = None,
    left_px: Optional[np.ndarray] = None,
    right_px: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Direct perspective lane overlay - renders polygon directly in camera view.
    No BEV intermediate step, resulting in perspective-correct overlay that sticks to road.

    Method:
    1. Sample points along left/right lane polynomials in vehicle frame
    2. Convert vehicle (x,y) → BEV pixels
    3. Use M_inv to warp BEV pixels → camera pixels
    4. Draw filled polygon directly in camera view

    Args:
        rgb: Original camera image (RGB)
        left_coeffs: Left lane polynomial coefficients
        right_coeffs: Right lane polynomial coefficients
        M_inv: BEV to camera inverse transform matrix
        bev_h, bev_w: BEV dimensions
        cam_h, cam_w: Camera dimensions
        lookahead_m: Maximum distance to draw
        half_width_m: Half width of BEV view
        fill_color: RGB color for lane fill
        alpha: Transparency (0=transparent, 1=opaque)
        center_coeffs: Optional center line coefficients
        max_x_m: Optional max distance cutoff

    Returns:
        RGB image with lane overlay
    """
    if left_coeffs is None or right_coeffs is None:
        return rgb

    overlay_bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    fill_bgr = (fill_color[2], fill_color[1], fill_color[0])  # RGB to BGR

    # Determine draw distance
    draw_to = min(lookahead_m, max_x_m) if max_x_m is not None else lookahead_m

    # Sample dense points along lanes for smooth curves
    n_pts = 150  # Dense sampling for perspective-correct smooth rendering
    x_m = np.linspace(0, draw_to, n_pts)

    # Evaluate polynomials in vehicle frame
    y_left = np.polyval(left_coeffs, x_m)
    y_right = np.polyval(right_coeffs, x_m)

    # Convert vehicle coordinates to BEV pixels
    rows_l, cols_l = _vehicle_xy_to_bev_px(
        x_m, y_left, bev_h, bev_w, lookahead_m, half_width_m
    )
    rows_r, cols_r = _vehicle_xy_to_bev_px(
        x_m, y_right, bev_h, bev_w, lookahead_m, half_width_m
    )

    # Create homogeneous coordinates for BEV points
    bev_pts_left = np.stack([cols_l, rows_l, np.ones_like(cols_l)], axis=1)
    bev_pts_right = np.stack([cols_r, rows_r, np.ones_like(cols_r)], axis=1)

    # Warp to camera space using M_inv (BEV → camera perspective)
    cam_pts_left = (M_inv @ bev_pts_left.T).T
    cam_pts_right = (M_inv @ bev_pts_right.T).T

    # Normalize homogeneous coordinates
    cam_pts_left = cam_pts_left[:, :2] / (cam_pts_left[:, 2:3] + 1e-8)
    cam_pts_right = cam_pts_right[:, :2] / (cam_pts_right[:, 2:3] + 1e-8)

    # Filter points within camera bounds
    mask_l = (
        (cam_pts_left[:, 0] >= 0) & (cam_pts_left[:, 0] < cam_w) &
        (cam_pts_left[:, 1] >= 0) & (cam_pts_left[:, 1] < cam_h)
    )
    mask_r = (
        (cam_pts_right[:, 0] >= 0) & (cam_pts_right[:, 0] < cam_w) &
        (cam_pts_right[:, 1] >= 0) & (cam_pts_right[:, 1] < cam_h)
    )

    if not (np.any(mask_l) and np.any(mask_r)):
        return rgb

    # Build polygon: left points + reversed right points
    left_pts = cam_pts_left[mask_l].astype(np.int32)
    right_pts = cam_pts_right[mask_r].astype(np.int32)

    # Draw lane markings that match actual road markings
    blended = overlay_bgr.copy()
    white_bgr = (255, 255, 255)  # White for lane markings

    # Helper function to draw dashed line (like real lane markings)
    def draw_dashed_line(img, pts, color, thickness=4, dash_length=20, gap_length=10):
        """Draw dashed line to match real road lane markings"""
        if len(pts) < 2:
            return
        total_length = 0
        segments = []
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            dist = np.linalg.norm(p2 - p1)
            segments.append((p1, p2, dist))
            total_length += dist

        draw_dash = True
        for p1, p2, seg_dist in segments:
            if seg_dist < 1:
                continue
            direction = (p2 - p1) / seg_dist
            local_pos = 0
            while local_pos < seg_dist:
                if draw_dash:
                    end_pos = min(local_pos + dash_length, seg_dist)
                    start_pt = (p1 + direction * local_pos).astype(np.int32)
                    end_pt = (p1 + direction * end_pos).astype(np.int32)
                    cv2.line(img, tuple(start_pt), tuple(end_pt), color, thickness, cv2.LINE_AA)
                    local_pos = end_pos
                    draw_dash = False
                else:
                    local_pos += gap_length
                    draw_dash = True

    # Draw left lane marking (white dashed line)
    if len(left_pts) >= 2:
        draw_dashed_line(blended, left_pts, white_bgr, thickness=3, dash_length=30, gap_length=15)

    # Draw right lane marking (white dashed line)
    if len(right_pts) >= 2:
        draw_dashed_line(blended, right_pts, white_bgr, thickness=3, dash_length=30, gap_length=15)

    # Optional: Draw semi-transparent lane area (ช่องใหญ่) with very low alpha
    if len(left_pts) >= 2 and len(right_pts) >= 2:
        polygon = np.vstack([left_pts, right_pts[::-1]])
        fill_canvas = np.zeros_like(overlay_bgr)
        cv2.fillPoly(fill_canvas, [polygon], fill_bgr, lineType=cv2.LINE_AA)
        fill_mask = (fill_canvas.sum(axis=2) > 0).astype(np.uint8)
        blended = blended.astype(np.float32)
        # Very low alpha (0.15) so lane area is barely visible, lane markings are primary
        lane_area_alpha = 0.15
        blended[fill_mask > 0] = (
            lane_area_alpha * fill_canvas[fill_mask > 0].astype(np.float32) +
            (1 - lane_area_alpha) * blended[fill_mask > 0].astype(np.float32)
        )
        blended = blended.astype(np.uint8)

    # Optional: Draw center line if provided
    if center_coeffs is not None:
        y_center = np.polyval(center_coeffs, x_m)
        rows_c, cols_c = _vehicle_xy_to_bev_px(
            x_m, y_center, bev_h, bev_w, lookahead_m, half_width_m
        )
        bev_pts_center = np.stack([cols_c, rows_c, np.ones_like(cols_c)], axis=1)
        cam_pts_center = (M_inv @ bev_pts_center.T).T
        cam_pts_center = cam_pts_center[:, :2] / (cam_pts_center[:, 2:3] + 1e-8)
        mask_c = (
            (cam_pts_center[:, 0] >= 0) & (cam_pts_center[:, 0] < cam_w) &
            (cam_pts_center[:, 1] >= 0) & (cam_pts_center[:, 1] < cam_h)
        )
        if np.any(mask_c):
            center_pts = cam_pts_center[mask_c].astype(np.int32)
            if len(center_pts) >= 2:
                cv2.polylines(blended, [center_pts.reshape(-1, 1, 2)], False, (0, 255, 0), 3, cv2.LINE_AA)

    return cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

class LaneTrajectoryPipeline:
    """
    การจับเส้นแบบ phase (P1→P2→P3→P4→P5):

    P1 Lane mask:     RGB → binary mask + confidence
    P2 BEV + quality: mask → BEV warp, fill gaps, mask_quality
    P3 Boundaries:    BEV → left/right/center coeffs (sliding window or ego tracker)
    P4 Smooth center: center_coeffs_raw + Kalman → center_coeffs, geometry_valid
    P5 Trajectory:    center_coeffs → (x_ref, y_ref, cte, heading, curvature, v_ref)
    P6 Visualization: BEV window vis + lane overlay

    Returns TrajectoryOutput with:
      .bev_binary     — BEV warped mask (Image 1 middle)
      .bev_window_vis — colored sliding windows (Image 1 bottom)
      .lane_overlay   — green filled lane on original frame (Image 2 bottom)
    """

    def __init__(
        self,
        detector=None,
        use_classical_detector: bool = False,
        cam_w: int = 640,
        cam_h: int = 360,
        v_nominal_ms: float = 25.0 / 3.6,
        lookahead_m: float = LOOKAHEAD_M,
        ds_m: float = DS_RESAMPLE_M,
        use_kalman: bool = True,
        kalman_q: float = KALMAN_Q,
        kalman_r: float = KALMAN_R,
        lane_confidence_waypoint_only_threshold: float = 0.4,
        use_ego_lane_tracker: bool = True,
        # BEV trapezoid — top ต่ำมาก = ใช้แถวบนเกือบทั้งภาพ = เห็นเลนเยอะมาก
        bev_top_ratio: float = 0.35,
        bev_bot_ratio: float = 0.98,
        bev_top_margin: float = 0.32,
        bev_bot_margin: float = 0.10,
        lightweight_vis: bool = False,
    ):
        # Initialize detector (UNet or CARLA waypoint-based)
        if detector is not None:
            self.detector = detector
        elif use_classical_detector:
            # Use CARLA waypoint-based lane detection (ground truth from simulator, no AI model)
            from .lane_detector import LaneDetector
            self.detector = LaneDetector(model_path=None, use_carla=True)
            logger.info("Using CARLA Waypoint Lane Detector (ground truth, no AI model)")
        else:
            # Default: use UNet detector
            self.detector = detector  # Will be set by caller
            logger.warning("No detector provided, must be set externally")

        self.use_classical = use_classical_detector
        self.lightweight_vis = lightweight_vis
        self.cam_w       = cam_w
        self.cam_h       = cam_h
        self.v_nominal_ms = v_nominal_ms
        self.lookahead_m = lookahead_m
        self.ds_m        = ds_m
        self._kalman     = KalmanPoly(q=kalman_q, r=kalman_r) if use_kalman else None
        self._prev_curvature = 0.0
        self._prev_center_coeffs: Optional[np.ndarray] = None
        self.lane_conf_threshold = lane_confidence_waypoint_only_threshold
        self._road_top_ratio = ROAD_TOP_RATIO
        self._use_ego_tracker = use_ego_lane_tracker
        self._ego_tracker = None
        if use_ego_lane_tracker:
            try:
                from .ego_lane_tracker import EgoLaneTracker
                self._ego_tracker = EgoLaneTracker()
            except ImportError:
                self._use_ego_tracker = False
        self._curvature_smoother = CurvatureSmoother()
        self._lane_tracker = KalmanLaneTracker()
        self._prev_mask: Optional[np.ndarray] = None
        self._frames_without_center: int = 0
        self._prev_draw_left: Optional[np.ndarray] = None
        self._prev_draw_right: Optional[np.ndarray] = None
        self._prev_draw_center: Optional[np.ndarray] = None
        self._prev_frame_width: Optional[float] = None
        self.half_width_m = HALF_WIDTH_M
        # Pre-computed constants for fast Phase 1
        self._road_top_px = int(cam_h * self._road_top_ratio)
        self._conf_roi_top_px = int(cam_h * LANE_CONF_ROI_TOP_RATIO)
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Tiny empty BEV for compatibility (no actual BEV is computed)
        self._empty_bev = np.zeros((cam_h, cam_w), dtype=np.uint8)
        # Ground projection storage (set by Phase 2)
        self._ground_x = np.array([])
        self._ground_y = np.array([])
        self._img_rows = np.array([])
        self._img_cols = np.array([])
        # Camera intrinsics for pinhole projection
        try:
            from config import CAM_FOV_DEG
            self._cam_fov = CAM_FOV_DEG
        except ImportError:
            self._cam_fov = 90.0
        self._cam_f = cam_w / (2.0 * np.tan(np.radians(self._cam_fov / 2.0)))
        self._cam_cx = cam_w / 2.0
        self._cam_cy = cam_h / 2.0
        # Camera extrinsics: pitch=-8°, height=2.0m, x_offset=1.5m
        self._cam_pitch_rad = np.radians(-8.0)
        self._cam_height = 2.0
        self._cam_x_offset = 1.5
        # Pre-compute rotation matrix: vehicle frame → camera frame
        # Vehicle: X=forward, Y=right, Z=up
        # Camera: X_cam=right, Y_cam=down, Z_cam=forward
        sp = np.sin(self._cam_pitch_rad)
        cp = np.cos(self._cam_pitch_rad)
        R_base = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
        R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
        self._R_v2c = R_pitch @ R_base       # vehicle → camera
        self._R_c2v = self._R_v2c.T          # camera → vehicle

    def reset(self) -> None:
        if self._kalman is not None:
            self._kalman.reset()
        if self._ego_tracker is not None:
            self._ego_tracker.reset()
        self._lane_tracker.reset()
        self._temporal_buffer.reset()
        self._curvature_smoother.reset()
        self._prev_curvature = 0.0
        self._prev_center_coeffs = None
        self._prev_mask = None
        self._frames_without_center = 0
        self._prev_draw_left = None
        self._prev_draw_right = None
        self._prev_draw_center = None
        self._prev_frame_width = None

    # ── Image → Ground projection (rotation matrix + ray intersection) ───
    def _image_to_ground(self, rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project image pixels (row, col) to vehicle ground plane (x_forward, y_lateral).

        Uses rotation matrix for exact camera geometry.
        Ray from camera through pixel intersects ground plane (Z=0).

        Returns: (x_ground, y_ground) in vehicle frame [meters]
          x = forward distance, y = lateral (positive = right)
        """
        f = self._cam_f
        cx, cy = self._cam_cx, self._cam_cy
        h_cam = self._cam_height
        R_c2v = self._R_c2v  # camera → vehicle rotation

        n = len(rows)
        x_ground = np.full(n, -1.0)
        y_ground = np.full(n, 0.0)

        # Normalized image coords → ray direction in camera frame
        nx = (cols - cx) / f
        ny = (rows - cy) / f
        # Ray in camera frame: (nx, ny, 1.0) for each pixel
        # Transform to vehicle frame: d_veh = R_c2v @ d_cam
        d_veh_x = R_c2v[0, 0] * nx + R_c2v[0, 1] * ny + R_c2v[0, 2]
        d_veh_y = R_c2v[1, 0] * nx + R_c2v[1, 1] * ny + R_c2v[1, 2]
        d_veh_z = R_c2v[2, 0] * nx + R_c2v[2, 1] * ny + R_c2v[2, 2]

        # Ground intersection: camera_pos + t * d_veh, z=0
        # Camera is at (cam_x_offset, 0, h_cam) in vehicle frame
        # h_cam + t * d_veh_z = 0  =>  t = -h_cam / d_veh_z
        valid = d_veh_z < -0.001  # ray must point downward
        t = np.where(valid, -h_cam / d_veh_z, 0.0)

        # Ground point = camera_pos + t * d_veh
        x_ground = np.where(valid, self._cam_x_offset + t * d_veh_x, -1.0)
        y_ground = np.where(valid, t * d_veh_y, 0.0)

        return x_ground, y_ground

    def _ground_to_image(self, x_ground: np.ndarray, y_ground: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project vehicle ground plane (x_forward, y_lateral) back to image pixels (row, col).

        Uses rotation matrix for exact inverse of _image_to_ground.
        Camera is at (cam_x_offset, 0, h_cam) in vehicle frame.
        """
        f = self._cam_f
        cx, cy = self._cam_cx, self._cam_cy
        h_cam = self._cam_height
        R_v2c = self._R_v2c  # vehicle → camera rotation

        # Vector from camera to ground point in vehicle frame
        # Camera is at (cam_x_offset, 0, h_cam), ground point at (x, y, 0)
        vx = x_ground - self._cam_x_offset
        vy = y_ground
        vz = np.full_like(x_ground, -h_cam)

        # Transform to camera frame
        X_cam = R_v2c[0, 0] * vx + R_v2c[0, 1] * vy + R_v2c[0, 2] * vz
        Y_cam = R_v2c[1, 0] * vx + R_v2c[1, 1] * vy + R_v2c[1, 2] * vz
        Z_cam = R_v2c[2, 0] * vx + R_v2c[2, 1] * vy + R_v2c[2, 2] * vz

        valid = Z_cam > 0.1
        cols = np.where(valid, cx + f * X_cam / Z_cam, -1.0)
        rows = np.where(valid, cy + f * Y_cam / Z_cam, -1.0)

        return rows, cols

    # ── Phase 1: Lane mask (pixel) ─────────────────────────────────────────
    def _phase1_lane_mask(self, rgb: np.ndarray, world=None, vehicle=None) -> Tuple[np.ndarray, float]:
        """P1: RGB → binary mask + confidence. Optimized for speed."""
        img = cv2.resize(rgb, (self.cam_w, self.cam_h))
        # use_model_priority=True: DSUNet is primary, CARLA is fallback
        mask_uint8, _, _ = self.detector.detect_lanes(
            img, world=world, vehicle=vehicle, use_model_priority=True,
        )
        mask = (mask_uint8 > 0).astype(np.uint8)
        mask[: self._road_top_px, :] = 0
        # Minimal morphology with pre-computed kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close)
        # Fast temporal smoothing: bitwise OR with previous mask (no float math)
        if self._prev_mask is not None and self._prev_mask.shape == mask.shape:
            mask = mask | self._prev_mask
        self._prev_mask = mask
        roi = mask[self._conf_roi_top_px :, :]
        conf = min(1.0, np.count_nonzero(roi) / max(roi.size * 0.05, 1))
        return mask, float(conf)

    # ── Phase 2: Image-space lane extraction (no BEV) ──────────────────────
    def _phase2_bev_and_quality(self, mask: np.ndarray) -> Tuple[np.ndarray, str]:
        """P2: mask → project lane pixels to ground plane. No BEV image created."""
        ys, xs = np.where(mask > 0)

        if len(ys) < MIN_PX_TOTAL:
            self._ground_x = np.array([])
            self._ground_y = np.array([])
            self._img_rows = np.array([])
            self._img_cols = np.array([])
            return self._empty_bev, "invalid"

        # Project image pixels to vehicle ground plane
        ground_x, ground_y = self._image_to_ground(ys.astype(np.float64), xs.astype(np.float64))

        # Filter: keep only points in valid range
        valid = (ground_x > 0.5) & (ground_x < self.lookahead_m) & \
                (np.abs(ground_y) < HALF_WIDTH_M) & np.isfinite(ground_x) & np.isfinite(ground_y)
        with np.errstate(divide='ignore', invalid='ignore'):
            angle_ratio = np.abs(ground_y) / np.maximum(ground_x, 0.1)
        valid = valid & (angle_ratio < 1.5)

        n_valid = np.count_nonzero(valid)

        self._ground_x = ground_x[valid] if n_valid > 0 else np.array([])
        self._ground_y = ground_y[valid] if n_valid > 0 else np.array([])
        self._img_rows = ys[valid] if n_valid > 0 else np.array([])
        self._img_cols = xs[valid] if n_valid > 0 else np.array([])

        quality = "valid" if n_valid >= MIN_PX_TOTAL else ("degraded" if n_valid >= 50 else "invalid")
        return self._empty_bev, quality

    # ── Phase 3: Boundaries + centerline (image-space, no BEV) ──────────
    def _phase3_boundaries_and_center(
        self, bev_bin: np.ndarray
    ) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
        Optional[np.ndarray], Optional[np.ndarray], List, List, bool, bool,
        str, str,  # p2_case, p3_case
    ]:
        """P3: Use ground-projected points from P2 to find left/right lanes and fit polynomials.

        Instead of sliding window on BEV, we:
        1. Split ground points into left (y < 0) and right (y > 0) groups
        2. Cluster each side to find the dominant lane line
        3. Fit polynomial in vehicle frame: y = f(x)
        4. Return center_coeffs = (left + right) / 2
        """
        center_raw, c_l, c_r = None, None, None
        left_px, right_px = None, None
        raw_left_wins, raw_right_wins = [], []
        filt_left_wins, filt_right_wins = [], []
        tracked, used_completion = False, False
        p2_case, p3_case = "none", "none"

        gx = self._ground_x
        gy = self._ground_y

        if len(gx) < MIN_POINTS_POLY * 2:
            return (center_raw, c_l, c_r, left_px, right_px,
                    raw_left_wins, raw_right_wins, filt_left_wins, filt_right_wins,
                    tracked, used_completion, p2_case, p3_case)

        # Split into left (y < 0) and right (y > 0)
        # No absolute Y filter needed — detect_lanes_carla already filters to ego lane.
        # On curving roads, ego lane edges can have large lateral offset at distance.
        gx_filt = gx
        gy_filt = gy
        img_rows_filt = self._img_rows
        img_cols_filt = self._img_cols

        left_mask = gy_filt < -0.3   # at least 30cm to the left
        right_mask = gy_filt > 0.3   # at least 30cm to the right

        left_x, left_y = gx_filt[left_mask], gy_filt[left_mask]
        right_x, right_y = gx_filt[right_mask], gy_filt[right_mask]

        # Fit polynomials: y = a*x^2 + b*x + c
        def fit_lane(x_pts, y_pts, label):
            if len(x_pts) < MIN_POINTS_POLY:
                return None
            try:
                coeffs = np.polyfit(x_pts, y_pts, POLY_ORDER)
                # Sanity check: curvature shouldn't be extreme
                if abs(coeffs[0]) > 0.1:  # extreme curvature
                    return None
                return coeffs
            except (np.linalg.LinAlgError, ValueError):
                return None

        c_l = fit_lane(left_x, left_y, "left")
        c_r = fit_lane(right_x, right_y, "right")

        # Build pixel arrays for visualization (image-space coords)
        if c_l is not None and len(img_rows_filt[left_mask]) > 0:
            left_px = np.column_stack((img_rows_filt[left_mask], img_cols_filt[left_mask]))
        if c_r is not None and len(img_rows_filt[right_mask]) > 0:
            right_px = np.column_stack((img_rows_filt[right_mask], img_cols_filt[right_mask]))

        # Determine center coefficients
        if c_l is not None and c_r is not None:
            center_raw = (c_l + c_r) / 2.0
            p2_case = "both"
            p3_case = "from_both"
        elif c_r is not None:
            # Only right lane: estimate center by shifting left
            center_raw = c_r.copy()
            center_raw[-1] -= 1.8  # shift ~1.8m left (half lane width)
            c_l = c_r.copy()
            c_l[-1] -= 3.6  # mirror at lane width
            p2_case = "mirror_left"
            p3_case = "from_both"
        elif c_l is not None:
            # Only left lane: estimate center by shifting right
            center_raw = c_l.copy()
            center_raw[-1] += 1.8
            c_r = c_l.copy()
            c_r[-1] += 3.6
            p2_case = "mirror_right"
            p3_case = "from_both"
        else:
            p3_case = "none"

        return (center_raw, c_l, c_r, left_px, right_px,
                raw_left_wins, raw_right_wins, filt_left_wins, filt_right_wins,
                tracked, used_completion, p2_case, p3_case)

    # ── Phase 4: Smooth center (Kalman) + geometry_valid ─────────────────
    def _phase4_smooth_center(
        self, center_coeffs_raw: Optional[np.ndarray], mask_quality: str
    ) -> Tuple[Optional[np.ndarray], bool]:
        """P4: center_coeffs_raw + Kalman → center_coeffs, geometry_valid. รอยต่อ: reset/re-init ไม่ค้างไม่เพี้ยน."""
        center_coeffs = None
        geometry_valid = False
        N = MAX_FRAMES_STALE_CENTER
        if self._kalman is not None:
            if center_coeffs_raw is not None:
                if self._frames_without_center >= N:
                    self._kalman.reset()
                    center_coeffs = self._kalman.update(center_coeffs_raw)
                    self._prev_center_coeffs = center_coeffs.copy()
                else:
                    center_coeffs = self._kalman.update(center_coeffs_raw)
                    self._prev_center_coeffs = center_coeffs.copy()
                geometry_valid = True
                self._frames_without_center = 0
            else:
                self._frames_without_center += 1
                if self._frames_without_center >= N:
                    self._prev_center_coeffs = None
                    self._kalman.reset()
                    center_coeffs = None
                else:
                    center_coeffs = self._kalman.predict()
                    self._prev_center_coeffs = center_coeffs.copy() if center_coeffs is not None else None
        else:
            if center_coeffs_raw is not None:
                center_coeffs = center_coeffs_raw.copy()
                self._prev_center_coeffs = center_coeffs.copy()
                geometry_valid = True
                self._frames_without_center = 0
            else:
                self._frames_without_center += 1
                if self._frames_without_center >= N:
                    self._prev_center_coeffs = None
                center_coeffs = None
        if mask_quality == "invalid":
            geometry_valid = False
        return center_coeffs, geometry_valid

    # ── Phase 5: Trajectory + state (cte, heading, curvature) ─────────────
    def _phase5_trajectory_and_state(
        self, center_coeffs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """P5: center_coeffs → x_ref, y_ref, yaw_ref, curv_ref, v_ref, cte, heading_err, curvature, v_at_ego."""
        if center_coeffs is not None:
            x_ref, y_ref, yaw_ref, curv_ref = generate_trajectory(
                center_coeffs, ds_m=self.ds_m, lookahead_m=self.lookahead_m,
                curvature_prev=self._prev_curvature,
            )
            self._prev_curvature = float(curv_ref[-1]) if len(curv_ref) else 0.0
        else:
            x_ref = np.array([0.0, self.lookahead_m * 0.5])
            y_ref = np.array([0.0, 0.0])
            yaw_ref = np.array([0.0, 0.0])
            curv_ref = np.array([0.0, 0.0])
        k_fac = curvature_to_speed_factor(curv_ref)
        v_ref = self.v_nominal_ms * k_fac
        cte, heading_err, curvature_raw, v_at_ego = mpc_reference_output(
            x_ref, y_ref, yaw_ref, v_ref, self.v_nominal_ms,
        )
        curvature = self._curvature_smoother.update(curvature_raw)
        self._prev_curvature = curvature
        return x_ref, y_ref, yaw_ref, curv_ref, v_ref, cte, heading_err, curvature, v_at_ego

    # ── Phase 6: Visualization (vehicle-frame projection) ──────────────────
    def _phase6_visualization(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        bev_bin: np.ndarray,
        left_px: Optional[np.ndarray],
        right_px: Optional[np.ndarray],
        left_wins: List,
        right_wins: List,
        c_l_fit: Optional[np.ndarray],
        c_r_fit: Optional[np.ndarray],
        center_coeffs: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
        """P6: Lane overlay using vehicle-frame coefficients projected to image space."""
        a = BEV_DISPLAY_EMA_ALPHA
        def blend_draw(curr: Optional[np.ndarray], prev: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if curr is None:
                return None
            curr = np.asarray(curr, dtype=np.float64)
            if prev is not None and prev.shape == curr.shape:
                return a * prev + (1.0 - a) * curr
            return curr.copy()

        draw_left = blend_draw(c_l_fit, self._prev_draw_left)
        draw_right = blend_draw(c_r_fit, self._prev_draw_right)
        draw_center = blend_draw(center_coeffs, self._prev_draw_center)
        if draw_left is not None:
            self._prev_draw_left = draw_left
        if draw_right is not None:
            self._prev_draw_right = draw_right
        if draw_center is not None:
            self._prev_draw_center = draw_center

        n_sample = 30 if self.lightweight_vis else 80

        def coeffs_to_image_pts(coeffs):
            """Convert vehicle-frame polynomial to image-space points."""
            if coeffs is None:
                return None
            x_pts = np.linspace(0.5, self.lookahead_m, n_sample)
            y_pts = np.polyval(coeffs, x_pts)
            rows, cols = self._ground_to_image(x_pts, y_pts)
            valid = (rows >= 0) & (rows < self.cam_h) & (cols >= 0) & (cols < self.cam_w)
            if np.count_nonzero(valid) < 2:
                return None
            return np.column_stack((cols[valid].astype(int), rows[valid].astype(int)))

        # No BEV visualization — raw image lane detection only
        bev_window_vis = None

        # ── Camera overlay: project curves to image ──
        rgb_resized = cv2.resize(rgb, (self.cam_w, self.cam_h))
        lane_overlay = rgb_resized.copy()

        left_img_pts = coeffs_to_image_pts(draw_left)
        right_img_pts = coeffs_to_image_pts(draw_right)
        center_img_pts = coeffs_to_image_pts(draw_center)

        # Fill lane area FIRST (so lines draw on top, not blended)
        if not self.lightweight_vis and left_img_pts is not None and right_img_pts is not None:
            poly = np.vstack([left_img_pts, right_img_pts[::-1]])
            fill = lane_overlay.copy()
            cv2.fillPoly(fill, [poly], (0, 180, 0))
            lane_overlay = cv2.addWeighted(lane_overlay, 0.75, fill, 0.25, 0)

        # Draw lines ON TOP of fill (with anti-aliasing)
        if left_img_pts is not None and len(left_img_pts) > 1:
            cv2.polylines(lane_overlay, [left_img_pts], False, (0, 255, 255), 3, cv2.LINE_AA)
        if right_img_pts is not None and len(right_img_pts) > 1:
            cv2.polylines(lane_overlay, [right_img_pts], False, (255, 100, 0), 3, cv2.LINE_AA)
        if center_img_pts is not None and len(center_img_pts) > 1:
            cv2.polylines(lane_overlay, [center_img_pts], False, (0, 255, 0), 2, cv2.LINE_AA)

        # ── Mask vis (lightweight: skip resize, use cam resolution) ──
        mask_vis = (mask * 255).astype(np.uint8) if self.lightweight_vis else cv2.resize(
            mask.astype(np.uint8) * 255, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST,
        )

        # Image-space lane points for output
        left_px_img = None
        right_px_img = None
        if left_img_pts is not None:
            left_px_img = np.column_stack((left_img_pts[:, 1], left_img_pts[:, 0]))
        if right_img_pts is not None:
            right_px_img = np.column_stack((right_img_pts[:, 1], right_img_pts[:, 0]))

        # Separated masks (skip in lightweight — just return full mask)
        if self.lightweight_vis:
            left_mask_vis = mask_vis
            right_mask_vis = mask_vis
        else:
            left_mask_vis = create_left_lane_mask(mask_vis, left_px_img, rgb.shape[1], rgb.shape[0])
            right_mask_vis = create_right_lane_mask(mask_vis, right_px_img, rgb.shape[1], rgb.shape[0])

        return bev_window_vis, lane_overlay, mask_vis, left_px_img, right_px_img, left_mask_vis, right_mask_vis

    def process(self, rgb: np.ndarray, world=None, vehicle=None) -> TrajectoryOutput:
        """การจับเส้นแบบ phase: P1 → P2 → P3 → P4 → P5 → P6 → TrajectoryOutput."""
        # P1: Lane mask (uses CARLA waypoints if world/vehicle provided, else UNet)
        mask, confidence = self._phase1_lane_mask(rgb, world=world, vehicle=vehicle)
        # P2: Ground projection + quality
        bev_bin, mask_quality = self._phase2_bev_and_quality(mask)
        # P3: Boundaries + center
        (center_raw, c_l, c_r, left_px, right_px,
         raw_left_wins, raw_right_wins, filt_left_wins, filt_right_wins,
         tracked, used_completion, p2_case, p3_case) = (
            self._phase3_boundaries_and_center(bev_bin)
        )
        # Kalman lane tracker: smooth left/right coeffs, extrapolate when UNet misses
        left_s, right_s = self._lane_tracker.update(c_l, c_r)
        c_l_fit = left_s if left_s is not None else c_l
        c_r_fit = right_s if right_s is not None else c_r
        confidence = compute_rich_confidence(
            confidence, c_l_fit, c_r_fit,
            min_lane_width_m=MIN_LANE_WIDTH_M,
            max_lane_width_m=MAX_LANE_WIDTH_M,
        )
        # P4: Smooth center
        center_coeffs, geometry_valid = self._phase4_smooth_center(center_raw, mask_quality)
        # P5: Trajectory + state
        x_ref, y_ref, yaw_ref, curv_ref, v_ref, cte, heading_err, curvature, v_at_ego = (
            self._phase5_trajectory_and_state(center_coeffs)
        )
        waypoint_only = confidence < self.lane_conf_threshold
        # P6: Visualization
        (bev_window_vis, lane_overlay, mask_vis, left_px_img, right_px_img,
         left_mask_vis, right_mask_vis) = self._phase6_visualization(
            rgb, mask, bev_bin, left_px, right_px, filt_left_wins, filt_right_wins,
            c_l_fit, c_r_fit, center_coeffs,
        )

        # Store windows for Peter Moran visualization
        self._raw_left_wins = raw_left_wins
        self._raw_right_wins = raw_right_wins
        self._filt_left_wins = filt_left_wins
        self._filt_right_wins = filt_right_wins
        # Compute per-lane curvature (radius = 1 / |2*c2| for poly c2*x^2 + c1*x + c0)
        left_curvature_m = 9999.0
        right_curvature_m = 9999.0
        if c_l_fit is not None and len(c_l_fit) >= 3:
            c2 = c_l_fit[0]
            if abs(c2) > 1e-6:
                left_curvature_m = float(abs(1.0 / (2.0 * c2)))
        if c_r_fit is not None and len(c_r_fit) >= 3:
            c2 = c_r_fit[0]
            if abs(c2) > 1e-6:
                right_curvature_m = float(abs(1.0 / (2.0 * c2)))
        # Option A: path (s_m, lat_m) from center_coeffs so control/display use same geometry when valid
        reference_path_from_perception = (
            [(float(x_ref[i]), float(y_ref[i])) for i in range(len(x_ref))]
            if center_coeffs is not None and geometry_valid and len(x_ref) >= 2
            else None
        )
        if self.lightweight_vis and lane_overlay is not None and lane_overlay.size > 0:
            h, w = lane_overlay.shape[:2]
            lane_overlay = cv2.resize(lane_overlay, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)

        return TrajectoryOutput(
            x_ref=x_ref, y_ref=y_ref, yaw_ref=yaw_ref, v_ref=v_ref,
            cte=cte, heading_err=heading_err, curvature=curvature,
            v_ref_at_ego=v_at_ego, confidence=confidence,
            mask_vis=mask_vis,
            bev_binary=bev_bin,
            bev_window_vis=bev_window_vis,
            lane_overlay=lane_overlay,
            left_mask_vis=left_mask_vis,
            right_mask_vis=right_mask_vis,
            raw_camera=rgb,
            left_curvature_m=left_curvature_m,
            right_curvature_m=right_curvature_m,
            waypoint_only=waypoint_only,
            geometry_valid=geometry_valid,
            tracked=tracked,
            used_completion=used_completion,
            left_px_img=left_px_img,
            right_px_img=right_px_img,
            phase_p2_case=p2_case,
            phase_p3_case=p3_case,
            reference_path_from_perception=reference_path_from_perception,
        )


# Alias
LaneTracker = LaneTrajectoryPipeline


# ═══════════════════════════════════════════════════════════════════════════
# Standalone visualization helper (for debugging / logging)
# ═══════════════════════════════════════════════════════════════════════════

def visualize_pipeline(
    rgb: np.ndarray,
    output: TrajectoryOutput,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Produce a 3-panel figure (like the two target images combined):
      Row 0: Original Frame
      Row 1: Warped Frame (BEV binary)
      Row 2: Warped Frame With Search Window
      Row 3: Original Frame With Lane Overlay

    Returns the combined image (RGB).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(7, 16))
    axes[0].imshow(rgb)
    axes[0].set_title("Original Frame")
    axes[1].imshow(output.bev_binary, cmap="gray")
    axes[1].set_title("Warped Frame")
    axes[2].imshow(cv2.cvtColor(output.bev_window_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Warped Frame With Search Window")
    if output.lane_overlay is not None:
        axes[3].imshow(output.lane_overlay)
        axes[3].set_title("Original Frame With Lane Overlay")
    else:
        axes[3].imshow(rgb)
        axes[3].set_title("Original Frame (no overlay)")

    for ax in axes:
        ax.axis("on")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Saved visualization to %s", save_path)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf
