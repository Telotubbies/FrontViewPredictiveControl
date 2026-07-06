"""
BEV (Bird's Eye View) Lane Detection Pipeline.

Provides robust lane polynomial fitting using:
1. UNet probability map → BEV warp
2. Histogram peak detection for lane starts
3. Bidirectional sliding window tracking
4. Width-constrained polynomial fitting

Returns left/right lane polynomials for MPC controller.
"""
import cv2
import numpy as np
from typing import Tuple
from scipy.interpolate import CubicSpline

# ── Config ──
PROB_THRESHOLD = 0.08       # Very low threshold to catch faint dash lines
POLY_ORDER = 2
N_WINDOWS = 25              # More windows for finer tracking
WINDOW_MARGIN = 40          # Wider margin to catch dash lines
MIN_PIX_RECENTER = 2        # Very low threshold for sparse dash lines
DILATE_V = 80               # Large vertical dilation to connect dash lines
DILATE_H = 8                # Wider horizontal dilation
MAX_JUMP_BEV = 20           # Allow more jump for curves

# BEV transform for CARLA 640x480 camera
# Same width as original, but extended length (lower horizon to see more road ahead)
BEV_SRC = np.float32([
    [140, 220],   # top-left: same x, but higher up (y=220 instead of 255)
    [500, 220],   # top-right: same x, but higher up
    [625, 475],   # bottom-right: same as original
    [15,  475],   # bottom-left: same as original
])
BEV_W, BEV_H = 640, 640
BEV_DST = np.float32([
    [60,  0],     # same width as original
    [580, 0],
    [580, BEV_H],
    [60,  BEV_H],
])

BEV_LANE_MIN = 150
BEV_LANE_MAX = 500
BEV_IDEAL_WIDTH = 320
MIN_ACTIVE_TRUST = 8

# ROI limit: only detect lanes in bottom portion of BEV (ego lane area)
# Top of BEV = far away, Bottom = close to car
BEV_ROI_TOP = int(BEV_H * 0.10)  # Start from 10% down (capture more lookahead distance)
BEV_ROI_BOTTOM = BEV_H           # Full bottom

# Lane marking half-width in BEV pixels (offset to get inner edge)
# Sliding window tracks center of marking; offset inward for true lane boundary
LANE_MARKING_HALF_WIDTH = 12

# Pixel to meter conversion (based on Udacity research)
# BEV: 640x640 pixels, lane width ~3.5m, lookahead ~32m
XM_PER_PIX = 3.5 / 520   # meters per pixel in x (lane width / BEV lane width in pixels)
YM_PER_PIX = 32.0 / 640  # meters per pixel in y (lookahead distance / BEV height)

# Precompute transforms
M_BEV = cv2.getPerspectiveTransform(BEV_SRC, BEV_DST)
M_INV = cv2.getPerspectiveTransform(BEV_DST, BEV_SRC)


def to_bev(img: np.ndarray) -> np.ndarray:
    """Warp image to Bird's Eye View."""
    return cv2.warpPerspective(img, M_BEV, (BEV_W, BEV_H))


def from_bev(img: np.ndarray, size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """Warp image from BEV back to perspective."""
    return cv2.warpPerspective(img, M_INV, size)


def find_lane_starts_bev(bev_dilated: np.ndarray):
    """Find lane start positions using histogram peaks.

    Prioritizes peaks closest to center (ego lane) with strong signal.
    """
    h, w = bev_dilated.shape[:2]
    mid_x = w // 2

    # Bottom half histogram (closer to car = more reliable)
    bot_half = bev_dilated[h // 2:, :]
    hist_bot = np.sum(bot_half, axis=0).astype(float)

    # Top half histogram
    top_half = bev_dilated[:h // 2, :]
    hist_top = np.sum(top_half, axis=0).astype(float)

    def pick_pair(hist, label=""):
        from scipy.signal import find_peaks
        # Lower threshold to catch more peaks
        peaks, props = find_peaks(hist, height=hist.max() * 0.05, distance=30)
        if len(peaks) == 0:
            return None, None

        left_peaks = peaks[peaks < mid_x]
        right_peaks = peaks[peaks >= mid_x]

        def best_peak(candidates, target_x, prefer_center=True):
            """Select best peak prioritizing proximity to center."""
            if len(candidates) == 0:
                return None
            scores = []
            for p in candidates:
                # Distance to ideal position
                dist_to_ideal = 1.0 - abs(p - target_x) / w
                # Distance to center (closer = better for ego lane)
                dist_to_center = 1.0 - abs(p - mid_x) / mid_x
                # Peak height (stronger signal = better)
                height_score = hist[p] / (hist.max() + 1e-6)

                # Weight: 50% center proximity, 30% ideal position, 20% height
                if prefer_center:
                    score = 0.5 * dist_to_center + 0.3 * dist_to_ideal + 0.2 * height_score
                else:
                    score = 0.4 * dist_to_ideal + 0.4 * height_score + 0.2 * dist_to_center
                scores.append(score)
            return candidates[np.argmax(scores)]

        # For left: prefer peak closest to center (rightmost left peak)
        lx = best_peak(left_peaks, mid_x - BEV_IDEAL_WIDTH // 2, prefer_center=True)
        # For right: prefer peak closest to center (leftmost right peak)
        rx = best_peak(right_peaks, mid_x + BEV_IDEAL_WIDTH // 2, prefer_center=True)
        return lx, rx

    bot_lx, bot_rx = pick_pair(hist_bot, "bot")
    top_lx, top_rx = pick_pair(hist_top, "top")

    return (bot_lx, bot_rx), (top_lx, top_rx), hist_bot


def _run_one_pass(bev_dilated, bev_prob, nz_y, nz_x, start_x, window_ranges,
                  vis_bev=None, color=(0, 255, 0)):
    """Single direction sliding window pass."""
    h, w = bev_dilated.shape[:2]
    current_x = start_x
    active_pts = []
    miss_count = 0

    for yt, yb in window_ranges:
        margin = WINDOW_MARGIN if miss_count < 3 else WINDOW_MARGIN * 2

        xl = max(0, current_x - margin)
        xr = min(w, current_x + margin)

        in_win = (nz_y >= yt) & (nz_y < yb) & (nz_x >= xl) & (nz_x < xr)
        win_x = nz_x[in_win]
        win_y = nz_y[in_win]

        if vis_bev is not None:
            cv2.rectangle(vis_bev, (xl, yt), (xr, yb), color, 1)

        if len(win_x) >= MIN_PIX_RECENTER:
            probs = bev_prob[win_y, win_x]
            weights = probs / (probs.sum() + 1e-9)
            new_x = int(np.sum(win_x * weights))
            jump = new_x - current_x
            if abs(jump) > MAX_JUMP_BEV:
                new_x = current_x + int(np.sign(jump) * MAX_JUMP_BEV)
            current_x = new_x
            row_c = (yt + yb) // 2
            active_pts.append((row_c, current_x))
            miss_count = 0
        else:
            miss_count += 1

    return active_pts


def sliding_window_bev(bev_dilated, bev_prob, bot_start_x, top_start_x,
                       vis_bev=None, color=(0, 255, 0)):
    """Bidirectional sliding window: bottom-up + top-down, merge, fit with CubicSpline.

    Returns:
        coeffs: polynomial coefficients (for MPC compatibility)
        vis_centers: list of (row, col) window center points
        n_active: number of active windows
        spline: CubicSpline object for accurate evaluation (or None)
    """
    h, w = bev_dilated.shape[:2]
    wh = h // N_WINDOWS

    nz = bev_dilated.nonzero()
    nz_y, nz_x = nz[0], nz[1]

    bot_up_ranges = [(h - (i + 1) * wh, h - i * wh) for i in range(N_WINDOWS) if h - (i + 1) * wh >= 0]
    top_down_ranges = [(i * wh, (i + 1) * wh) for i in range(N_WINDOWS) if (i + 1) * wh <= h]

    bot_pts = []
    top_pts = []

    if bot_start_x is not None:
        bot_pts = _run_one_pass(bev_dilated, bev_prob, nz_y, nz_x, bot_start_x,
                                bot_up_ranges, vis_bev, color)
    if top_start_x is not None:
        top_pts = _run_one_pass(bev_dilated, bev_prob, nz_y, nz_x, top_start_x,
                                top_down_ranges, vis_bev, color)

    # Merge: prefer bottom-up
    merged = {}
    for r, c in top_pts:
        merged[r] = c
    for r, c in bot_pts:
        merged[r] = c

    active_pts = sorted(merged.items())
    n_active = len(active_pts)
    if n_active < 3:
        return None, [], n_active, None

    rows = np.array([p[0] for p in active_pts], dtype=float)
    cols = np.array([p[1] for p in active_pts], dtype=float)

    # Median-filter outliers
    if n_active >= 5:
        for i in range(1, len(cols) - 1):
            local_med = np.median(cols[max(0, i - 2):min(len(cols), i + 3)])
            if abs(cols[i] - local_med) > 20:
                cols[i] = local_med

    # Fit CubicSpline for accurate curve (needs at least 10 points for reliable fit)
    # CubicSpline requires strictly increasing x, so sort by rows
    # For fewer points, polynomial is more stable
    spline = None
    if n_active >= 10:
        try:
            sort_idx = np.argsort(rows)
            sorted_rows = rows[sort_idx]
            sorted_cols = cols[sort_idx]
            spline = CubicSpline(sorted_rows, sorted_cols, bc_type='natural')
        except Exception:
            spline = None

    # Also fit polynomial for MPC compatibility
    try:
        coeffs = np.polyfit(rows, cols, POLY_ORDER)
    except Exception:
        return None, [], n_active, spline

    # Generate vis_centers using spline if available, else polynomial
    vis_centers = []
    for yt, yb in bot_up_ranges:
        r = (yt + yb) // 2
        if spline is not None and rows.min() <= r <= rows.max():
            fc = int(np.clip(spline(r), 0, w - 1))
        else:
            fc = int(np.clip(np.polyval(coeffs, r), 0, w - 1))
        vis_centers.append((r, fc))

    return coeffs, vis_centers, n_active, spline


def validate_bev_pair(left_x, right_x, w):
    """Validate lane pair in BEV."""
    if left_x is None or right_x is None:
        return left_x, right_x
    wid = right_x - left_x
    if wid < BEV_LANE_MIN or wid > BEV_LANE_MAX:
        mid = w // 2
        if abs(left_x - (mid - BEV_IDEAL_WIDTH // 2)) < abs(right_x - (mid + BEV_IDEAL_WIDTH // 2)):
            return left_x, None
        else:
            return None, right_x
    return left_x, right_x


def _offset_poly_inward(coeffs, offset):
    """Offset polynomial by adding to the constant term (c[-1])."""
    if coeffs is None:
        return None
    c = coeffs.copy()
    c[-1] += offset
    return c


def constrain_bev_pair(left_coeffs, right_coeffs, l_active, r_active, h, w):
    """
    Ensure L/R form valid ego lane using IDEAL_WIDTH.
    Also offset polynomials inward by LANE_MARKING_HALF_WIDTH to get inner edge.
    """
    if left_coeffs is None and right_coeffs is None:
        return left_coeffs, right_coeffs

    # Effective lane width (accounting for marking offset on both sides)
    effective_width = BEV_IDEAL_WIDTH - 2 * LANE_MARKING_HALF_WIDTH

    def _gen_right(lc):
        rc = lc.copy()
        rc[-1] += effective_width
        return rc

    def _gen_left(rc):
        lc = rc.copy()
        lc[-1] -= effective_width
        return lc

    # First, offset detected polynomials inward (L→right, R→left)
    if left_coeffs is not None:
        left_coeffs = _offset_poly_inward(left_coeffs, LANE_MARKING_HALF_WIDTH)
    if right_coeffs is not None:
        right_coeffs = _offset_poly_inward(right_coeffs, -LANE_MARKING_HALF_WIDTH)

    if left_coeffs is not None and right_coeffs is None:
        return left_coeffs, _gen_right(left_coeffs)
    if right_coeffs is not None and left_coeffs is None:
        return _gen_left(right_coeffs), right_coeffs

    # Both exist — check width
    check_rows = np.linspace(0, h - 1, 10)
    l_cols = np.polyval(left_coeffs, check_rows)
    r_cols = np.polyval(right_coeffs, check_rows)
    widths = r_cols - l_cols
    median_w = np.median(widths)
    width_spread = widths.max() - widths.min()

    width_ok = (effective_width * 0.6 < median_w < effective_width * 1.4
                and np.all(widths > 50)
                and width_spread < effective_width * 0.5)

    if width_ok:
        l_strong = l_active >= MIN_ACTIVE_TRUST
        r_strong = r_active >= MIN_ACTIVE_TRUST
        if l_strong and not r_strong:
            return left_coeffs, _gen_right(left_coeffs)
        if r_strong and not l_strong:
            return _gen_left(right_coeffs), right_coeffs
        return left_coeffs, right_coeffs

    if l_active >= r_active:
        return left_coeffs, _gen_right(left_coeffs)
    else:
        return _gen_left(right_coeffs), right_coeffs


class BEVLanePipeline:
    """
    BEV Lane Detection Pipeline for integration with main system.

    Usage:
        pipeline = BEVLanePipeline()
        left_coeffs, right_coeffs, vis = pipeline.process(prob_map)
    """

    def __init__(self):
        self.kernel = np.ones((DILATE_V, DILATE_H), np.uint8)

    def process(self, prob_map: np.ndarray, return_vis: bool = False):
        """
        Process probability map and return lane polynomials.

        Args:
            prob_map: Lane probability map (H, W) in [0, 1] from UNet
            return_vis: If True, return visualization images

        Returns:
            left_coeffs: Polynomial coefficients for left lane in BEV
            right_coeffs: Polynomial coefficients for right lane in BEV
            l_active: Number of active windows for left lane
            r_active: Number of active windows for right lane
            vis_bev: (optional) BEV visualization image
        """
        # Warp to BEV
        bev_prob = to_bev(prob_map.astype(np.float32))

        # Apply ROI mask - ignore distant lanes (top portion of BEV)
        bev_prob[:BEV_ROI_TOP, :] = 0

        # Threshold + morphological operations for dash line connection
        bev_binary = (bev_prob > PROB_THRESHOLD).astype(np.uint8)

        # Step 1: Initial dilation to thicken faint dash marks (wider kernel)
        dilate_kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 25))
        bev_dilated_1 = cv2.dilate(bev_binary, dilate_kernel_1)

        # Step 2: Vertical closing to connect dash lines (very tall kernel for large gaps)
        close_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 150))
        bev_closed = cv2.morphologyEx(bev_dilated_1, cv2.MORPH_CLOSE, close_kernel_v)

        # Step 3: Final dilation to ensure continuous lanes
        bev_dilated = cv2.dilate(bev_closed, self.kernel)

        # Find lane starts
        (bot_lx, bot_rx), (top_lx, top_rx), _ = find_lane_starts_bev(bev_dilated)
        bot_lx, bot_rx = validate_bev_pair(bot_lx, bot_rx, BEV_W)
        top_lx, top_rx = validate_bev_pair(top_lx, top_rx, BEV_W)

        # Sliding window
        vis_bev = cv2.cvtColor(bev_dilated * 255, cv2.COLOR_GRAY2BGR) if return_vis else None

        left_coeffs, left_wins, l_active, left_spline = None, [], 0, None
        right_coeffs, right_wins, r_active, right_spline = None, [], 0, None

        if bot_lx is not None or top_lx is not None:
            left_coeffs, left_wins, l_active, left_spline = sliding_window_bev(
                bev_dilated, bev_prob, bot_lx, top_lx,
                vis_bev=vis_bev, color=(255, 100, 0))

        if bot_rx is not None or top_rx is not None:
            right_coeffs, right_wins, r_active, right_spline = sliding_window_bev(
                bev_dilated, bev_prob, bot_rx, top_rx,
                vis_bev=vis_bev, color=(0, 100, 255))

        # Store splines and window centers for later use
        self._left_spline = left_spline
        self._right_spline = right_spline
        self._left_wins = left_wins
        self._right_wins = right_wins

        # Constrain pair
        left_coeffs, right_coeffs = constrain_bev_pair(
            left_coeffs, right_coeffs, l_active, r_active, BEV_H, BEV_W)

        # Always return 5 values for consistency
        return left_coeffs, right_coeffs, l_active, r_active, vis_bev

    def bev_to_perspective_points(self, coeffs: np.ndarray, n_points: int = 20,
                                    spline=None, row_range=None):
        """Convert BEV lane to perspective image points using spline if available.

        Args:
            row_range: Optional (min, max) tuple to use common range for symmetric trapezoid
        """
        if coeffs is None and spline is None:
            return []

        # Determine row range
        if row_range is not None:
            # Use provided common range for symmetric trapezoid
            row_min, row_max = row_range
        elif spline is not None:
            # Use spline's actual data range
            row_min, row_max = spline.x.min(), spline.x.max()
        else:
            # Fallback to ROI range
            row_min, row_max = BEV_ROI_TOP, BEV_H - 1

        rows = np.linspace(row_min, row_max, n_points)

        # Evaluate curve
        if spline is not None:
            # Clamp to spline's valid range to avoid extrapolation
            valid_rows = rows[(rows >= spline.x.min()) & (rows <= spline.x.max())]
            if len(valid_rows) > 0:
                cols = spline(valid_rows)
                rows = valid_rows
            elif coeffs is not None:
                cols = np.polyval(coeffs, rows)
            else:
                return []
        elif coeffs is not None:
            cols = np.polyval(coeffs, rows)
        else:
            return []

        # Filter out-of-bounds columns
        valid_mask = (cols >= 0) & (cols < BEV_W)
        rows = rows[valid_mask]
        cols = cols[valid_mask]

        points = []
        for r, c in zip(rows, cols):
            pt = np.array([[[c, r]]], dtype=np.float32)
            pt_persp = cv2.perspectiveTransform(pt, M_INV)[0][0]
            points.append((int(pt_persp[0]), int(pt_persp[1])))

        return points

    def bev_to_perspective_points_left(self, coeffs: np.ndarray, n_points: int = 50):
        """Convert left lane to perspective points using stored spline."""
        return self.bev_to_perspective_points(coeffs, n_points,
                                               getattr(self, '_left_spline', None),
                                               self._get_common_row_range())

    def bev_to_perspective_points_right(self, coeffs: np.ndarray, n_points: int = 50):
        """Convert right lane to perspective points using stored spline."""
        return self.bev_to_perspective_points(coeffs, n_points,
                                               getattr(self, '_right_spline', None),
                                               self._get_common_row_range())

    def _get_common_row_range(self):
        """Get common row range between left and right lanes for symmetric trapezoid.

        Uses spline ranges if available, otherwise uses a fixed bottom portion of BEV.
        """
        left_spline = getattr(self, '_left_spline', None)
        right_spline = getattr(self, '_right_spline', None)

        # Collect available ranges
        ranges = []
        if left_spline is not None:
            ranges.append((left_spline.x.min(), left_spline.x.max()))
        if right_spline is not None:
            ranges.append((right_spline.x.min(), right_spline.x.max()))

        if len(ranges) == 2:
            # Use intersection of both ranges
            row_min = max(ranges[0][0], ranges[1][0])
            row_max = min(ranges[0][1], ranges[1][1])
            if row_min < row_max:
                return (row_min, row_max)
        elif len(ranges) == 1:
            # Use the single available range
            return ranges[0]

        # Fallback: use bottom 70% of BEV (where lanes are most visible)
        return (BEV_H * 0.3, BEV_H - 1)

    def create_lane_overlay_addison(self, left_coeffs, right_coeffs,
                                     orig_frame_shape, n_points: int = 50):
        """Create lane overlay using Automatic Addison method.

        Draw polygon in BEV space, then warp back to perspective.
        This is more accurate than warping individual points.

        Args:
            left_coeffs: Left lane polynomial coefficients
            right_coeffs: Right lane polynomial coefficients
            orig_frame_shape: (height, width) of original frame
            n_points: Number of points for smooth curve

        Returns:
            Lane overlay image (same size as original frame)
        """
        if left_coeffs is None or right_coeffs is None:
            return np.zeros((orig_frame_shape[0], orig_frame_shape[1], 3), dtype=np.uint8)

        # Get common row range for symmetric polygon
        row_range = self._get_common_row_range()
        if row_range is None:
            row_range = (BEV_ROI_TOP, BEV_H - 1)

        row_min, row_max = row_range
        ploty = np.linspace(row_min, row_max, n_points)

        # Evaluate curves using spline if available, else polynomial
        left_spline = getattr(self, '_left_spline', None)
        right_spline = getattr(self, '_right_spline', None)

        if left_spline is not None and row_min >= left_spline.x.min() and row_max <= left_spline.x.max():
            left_fitx = left_spline(ploty)
        else:
            left_fitx = np.polyval(left_coeffs, ploty)

        if right_spline is not None and row_min >= right_spline.x.min() and row_max <= right_spline.x.max():
            right_fitx = right_spline(ploty)
        else:
            right_fitx = np.polyval(right_coeffs, ploty)

        # Clip to BEV bounds
        left_fitx = np.clip(left_fitx, 0, BEV_W - 1)
        right_fitx = np.clip(right_fitx, 0, BEV_W - 1)

        # Create polygon points in BEV space (like Addison's method)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw polygon in BEV space
        color_warp = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 200, 0))  # Green fill

        # Draw lane lines in BEV
        for i in range(len(ploty) - 1):
            cv2.line(color_warp,
                     (int(left_fitx[i]), int(ploty[i])),
                     (int(left_fitx[i+1]), int(ploty[i+1])),
                     (255, 100, 0), 3)  # Blue for left
            cv2.line(color_warp,
                     (int(right_fitx[i]), int(ploty[i])),
                     (int(right_fitx[i+1]), int(ploty[i+1])),
                     (0, 100, 255), 3)  # Red for right

        # Warp back to perspective using inverse transform (Addison's key insight)
        newwarp = cv2.warpPerspective(color_warp, M_INV,
                                       (orig_frame_shape[1], orig_frame_shape[0]))

        return newwarp

    def create_lane_overlay_with_dots(self, left_coeffs, right_coeffs,
                                       orig_frame_shape, left_wins=None, right_wins=None,
                                       extend_pct: float = 0.3):
        """Create lane overlay using actual window centers with 30% extension.

        This ensures dots match the green fill exactly (migrated from test_unet_lane.py).

        Args:
            left_coeffs: Left lane polynomial coefficients
            right_coeffs: Right lane polynomial coefficients
            orig_frame_shape: (height, width) of original frame
            left_wins: List of (row, col) actual window centers for left lane
            right_wins: List of (row, col) actual window centers for right lane
            extend_pct: Percentage to extend visibility upward (default 30%)

        Returns:
            Lane overlay image (same size as original frame)
        """
        h, w = orig_frame_shape[:2]
        bev_overlay = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

        # Use actual window centers for drawing (not polynomial)
        left_pts_bev = []
        right_pts_bev = []

        if left_wins and len(left_wins) >= 2:
            left_sorted = sorted(left_wins, key=lambda x: x[0])
            left_pts_bev = [(int(c), int(r)) for r, c in left_sorted]
            pts = np.array(left_pts_bev, dtype=np.int32)
            cv2.polylines(bev_overlay, [pts], False, (255, 100, 0), 4)

        if right_wins and len(right_wins) >= 2:
            right_sorted = sorted(right_wins, key=lambda x: x[0])
            right_pts_bev = [(int(c), int(r)) for r, c in right_sorted]
            pts = np.array(right_pts_bev, dtype=np.int32)
            cv2.polylines(bev_overlay, [pts], False, (0, 100, 255), 4)

        # Fill area between lanes using actual window centers
        if len(left_pts_bev) >= 2 and len(right_pts_bev) >= 2:
            poly_pts = np.array(left_pts_bev + list(reversed(right_pts_bev)), dtype=np.int32)
            fill = bev_overlay.copy()
            cv2.fillPoly(fill, [poly_pts], (0, 180, 0))
            cv2.addWeighted(fill, 0.5, bev_overlay, 0.5, 0, bev_overlay)

            # Draw centerline
            if len(left_pts_bev) == len(right_pts_bev):
                center_pts = [((lp[0]+rp[0])//2, (lp[1]+rp[1])//2) for lp, rp in zip(left_pts_bev, right_pts_bev)]
                if len(center_pts) >= 2:
                    cv2.polylines(bev_overlay, [np.array(center_pts, dtype=np.int32)], False, (0, 255, 0), 2)

        # Warp back to perspective
        persp_overlay = cv2.warpPerspective(bev_overlay, M_INV, (w, h))

        # Draw extended dots on perspective overlay
        vis_overlay = persp_overlay.copy()

        def draw_extended_lane_dots(wins, coeffs, color):
            if not wins or len(wins) < 2:
                return
            sorted_wins = sorted(wins, key=lambda x: x[0])
            rows = [r for r, c in sorted_wins]

            # Draw actual window centers
            for r, c in sorted_wins:
                pt = np.array([[[c, r]]], dtype=np.float32)
                pt_persp = cv2.perspectiveTransform(pt, M_INV)[0][0]
                px, py = int(pt_persp[0]), int(pt_persp[1])
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(vis_overlay, (px, py), 5, color, -1)

            # Extend 30% upward using polynomial
            row_min = min(rows)
            row_range = max(rows) - row_min
            extended_min = max(0, row_min - int(row_range * extend_pct))

            if coeffs is not None and extended_min < row_min:
                ext_rows = np.linspace(extended_min, row_min, 15)
                ext_cols = np.polyval(coeffs, ext_rows)
                for r, c in zip(ext_rows, ext_cols):
                    if 0 <= c < BEV_W:
                        pt = np.array([[[c, r]]], dtype=np.float32)
                        pt_persp = cv2.perspectiveTransform(pt, M_INV)[0][0]
                        px, py = int(pt_persp[0]), int(pt_persp[1])
                        if 0 <= px < w and 0 <= py < h:
                            cv2.circle(vis_overlay, (px, py), 4, color, -1)

        draw_extended_lane_dots(left_wins, left_coeffs, (255, 100, 0))
        draw_extended_lane_dots(right_wins, right_coeffs, (0, 100, 255))

        return vis_overlay

    def get_centerline_bev(self, left_coeffs, right_coeffs, n_points: int = 20):
        """Get centerline polynomial coefficients in BEV."""
        if left_coeffs is None or right_coeffs is None:
            return None
        return (left_coeffs + right_coeffs) / 2

    def compute_cte_heading(self, left_coeffs, right_coeffs, image_height: int = 480):
        """
        Compute cross-track error and heading from BEV lane polynomials.

        Returns:
            cte: Cross-track error normalized to [-1, 1]
            heading: Heading error in radians
            curvature: Road curvature (scaled for MPC control)
        """
        center_coeffs = self.get_centerline_bev(left_coeffs, right_coeffs)
        if center_coeffs is None:
            return 0.0, 0.0, 0.0

        # CTE at bottom of BEV (closest to car)
        center_x_bottom = np.polyval(center_coeffs, BEV_H - 1)
        cte = (center_x_bottom - BEV_W / 2) / (BEV_W / 2)

        # Heading from derivative at bottom
        # For polynomial ax^2 + bx + c, derivative is 2ax + b
        if len(center_coeffs) >= 2:
            deriv = 2 * center_coeffs[0] * (BEV_H - 1) + center_coeffs[1]
            # Convert pixel slope to radians (BEV_H pixels = ~30m lookahead)
            heading = np.arctan(deriv * 0.05)  # Scale factor for realistic heading
        else:
            heading = 0.0

        # Curvature calculation in METER SPACE (based on Udacity research)
        # Convert polynomial from pixel space to meter space
        # pixel poly: x = a*y^2 + b*y + c
        # meter poly: x_m = a_m*y_m^2 + b_m*y_m + c_m
        # where: a_m = a * XM_PER_PIX / YM_PER_PIX^2
        #        b_m = b * XM_PER_PIX / YM_PER_PIX

        if len(center_coeffs) >= 3:
            a_px, b_px, c_px = center_coeffs

            # Convert to meter space
            a_m = a_px * XM_PER_PIX / (YM_PER_PIX ** 2)
            b_m = b_px * XM_PER_PIX / YM_PER_PIX

            # Evaluate at lookahead point (bottom of BEV = closest to car)
            y_eval_m = (BEV_H - 1) * YM_PER_PIX  # ~32m from top

            # Curvature formula: R = (1 + (2Ay + B)^2)^1.5 / |2A|
            # curvature = 1/R = |2A| / (1 + (2Ay + B)^2)^1.5
            dxdy = 2 * a_m * y_eval_m + b_m
            d2xdy2 = 2 * a_m

            if abs(d2xdy2) > 1e-9:
                curvature = d2xdy2 / ((1 + dxdy**2) ** 1.5)
            else:
                curvature = 0.0
        else:
            curvature = 0.0

        return float(np.clip(cte, -1, 1)), float(np.clip(heading, -0.5, 0.5)), float(np.clip(curvature, -0.05, 0.05))

    def create_visualization(self, image: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create lane overlay visualization using BEV pipeline.

        Args:
            image: RGB image (H, W, 3)
            model: UNet model for inference

        Returns:
            vis_rgb: Image with lane overlay
            bev_thumb: BEV visualization thumbnail
        """
        import torch

        vis_rgb = image.copy()
        h, w = image.shape[:2]

        # Get probability map from UNet
        with torch.no_grad():
            img_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(model.device)
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            prob_map = probs[0, 1].cpu().numpy()

        # Process through BEV pipeline with visualization
        left_coeffs, right_coeffs, l_active, r_active, bev_vis = self.process(prob_map, return_vis=True)

        # Create lane overlay on original image
        if left_coeffs is not None or right_coeffs is not None:
            # Get perspective points for both lanes
            left_pts = self.bev_to_perspective_points(left_coeffs, n_points=30, spline=self._left_spline)
            right_pts = self.bev_to_perspective_points(right_coeffs, n_points=30, spline=self._right_spline)

            # Create filled polygon overlay
            if len(left_pts) > 2 and len(right_pts) > 2:
                overlay = vis_rgb.copy()

                # Combine points for polygon (left bottom to top, right top to bottom)
                left_arr = np.array(left_pts, dtype=np.int32)
                right_arr = np.array(right_pts, dtype=np.int32)

                # Sort by y (row) - left ascending, right descending
                left_sorted = left_arr[left_arr[:, 1].argsort()]
                right_sorted = right_arr[right_arr[:, 1].argsort()[::-1]]

                polygon = np.vstack([left_sorted, right_sorted])

                # Draw filled polygon
                cv2.fillPoly(overlay, [polygon], (0, 255, 0))

                # Blend with original
                alpha = 0.3
                vis_rgb = cv2.addWeighted(overlay, alpha, vis_rgb, 1 - alpha, 0)

                # Draw lane lines
                cv2.polylines(vis_rgb, [left_sorted], False, (255, 0, 0), 2)
                cv2.polylines(vis_rgb, [right_sorted], False, (0, 0, 255), 2)

        return vis_rgb, bev_vis
