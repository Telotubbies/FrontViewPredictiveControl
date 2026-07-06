"""
Standalone UNet lane detection test — BEV (Bird's Eye View) approach.

Pipeline:
1. UNet → probability map of lane markings
2. Perspective warp prob map → Bird's Eye View (BEV)
3. In BEV: threshold → dilate → histogram → sliding windows → polynomial fit
4. Inverse warp lane overlay back to perspective view for visualization

Usage:
    python test_unet_lane.py                          # scenario_*.png
    python test_unet_lane.py --live                   # live from CARLA
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sys
import time
import glob
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from perception.lane_detector import LaneDetector

# ── Config ──
PROB_THRESHOLD = 0.12       # probability threshold in BEV
POLY_ORDER = 2              # polynomial degree for lane fitting
N_WINDOWS = 20              # sliding windows in BEV
WINDOW_MARGIN = 25          # half-width of sliding window in BEV (pixels)
MIN_PIX_RECENTER = 5        # min pixels to recenter window
DILATE_V = 40               # vertical dilation in BEV to connect dashes
DILATE_H = 3                # horizontal dilation in BEV (keep thin)
MAX_JUMP_BEV = 12           # max horizontal jump per window in BEV (curves need ~10)

# BEV perspective transform — calibrated for CARLA 640x480 camera
# Expanded trapezoid: sees more road (higher top = further distance, wider = more lanes)
BEV_SRC = np.float32([
    [140, 255],   # top-left — slightly raised from original (270→255)
    [500, 255],   # top-right
    [625, 475],   # bottom-right
    [15,  475],   # bottom-left
])
BEV_W, BEV_H = 640, 640    # larger BEV for more detail
BEV_DST = np.float32([
    [60,  0],
    [580, 0],
    [580, BEV_H],
    [60,  BEV_H],
])

# Lane width constraints in BEV pixels
BEV_LANE_MIN = 150
BEV_LANE_MAX = 500
BEV_IDEAL_WIDTH = 320

# Precompute transform matrices
M_BEV = cv2.getPerspectiveTransform(BEV_SRC, BEV_DST)
M_INV = cv2.getPerspectiveTransform(BEV_DST, BEV_SRC)


def unet_to_prob(model, rgb: np.ndarray) -> np.ndarray:
    """Run UNet inference, return lane probability map (H, W) in [0, 1]."""
    with torch.no_grad():
        t = torch.FloatTensor(rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
        t = t.to(model.device)
        out = model(t)
        probs = torch.softmax(out, dim=1)
        return probs[0, 1].cpu().numpy()


def to_bev(img, size=None):
    """Warp image to Bird's Eye View."""
    if size is None:
        size = (BEV_W, BEV_H)
    return cv2.warpPerspective(img, M_BEV, size)


def from_bev(img, size=(640, 480)):
    """Warp image from BEV back to perspective."""
    return cv2.warpPerspective(img, M_INV, size)


def find_lane_starts_bev(bev_dilated):
    """Find left/right lane starting x from bottom-half AND top-half histograms."""
    h, w = bev_dilated.shape[:2]
    mid = w // 2
    ks = 15

    def smooth(hist):
        return np.convolve(hist, np.ones(ks) / ks, mode='same')

    def find_peaks(hist_1d, min_height=3, min_dist=30):
        raw = []
        for i in range(min_dist, len(hist_1d) - min_dist):
            if hist_1d[i] > min_height:
                if hist_1d[i] >= max(hist_1d[max(0, i - min_dist):i]) and \
                   hist_1d[i] >= max(hist_1d[i + 1:min(len(hist_1d), i + min_dist + 1)]):
                    raw.append((i, hist_1d[i]))
        merged = []
        i = 0
        while i < len(raw):
            group = [raw[i]]
            while i + 1 < len(raw) and raw[i + 1][0] - raw[i][0] < 15:
                i += 1
                group.append(raw[i])
            best_ht = max(g[1] for g in group)
            center = int(np.mean([g[0] for g in group]))
            merged.append((center, best_ht))
            i += 1
        return merged

    def pick_pair(histogram):
        left_peaks = find_peaks(histogram[:mid])
        right_peaks_raw = find_peaks(histogram[mid:])
        right_peaks = [(c + mid, ht) for c, ht in right_peaks_raw]
        lx, rx = None, None
        vehicle_x = mid
        if left_peaks and right_peaks:
            best = float('inf')
            for lc, lh in left_peaks:
                for rc, rh in right_peaks:
                    wid = rc - lc
                    if wid < BEV_LANE_MIN or wid > BEV_LANE_MAX:
                        continue
                    pair_center = (lc + rc) / 2
                    center_err = abs(pair_center - vehicle_x) / (w / 2)
                    width_err = abs(wid - BEV_IDEAL_WIDTH) / BEV_IDEAL_WIDTH
                    score = center_err + width_err * 0.1
                    if score < best:
                        best = score
                        lx, rx = lc, rc
        if lx is None and left_peaks:
            left_peaks.sort(key=lambda p: -p[1])
            lx = left_peaks[0][0]
        if rx is None and right_peaks:
            right_peaks.sort(key=lambda p: -p[1])
            rx = right_peaks[0][0]
        return lx, rx

    # Bottom-half histogram (for bottom-up sliding window)
    hist_bot = smooth(np.sum(bev_dilated[h // 2:, :], axis=0).astype(float))
    bot_lx, bot_rx = pick_pair(hist_bot)

    # Top-half histogram (for top-down sliding window)
    hist_top = smooth(np.sum(bev_dilated[:h // 2, :], axis=0).astype(float))
    top_lx, top_rx = pick_pair(hist_top)

    return (bot_lx, bot_rx), (top_lx, top_rx), hist_bot


def _run_one_pass(bev_dilated, bev_prob, nz_y, nz_x, start_x,
                   window_ranges, vis_bev=None, color=(0, 255, 0)):
    """Single-direction sliding window pass. Returns list of (row_center, x)."""
    w = bev_dilated.shape[1]
    current_x = start_x
    active_pts = []
    miss_count = 0

    for yt, yb in window_ranges:
        margin = WINDOW_MARGIN if miss_count < 3 else WINDOW_MARGIN * 2
        xl = max(0, current_x - margin)
        xr = min(w, current_x + margin)
        inds = (nz_y >= yt) & (nz_y < yb) & (nz_x >= xl) & (nz_x < xr)

        if vis_bev is not None:
            cv2.rectangle(vis_bev, (xl, yt), (xr, yb), color, 1)

        row_c = (yt + yb) // 2
        if np.sum(inds) >= MIN_PIX_RECENTER:
            miss_count = 0
            prob_win = bev_prob[yt:yb, xl:xr]
            col_sum = np.sum(prob_win, axis=0)
            if col_sum.sum() > 0:
                col_idx = np.arange(xl, xr)
                center_x = int(np.average(col_idx, weights=col_sum))
            else:
                center_x = int(np.median(nz_x[inds]))
            jump = center_x - current_x
            if abs(jump) > MAX_JUMP_BEV:
                center_x = current_x + MAX_JUMP_BEV * (1 if jump > 0 else -1)
            current_x = center_x
            active_pts.append((row_c, center_x))
        else:
            miss_count += 1

    return active_pts


def sliding_window_bev(bev_dilated, bev_prob, bot_start_x, top_start_x,
                       vis_bev=None, color=(0, 255, 0)):
    """
    Bidirectional sliding window in BEV:
    1. Bottom-up pass: starts from bot_start_x at bottom, slides upward
    2. Top-down pass: starts from top_start_x at top, slides downward
    3. Merge active points from both passes (prefer bottom-up where both exist)
    4. Polyfit from merged points
    """
    h, w = bev_dilated.shape[:2]
    wh = h // N_WINDOWS

    nz = bev_dilated.nonzero()
    nz_y, nz_x = nz[0], nz[1]

    # Bottom-up window ranges: from bottom to top
    bot_up_ranges = []
    for i in range(N_WINDOWS):
        yt = h - (i + 1) * wh
        yb = h - i * wh
        if yt >= 0:
            bot_up_ranges.append((yt, yb))

    # Top-down window ranges: from top to bottom
    top_down_ranges = []
    for i in range(N_WINDOWS):
        yt = i * wh
        yb = (i + 1) * wh
        if yb <= h:
            top_down_ranges.append((yt, yb))

    # Run both passes
    bot_pts = []
    if bot_start_x is not None:
        bot_pts = _run_one_pass(bev_dilated, bev_prob, nz_y, nz_x,
                                bot_start_x, bot_up_ranges, vis_bev, color)

    top_pts = []
    if top_start_x is not None:
        top_pts = _run_one_pass(bev_dilated, bev_prob, nz_y, nz_x,
                                top_start_x, top_down_ranges, vis_bev, color)

    # Merge: index by row_c, prefer bottom-up (closer to car = more reliable)
    merged = {}
    for r, c in top_pts:
        merged[r] = c
    for r, c in bot_pts:
        merged[r] = c  # bottom-up overwrites top-down

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

    try:
        coeffs = np.polyfit(rows, cols, POLY_ORDER)
    except:
        return None, [], n_active, None

    # Return actual merged centers (not polynomial-generated)
    # This ensures dots match the actual sliding window detections
    actual_centers = [(int(r), int(c)) for r, c in zip(rows, cols)]

    return coeffs, actual_centers, n_active, None  # No spline in test file


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


MIN_ACTIVE_TRUST = 8  # min active windows to trust a side's polynomial
LANE_MARKING_HALF_WIDTH = 12  # offset to get inner edge of lane marking


def _offset_poly_inward(coeffs, offset):
    """Offset polynomial by adding to the constant term."""
    if coeffs is None:
        return None
    c = coeffs.copy()
    c[-1] += offset
    return c


def constrain_bev_pair(left_coeffs, right_coeffs, left_x, right_x,
                       l_active, r_active, h, w):
    """
    Ensure L/R form valid ego lane in BEV.
    Offsets polynomials inward by LANE_MARKING_HALF_WIDTH to get inner edge.
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

    # Offset detected polynomials inward (L→right, R→left)
    if left_coeffs is not None:
        left_coeffs = _offset_poly_inward(left_coeffs, LANE_MARKING_HALF_WIDTH)
    if right_coeffs is not None:
        right_coeffs = _offset_poly_inward(right_coeffs, -LANE_MARKING_HALF_WIDTH)

    # If only one side detected → generate the other
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

    # Width wrong — pick the more trustworthy side as anchor
    if l_active >= r_active:
        return left_coeffs, _gen_right(left_coeffs)
    else:
        return _gen_left(right_coeffs), right_coeffs


def draw_bev_overlay(bgr, left_coeffs, right_coeffs, left_wins=None, right_wins=None):
    """Draw lane overlay using actual window centers (not polynomial).
    
    This ensures green fill matches the dots exactly.
    """
    h, w = bgr.shape[:2]
    vis = bgr.copy()

    bev_overlay = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    
    # Use actual window centers for drawing (not polynomial)
    left_pts_bev = []
    right_pts_bev = []
    
    if left_wins and len(left_wins) >= 2:
        # Sort by row (top to bottom)
        left_sorted = sorted(left_wins, key=lambda x: x[0])
        left_pts_bev = [(int(c), int(r)) for r, c in left_sorted]
        pts = np.array(left_pts_bev, dtype=np.int32)
        cv2.polylines(bev_overlay, [pts], False, (255, 0, 0), 4)
    
    if right_wins and len(right_wins) >= 2:
        right_sorted = sorted(right_wins, key=lambda x: x[0])
        right_pts_bev = [(int(c), int(r)) for r, c in right_sorted]
        pts = np.array(right_pts_bev, dtype=np.int32)
        cv2.polylines(bev_overlay, [pts], False, (0, 0, 255), 4)

    # Fill area between lanes using actual window centers
    if len(left_pts_bev) >= 2 and len(right_pts_bev) >= 2:
        # Create polygon from actual centers
        poly_pts = np.array(left_pts_bev + list(reversed(right_pts_bev)), dtype=np.int32)
        fill = bev_overlay.copy()
        cv2.fillPoly(fill, [poly_pts], (0, 180, 0))
        cv2.addWeighted(fill, 0.5, bev_overlay, 0.5, 0, bev_overlay)
        
        # Draw centerline
        if len(left_pts_bev) == len(right_pts_bev):
            center_pts = [((l[0]+r[0])//2, (l[1]+r[1])//2) for l, r in zip(left_pts_bev, right_pts_bev)]
            if len(center_pts) >= 2:
                cv2.polylines(bev_overlay, [np.array(center_pts, dtype=np.int32)], False, (0, 255, 0), 2)

    persp_overlay = from_bev(bev_overlay, size=(w, h))
    mask = (persp_overlay.sum(axis=2) > 0)
    vis[mask] = cv2.addWeighted(vis, 0.4, persp_overlay, 0.6, 0)[mask]

    return vis, bev_overlay


def process_frame(model, rgb: np.ndarray, frame_idx: int = 0, label: str = ""):
    """Full BEV pipeline: UNet → prob → BEV warp → sliding window → fit → inverse warp."""
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()
    prob = unet_to_prob(model, rgb)
    t_unet = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()

    # Warp prob to BEV
    bev_prob = to_bev(prob.astype(np.float32))

    # Threshold + dilate in BEV
    bev_binary = (bev_prob > PROB_THRESHOLD).astype(np.uint8)
    kernel = np.ones((DILATE_V, DILATE_H), np.uint8)
    bev_dilated = cv2.dilate(bev_binary, kernel)

    # Find lane starts via histogram (bottom-half + top-half)
    (bot_lx, bot_rx), (top_lx, top_rx), histogram = find_lane_starts_bev(bev_dilated)
    bot_lx, bot_rx = validate_bev_pair(bot_lx, bot_rx, BEV_W)
    top_lx, top_rx = validate_bev_pair(top_lx, top_rx, BEV_W)

    # Bidirectional sliding window in BEV
    vis_bev = cv2.cvtColor(bev_dilated * 255, cv2.COLOR_GRAY2BGR)
    left_coeffs, left_wins, l_active, _ = None, [], 0, None
    right_coeffs, right_wins, r_active, _ = None, [], 0, None

    if bot_lx is not None or top_lx is not None:
        left_coeffs, left_wins, l_active, _ = sliding_window_bev(
            bev_dilated, bev_prob, bot_lx, top_lx,
            vis_bev=vis_bev, color=(255, 100, 0))
    if bot_rx is not None or top_rx is not None:
        right_coeffs, right_wins, r_active, _ = sliding_window_bev(
            bev_dilated, bev_prob, bot_rx, top_rx,
            vis_bev=vis_bev, color=(0, 100, 255))

    # Use bot starts for width reference in constrain
    left_x = bot_lx if bot_lx is not None else top_lx
    right_x = bot_rx if bot_rx is not None else top_rx

    # Constrain pair (uses active counts to trust strong side)
    left_coeffs, right_coeffs = constrain_bev_pair(
        left_coeffs, right_coeffs, left_x, right_x,
        l_active, r_active, BEV_H, BEV_W)

    t_post = (time.perf_counter() - t0) * 1000

    # Draw overlay (pass window centers to avoid extrapolation)
    vis_overlay, bev_overlay = draw_bev_overlay(bgr, left_coeffs, right_coeffs, left_wins, right_wins)

    # Draw lane lines as dots: actual centers + 30% extension using polynomial
    def draw_extended_lane_dots(wins, coeffs, color, extend_pct=0.3):
        if len(wins) < 2:
            return
        # Sort by row (top to bottom)
        sorted_wins = sorted(wins, key=lambda x: x[0])
        rows = [r for r, c in sorted_wins]
        
        # First draw actual window centers (these match green fill exactly)
        for r, c in sorted_wins:
            pt = np.array([[[c, r]]], dtype=np.float32)
            pt_persp = cv2.perspectiveTransform(pt, M_INV)[0][0]
            px, py = int(pt_persp[0]), int(pt_persp[1])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(vis_overlay, (px, py), 5, color, -1)
        
        # Then extend 30% upward using polynomial (beyond detected range)
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

    # BEV thumbnail
    thumb_h = h // 3
    thumb_w = thumb_h
    bev_small = cv2.resize(bev_overlay, (thumb_w, thumb_h))
    vis_overlay[5:5 + thumb_h, w - 5 - thumb_w:w - 5] = bev_small

    # Info
    has_l = left_coeffs is not None
    has_r = right_coeffs is not None
    width_str = ""
    if left_x is not None and right_x is not None:
        width_str = f" bw={right_x - left_x}px"
    info = f"UNet:{t_unet:.0f}ms Post:{t_post:.1f}ms L:{has_l}({l_active}) R:{has_r}({r_active}){width_str}"
    if label:
        info = f"{label} | {info}"
    cv2.putText(vis_overlay, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_path = f"/tmp/test_unet_result_{frame_idx}.png"
    cv2.imwrite(out_path, vis_overlay)
    print(f"[{frame_idx:2d}] {info} → {out_path}")

    return left_coeffs, right_coeffs


def main():
    live = "--live" in sys.argv

    detector = LaneDetector("model/lane_unet_final.pth", use_carla=False, model_type="unet")
    model = detector.model
    print(f"Model loaded on {model.device}")

    if live:
        import carla, queue as Q
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(10)
        # Use Town04 for highway curves
        world = client.load_world("Town04")
        time.sleep(3)
        bp = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        # Town04 spawn 200+ has highway curves
        spawn_idx = min(250, len(spawn_points) - 1)
        veh = world.spawn_actor(bp.find("vehicle.tesla.model3"),
                                spawn_points[spawn_idx])
        veh.set_autopilot(True)
        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "480")
        cam_bp.set_attribute("fov", "90")
        cam_tf = carla.Transform(carla.Location(x=1.5, z=2.0), carla.Rotation(pitch=-8.0))
        cam = world.spawn_actor(cam_bp, cam_tf, attach_to=veh)
        cq = Q.Queue()
        cam.listen(cq.put)
        time.sleep(3)

        for i in range(15):
            time.sleep(0.4)
            img = None
            while not cq.empty():
                img = cq.get()
            if img is None:
                continue
            raw = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))
            rgb = raw[:, :, 2::-1].copy()
            process_frame(model, rgb, i, label=f"live_{i}")

        cam.stop()
        cam.destroy()
        veh.destroy()
    else:
        # Use scenario images
        files = sorted(glob.glob("/tmp/scenario_*.png"))
        if not files:
            # Fallback to test_frame_*
            files = sorted(glob.glob("/tmp/test_frame_*.png"))
        if not files:
            print("No test images found! Run with --live or capture scenarios first.")
            return

        print(f"Testing {len(files)} images")
        for i, f in enumerate(files):
            bgr = cv2.imread(f)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            label = Path(f).stem
            process_frame(model, rgb, i, label=label)


if __name__ == "__main__":
    main()
