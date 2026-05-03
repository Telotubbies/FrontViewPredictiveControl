"""
Peter Moran Lane Tracking Visualization — Overlay Style
Composites diagnostic thumbnails directly onto the camera+lane overlay image.

Layout matches Peter Moran's original output:
  - 6 small BEV diagnostic thumbnails across the top-left
  - Small raw camera thumbnail at top-right
  - Metrics text: Left Curvature / Right Curvature / Center Alignment
  - Main view: green filled lane polygon + yellow boundaries
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Thumbnail size for the 6 diagnostic panels
THUMB_W = 80
THUMB_H = 130
THUMB_GAP = 4


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _make_window_thumb(
    bev_binary: np.ndarray,
    windows: List[Tuple[int, int, int, int]],
    box_color: Tuple[int, int, int],
    fill: bool = False,
) -> np.ndarray:
    """Draw sliding-window boxes on BEV pixels, return thumbnail."""
    h, w = bev_binary.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if bev_binary.ndim == 2:
        canvas[bev_binary > 0] = (255, 255, 255)
    else:
        gray = cv2.cvtColor(bev_binary, cv2.COLOR_BGR2GRAY)
        canvas[gray > 0] = (255, 255, 255)
    for x1, y1, x2, y2 in windows:
        if fill:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
    return cv2.resize(canvas, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


def _make_lane_fill_thumb(
    bev_binary: np.ndarray,
    lane_pixels: Optional[np.ndarray],
    color: Tuple[int, int, int],
) -> np.ndarray:
    """Solid-fill lane region on black, return thumbnail."""
    h, w = bev_binary.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if lane_pixels is not None and len(lane_pixels) >= 2:
        pts = np.column_stack((
            lane_pixels[:, 1].astype(np.int32),
            lane_pixels[:, 0].astype(np.int32),
        ))
        # thick polyline to approximate filled lane
        cv2.polylines(canvas, [pts], False, color, thickness=14)
    return cv2.resize(canvas, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


def _make_center_thumb(
    bev_binary: np.ndarray,
    left_pixels: Optional[np.ndarray],
    right_pixels: Optional[np.ndarray],
) -> np.ndarray:
    """Yellow left + blue right on BEV, return thumbnail."""
    h, w = bev_binary.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if bev_binary.ndim == 2:
        canvas[bev_binary > 0] = (60, 60, 60)
    if left_pixels is not None and len(left_pixels) >= 2:
        pts = np.column_stack((
            left_pixels[:, 1].astype(np.int32),
            left_pixels[:, 0].astype(np.int32),
        ))
        cv2.polylines(canvas, [pts], False, (0, 255, 255), 3)  # Yellow (BGR)
    if right_pixels is not None and len(right_pixels) >= 2:
        pts = np.column_stack((
            right_pixels[:, 1].astype(np.int32),
            right_pixels[:, 0].astype(np.int32),
        ))
        cv2.polylines(canvas, [pts], False, (255, 0, 0), 3)  # Blue (BGR)
    return cv2.resize(canvas, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


def compose_peter_moran_overlay(
    lane_overlay_bgr: np.ndarray,
    raw_camera_bgr: np.ndarray,
    bev_binary: np.ndarray,
    raw_windows_left: List[Tuple[int, int, int, int]],
    raw_windows_right: List[Tuple[int, int, int, int]],
    filt_windows_left: List[Tuple[int, int, int, int]],
    filt_windows_right: List[Tuple[int, int, int, int]],
    left_px_bev: Optional[np.ndarray],
    right_px_bev: Optional[np.ndarray],
    left_curvature_m: float,
    right_curvature_m: float,
    cte_m: float,
) -> np.ndarray:
    """
    Compose Peter Moran-style overlay onto the lane_overlay image.

    Returns BGR image ready for pygame blit or cv2.imshow.
    """
    out = lane_overlay_bgr.copy()
    H, W = out.shape[:2]

    # ── 1) Six diagnostic thumbnails at top-left ──────────────────────────
    thumbs = []
    # (a) Raw windows LEFT (red)
    thumbs.append(_make_window_thumb(bev_binary, raw_windows_left, (0, 0, 255)))
    # (b) Left lane fill (green)
    thumbs.append(_make_lane_fill_thumb(bev_binary, left_px_bev, (0, 255, 0)))
    # (c) Raw windows RIGHT (red)
    thumbs.append(_make_window_thumb(bev_binary, raw_windows_right, (0, 0, 255)))
    # (d) Right lane fill (green)
    thumbs.append(_make_lane_fill_thumb(bev_binary, right_px_bev, (0, 255, 0)))
    # (e) Center detection (yellow left + blue right)
    thumbs.append(_make_center_thumb(bev_binary, left_px_bev, right_px_bev))
    # (f) Filtered combined (green windows both sides)
    thumbs.append(_make_window_thumb(
        bev_binary, filt_windows_left + filt_windows_right, (0, 255, 0), fill=True
    ))

    # Paste thumbnails across top
    tx, ty = 4, 4
    for thumb in thumbs:
        th, tw = thumb.shape[:2]
        if tx + tw > W - 120:
            break
        y1, y2 = ty, ty + th
        x1, x2 = tx, tx + tw
        if y2 <= H and x2 <= W:
            out[y1:y2, x1:x2] = thumb
        tx += tw + THUMB_GAP

    # ── 2) Raw camera thumbnail at top-right ──────────────────────────────
    raw_thumb_w, raw_thumb_h = 120, 80
    raw_thumb = cv2.resize(_ensure_bgr(raw_camera_bgr), (raw_thumb_w, raw_thumb_h))
    rx = W - raw_thumb_w - 4
    ry = 4
    if ry + raw_thumb_h <= H and rx >= 0:
        out[ry:ry + raw_thumb_h, rx:rx + raw_thumb_w] = raw_thumb

    # ── 3) Metrics text below thumbnails ──────────────────────────────────
    text_y = ty + THUMB_H + 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    shadow = (0, 0, 0)

    # Left Curvature
    lc_label = "Left Curvature"
    lc_value = f"{abs(left_curvature_m):.2f}m"
    cv2.putText(out, lc_label, (10, text_y), font, 0.55, shadow, 3, cv2.LINE_AA)
    cv2.putText(out, lc_label, (10, text_y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, lc_value, (10, text_y + 28), font, 0.7, shadow, 3, cv2.LINE_AA)
    cv2.putText(out, lc_value, (10, text_y + 28), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Right Curvature
    rc_x = W // 3
    rc_label = "Right Curvature"
    rc_value = f"{abs(right_curvature_m):.2f}m"
    cv2.putText(out, rc_label, (rc_x, text_y), font, 0.55, shadow, 3, cv2.LINE_AA)
    cv2.putText(out, rc_label, (rc_x, text_y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, rc_value, (rc_x, text_y + 28), font, 0.7, shadow, 3, cv2.LINE_AA)
    cv2.putText(out, rc_value, (rc_x, text_y + 28), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Center Alignment
    ca_x = 2 * W // 3
    direction = "Left" if cte_m < 0 else "Right"
    ca_label = "Center Alignment"
    ca_value = f"{abs(cte_m):.2f}m {direction}"
    cv2.putText(out, ca_label, (ca_x, text_y), font, 0.55, shadow, 3, cv2.LINE_AA)
    cv2.putText(out, ca_label, (ca_x, text_y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, ca_value, (ca_x, text_y + 28), font, 0.7, shadow, 3, cv2.LINE_AA)
    cv2.putText(out, ca_value, (ca_x, text_y + 28), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return out
