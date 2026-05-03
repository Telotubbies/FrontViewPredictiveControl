"""
Lane candidate clustering from BEV binary mask.

Why: ข้ามเลนบ่อยเพราะไม่มี ego-relative filtering และเลือกแค่ left/right จาก histogram
→ ได้หลาย lane candidates แล้ว filter เฉพาะที่อยู่ใกล้ ego (ego-lane selection ใช้ต่อ)

Flow: BEV mask → histogram peaks (หลายจุด) → sliding window ต่อ peak → fit poly ต่อ cluster
      → score + lateral offset → list of LaneCandidate (ego-relative filter อยู่ที่ caller)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from config import (
        BEV_LANE_SEARCH_TOP_RATIO,
        BEV_HISTOGRAM_WEIGHT_FAR,
        BEV_HISTOGRAM_WEIGHT_NEAR,
    )
except ImportError:
    BEV_LANE_SEARCH_TOP_RATIO = 0.25
    BEV_HISTOGRAM_WEIGHT_FAR = 1.5
    BEV_HISTOGRAM_WEIGHT_NEAR = 0.5

# Defaults (caller can override via module or env) - Optimized for speed
N_WINDOWS = 6  # Reduced from 9 → 33% faster
MARGIN_WINDOW = 50  # Reduced from 60 → smaller search area
MIN_PIX_WINDOW = 15  # Reduced from 20 → more lenient
POLY_ORDER = 2
MIN_POINTS_POLY = 6  # Reduced from 8 → faster fitting
OUTLIER_RESIDUAL_THRESH = 2.5  # Increased tolerance → less iterations
# Ego-relative: keep only lanes with |lateral_offset_m| <= this at x=0
EGO_LATERAL_FILTER_M = 5.0
# Histogram: min peak prominence to consider a lane
HIST_PEAK_MIN_HEIGHT = 10
HIST_SMOOTH_K = 31


@dataclass
class LaneCandidate:
    """Single lane line candidate (left or right boundary)."""
    coeffs: np.ndarray       # poly coeffs in vehicle frame (x forward, y lateral): y = poly(x)
    score: float             # 0..1, from point count + fit residual
    lateral_at_ego_m: float  # y at x=0 (vehicle frame)
    is_left: bool            # True = left of center (y < 0 in vehicle frame)
    n_pts: int


def _bev_px_to_vehicle_xy(
    row: np.ndarray, col: np.ndarray,
    bev_h: int, bev_w: int,
    lookahead_m: float, half_width_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = (bev_h - 1 - row) * (lookahead_m / max(bev_h - 1, 1))
    y = (col - (bev_w - 1) / 2.0) * (half_width_m * 2.0 / max(bev_w - 1, 1))
    return x, y


def _fit_poly_robust(x: np.ndarray, y: np.ndarray, order: int = POLY_ORDER) -> Optional[np.ndarray]:
    if len(x) < MIN_POINTS_POLY:
        return None
    coeffs = np.polyfit(x, y, order)
    # Single outlier rejection pass for speed (was 3 iterations)
    pred = np.polyval(coeffs, x)
    res = np.abs(y - pred)
    std = np.std(res)
    thresh = std * OUTLIER_RESIDUAL_THRESH if std > 1e-6 else np.max(res) + 1
    mask = res <= thresh
    if np.sum(mask) >= MIN_POINTS_POLY:
        coeffs = np.polyfit(x[mask], y[mask], order)
    return coeffs


def _histogram_peaks(hist: np.ndarray, min_height: int = HIST_PEAK_MIN_HEIGHT) -> np.ndarray:
    """Return indices of peaks (local maxima) in 1D histogram."""
    h = np.asarray(hist, dtype=np.float64)
    if len(h) < 5:
        return np.array([], dtype=np.intp)
    # Smooth
    k = min(HIST_SMOOTH_K, len(h) // 2)
    if k >= 3:
        kernel = np.ones(k) / k
        h = np.convolve(h, kernel, mode="same")
    # Local maxima
    peaks = []
    for i in range(1, len(h) - 1):
        if h[i] >= min_height and h[i] >= h[i - 1] and h[i] >= h[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=np.intp)


def _trace_lane_from_base(
    bev_binary: np.ndarray,
    base_col: int,
    mid_col: int,
    n_windows: int = N_WINDOWS,
    margin: int = MARGIN_WINDOW,
    min_pix: int = MIN_PIX_WINDOW,
    lookahead_m: float = 20.0,
    half_width_m: float = 10.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Trace one lane upward from base_col. Returns (x_vehicle, y_vehicle, n_pts) or None.
    """
    h, w = bev_binary.shape
    window_h = max(1, h // n_windows)
    pts_px = []
    col_current = base_col
    margin_min = 15

    for win_i in range(n_windows):
        y_lo = h - (win_i + 1) * window_h
        y_hi = h - win_i * window_h
        if y_lo < 0:
            break
        y_mid = (y_lo + y_hi) / 2.0
        y_norm = y_mid / max(h, 1)
        # Adaptive margin: ใกล้รถ (y_norm=1) → margin น้อย, ไกลรถ (y_norm=0) → margin มาก
        # เพิ่ม expansion factor จาก 0.5 → 1.0 เพื่อให้ margin ขยายมากขึ้นตามระยะ
        margin_dyn = max(margin_min, int(margin * (1 - y_norm * 1.0)))
        x1 = max(0, col_current - margin_dyn)
        x2 = min(w, col_current + margin_dyn)
        ys, xs = np.where(bev_binary[y_lo:y_hi, x1:x2] > 0)
        if len(ys) >= min_pix:
            pts_px.append(np.column_stack((ys + y_lo, xs + x1)))
            col_current = int(np.mean(xs + x1))

    if not pts_px:
        return None
    pts_px = np.vstack(pts_px)
    x_veh, y_veh = _bev_px_to_vehicle_xy(
        pts_px[:, 0], pts_px[:, 1], h, w, lookahead_m, half_width_m
    )
    return x_veh, y_veh, len(pts_px)


def cluster_lane_candidates(
    bev_binary: np.ndarray,
    lookahead_m: float = 20.0,
    half_width_m: float = 10.0,
    ego_lateral_filter_m: float = EGO_LATERAL_FILTER_M,
) -> List[LaneCandidate]:
    """
    From BEV binary mask, produce a list of lane line candidates (each with poly, score, lateral at ego).

    Ego-relative: only candidates with |lateral_at_ego_m| <= ego_lateral_filter_m are kept.
    """
    h, w = bev_binary.shape
    mid_col = w // 2

    bottom = bev_binary[int(h * BEV_LANE_SEARCH_TOP_RATIO):, :]
    n_rows = bottom.shape[0]
    # ปรับ weights ให้ focus ใกล้รถมากขึ้น: ใกล้ (bottom) = 1.2, ไกล (top) = 0.8
    weights = np.linspace(0.8, 1.2, n_rows)[:, np.newaxis]
    hist = np.sum((bottom > 0).astype(np.float32) * weights, axis=0)
    hist_sm = np.convolve(hist, np.ones(31) / 31, mode="same")

    if np.max(hist_sm) < 5:
        return []

    peaks = _histogram_peaks(hist_sm, min_height=5)
    if len(peaks) == 0:
        peaks = np.array([np.argmax(hist_sm[:mid_col]), np.argmax(hist_sm[mid_col:]) + mid_col])

    candidates: List[LaneCandidate] = []
    for base_col in peaks:
        if base_col < 5 or base_col > w - 5:
            continue
        out = _trace_lane_from_base(
            bev_binary, base_col, mid_col,
            lookahead_m=lookahead_m, half_width_m=half_width_m,
        )
        if out is None:
            continue
        x_veh, y_veh, n_pts = out
        coeffs = _fit_poly_robust(x_veh, y_veh)
        if coeffs is None:
            continue
        lateral_at_ego = float(np.polyval(coeffs, 0.0))
        if abs(lateral_at_ego) > ego_lateral_filter_m:
            continue
        residual = np.mean(np.abs(y_veh - np.polyval(coeffs, x_veh)))
        score = min(1.0, n_pts / 80.0) * (1.0 / (1.0 + residual * 2.0))
        is_left = lateral_at_ego < 0
        candidates.append(LaneCandidate(
            coeffs=coeffs,
            score=float(score),
            lateral_at_ego_m=lateral_at_ego,
            is_left=is_left,
            n_pts=n_pts,
        ))

    return candidates
