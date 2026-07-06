"""
Legacy lane perception: UNet → row-wise poly fit (no BEV).
ใช้เมื่อ --legacy-perception เท่านั้น
"""
import math
import logging
from typing import Tuple, List

import cv2
import numpy as np

from perception.lane_detector import LaneDetector

logger = logging.getLogger(__name__)

# Config (import from parent to avoid circular import)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CAM_W, CAM_H, ROAD_TOP_RATIO, ROI_BOTTOM_RATIO, ROI_TOP_PENALTY_RATIO,
    ROAD_MARGIN_PX, MIN_LANE_WIDTH_PX, MAX_LANE_WIDTH_PX,
    CENTER_LANE_WIDTH_RATIO, LANE_MEMORY_MARGIN_PX, LANE_EXTEND_ROWS_UP,
    LANE_FILL_STEP, DASHED_LANE_MIN_PIXELS,
)


class RoadPerception:
    """Legacy UNet lane detection (row-wise poly fit, no BEV)."""

    EMA = 0.25

    def __init__(self, model_path: str, device) -> None:
        self.dev = device
        self.detector = LaneDetector(
            model_path=model_path, use_carla=False, model_type="unet"
        )
        logger.info("UNet loaded (legacy RoadPerception)")
        self.reset()

    def reset(self) -> None:
        self._prev_cte = 0.0
        self._prev_head = 0.0
        self._prev_curv = 0.0
        self._prev_rows = np.array([], dtype=np.float64)
        self._prev_centers = np.array([], dtype=np.float64)
        self._prev_poly = None
        self._prev_center_bottom = None

    def process(
        self, rgb: np.ndarray
    ) -> Tuple[float, float, float, np.ndarray, float, List]:
        """
        Returns: (cte_norm, heading_rad, curvature, mask_vis, confidence, centerline)
        cte_norm is in [-1,1]; multiply by CTE_TO_METERS for meters.
        """
        h0, w0 = rgb.shape[:2]
        img = cv2.resize(rgb, (CAM_W, CAM_H))
        mask_uint8, _, _ = self.detector.detect_lanes(
            img, world=None, vehicle=None
        )
        mask = (mask_uint8 > 0).astype(np.uint8)
        mask[: int(CAM_H * ROAD_TOP_RATIO), :] = 0

        road_roi = mask[int(CAM_H * (1 - ROI_BOTTOM_RATIO)) :, :]
        sky_roi = mask[: int(CAM_H * ROI_TOP_PENALTY_RATIO), :]
        cov_road = min(
            1.0, np.sum(road_roi > 0) / max(road_roi.size * 0.05, 1)
        )
        cov_sky = min(
            1.0, np.sum(sky_roi > 0) / max(sky_roi.size * 0.15, 1)
        )
        raw_conf = cov_road * (1.0 - 0.85 * cov_sky)

        cx = CAM_W / 2.0
        row_start = int(CAM_H * ROAD_TOP_RATIO)
        center_lo = int(CAM_W * (0.5 - CENTER_LANE_WIDTH_RATIO / 2))
        center_hi = int(CAM_W * (0.5 + CENTER_LANE_WIDTH_RATIO / 2))

        rows_obs: List[int] = []
        centers_obs: List[float] = []
        for r in range(row_start, CAM_H, 2):
            cols = np.where(mask[r] > 0)[0]
            if len(cols) < DASHED_LANE_MIN_PIXELS:
                continue
            left, right = cols[0], cols[-1]
            lane_w = right - left
            if lane_w < MIN_LANE_WIDTH_PX or lane_w > MAX_LANE_WIDTH_PX:
                continue
            if left < ROAD_MARGIN_PX or right > (CAM_W - ROAD_MARGIN_PX):
                continue
            mid = (left + right) / 2.0
            if center_lo <= mid <= center_hi:
                if (
                    self._prev_center_bottom is not None
                    and abs(mid - self._prev_center_bottom)
                    > LANE_MEMORY_MARGIN_PX
                ):
                    continue
                centers_obs.append(mid)
                rows_obs.append(r)

        poly = None
        if len(rows_obs) >= 3:
            rows_a = np.array(rows_obs, dtype=np.float64)
            ctrs_a = np.array(centers_obs, dtype=np.float64)
            r_bot, r_top = rows_a[-1], rows_a[0]
            rn_obs = (rows_a - r_bot) / max(1.0, r_top - r_bot)
            cn_obs = (ctrs_a - cx) / (CAM_W / 2.0)
            try:
                poly = np.polyfit(rn_obs, cn_obs, 2)
                self._prev_poly = tuple(poly)
            except Exception:
                poly = self._prev_poly
        else:
            poly = self._prev_poly

        if poly is None or len(rows_obs) < 2:
            conf = max(0.0, raw_conf - 0.2)
            mask_vis = cv2.resize(
                mask * 255, (w0, h0), interpolation=cv2.INTER_NEAREST
            )
            return (
                self._prev_cte,
                self._prev_head,
                self._prev_curv,
                mask_vis,
                conf,
                [],
            )

        row_extend = max(0, row_start - LANE_EXTEND_ROWS_UP)
        all_rows = np.arange(
            row_extend, CAM_H, LANE_FILL_STEP, dtype=np.float64
        )
        poly_arr = np.array(poly)
        r_bot = float(CAM_H - 1)
        r_top = float(row_start)
        rn_all = (all_rows - r_bot) / max(1.0, r_top - r_bot)
        cn_all = np.polyval(poly_arr, rn_all)
        centers_filled = np.clip(
            cx + cn_all * (CAM_W / 2.0), center_lo, center_hi
        )
        rows_a, ctrs_a = all_rows, centers_filled

        raw_cte = (ctrs_a[-1] - cx) / (CAM_W / 2.0)
        la = max(0, len(rows_a) // 3)
        dx = ctrs_a[la] - ctrs_a[-1]
        dy = rows_a[-1] - rows_a[la]
        raw_head = math.atan2(dx, max(dy, 5.0))
        rn = (rows_a - rows_a[-1]) / max(1.0, rows_a[-1] - rows_a[0])
        cn = (ctrs_a - cx) / (CAM_W / 2.0)
        raw_curv = 0.0
        if len(rn) >= 4:
            try:
                p = np.polyfit(rn, cn, 2)
                raw_curv = p[0] * 0.03
            except Exception:
                pass

        a = self.EMA
        cte = a * self._prev_cte + (1 - a) * float(
            np.clip(raw_cte, -1, 1)
        )
        head = a * self._prev_head + (1 - a) * float(
            np.clip(raw_head, -0.5, 0.5)
        )
        curv = a * self._prev_curv + (1 - a) * float(
            np.clip(raw_curv, -0.04, 0.04)
        )
        self._prev_cte = cte
        self._prev_head = head
        self._prev_curv = curv
        self._prev_rows = rows_a
        self._prev_centers = ctrs_a
        self._prev_center_bottom = float(ctrs_a[-1])

        conf = min(
            1.0,
            raw_conf + 0.15 if len(rows_obs) >= 10 else raw_conf,
        )
        mask_vis = cv2.resize(
            mask * 255, (w0, h0), interpolation=cv2.INTER_NEAREST
        )
        centerline = [
            (int(rows_a[i]), int(np.clip(ctrs_a[i], 0, CAM_W - 1)))
            for i in range(len(rows_a))
        ]
        return cte, head, curv, mask_vis, conf, centerline


class BEVRoadPerception:
    """BEV-based lane detection with polynomial fitting (robust for dashed lines)."""

    EMA = 0.25

    def __init__(self, model_path: str, device) -> None:
        self.dev = device
        self.detector = LaneDetector(
            model_path=model_path, use_carla=False, model_type="unet"
        )
        logger.info("UNet loaded with BEV pipeline (BEVRoadPerception)")
        self.reset()

    def reset(self) -> None:
        self._prev_cte = 0.0
        self._prev_head = 0.0
        self._prev_curv = 0.0
        self._prev_left_coeffs = None
        self._prev_right_coeffs = None

    def process(
        self, rgb: np.ndarray
    ) -> Tuple[float, float, float, np.ndarray, float, List, np.ndarray]:
        """
        Process RGB image using BEV pipeline.
        Returns: (cte_norm, heading_rad, curvature, mask_vis, confidence, centerline, bev_vis)
        """
        h0, w0 = rgb.shape[:2]
        img = cv2.resize(rgb, (CAM_W, CAM_H))

        # Use BEV pipeline with visualization
        left_c, right_c, raw_cte, raw_head, raw_curv, conf, bev_vis = \
            self.detector.detect_lanes_bev(img, return_vis=True)

        # Debug: log detection results every 100 frames
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        if self._frame_count % 100 == 1:
            logger.info(f"BEV detect: L={left_c is not None}, R={right_c is not None}, conf={conf:.2f}")

        if left_c is None and right_c is None:
            # Fallback to previous
            mask_vis = np.zeros((h0, w0), dtype=np.uint8)
            empty_bev = np.zeros((640, 640, 3), dtype=np.uint8) if bev_vis is None else bev_vis
            return self._prev_cte, self._prev_head, self._prev_curv, \
                   mask_vis, max(0.0, conf - 0.3), [], empty_bev

        # EMA smoothing
        a = self.EMA
        cte = a * self._prev_cte + (1 - a) * float(np.clip(raw_cte, -1, 1))
        head = a * self._prev_head + (1 - a) * float(np.clip(raw_head, -0.5, 0.5))
        curv = a * self._prev_curv + (1 - a) * float(np.clip(raw_curv, -0.04, 0.04))

        self._prev_cte = cte
        self._prev_head = head
        self._prev_curv = curv
        self._prev_left_coeffs = left_c
        self._prev_right_coeffs = right_c

        # Generate lane overlay using actual window centers with 30% extension
        # This ensures dots match the green fill exactly (migrated from test_unet_lane.py)
        left_wins = getattr(self.detector.bev_pipeline, '_left_wins', None)
        right_wins = getattr(self.detector.bev_pipeline, '_right_wins', None)

        overlay_warp = self.detector.bev_pipeline.create_lane_overlay_with_dots(
            left_c, right_c, (h0, w0), left_wins=left_wins, right_wins=right_wins)

        # Blend overlay with original image
        lane_overlay = cv2.resize(rgb, (w0, h0))
        if overlay_warp is not None and overlay_warp.any():
            cv2.addWeighted(overlay_warp, 0.5, lane_overlay, 1.0, 0, lane_overlay)

        # Generate visualization mask
        mask_vis = np.zeros((h0, w0), dtype=np.uint8)

        # Generate centerline for visualization
        centerline = []
        if left_c is not None and right_c is not None:
            center_c = (left_c + right_c) / 2
            pts = self.detector.bev_pipeline.bev_to_perspective_points(center_c, n_points=20)
            centerline = [(int(y), int(x)) for x, y in pts if 0 <= x < w0 and 0 <= y < h0]

        return cte, head, curv, mask_vis, conf, centerline, bev_vis, lane_overlay
