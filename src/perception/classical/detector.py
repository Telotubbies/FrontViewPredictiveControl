"""
Classical lane detection: perspective transform + sliding window + polynomial fit.

Output: CTE (m), heading error (rad), curvature for LaneMPC.
Based on automaticaddison.com pipeline (Addison Sears-Collins).
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

from .. import edge_detection as edge


def _default_roi_points(width: int, height: int) -> np.ndarray:
    """Trapezoid ROI for bird's-eye. Scaled from reference 600x338."""
    sx, sy = width / 600.0, height / 338.0
    return np.float32([
        (274 * sx, 184 * sy),
        (0, height - 1),
        (width - 1, height - 1),
        (371 * sx, 184 * sy),
    ])


def _default_desired_roi(width: int, height: int, padding_ratio: float = 0.25) -> np.ndarray:
    """Desired rectangle after warp (padding from sides)."""
    pad = int(padding_ratio * width)
    return np.float32([
        [pad, 0],
        [pad, height],
        [width - pad, height],
        [width - pad, 0],
    ])


class ClassicalLane:
    """
    Lane detection ด้วย classical CV: edge → warp → sliding window → poly fit.
    Outputs cte_m, heading_rad, curvature for MPC.
    """

    def __init__(
        self,
        image_width: int = 320,
        image_height: int = 240,
        roi_points: Optional[np.ndarray] = None,
        desired_roi_points: Optional[np.ndarray] = None,
        no_of_windows: int = 10,
        margin_ratio: float = 1.0 / 12,
        minpix_ratio: float = 1.0 / 24,
        ym_per_pix: float = 10.0 / 1000,
        xm_per_pix: float = 3.7 / 781,
    ):
        self.width = image_width
        self.height = image_height
        self.roi_points = roi_points if roi_points is not None else _default_roi_points(image_width, image_height)
        self.desired_roi_points = desired_roi_points if desired_roi_points is not None else _default_desired_roi(image_width, image_height)
        self.no_of_windows = no_of_windows
        self.margin = int(margin_ratio * image_width)
        self.minpix = max(1, int(minpix_ratio * image_width))
        self.YM_PER_PIX = ym_per_pix
        self.XM_PER_PIX = xm_per_pix

        self.warped_frame: Optional[np.ndarray] = None
        self.transformation_matrix: Optional[np.ndarray] = None
        self.inv_transformation_matrix: Optional[np.ndarray] = None
        self.histogram: Optional[np.ndarray] = None
        self.left_fit: Optional[np.ndarray] = None
        self.right_fit: Optional[np.ndarray] = None
        self.left_fitx: Optional[np.ndarray] = None
        self.right_fitx: Optional[np.ndarray] = None
        self.ploty: Optional[np.ndarray] = None
        self.leftx: Optional[np.ndarray] = None
        self.rightx: Optional[np.ndarray] = None
        self.lefty: Optional[np.ndarray] = None
        self.righty: Optional[np.ndarray] = None
        self.left_curvem: Optional[float] = None
        self.right_curvem: Optional[float] = None
        self.center_offset_m: Optional[float] = None
        self._last_valid: Optional[Tuple[float, float, float]] = None

    def get_line_markings(self, frame: np.ndarray) -> np.ndarray:
        """Isolate lane lines: HLS + Sobel + saturation + red channel."""
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(120, 255))
        sxbinary = edge.blur_gaussian(sxbinary, ksize=3)
        sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))
        s_channel = hls[:, :, 2]
        _, s_binary = edge.threshold(s_channel, (80, 255))
        _, r_thresh = edge.threshold(frame[:, :, 2], thresh=(120, 255))
        rs_binary = cv2.bitwise_and(s_binary, r_thresh)
        return cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))

    def perspective_transform(self, frame: np.ndarray) -> np.ndarray:
        """Warp to bird's-eye and binarize."""
        self.transformation_matrix = cv2.getPerspectiveTransform(self.roi_points, self.desired_roi_points)
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.desired_roi_points, self.roi_points)
        size = (self.width, self.height)
        warped = cv2.warpPerspective(frame, self.transformation_matrix, size, flags=cv2.INTER_LINEAR)
        _, binary = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary
        return binary

    def calculate_histogram(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        if frame is None:
            frame = self.warped_frame
        if frame is None:
            return np.zeros(self.width, dtype=np.float64)
        self.histogram = np.sum(frame[int(frame.shape[0] / 2) :, :], axis=0).astype(np.float64)
        return self.histogram

    def histogram_peak(self) -> Tuple[int, int]:
        if self.histogram is None or len(self.histogram) < 2:
            mid = self.width // 2
            return max(0, mid - 50), min(self.width, mid + 50)
        midpoint = len(self.histogram) // 2
        leftx_base = int(np.argmax(self.histogram[:midpoint]))
        rightx_base = int(np.argmax(self.histogram[midpoint:]) + midpoint)
        return leftx_base, rightx_base

    def get_lane_line_indices_sliding_windows(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.warped_frame is None:
            return None, None
        window_height = int(self.warped_frame.shape[0] / self.no_of_windows)
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = []
        right_lane_inds = []
        leftx_base, rightx_base = self.histogram_peak()
        leftx_current = leftx_base
        rightx_current = rightx_base

        for window in range(self.no_of_windows):
            win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)
            if len(good_left) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=np.intp)
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=np.intp)

        if len(left_lane_inds) < 10 or len(right_lane_inds) < 10:
            return None, None

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        self.leftx, self.lefty = leftx, lefty
        self.rightx, self.righty = rightx, righty
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        self.ploty = np.linspace(0, self.warped_frame.shape[0] - 1, self.warped_frame.shape[0])
        self.left_fitx = self.left_fit[0] * self.ploty**2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty**2 + self.right_fit[1] * self.ploty + self.right_fit[2]
        return self.left_fit, self.right_fit

    def get_lane_line_previous_window(self, left_fit: np.ndarray, right_fit: np.ndarray) -> None:
        if self.warped_frame is None:
            return
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.margin
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                           (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds]
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]
        if len(self.leftx) < 5 or len(self.rightx) < 5:
            return
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)
        self.ploty = np.linspace(0, self.warped_frame.shape[0] - 1, self.warped_frame.shape[0])
        self.left_fitx = self.left_fit[0] * self.ploty**2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty**2 + self.right_fit[1] * self.ploty + self.right_fit[2]

    def calculate_curvature(self) -> Tuple[float, float]:
        if self.ploty is None or self.left_fit is None or self.right_fit is None or self.lefty is None or self.righty is None:
            return 0.0, 0.0
        y_eval = np.max(self.ploty)
        left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * self.XM_PER_PIX, 2)
        right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * self.XM_PER_PIX, 2)
        denom_left = np.absolute(2 * left_fit_cr[0])
        denom_right = np.absolute(2 * right_fit_cr[0])
        left_curvem = ((1 + (2 * left_fit_cr[0] * y_eval * self.YM_PER_PIX + left_fit_cr[1])**2)**1.5) / (denom_left if denom_left > 1e-10 else 1e10)
        right_curvem = ((1 + (2 * right_fit_cr[0] * y_eval * self.YM_PER_PIX + right_fit_cr[1])**2)**1.5) / (denom_right if denom_right > 1e-10 else 1e10)
        self.left_curvem = left_curvem
        self.right_curvem = right_curvem
        return left_curvem, right_curvem

    def calculate_car_position(self) -> float:
        if self.left_fit is None or self.right_fit is None:
            return 0.0
        car_location = self.width / 2
        height = self.height
        bottom_left = self.left_fit[0] * height**2 + self.left_fit[1] * height + self.left_fit[2]
        bottom_right = self.right_fit[0] * height**2 + self.right_fit[1] * height + self.right_fit[2]
        center_lane = (bottom_right - bottom_left) / 2 + bottom_left
        center_offset_px = car_location - center_lane
        center_offset_m = center_offset_px * self.XM_PER_PIX
        self.center_offset_m = center_offset_m
        return center_offset_m

    def heading_from_fit(self) -> float:
        if self.left_fit is None or self.right_fit is None or self.ploty is None:
            return 0.0
        y = np.max(self.ploty)
        center_fit = (self.left_fit + self.right_fit) / 2
        dxy_px = 2 * center_fit[0] * y + center_fit[1]
        angle_rad = np.arctan2(dxy_px * self.XM_PER_PIX, self.YM_PER_PIX)
        return float(np.clip(angle_rad, -0.5, 0.5))

    def process(self, frame: np.ndarray) -> Tuple[float, float, float, bool]:
        """Full pipeline: frame -> (cte_m, heading_rad, curvature, valid)."""
        try:
            markings = self.get_line_markings(frame)
            self.perspective_transform(markings)
            self.calculate_histogram()
            left_fit, right_fit = self.get_lane_line_indices_sliding_windows()
            if left_fit is None or right_fit is None:
                if self._last_valid is not None:
                    return (*self._last_valid, False)
                return 0.0, 0.0, 0.0, False
            self.get_lane_line_previous_window(left_fit, right_fit)
            self.calculate_curvature()
            cte_m = self.calculate_car_position()
            heading_rad = self.heading_from_fit()
            r_left, r_right = self.left_curvem or 1e4, self.right_curvem or 1e4
            curvature = 0.5 * (1.0 / r_left + 1.0 / r_right)
            curvature = float(np.clip(curvature, -0.08, 0.08))
            cte_m = float(np.clip(cte_m, -2.0, 2.0))
            self._last_valid = (cte_m, heading_rad, curvature)
            return cte_m, heading_rad, curvature, True
        except Exception:
            if self._last_valid is not None:
                return (*self._last_valid, False)
            return 0.0, 0.0, 0.0, False

    def overlay_lane_lines(self, orig_frame: np.ndarray) -> np.ndarray:
        """Draw green lane overlay on original frame (for visualization)."""
        if self.warped_frame is None or self.left_fitx is None or self.right_fitx is None or self.ploty is None or self.inv_transformation_matrix is None:
            return orig_frame
        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_(pts), (0, 255, 0))
        newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (self.width, self.height))
        return cv2.addWeighted(orig_frame, 1, newwarp, 0.3, 0)
