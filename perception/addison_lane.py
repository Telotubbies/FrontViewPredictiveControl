#!/usr/bin/env python3
"""
Addison-Style Lane Detection Pipeline.

Based on: https://automaticaddison.com/the-ultimate-guide-to-real-time-lane-detection-using-opencv/

Key steps:
1. ROI mask (trapezoid) to focus on ego-lane only
2. Perspective transform to BEV
3. Histogram to find lane starts
4. Sliding window to track lanes
5. Polynomial fit (degree 2)
6. Draw overlay in BEV, warp back to perspective
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

# ── Image dimensions ──
IMG_W, IMG_H = 640, 480

# ── ROI trapezoid (perspective view) ──
# This defines the region of interest - only ego-lane
# Calibrated for CARLA 640x480 camera
# Limited to road area only (not sky/horizon)
# Points: top-left, top-right, bottom-right, bottom-left
ROI_VERTICES = np.array([
    [(220, 270),   # top-left: lower horizon (road only)
     (420, 270),   # top-right: wider at top
     (620, IMG_H - 1),   # bottom-right: wider
     (20, IMG_H - 1)]    # bottom-left: wider
], dtype=np.int32)

# ── BEV transform points ──
# Source: trapezoid in perspective (matches ROI) - limited to road area
BEV_SRC = np.float32([
    [220, 270],   # top-left: lower horizon
    [420, 270],   # top-right: wider
    [620, IMG_H - 1],   # bottom-right: wider
    [20, IMG_H - 1],    # bottom-left: wider
])

# Destination: rectangle in BEV
BEV_W, BEV_H = 640, 800
BEV_DST = np.float32([
    [80, 0],           # top-left
    [BEV_W - 80, 0],   # top-right
    [BEV_W - 80, BEV_H],  # bottom-right
    [80, BEV_H],       # bottom-left
])

# Precompute transforms
M_BEV = cv2.getPerspectiveTransform(BEV_SRC, BEV_DST)
M_INV = cv2.getPerspectiveTransform(BEV_DST, BEV_SRC)

# ── Sliding window parameters ──
N_WINDOWS = 20
WINDOW_MARGIN = 80  # Increased for better curve tracking
MIN_PIX_RECENTER = 10

# ── Lane width constraints ──
MIN_LANE_WIDTH = 150  # pixels in BEV
MAX_LANE_WIDTH = 450
IDEAL_LANE_WIDTH = 300


class LaneKalmanFilter:
    """Simple Kalman filter for smoothing lane polynomial coefficients."""
    
    def __init__(self, dim: int = 3):
        """
        Initialize Kalman filter for polynomial coefficients.
        
        Args:
            dim: Number of polynomial coefficients (3 for degree 2)
        """
        self.dim = dim
        
        # State estimate
        self.x = np.zeros(dim)
        
        # State covariance
        self.P = np.eye(dim) * 1.0
        
        # Process noise
        self.Q = np.eye(dim) * 0.001
        
        # Measurement noise
        self.R = np.eye(dim) * 0.05
        
        # State transition (identity - coefficients change slowly)
        self.F = np.eye(dim)
        
        # Measurement matrix
        self.H = np.eye(dim)
        
        self.initialized = False
    
    def update(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Update filter with new measurement and return smoothed coefficients.
        """
        z = coeffs.flatten()
        
        if not self.initialized:
            self.x = z.copy()
            self.initialized = True
            return self.x.copy()
        
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        y = z - self.H @ x_pred  # Innovation
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = x_pred + K @ y
        self.P = (np.eye(self.dim) - K @ self.H) @ P_pred
        
        return self.x.copy()
    
    def predict_only(self) -> np.ndarray:
        """Predict next state without measurement (for missing detections)."""
        if not self.initialized:
            return None
        
        # Predict only
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy()


def apply_roi_mask(image: np.ndarray) -> np.ndarray:
    """Apply ROI mask to focus on ego-lane only."""
    mask = np.zeros_like(image)
    if len(image.shape) == 3:
        cv2.fillPoly(mask, ROI_VERTICES, (255, 255, 255))
    else:
        cv2.fillPoly(mask, ROI_VERTICES, 255)
    return cv2.bitwise_and(image, mask)


def to_bev(image: np.ndarray) -> np.ndarray:
    """Warp image to Bird's Eye View."""
    return cv2.warpPerspective(image, M_BEV, (BEV_W, BEV_H))


def from_bev(image: np.ndarray, size: Tuple[int, int] = (IMG_W, IMG_H)) -> np.ndarray:
    """Warp image from BEV back to perspective."""
    return cv2.warpPerspective(image, M_INV, size)


def find_lane_starts(binary_bev: np.ndarray, 
                     locked_left: int = None, locked_right: int = None) -> Tuple[int, int]:
    """
    Find lane start positions using histogram of bottom half.
    IMPORTANT: Left lane MUST be in left half, Right lane MUST be in right half.
    """
    # Take bottom half of image
    bottom_half = binary_bev[BEV_H // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    
    midpoint = BEV_W // 2
    search_margin = 80  # pixels to search around locked position
    min_peak_height = 500  # minimum histogram height to be valid
    
    # Find left lane - MUST stay in left half (0 to midpoint)
    if locked_left is not None and locked_left < midpoint:
        # Search around locked position but stay in left half
        search_left = max(0, locked_left - search_margin)
        search_right = min(midpoint, locked_left + search_margin)
        if search_right > search_left:
            local_hist = histogram[search_left:search_right]
            if len(local_hist) > 0 and np.max(local_hist) > min_peak_height:
                left_x = search_left + np.argmax(local_hist)
            else:
                left_x = np.argmax(histogram[:midpoint])
        else:
            left_x = np.argmax(histogram[:midpoint])
    else:
        # No lock or lock crossed midpoint - search left half only
        left_x = np.argmax(histogram[:midpoint])
    
    # Find right lane - MUST stay in right half (midpoint to BEV_W)
    if locked_right is not None and locked_right >= midpoint:
        # Search around locked position but stay in right half
        search_left = max(midpoint, locked_right - search_margin)
        search_right = min(BEV_W, locked_right + search_margin)
        if search_right > search_left:
            local_hist = histogram[search_left:search_right]
            if len(local_hist) > 0 and np.max(local_hist) > min_peak_height:
                right_x = search_left + np.argmax(local_hist)
            else:
                right_x = np.argmax(histogram[midpoint:]) + midpoint
        else:
            right_x = np.argmax(histogram[midpoint:]) + midpoint
    else:
        # No lock or lock crossed midpoint - search right half only
        right_x = np.argmax(histogram[midpoint:]) + midpoint
    
    # Validate: peaks should have significant height
    if histogram[left_x] < min_peak_height:
        left_x = None
    if histogram[right_x] < min_peak_height:
        right_x = None
    
    # Final sanity check: left must be < midpoint, right must be >= midpoint
    if left_x is not None and left_x >= midpoint:
        left_x = None
    if right_x is not None and right_x < midpoint:
        right_x = None
    
    return left_x, right_x


def sliding_window_fit(binary_bev: np.ndarray, start_x: int, 
                       vis_img: np.ndarray = None, color: Tuple = (0, 255, 0),
                       side: str = None
                       ) -> Tuple[Optional[np.ndarray], List[Tuple[int, int]], int]:
    """
    Sliding window lane tracking from bottom to top.
    
    Args:
        side: 'left' or 'right' to constrain window to that half
    
    Returns:
        coeffs: Polynomial coefficients (degree 2)
        window_centers: List of (row, col) window centers
        active_count: Number of windows with detected pixels
    """
    window_height = BEV_H // N_WINDOWS
    midpoint = BEV_W // 2
    
    # Get all nonzero pixel positions
    nonzero = binary_bev.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    
    # Current x position
    current_x = start_x
    
    # Collect lane pixel indices
    lane_inds = []
    window_centers = []
    active_count = 0
    
    for win_idx in range(N_WINDOWS):
        # Window boundaries
        win_y_low = BEV_H - (win_idx + 1) * window_height
        win_y_high = BEV_H - win_idx * window_height
        win_x_low = max(0, current_x - WINDOW_MARGIN)
        win_x_high = min(BEV_W, current_x + WINDOW_MARGIN)
        
        # Constrain window to correct side
        if side == 'left':
            win_x_high = min(win_x_high, midpoint)
        elif side == 'right':
            win_x_low = max(win_x_low, midpoint)
        
        # Draw window on visualization
        if vis_img is not None:
            cv2.rectangle(vis_img, (win_x_low, win_y_low), (win_x_high, win_y_high), 
                         color, 2)
        
        # Find pixels in window
        good_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                     (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)).nonzero()[0]
        
        lane_inds.append(good_inds)
        
        # Recenter if enough pixels found, otherwise use previous position
        if len(good_inds) > MIN_PIX_RECENTER:
            new_x = int(np.mean(nonzero_x[good_inds]))
            # Constrain new_x to correct side
            if side == 'left':
                new_x = min(new_x, midpoint - 1)
            elif side == 'right':
                new_x = max(new_x, midpoint)
            current_x = new_x
            active_count += 1
        else:
            # No pixels found - keep current position (maintain lane shape)
            pass
        
        window_centers.append(((win_y_low + win_y_high) // 2, current_x))
    
    # Concatenate all indices
    try:
        lane_inds = np.concatenate(lane_inds)
    except ValueError:
        return None, window_centers, 0
    
    if len(lane_inds) < 50:
        return None, window_centers, active_count
    
    # Extract pixel positions
    x = nonzero_x[lane_inds]
    y = nonzero_y[lane_inds]
    
    # Fit polynomial (degree 2)
    try:
        coeffs = np.polyfit(y, x, 2)
    except:
        return None, window_centers, active_count
    
    return coeffs, window_centers, active_count


def constrain_lane_pair(left_coeffs: np.ndarray, right_coeffs: np.ndarray,
                        left_active: int, right_active: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Constrain lane pair to have valid width."""
    midpoint = BEV_W // 2
    y_eval = BEV_H - 1
    
    if left_coeffs is None and right_coeffs is None:
        return None, None
    
    # Generate from one if other is missing
    if left_coeffs is None and right_coeffs is not None:
        left_coeffs = right_coeffs.copy()
        left_coeffs[2] -= IDEAL_LANE_WIDTH
        # Ensure generated left is actually in left half
        gen_x = np.polyval(left_coeffs, y_eval)
        if gen_x >= midpoint:
            left_coeffs[2] = midpoint - 50  # Force to left side
        return left_coeffs, right_coeffs
    
    if right_coeffs is None and left_coeffs is not None:
        right_coeffs = left_coeffs.copy()
        right_coeffs[2] += IDEAL_LANE_WIDTH
        # Ensure generated right is actually in right half
        gen_x = np.polyval(right_coeffs, y_eval)
        if gen_x < midpoint:
            right_coeffs[2] = midpoint + 50  # Force to right side
        return left_coeffs, right_coeffs
    
    # Validate both lanes are on correct sides
    left_x = np.polyval(left_coeffs, y_eval)
    right_x = np.polyval(right_coeffs, y_eval)
    
    # If left is on wrong side, regenerate from right
    if left_x >= midpoint:
        left_coeffs = right_coeffs.copy()
        left_coeffs[2] -= IDEAL_LANE_WIDTH
        left_x = np.polyval(left_coeffs, y_eval)
        if left_x >= midpoint:
            left_coeffs[2] = midpoint - 50
    
    # If right is on wrong side, regenerate from left
    if right_x < midpoint:
        right_coeffs = left_coeffs.copy()
        right_coeffs[2] += IDEAL_LANE_WIDTH
        right_x = np.polyval(right_coeffs, y_eval)
        if right_x < midpoint:
            right_coeffs[2] = midpoint + 50
    
    # Check width
    left_x = np.polyval(left_coeffs, y_eval)
    right_x = np.polyval(right_coeffs, y_eval)
    width = right_x - left_x
    
    if width < MIN_LANE_WIDTH or width > MAX_LANE_WIDTH:
        # Use more trusted side but keep lanes on correct sides
        if left_active >= right_active:
            right_coeffs = left_coeffs.copy()
            right_coeffs[2] += IDEAL_LANE_WIDTH
            if np.polyval(right_coeffs, y_eval) < midpoint:
                right_coeffs[2] = midpoint + 50
        else:
            left_coeffs = right_coeffs.copy()
            left_coeffs[2] -= IDEAL_LANE_WIDTH
            if np.polyval(left_coeffs, y_eval) >= midpoint:
                left_coeffs[2] = midpoint - 50
    
    return left_coeffs, right_coeffs


def draw_lane_overlay(orig_img: np.ndarray, left_coeffs: np.ndarray, 
                      right_coeffs: np.ndarray) -> np.ndarray:
    """
    Draw lane overlay using Addison's method:
    1. Create polygon in BEV space
    2. Fill with green
    3. Draw lane lines
    4. Warp back to perspective
    """
    if left_coeffs is None or right_coeffs is None:
        return orig_img
    
    # Generate y values from top to bottom of BEV
    ploty = np.linspace(0, BEV_H - 1, 50)
    
    # Evaluate polynomials
    left_fitx = np.polyval(left_coeffs, ploty)
    right_fitx = np.polyval(right_coeffs, ploty)
    
    # Clip to BEV bounds
    left_fitx = np.clip(left_fitx, 0, BEV_W - 1)
    right_fitx = np.clip(right_fitx, 0, BEV_W - 1)
    
    # Create BEV overlay
    bev_overlay = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    
    # Create polygon points
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Fill polygon with green
    cv2.fillPoly(bev_overlay, np.int_([pts]), (0, 200, 0))
    
    # Draw lane lines (white)
    for i in range(len(ploty) - 1):
        cv2.line(bev_overlay, 
                 (int(left_fitx[i]), int(ploty[i])), 
                 (int(left_fitx[i+1]), int(ploty[i+1])), 
                 (255, 255, 255), 3)
        cv2.line(bev_overlay, 
                 (int(right_fitx[i]), int(ploty[i])), 
                 (int(right_fitx[i+1]), int(ploty[i+1])), 
                 (255, 255, 255), 3)
    
    # Warp back to perspective
    persp_overlay = from_bev(bev_overlay)
    
    # Blend with original
    result = cv2.addWeighted(orig_img, 1, persp_overlay, 0.5, 0)
    
    return result


def compute_curvature_and_offset(left_coeffs: np.ndarray, right_coeffs: np.ndarray,
                                  xm_per_pix: float = 3.7/200,  # meters per pixel
                                  ym_per_pix: float = 30.0/BEV_H  # meters per pixel
                                  ) -> Tuple[float, float]:
    """
    Compute radius of curvature and center offset.
    
    Based on Addison's formulas:
    - Curvature: R = (1 + (2Ay + B)^2)^(3/2) / |2A|
    - Offset: distance from lane center to image center
    """
    if left_coeffs is None or right_coeffs is None:
        return float('inf'), 0.0
    
    # Evaluate at bottom of image (closest to car)
    y_eval = BEV_H - 1
    
    # Convert to real-world coordinates
    left_fit_cr = np.array([
        left_coeffs[0] * xm_per_pix / (ym_per_pix ** 2),
        left_coeffs[1] * xm_per_pix / ym_per_pix,
        left_coeffs[2] * xm_per_pix
    ])
    right_fit_cr = np.array([
        right_coeffs[0] * xm_per_pix / (ym_per_pix ** 2),
        right_coeffs[1] * xm_per_pix / ym_per_pix,
        right_coeffs[2] * xm_per_pix
    ])
    
    # Curvature formula: R = (1 + (2Ay + B)^2)^(3/2) / |2A|
    y_eval_m = y_eval * ym_per_pix
    
    left_curv = ((1 + (2*left_fit_cr[0]*y_eval_m + left_fit_cr[1])**2)**1.5) / abs(2*left_fit_cr[0] + 1e-6)
    right_curv = ((1 + (2*right_fit_cr[0]*y_eval_m + right_fit_cr[1])**2)**1.5) / abs(2*right_fit_cr[0] + 1e-6)
    
    avg_curvature = (left_curv + right_curv) / 2
    
    # Center offset
    left_x = np.polyval(left_coeffs, y_eval)
    right_x = np.polyval(right_coeffs, y_eval)
    lane_center = (left_x + right_x) / 2
    image_center = BEV_W / 2
    offset_pixels = lane_center - image_center
    offset_meters = offset_pixels * xm_per_pix
    
    return avg_curvature, offset_meters


class AddisonLaneDetector:
    """
    Complete Addison-style lane detection pipeline.
    """
    
    def __init__(self, use_unet: bool = True, model_path: str = None):
        self.use_unet = use_unet
        self.model = None
        
        # Simple temporal smoothing
        self.prev_left_coeffs = None
        self.prev_right_coeffs = None
        self.alpha = 0.9  # smoothing factor (higher = more weight to new, faster response to curves)
        
        # Curve equation memory for dash gaps
        self.curve_memory_left = None
        self.curve_memory_right = None
        self.curve_memory_frames = 0
        self.max_curve_memory_frames = 15  # Remember curve for 15 frames during dash gaps
        
        # Previous lane positions for sliding window (use as starting point)
        self.prev_left_x = None
        self.prev_right_x = None
        
        # Track consecutive misses for fallback
        self.left_miss_count = 0
        self.right_miss_count = 0
        self.max_miss_frames = 10  # Use previous coeffs for up to 10 frames
        
        # Lane locking - remember lane position and don't jump to other lanes
        self.locked_left_x = None
        self.locked_right_x = None
        self.lock_tolerance = 100  # pixels - max allowed jump from locked position (match search_margin)
        self.lock_frames = 0
        self.lock_threshold = 5  # frames before locking
        
        if use_unet and model_path:
            self._load_unet(model_path)
    
    def _load_unet(self, model_path: str):
        """Load UNet model for lane segmentation."""
        try:
            import torch
            from perception.lane_detector import LaneUNet
            
            self.model = LaneUNet()
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            
            # Try GPU but fallback to CPU if OOM
            if torch.cuda.is_available():
                try:
                    # Check if enough GPU memory (need ~500MB)
                    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    if free_mem > 600 * 1024 * 1024:  # 600MB
                        self.model = self.model.cuda()
                        self.device = 'cuda'
                    else:
                        self.device = 'cpu'
                        print("GPU memory low, using CPU for UNet")
                except:
                    self.device = 'cpu'
            else:
                self.device = 'cpu'
        except Exception as e:
            print(f"Failed to load UNet: {e}")
            self.device = 'cpu'
            self.use_unet = False
    
    def _get_binary(self, rgb: np.ndarray) -> np.ndarray:
        """Get binary lane mask from image."""
        if self.use_unet and self.model is not None:
            return self._unet_binary(rgb)
        else:
            return self._edge_binary(rgb)
    
    def _unet_binary(self, rgb: np.ndarray) -> np.ndarray:
        """Get binary mask from UNet."""
        import torch
        
        with torch.no_grad():
            img_tensor = torch.FloatTensor(rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            prob_map = probs[0, 1].cpu().numpy()
        
        # Threshold
        binary = (prob_map > 0.15).astype(np.uint8) * 255
        return binary
    
    def _edge_binary(self, rgb: np.ndarray) -> np.ndarray:
        """Get binary mask using edge detection (fallback)."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges
    
    def process(self, rgb: np.ndarray, return_vis: bool = True
                ) -> Tuple[np.ndarray, float, float, float, Optional[np.ndarray]]:
        """
        Process frame and detect lanes.
        
        Returns:
            result: Image with lane overlay
            curvature: Radius of curvature (meters)
            offset: Center offset (meters, positive = right of center)
            confidence: Detection confidence
            bev_vis: BEV visualization (optional)
        """
        h, w = rgb.shape[:2]
        
        # 1. Get binary lane mask
        binary = self._get_binary(rgb)
        
        # 2. Apply ROI mask
        binary_roi = apply_roi_mask(binary)
        
        # 3. Warp to BEV
        binary_bev = to_bev(binary_roi)
        
        # 3.5 Dilate vertically to connect dash lines (critical for dash lane detection)
        # Use larger kernel to connect dashes that are far apart in BEV
        dilate_kernel_v = np.ones((160, 5), np.uint8)  # very tall kernel to connect dashes
        binary_bev = cv2.dilate(binary_bev, dilate_kernel_v)
        # Also dilate horizontally to thicken thin lines
        dilate_kernel_h = np.ones((5, 12), np.uint8)
        binary_bev = cv2.dilate(binary_bev, dilate_kernel_h)
        
        # 4. Find lane starts with ego-lane selection (multi-peak + center proximity)
        left_start, right_start = find_lane_starts(
            binary_bev, 
            locked_left=self.locked_left_x, 
            locked_right=self.locked_right_x
        )
        
        # Lane locking DISABLED - allow histogram to guide directly
        # This prevents getting stuck during dash gaps
        # if self.locked_left_x is not None and left_start is not None:
        #     if abs(left_start - self.locked_left_x) > self.lock_tolerance:
        #         # New detection is too far from locked position - reject it
        #         left_start = self.locked_left_x
        # 
        # if self.locked_right_x is not None and right_start is not None:
        #     if abs(right_start - self.locked_right_x) > self.lock_tolerance:
        #         # New detection is too far from locked position - reject it
        #         right_start = self.locked_right_x
        
        # Use previous lane position as fallback if histogram fails
        if left_start is None and self.prev_left_x is not None:
            left_start = self.prev_left_x
        if right_start is None and self.prev_right_x is not None:
            right_start = self.prev_right_x
        
        # 5. Sliding window fit
        vis_bev = cv2.cvtColor(binary_bev, cv2.COLOR_GRAY2BGR) if return_vis else None
        
        left_coeffs, left_wins, left_active = None, [], 0
        right_coeffs, right_wins, right_active = None, [], 0
        
        if left_start is not None:
            left_coeffs, left_wins, left_active = sliding_window_fit(
                binary_bev, left_start, vis_bev, (255, 100, 0), side='left')
            # Save position for next frame
            if left_coeffs is not None:
                new_left_x = int(np.polyval(left_coeffs, BEV_H - 1))
                self.prev_left_x = new_left_x
                # Update locked position (gradual adjustment)
                if self.locked_left_x is None:
                    self.locked_left_x = new_left_x
                else:
                    self.locked_left_x = int(0.9 * self.locked_left_x + 0.1 * new_left_x)
        
        if right_start is not None:
            right_coeffs, right_wins, right_active = sliding_window_fit(
                binary_bev, right_start, vis_bev, (0, 100, 255), side='right')
            # Save position for next frame
            if right_coeffs is not None:
                new_right_x = int(np.polyval(right_coeffs, BEV_H - 1))
                self.prev_right_x = new_right_x
                # Update locked position (gradual adjustment)
                if self.locked_right_x is None:
                    self.locked_right_x = new_right_x
                else:
                    self.locked_right_x = int(0.9 * self.locked_right_x + 0.1 * new_right_x)
        
        # 6. Constrain pair
        left_coeffs, right_coeffs = constrain_lane_pair(
            left_coeffs, right_coeffs, left_active, right_active)
        
        # 7. Curve memory management - ALWAYS ACTIVE to maintain curve shape
        total_active = left_active + right_active
        active_ratio = total_active / (2 * N_WINDOWS)
        
        # ALWAYS use curve memory to maintain smooth curves
        if self.curve_memory_left is not None and left_coeffs is not None:
            # Always blend with remembered curve for smoothness
            if active_ratio < 0.7:  # Low detection - strong memory
                left_coeffs = 0.1 * left_coeffs + 0.9 * self.curve_memory_left
            else:  # Good detection - light memory for smoothness
                left_coeffs = 0.7 * left_coeffs + 0.3 * self.curve_memory_left
        elif left_coeffs is None and self.curve_memory_left is not None:
            # Use remembered curve completely
            left_coeffs = self.curve_memory_left.copy()
        
        if self.curve_memory_right is not None and right_coeffs is not None:
            # Always blend with remembered curve for smoothness
            if active_ratio < 0.7:  # Low detection - strong memory
                right_coeffs = 0.1 * right_coeffs + 0.9 * self.curve_memory_right
            else:  # Good detection - light memory for smoothness
                right_coeffs = 0.7 * right_coeffs + 0.3 * self.curve_memory_right
        elif right_coeffs is None and self.curve_memory_right is not None:
            # Use remembered curve completely
            right_coeffs = self.curve_memory_right.copy()
        
        # Update curve memory with latest good detection
        if left_coeffs is not None:
            if self.curve_memory_left is None:
                self.curve_memory_left = left_coeffs.copy()
            else:
                # Gradually update memory
                self.curve_memory_left = 0.8 * self.curve_memory_left + 0.2 * left_coeffs
        
        if right_coeffs is not None:
            if self.curve_memory_right is None:
                self.curve_memory_right = right_coeffs.copy()
            else:
                # Gradually update memory
                self.curve_memory_right = 0.8 * self.curve_memory_right + 0.2 * right_coeffs
        
        # 8. Simple temporal smoothing with fallback
        if left_coeffs is not None:
            if self.prev_left_coeffs is not None:
                left_coeffs = self.alpha * left_coeffs + (1 - self.alpha) * self.prev_left_coeffs
            self.prev_left_coeffs = left_coeffs.copy()
            self.left_miss_count = 0
        else:
            # Use previous coeffs if we have history
            self.left_miss_count += 1
            if self.left_miss_count <= self.max_miss_frames and self.prev_left_coeffs is not None:
                left_coeffs = self.prev_left_coeffs.copy()
        
        if right_coeffs is not None:
            if self.prev_right_coeffs is not None:
                right_coeffs = self.alpha * right_coeffs + (1 - self.alpha) * self.prev_right_coeffs
            self.prev_right_coeffs = right_coeffs.copy()
            self.right_miss_count = 0
        else:
            # Use previous coeffs if we have history
            self.right_miss_count += 1
            if self.right_miss_count <= self.max_miss_frames and self.prev_right_coeffs is not None:
                right_coeffs = self.prev_right_coeffs.copy()
        
        # 8. Compute metrics
        curvature, offset = compute_curvature_and_offset(left_coeffs, right_coeffs)
        
        # 9. Draw overlay
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = draw_lane_overlay(bgr, left_coeffs, right_coeffs)
        
        # Confidence
        confidence = min(1.0, (left_active + right_active) / (2 * N_WINDOWS) + 0.3)
        if left_coeffs is None or right_coeffs is None:
            confidence *= 0.5
        
        return result, curvature, offset, confidence, vis_bev


def test_addison_pipeline():
    """Test the Addison lane detection pipeline."""
    import sys
    sys.path.insert(0, '/home/supawich/Desktop/CARLA_0.9.16/carla_mpc_classical')
    
    detector = AddisonLaneDetector(
        use_unet=True, 
        model_path='/home/supawich/Desktop/CARLA_0.9.16/carla_mpc_classical/model/lane_unet_final.pth'
    )
    
    # Connect to CARLA
    import carla
    import queue
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    
    # Spawn vehicle
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[50])
    
    # Attach camera
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')
    cam_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    
    img_queue = queue.Queue()
    camera.listen(lambda img: img_queue.put(img))
    
    try:
        for _ in range(5):
            world.tick()
        
        for frame in range(100):
            world.tick()
            
            try:
                img = img_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Convert to RGB
            raw = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))
            rgb = raw[:, :, 2::-1].copy()
            
            # Process
            result, curvature, offset, confidence, bev_vis = detector.process(rgb)
            
            # Display
            cv2.putText(result, f"Curve Radius: {curvature:.1f} m", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, f"Center Offset: {offset*100:.2f} cm", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, f"Confidence: {confidence:.0%}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Addison Lane Detection", result)
            if bev_vis is not None:
                cv2.imshow("BEV", bev_vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_addison_pipeline()
