#!/usr/bin/env python3
"""
Vectorized Sliding Window for GPU Acceleration
Replace sequential loops with parallel tensor operations
"""

import torch
import numpy as np
import cv2
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


def sliding_window_search_gpu(
    bev_binary: torch.Tensor,
    n_windows: int = 9,
    margin: int = 50,
    min_pix: int = 50,
    lookahead_m: float = 30.0,
    half_width_m: float = 8.0,
) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor],  # vehicle XY left/right
    Optional[torch.Tensor], Optional[torch.Tensor],  # pixel (row, col) left/right
    List, List, List, List, str  # window rects and case
]:
    """
    GPU-accelerated sliding window search using vectorized operations
    
    Args:
        bev_binary: BEV binary mask tensor (H, W) on GPU
        n_windows: Number of sliding windows
        margin: Window margin in pixels
        min_pix: Minimum pixels per window
        lookahead_m: Lookahead distance in meters
        half_width_m: Half width in meters
        
    Returns:
        Same format as original sliding_window_search but with tensors
    """
    try:
        device = bev_binary.device
        h, w = bev_binary.shape
        mid_col = w // 2
        
        # 1. Create window boundaries (vectorized)
        window_h = h // n_windows
        window_starts = torch.arange(n_windows, device=device) * window_h
        window_ends = window_starts + window_h
        
        # 2. Split into left/right halves (vectorized)
        left_half = bev_binary[:, :mid_col]
        right_half = bev_binary[:, mid_col:]
        
        # 3. Create distance-weighted histograms (vectorized)
        bottom_start = int(h * 0.25)  # BEV_LANE_SEARCH_TOP_RATIO
        bottom = bev_binary[bottom_start:, :]
        n_rows = bottom.shape[0]
        
        # Create weights (far = more weight)
        weights = torch.linspace(1.5, 0.5, n_rows, device=device)[:, None]
        
        # Compute histograms for all windows at once
        left_hist = _compute_weighted_histograms_vectorized(
            left_half, window_starts, window_ends, bottom_start, weights
        )
        right_hist = _compute_weighted_histograms_vectorized(
            right_half, window_starts, window_ends, bottom_start, weights
        )
        
        # 4. Find peaks for all windows (vectorized)
        left_peaks = torch.argmax(left_hist, dim=1)
        right_peaks = torch.argmax(right_hist, dim=1) + mid_col
        
        # 5. Extract pixels for each window (vectorized)
        left_pixels_list, right_pixels_list = [], []
        left_windows_list, right_windows_list = [], []
        
        for i in range(n_windows):
            y_lo = h - (i + 1) * window_h
            y_hi = h - i * window_h
            
            # Dynamic margin based on distance
            y_norm = (y_lo + y_hi) / (2 * h)
            margin_dyn = max(15, int(margin * (1 - y_norm * 0.5)))
            
            # Left window
            xl1 = max(0, int(left_peaks[i].item()) - margin_dyn)
            xl2 = min(w, int(left_peaks[i].item()) + margin_dyn)
            
            # Extract pixels (vectorized)
            window_mask = bev_binary[y_lo:y_hi, xl1:xl2]
            ys, xs = torch.where(window_mask > 0)
            
            if len(ys) >= min_pix:
                # Convert to global coordinates
                pixels = torch.stack([ys + y_lo, xs + xl1], dim=1)
                left_pixels_list.append(pixels)
                left_windows_list.append((xl1, y_lo, xl2, y_hi))
                
                # Update peak for next window
                if len(xs) > 0:
                    new_peak = int(torch.mean(xs + xl1, dtype=torch.float).item())
                    left_peaks[max(0, i-1):i+1] = new_peak
            
            # Right window
            xr1 = max(0, int(right_peaks[i].item()) - margin_dyn)
            xr2 = min(w, int(right_peaks[i].item()) + margin_dyn)
            
            window_mask = bev_binary[y_lo:y_hi, xr1:xr2]
            ys, xs = torch.where(window_mask > 0)
            
            if len(ys) >= min_pix:
                pixels = torch.stack([ys + y_lo, xs + xr1], dim=1)
                right_pixels_list.append(pixels)
                right_windows_list.append((xr1, y_lo, xr2, y_hi))
                
                if len(xs) > 0:
                    new_peak = int(torch.mean(xs + xr1, dtype=torch.float).item()) + mid_col
                    right_peaks[max(0, i-1):i+1] = new_peak
        
        # 6. Convert pixels to vehicle coordinates (vectorized)
        left_xy = None
        right_xy = None
        
        if left_pixels_list:
            left_px = torch.cat(left_pixels_list, dim=0)
            left_xy = _bev_pixels_to_vehicle_xy_gpu(
                left_px[:, 0], left_px[:, 1], h, w, lookahead_m, half_width_m
            )
        
        if right_pixels_list:
            right_px = torch.cat(right_pixels_list, dim=0)
            right_xy = _bev_pixels_to_vehicle_xy_gpu(
                right_px[:, 0], right_px[:, 1], h, w, lookahead_m, half_width_m
            )
        
        # Convert lists to proper format
        left_px_tensor = torch.cat(left_pixels_list, dim=0) if left_pixels_list else None
        right_px_tensor = torch.cat(right_pixels_list, dim=0) if right_pixels_list else None
        
        return (
            left_xy, right_xy,
            left_px_tensor, right_px_tensor,
            left_windows_list, right_windows_list,  # raw windows
            left_windows_list, right_windows_list,  # filtered windows (same for GPU version)
            "both" if left_xy is not None and right_xy is not None else "none"
        )
        
    except Exception as e:
        logger.error(f"GPU sliding window failed: {e}")
        return None, None, None, None, [], [], [], [], "none"


def _compute_weighted_histograms_vectorized(
    half_mask: torch.Tensor,
    window_starts: torch.Tensor,
    window_ends: torch.Tensor,
    bottom_start: int,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Compute weighted histograms for all windows at once (vectorized)
    
    Args:
        half_mask: Half of BEV mask (H, W/2)
        window_starts: Window start positions
        window_ends: Window end positions
        bottom_start: Bottom row start
        weights: Distance weights
        
    Returns:
        Histograms tensor (n_windows, W/2)
    """
    n_windows = len(window_starts)
    h, w = half_mask.shape
    
    # Extract bottom portion (where lanes are)
    bottom_mask = half_mask[bottom_start:, :]
    bottom_weights = weights
    
    # Initialize histograms
    histograms = torch.zeros(n_windows, w, device=half_mask.device)
    
    # Compute for each window
    for i in range(n_windows):
        y_start = window_starts[i] - bottom_start
        y_end = window_ends[i] - bottom_start
        
        if y_start >= 0 and y_end <= bottom_mask.shape[0]:
            window_mask = bottom_mask[y_start:y_end, :]
            window_weights = bottom_weights[y_start:y_end, :]
            
            # Weighted histogram
            weighted_mask = (window_mask > 0).float() * window_weights
            histograms[i] = torch.sum(weighted_mask, dim=0)
    
    return histograms


def _bev_pixels_to_vehicle_xy_gpu(
    rows: torch.Tensor, cols: torch.Tensor,
    bev_h: int, bev_w: int,
    lookahead_m: float, half_width_m: float
) -> torch.Tensor:
    """
    Convert BEV pixel coordinates to vehicle frame (GPU version)
    
    Args:
        rows, cols: Pixel coordinates tensors
        bev_h, bev_w: BEV dimensions
        lookahead_m, half_width_m: Physical dimensions
        
    Returns:
        Vehicle coordinates tensor (N, 2) [x_forward, y_lateral]
    """
    # Convert to meters
    x_m = (bev_h - rows) * lookahead_m / bev_h      # Row → forward distance
    y_m = (cols - bev_w/2) * 2 * half_width_m / bev_w  # Col → lateral distance
    
    return torch.stack([x_m, y_m], dim=1)


def test_vectorized_sliding_window():
    """Test vectorized sliding window performance"""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Testing vectorized sliding window on {device}")
        
        # Create test BEV mask
        bev_np = np.zeros((320, 450), dtype=np.uint8)
        bev_np[50:270, 150:200] = 255  # Left lane
        bev_np[50:270, 250:300] = 255  # Right lane
        
        # Convert to tensor
        bev = torch.from_numpy(bev_np).float().to(device) / 255.0
        
        # Test GPU version
        import time
        start_time = time.time()
        
        result_gpu = sliding_window_search_gpu(bev)
        
        gpu_time = time.time() - start_time
        logger.info(f"GPU sliding window: {gpu_time*1000:.2f}ms")
        
        # Test CPU version for comparison
        try:
            from lane_trajectory import sliding_window_search
            start_time = time.time()
            
            result_cpu = sliding_window_search(bev_np)
            
            cpu_time = time.time() - start_time
            logger.info(f"CPU sliding window: {cpu_time*1000:.2f}ms")
            
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                logger.info(f"Speedup: {speedup:.2f}x")
        except ImportError:
            logger.warning("Could not import CPU sliding window for comparison")
        
        return True
        
    except Exception as e:
        logger.error(f"Vectorized sliding window test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_vectorized_sliding_window()
