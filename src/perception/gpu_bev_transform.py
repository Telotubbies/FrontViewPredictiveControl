#!/usr/bin/env python3
"""
GPU-Accelerated BEV Transform using Kornia
Eliminate GPU→CPU→GPU transfers for better performance
"""

import torch
import numpy as np
import cv2
import kornia.geometry.transform as K
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def bev_transform_gpu(mask: torch.Tensor, 
                      M: torch.Tensor,
                      dsize: Tuple[int, int] = (450, 320)) -> torch.Tensor:
    """
    GPU-accelerated BEV transform using Kornia
    
    Args:
        mask: Input mask tensor (H, W) on GPU
        M: Transform matrix tensor (3, 3) on GPU  
        dsize: Output size (width, height)
        
    Returns:
        BEV transformed mask tensor on GPU
    """
    try:
        # Ensure mask is float and add batch/channel dims
        if mask.dtype != torch.float32:
            mask = mask.float()
        
        # Add batch and channel dims: (H, W) -> (1, 1, H, W)
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
        
        # Convert transform matrix to the right format
        # Kornia expects (batch, 3, 3)
        M_batch = M.unsqueeze(0)
        
        # Apply perspective transform
        # Note: Kornia dsize is (height, width) - opposite of OpenCV
        bev_mask = K.warp_perspective(
            mask_4d, 
            M_batch, 
            dsize=(dsize[1], dsize[0]),  # (height, width)
            mode='bilinear',
            padding_mode='zeros'
        )
        
        # Remove batch/channel dims: (1, 1, H, W) -> (H, W)
        bev_mask = bev_mask.squeeze(0).squeeze(0)
        
        return bev_mask
        
    except Exception as e:
        logger.error(f"GPU BEV transform failed: {e}")
        # Fallback to CPU
        return bev_transform_cpu_fallback(mask.cpu().numpy(), M.cpu().numpy(), dsize)


def bev_transform_cpu_fallback(mask: np.ndarray, 
                              M: np.ndarray,
                              dsize: Tuple[int, int] = (450, 320)) -> torch.Tensor:
    """
    Fallback CPU BEV transform (for compatibility)
    """
    bev = cv2.warpPerspective(
        (mask > 0).astype(np.uint8) * 255,
        M, dsize,
        flags=cv2.INTER_NEAREST
    )
    
    # Convert back to tensor on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.from_numpy(bev).float().to(device)


def get_bev_transform_matrix_gpu(cam_w: int, cam_h: int,
                               src_top_ratio: float = 0.45,
                               src_bot_ratio: float = 0.97,
                               src_top_margin: float = 0.42,
                               src_bot_margin: float = 0.10,
                               device: torch.device = None) -> torch.Tensor:
    """
    Get BEV transform matrix as GPU tensor
    
    Args:
        cam_w, cam_h: Camera dimensions
        src_* ratios/margins: Transform parameters
        device: Target device (defaults to CUDA if available)
        
    Returns:
        Transform matrix tensor (3, 3) on specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Source points (camera trapezoid)
    top_y = int(cam_h * src_top_ratio)
    bot_y = int(cam_h * src_bot_ratio)
    
    src = torch.tensor([
        [int(cam_w * src_bot_margin),       bot_y],   # bottom-left
        [int(cam_w * (1 - src_bot_margin)), bot_y],   # bottom-right  
        [int(cam_w * (1 - src_top_margin)), top_y],   # top-right
        [int(cam_w * src_top_margin),       top_y],   # top-left
    ], dtype=torch.float32, device=device)
    
    # Destination points (BEV rectangle)
    dst = torch.tensor([
        [0,           cam_h - 1],
        [cam_w - 1,   cam_h - 1],
        [cam_w - 1,   0],
        [0,           0],
    ], dtype=torch.float32, device=device)
    
    # Compute perspective transform matrix
    # Using OpenCV for compatibility, then convert to tensor
    import cv2
    M_cv = cv2.getPerspectiveTransform(src.cpu().numpy(), dst.cpu().numpy())
    M_tensor = torch.from_numpy(M_cv).float().to(device)
    
    return M_tensor


def test_gpu_bev_transform():
    """Test GPU BEV transform performance"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Testing GPU BEV transform on {device}")
        
        # Create test mask
        mask_np = np.zeros((480, 640), dtype=np.uint8)
        mask_np[100:380, 200:440] = 255  # Simple lane
        
        # Convert to tensor
        mask = torch.from_numpy(mask_np).float().to(device) / 255.0
        
        # Get transform matrix
        M = get_bev_transform_matrix_gpu(640, 480, device=device)
        
        # Apply GPU transform
        import time
        start_time = time.time()
        
        bev_mask = bev_transform_gpu(mask, M, (450, 320))
        
        gpu_time = time.time() - start_time
        logger.info(f"GPU BEV transform: {gpu_time*1000:.2f}ms")
        
        # Test CPU fallback
        start_time = time.time()
        bev_cpu = bev_transform_cpu_fallback(mask_np, M.cpu().numpy(), (450, 320))
        cpu_time = time.time() - start_time
        logger.info(f"CPU BEV transform: {cpu_time*1000:.2f}ms")
        
        logger.info(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"GPU BEV transform test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_gpu_bev_transform()
