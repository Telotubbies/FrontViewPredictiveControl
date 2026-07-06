"""
Device utilities for GPU detection (CUDA/ROCm).
"""

import torch
import logging
import os

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Get the best available device (ROCm > CUDA > CPU).
    
    Returns:
        torch.device object
    """
    # Force CPU if FORCE_CPU env var is set (avoid CUDA OOM when training)
    if os.environ.get('FORCE_CPU', '0') == '1':
        device = torch.device('cpu')
        logger.info("⚠️  FORCE_CPU=1, using CPU")
        return device
    
    # Check for ROCm (AMD GPU)
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"✅ Using ROCm device: {torch.cuda.get_device_name(0)}")
            return device
    
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    
    # Fallback to CPU
    device = torch.device('cpu')
    logger.info("⚠️  No GPU available, using CPU")
    return device


def is_rocm_available() -> bool:
    """Check if ROCm is available."""
    return hasattr(torch.version, 'hip') and torch.version.hip is not None and torch.cuda.is_available()


def is_cuda_available() -> bool:
    """Check if CUDA is available (NVIDIA)."""
    return torch.cuda.is_available() and not is_rocm_available()


def get_device_info() -> dict:
    """Get device information."""
    info = {
        'device_type': 'cpu',
        'device_name': 'CPU',
        'is_rocm': False,
        'is_cuda': False,
        'is_cpu': True
    }
    
    if is_rocm_available():
        info['device_type'] = 'rocm'
        info['device_name'] = torch.cuda.get_device_name(0)
        info['is_rocm'] = True
        info['is_cpu'] = False
    elif is_cuda_available():
        info['device_type'] = 'cuda'
        info['device_name'] = torch.cuda.get_device_name(0)
        info['is_cuda'] = True
        info['is_cpu'] = False
    
    return info

