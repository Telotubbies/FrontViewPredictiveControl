"""
Device utilities for GPU detection (CUDA/ROCm/DirectML).
"""

import torch
import logging
import os

logger = logging.getLogger(__name__)

# Try to import DirectML for AMD GPUs on Windows
try:
    import torch_directml
    _DML_AVAILABLE = torch_directml.device_count() > 0
except ImportError:
    torch_directml = None
    _DML_AVAILABLE = False


def get_device() -> torch.device:
    """
    Get the best available device (DirectML > ROCm > CUDA > CPU).

    Returns:
        torch.device object
    """
    # Force CPU if FORCE_CPU env var is set (avoid CUDA OOM when training)
    if os.environ.get('FORCE_CPU', '0') == '1':
        device = torch.device('cpu')
        logger.info("⚠️  FORCE_CPU=1, using CPU")
        return device

    # Check for DirectML (AMD GPU on Windows)
    if _DML_AVAILABLE:
        device = torch_directml.device()
        logger.info(f"✅ Using DirectML device: {torch_directml.device_name(0)}")
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


def is_directml_available() -> bool:
    """Check if DirectML is available (AMD GPU on Windows)."""
    return _DML_AVAILABLE


def is_rocm_available() -> bool:
    """Check if ROCm is available."""
    return hasattr(torch.version, 'hip') and torch.version.hip is not None and torch.cuda.is_available()


def is_cuda_available() -> bool:
    """Check if CUDA is available (NVIDIA)."""
    return torch.cuda.is_available() and not is_rocm_available() and not _DML_AVAILABLE


def get_device_info() -> dict:
    """Get device information."""
    info = {
        'device_type': 'cpu',
        'device_name': 'CPU',
        'is_rocm': False,
        'is_cuda': False,
        'is_directml': False,
        'is_cpu': True
    }

    if _DML_AVAILABLE:
        info['device_type'] = 'directml'
        info['device_name'] = torch_directml.device_name(0)
        info['is_directml'] = True
        info['is_cpu'] = False
    elif is_rocm_available():
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
