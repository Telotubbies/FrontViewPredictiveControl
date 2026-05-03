#!/usr/bin/env python3
"""
RTX 3080 Ti Optimization Setup Script

Installs and configures:
- TensorRT
- CuPy
- Numba CUDA
- PyTorch optimizations
- CUDA kernels
"""

import subprocess
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def check_cuda_version():
    """Check CUDA version and compatibility"""
    try:
        import torch
        cuda_version = torch.version.cuda
        compute_capability = torch.cuda.get_device_capability()
        
        logger.info(f"CUDA Version: {cuda_version}")
        logger.info(f"Compute Capability: {compute_capability}")
        
        # Check 3080 Ti compatibility
        if compute_capability[0] >= 8:  # Ampere and later
            logger.info("✅ RTX 3080 Ti fully supported")
            return True
        else:
            logger.warning("⚠️ GPU may not support all optimizations")
            return False
            
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False


def install_tensorrt():
    """Install TensorRT for maximum performance"""
    logger.info("🔧 Installing TensorRT...")
    
    try:
        # Check if TensorRT is available
        import tensorrt
        logger.info("✅ TensorRT already installed")
        return True
    except ImportError:
        pass
    
    # Install torch2trt (TensorRT wrapper for PyTorch)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torch2trt"
        ])
        logger.info("✅ torch2trt installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install torch2trt: {e}")
        return False


def install_cupy():
    """Install CuPy for CUDA programming"""
    logger.info("🔧 Installing CuPy...")
    
    try:
        import cupy
        logger.info("✅ CuPy already installed")
        return True
    except ImportError:
        pass
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "cupy-cuda11x"  # Adjust for CUDA version
        ])
        logger.info("✅ CuPy installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install CuPy: {e}")
        return False


def install_numba_cuda():
    """Install Numba CUDA for custom kernels"""
    logger.info("🔧 Installing Numba CUDA...")
    
    try:
        from numba import cuda
        logger.info("✅ Numba CUDA already installed")
        return True
    except ImportError:
        pass
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "numba"
        ])
        logger.info("✅ Numba CUDA installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install Numba CUDA: {e}")
        return False


def optimize_pytorch():
    """Optimize PyTorch settings for 3080 Ti"""
    logger.info("⚙️ Optimizing PyTorch settings...")
    
    try:
        import torch
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        
        logger.info("✅ PyTorch optimizations enabled")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to optimize PyTorch: {e}")
        return False


def create_optimized_config():
    """Create optimized configuration for 3080 Ti"""
    logger.info("📝 Creating optimized configuration...")
    
    config_content = """
# RTX 3080 Ti Optimized Configuration
camera:
  CAM_W: 1280  # Higher resolution for better accuracy
  CAM_H: 720
  CAM_FOV_DEG: 110.0
  UNET_INPUT_W: 1280
  UNET_INPUT_H: 720
  UNET_INPUT_W_INFER: 640   # Lower resolution for inference
  UNET_INPUT_H_INFER: 360
  TARGET_SPEED_KMH: 30.0    # Higher target speed

# Optimized perception settings
perception:
  BATCH_SIZE: 4              # Process 4 frames simultaneously
  USE_TENSORRT: true         # Enable TensorRT
  USE_FP16: true             # Use mixed precision
  USE_CUDA_STREAMS: true     # Enable parallel processing
  MEMORY_POOL_SIZE: 1000     # Pre-allocated memory pool

# MPC optimizations
mpc:
  MPC_N: 12                  # Longer horizon for high speed
  MPC_DT: 0.08               # Shorter timestep for stability
  MPC_W_CTE: 400.0           # Higher CTE weight for accuracy
  MPC_W_HEADING: 100.0
  MPC_W_STEER_RATE: 1500.0   # Higher smoothness weight

# Performance settings
runtime:
  CONTROL_HZ: 15             # Higher control frequency
  USE_PERCEPTION_THREAD: true
  PERCEPTION_SKIP_FRAME: 1   # Process every frame
  MAIN_LOOP_SLEEP_S: 0.005   # Minimal sleep for high performance
  DASHBOARD_DRAW_EVERY_N: 2  # Reduce GUI overhead

# 3080 Ti specific optimizations
gpu:
  TENSOR_CORES_ENABLED: true
  MEMORY_BANDWIDTH_OPTIMIZED: true
  CUDA_KERNELS_ENABLED: true
  PARALLEL_PROCESSING: true
"""
    
    config_path = Path("config_3080ti.yaml")
    try:
        with open(config_path, 'w') as f:
            f.write(config_content)
        logger.info(f"✅ Optimized config created: {config_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create config: {e}")
        return False


def test_performance():
    """Test 3080 Ti performance"""
    logger.info("🧪 Testing RTX 3080 Ti performance...")
    
    try:
        import torch
        import time
        
        # Test matrix multiplication (Tensor Cores)
        size = 1024
        a = torch.randn(size, size, device='cuda', dtype=torch.float16)
        b = torch.randn(size, size, device='cuda', dtype=torch.float16)
        
        # Warm up
        for _ in range(5):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        gflops = (2 * size ** 3) / (avg_time * 1e9)
        
        logger.info(f"✅ Tensor Core Performance: {gflops:.1f} GFLOPS")
        
        # Test memory bandwidth
        data_size = 100 * 1024 * 1024  # 100MB
        data = torch.randn(data_size // 4, device='cuda')
        
        start_time = time.time()
        for _ in range(10):
            _ = torch.sum(data)
        torch.cuda.synchronize()
        end_time = time.time()
        
        bandwidth = (data_size * 10) / (end_time - start_time) / (1024**3)  # GB/s
        logger.info(f"✅ Memory Bandwidth: {bandwidth:.1f} GB/s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False


def main():
    """Main setup function"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    logger.info("🚀 RTX 3080 Ti Optimization Setup")
    logger.info("=" * 50)
    
    # Check CUDA compatibility
    if not check_cuda_version():
        logger.error("❌ CUDA compatibility check failed")
        return False
    
    # Install optimizations
    success = True
    
    success &= install_tensorrt()
    success &= install_cupy()
    success &= install_numba_cuda()
    success &= optimize_pytorch()
    success &= create_optimized_config()
    
    if success:
        logger.info("✅ Setup completed successfully!")
        logger.info("🧪 Running performance test...")
        test_performance()
        
        logger.info("🎉 RTX 3080 Ti optimizations ready!")
        logger.info("📝 Use config_3080ti.yaml for best performance")
        logger.info("🚀 Run with: python main_simple.py --config config_3080ti.yaml")
        
    else:
        logger.error("❌ Setup failed. Check logs for details.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
