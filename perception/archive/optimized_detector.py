#!/usr/bin/env python3
"""
RTX 3080 Ti Optimized Lane Detector

Optimizations:
- FP16 mixed precision
- TensorRT acceleration  
- CUDA streams
- Memory pooling
- Batch processing
"""

import logging
import time
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

try:
    import torch2trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT not available, using PyTorch only")

from utils.type_hints import CameraFrame, PerceptionResult
from config_clean import get_config

logger = logging.getLogger(__name__)


class OptimizedLaneDetector:
    """RTX 3080 Ti optimized lane detection"""
    
    def __init__(self, 
                 model_path: str,
                 use_tensorrt: bool = True,
                 use_fp16: bool = True,
                 batch_size: int = 4):
        self.model_path = model_path
        self.use_tensorrt = use_tensorrt and TENSORRT_AVAILABLE
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        
        # Configuration
        self.config = get_config()
        
        # Model and optimization
        self.model: Optional[nn.Module] = None
        self.model_trt = None
        
        # CUDA streams for parallel processing
        self.streams = [
            torch.cuda.Stream() for _ in range(3)
        ]
        
        # Memory pool
        self.memory_pool = {}
        
        # Performance metrics
        self.inference_times = []
        self.transfer_times = []
        
        # Initialize
        self._initialize_model()
        self._setup_memory_pool()
    
    def _initialize_model(self):
        """Initialize and optimize model"""
        try:
            # Load base model
            self.model = torch.load(self.model_path, map_location='cuda')
            self.model.cuda()
            self.model.eval()
            
            # Disable gradients for inference
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Convert to FP16 if enabled
            if self.use_fp16:
                self.model.half()
                logger.info("Model converted to FP16")
            
            # TensorRT optimization
            if self.use_tensorrt:
                self._optimize_with_tensorrt()
            
            # Warm up
            self._warm_up()
            
            logger.info(f"Optimized detector initialized: "
                       f"FP16={self.use_fp16}, TensorRT={self.use_tensorrt}")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized detector: {e}")
            raise
    
    def _optimize_with_tensorrt(self):
        """Optimize model with TensorRT"""
        try:
            # Create sample input
            sample_input = torch.randn(
                1, 3, self.config.CAM_H, self.config.CAM_W,
                device='cuda',
                dtype=torch.float16 if self.use_fp16 else torch.float32
            )
            
            # Convert to TensorRT
            self.model_trt = torch2trt(
                self.model,
                [sample_input],
                fp16_mode=self.use_fp16,
                max_batch_size=self.batch_size,
                max_workspace_size=1 << 30  # 1GB
            )
            
            logger.info("Model optimized with TensorRT")
            
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}, using PyTorch")
            self.use_tensorrt = False
    
    def _setup_memory_pool(self):
        """Setup memory pool for reusable tensors"""
        try:
            # Pre-allocate commonly used tensors
            self.memory_pool = {
                'input_fp32': torch.empty(
                    (self.batch_size, 3, self.config.CAM_H, self.config.CAM_W),
                    dtype=torch.float32, device='cuda'
                ),
                'input_fp16': torch.empty(
                    (self.batch_size, 3, self.config.CAM_H, self.config.CAM_W),
                    dtype=torch.float16, device='cuda'
                ) if self.use_fp16 else None,
                'output': torch.empty(
                    (self.batch_size, 1, self.config.CAM_H, self.config.CAM_W),
                    dtype=torch.float32, device='cuda'
                )
            }
            
            logger.info("Memory pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup memory pool: {e}")
    
    def _warm_up(self):
        """Warm up model and GPU"""
        try:
            logger.info("Warming up model...")
            
            with torch.no_grad():
                for _ in range(5):
                    dummy_input = torch.randn(
                        1, 3, self.config.CAM_H, self.config.CAM_W,
                        device='cuda',
                        dtype=torch.float16 if self.use_fp16 else torch.float32
                    )
                    
                    if self.use_tensorrt and self.model_trt:
                        _ = self.model_trt(dummy_input)
                    else:
                        with autocast(enabled=self.use_fp16):
                            _ = self.model(dummy_input)
            
            torch.cuda.synchronize()
            logger.info("Model warm-up completed")
            
        except Exception as e:
            logger.error(f"Warm-up failed: {e}")
    
    def process_frame(self, frame: CameraFrame) -> np.ndarray:
        """Process single frame with optimization"""
        start_time = time.time()
        
        try:
            # Convert frame to tensor
            input_tensor = self._frame_to_tensor(frame.rgb)
            
            # Run inference
            with torch.no_grad():
                if self.use_tensorrt and self.model_trt:
                    output = self.model_trt(input_tensor)
                else:
                    with autocast(enabled=self.use_fp16):
                        output = self.model(input_tensor)
            
            # Convert to numpy
            mask = output.squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            
            # Record performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return mask
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            # Return fallback
            return np.zeros((self.config.CAM_H, self.config.CAM_W), dtype=np.uint8)
    
    def process_batch(self, frames: List[CameraFrame]) -> List[np.ndarray]:
        """Process batch of frames for maximum throughput"""
        if len(frames) != self.batch_size:
            # Fall back to individual processing
            return [self.process_frame(frame) for frame in frames]
        
        start_time = time.time()
        
        try:
            # Stack frames into batch
            batch_tensor = torch.stack([
                self._frame_to_tensor(frame.rgb) for frame in frames
            ])
            
            # Run batch inference
            with torch.no_grad():
                if self.use_tensorrt and self.model_trt:
                    batch_output = self.model_trt(batch_tensor)
                else:
                    with autocast(enabled=self.use_fp16):
                        batch_output = self.model(batch_tensor)
            
            # Convert to list of masks
            masks = []
            for i in range(self.batch_size):
                mask = batch_output[i].squeeze().cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8)
                masks.append(mask)
            
            # Record performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return masks
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [np.zeros((self.config.CAM_H, self.config.CAM_W), dtype=np.uint8) 
                   for _ in frames]
    
    def _frame_to_tensor(self, rgb_array: np.ndarray) -> torch.Tensor:
        """Convert RGB array to tensor with memory optimization"""
        try:
            # Resize if needed
            if rgb_array.shape[:2] != (self.config.CAM_H, self.config.CAM_W):
                import cv2
                rgb_array = cv2.resize(rgb_array, (self.config.CAM_W, self.config.CAM_H))
            
            # Convert to tensor
            tensor = torch.from_numpy(rgb_array).permute(2, 0, 1)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            # Convert to FP16 if needed
            if self.use_fp16:
                tensor = tensor.half()
            
            # Move to GPU
            tensor = tensor.cuda(non_blocking=True)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Tensor conversion error: {e}")
            # Return fallback tensor
            dtype = torch.float16 if self.use_fp16 else torch.float32
            return torch.zeros(
                1, 3, self.config.CAM_H, self.config.CAM_W,
                dtype=dtype, device='cuda'
            )
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics"""
        metrics = {
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'min_inference_time': np.min(self.inference_times) if self.inference_times else 0,
            'max_inference_time': np.max(self.inference_times) if self.inference_times else 0,
            'total_inferences': len(self.inference_times),
            'fp16_enabled': self.use_fp16,
            'tensorrt_enabled': self.use_tensorrt,
            'batch_size': self.batch_size
        }
        
        if self.inference_times:
            metrics['fps'] = 1.0 / metrics['avg_inference_time']
            metrics['throughput_fps'] = metrics['fps'] * self.batch_size
        
        return metrics
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.inference_times.clear()
        self.transfer_times.clear()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear memory pool
            for tensor in self.memory_pool.values():
                if tensor is not None:
                    del tensor
            self.memory_pool.clear()
            
            # Clear model
            if self.model is not None:
                del self.model
            if self.model_trt is not None:
                del self.model_trt
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            logger.info("Optimized detector cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


class BatchProcessingManager:
    """Manager for batch processing to maximize 3080 Ti utilization"""
    
    def __init__(self, 
                 detector: OptimizedLaneDetector,
                 batch_size: int = 4):
        self.detector = detector
        self.batch_size = batch_size
        self.frame_buffer = []
        self.result_callbacks = []
    
    def add_frame(self, frame: CameraFrame, callback):
        """Add frame to batch processing queue"""
        self.frame_buffer.append(frame)
        self.result_callbacks.append(callback)
        
        # Process batch when full
        if len(self.frame_buffer) >= self.batch_size:
            self._process_batch()
    
    def _process_batch(self):
        """Process current batch"""
        if not self.frame_buffer:
            return
        
        try:
            # Process batch
            masks = self.detector.process_batch(self.frame_buffer)
            
            # Call callbacks with results
            for mask, callback in zip(masks, self.result_callbacks):
                callback(mask)
            
            # Clear buffers
            self.frame_buffer.clear()
            self.result_callbacks.clear()
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Call callbacks with empty results
            for callback in self.result_callbacks:
                callback(np.zeros((480, 640), dtype=np.uint8))
            
            self.frame_buffer.clear()
            self.result_callbacks.clear()
    
    def flush(self):
        """Process remaining frames"""
        if self.frame_buffer:
            # Pad batch to required size
            while len(self.frame_buffer) < self.batch_size:
                last_frame = self.frame_buffer[-1]
                self.frame_buffer.append(last_frame)
                self.result_callbacks.append(lambda x: None)
            
            self._process_batch()
