#!/usr/bin/env python3
"""
RTX 3080 Ti Optimized Perception Manager

Maximizes GPU utilization with:
- Batch processing
- CUDA streams
- Tensor cores
- Memory pooling
- Parallel processing
"""

import logging
import threading
import time
import queue
from typing import Optional, Any, List, Dict
import numpy as np
import torch

from utils.type_hints import (
    CameraFrame, PerceptionResult, DetectionMode
)
from config_clean import get_config
from perception.optimized_detector import OptimizedLaneDetector, BatchProcessingManager
from utils.cuda_optimizations import cuda_optimizer, tensor_core_optimizer, memory_pool_manager

logger = logging.getLogger(__name__)


class RTX3080TiPerceptionManager:
    """RTX 3080 Ti optimized perception management"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 detection_mode: DetectionMode = DetectionMode.UNET,
                 batch_size: int = 4,
                 use_tensorrt: bool = True,
                 use_fp16: bool = True,
                 enable_cuda_optimizations: bool = True):
        
        self.model_path = model_path
        self.detection_mode = detection_mode
        self.batch_size = batch_size
        self.use_tensorrt = use_tensorrt
        self.use_fp16 = use_fp16
        self.enable_cuda_optimizations = enable_cuda_optimizations
        
        # Configuration
        self.config = get_config()
        
        # Optimized components
        self.detector: Optional[OptimizedLaneDetector] = None
        self.batch_manager: Optional[BatchProcessingManager] = None
        
        # Threading with CUDA streams
        self.frame_queue: queue.Queue = queue.Queue(maxsize=8)  # Larger queue for batching
        self.result_queue: queue.Queue = queue.Queue(maxsize=16)
        self.processing_thread: Optional[threading.Thread] = None
        
        # CUDA streams for parallel processing
        self.cuda_streams = [
            torch.cuda.Stream() for _ in range(3)
        ]
        
        # Performance tracking
        self.latest_result: Optional[PerceptionResult] = None
        self.frame_count = 0
        self.batch_count = 0
        self.start_time = time.time()
        
        # Initialize
        self._initialize_optimized_pipeline()
    
    def _initialize_optimized_pipeline(self):
        """Initialize optimized perception pipeline"""
        try:
            # Initialize optimized detector
            self.detector = OptimizedLaneDetector(
                model_path=self.model_path,
                use_tensorrt=self.use_tensorrt,
                use_fp16=self.use_fp16,
                batch_size=self.batch_size
            )
            
            # Initialize batch processing manager
            self.batch_manager = BatchProcessingManager(
                detector=self.detector,
                batch_size=self.batch_size
            )
            
            # Test CUDA optimizations
            if self.enable_cuda_optimizations:
                self._test_cuda_performance()
            
            logger.info(f"RTX 3080 Ti Perception Manager initialized: "
                       f"batch_size={self.batch_size}, "
                       f"TensorRT={self.use_tensorrt}, "
                       f"FP16={self.use_fp16}")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized perception: {e}")
            raise
    
    def _test_cuda_performance(self):
        """Test CUDA performance and optimizations"""
        try:
            # Test memory bandwidth
            bandwidth_stats = cuda_optimizer.memory_bandwidth_test()
            logger.info(f"Memory bandwidth: {bandwidth_stats}")
            
            # Test Tensor Core performance
            if tensor_core_optimizer.tensor_core_supported:
                logger.info("Tensor Cores available and optimized")
            
            # Test memory pool
            memory_stats = memory_pool_manager.get_memory_stats()
            logger.info(f"Memory pool: {memory_stats}")
            
        except Exception as e:
            logger.error(f"CUDA performance test failed: {e}")
    
    def start_processing(self):
        """Start optimized processing thread"""
        if self.processing_thread:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._optimized_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        logger.info("Optimized perception processing started")
    
    def stop_processing(self):
        """Stop processing thread"""
        if not self.processing_thread:
            return
        
        self.running = False
        self.processing_thread.join(timeout=3.0)
        self.processing_thread = None
        logger.info("Optimized perception processing stopped")
    
    def _optimized_processing_loop(self):
        """Optimized processing loop with CUDA streams"""
        batch_frames = []
        batch_callbacks = []
        
        while self.running:
            try:
                # Collect frames for batching
                try:
                    frame_data = self.frame_queue.get(timeout=0.01)
                    batch_frames.append(frame_data['frame'])
                    batch_callbacks.append(frame_data['callback'])
                except queue.Empty:
                    # Process batch if we have frames
                    if batch_frames:
                        self._process_batch_with_streams(batch_frames, batch_callbacks)
                        batch_frames.clear()
                        batch_callbacks.clear()
                    continue
                
                # Process batch when full or timeout
                if len(batch_frames) >= self.batch_size:
                    self._process_batch_with_streams(batch_frames, batch_callbacks)
                    batch_frames.clear()
                    batch_callbacks.clear()
                
            except Exception as e:
                logger.error(f"Optimized processing loop error: {e}")
                continue
    
    def _process_batch_with_streams(self, frames: List[CameraFrame], callbacks: List):
        """Process batch using CUDA streams for maximum parallelism"""
        if not frames:
            return
        
        start_time = time.time()
        
        try:
            # Use different CUDA streams for parallel operations
            with torch.cuda.stream(self.cuda_streams[0]):
                # Convert frames to tensors (Stream 0)
                tensors = []
                for frame in frames:
                    tensor = self._frame_to_tensor_optimized(frame.rgb)
                    tensors.append(tensor)
            
            with torch.cuda.stream(self.cuda_streams[1]):
                # Batch processing (Stream 1)
                batch_tensor = torch.stack(tensors)
                
                if self.use_tensorrt and self.detector.model_trt:
                    batch_output = self.detector.model_trt(batch_tensor)
                else:
                    with torch.cuda.amp.autocast(enabled=self.use_fp16):
                        batch_output = self.detector.model(batch_tensor)
            
            with torch.cuda.stream(self.cuda_streams[2]):
                # Post-processing and callbacks (Stream 2)
                results = []
                for i, frame in enumerate(frames):
                    mask = batch_output[i].squeeze().cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)
                    
                    # Create perception result
                    perception = self._create_perception_result(mask, frame)
                    results.append(perception)
                
                # Synchronize all streams
                torch.cuda.synchronize()
                
                # Call callbacks
                for result, callback in zip(results, callbacks):
                    if callback:
                        callback(result)
                
                # Update latest result
                self.latest_result = results[-1] if results else None
                self.batch_count += 1
            
            # Record performance
            batch_time = time.time() - start_time
            self.frame_count += len(frames)
            
            if self.batch_count % 10 == 0:
                fps = self.frame_count / (time.time() - self.start_time)
                logger.info(f"Batch processing: {self.batch_count} batches, "
                           f"{fps:.1f} FPS, {batch_time*1000:.1f}ms/batch")
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Call callbacks with empty results
            for callback in callbacks:
                if callback:
                    callback(None)
    
    def _frame_to_tensor_optimized(self, rgb_array: np.ndarray) -> torch.Tensor:
        """Optimized frame to tensor conversion"""
        try:
            # Use memory pool if available
            if self.enable_cuda_optimizations:
                # Get tensor from pool
                shape = (3, self.config.CAM_H, self.config.CAM_W)
                dtype = torch.float16 if self.use_fp16 else torch.float32
                tensor = memory_pool_manager.get_tensor(shape, dtype)
                
                # Convert and copy
                if rgb_array.shape[:2] != (self.config.CAM_H, self.config.CAM_W):
                    import cv2
                    rgb_array = cv2.resize(rgb_array, (self.config.CAM_W, self.config.CAM_H))
                
                # Convert HWC to CHW
                tensor_np = np.transpose(rgb_array, (2, 0, 1))
                tensor.copy_(torch.from_numpy(tensor_np))
                
                return tensor
            else:
                # Standard conversion
                if rgb_array.shape[:2] != (self.config.CAM_H, self.config.CAM_W):
                    import cv2
                    rgb_array = cv2.resize(rgb_array, (self.config.CAM_W, self.config.CAM_H))
                
                tensor = torch.from_numpy(rgb_array).permute(2, 0, 1)
                
                if self.use_fp16:
                    tensor = tensor.half()
                
                return tensor.cuda(non_blocking=True)
                
        except Exception as e:
            logger.error(f"Optimized tensor conversion error: {e}")
            # Return fallback tensor
            dtype = torch.float16 if self.use_fp16 else torch.float32
            return torch.zeros(
                3, self.config.CAM_H, self.config.CAM_W,
                dtype=dtype, device='cuda'
            )
    
    def _create_perception_result(self, mask: np.ndarray, frame: CameraFrame) -> PerceptionResult:
        """Create perception result from mask"""
        try:
            # Simplified trajectory computation
            # In production, this would use the full pipeline
            h, w = mask.shape
            
            # Find lane center (simplified)
            lane_pixels = np.where(mask > 0)
            if len(lane_pixels[0]) > 0:
                center_y = np.mean(lane_pixels[0])
                center_x = np.mean(lane_pixels[1])
                
                # Compute CTE (cross-track error)
                cte = (center_x - w/2) * 0.01  # Convert to meters
                
                # Compute heading error
                heading_error = 0.0  # Simplified
                
                # Compute curvature
                curvature = 0.0  # Simplified
                
                confidence = np.sum(mask > 0) / (h * w)
                geometry_valid = confidence > 0.1
                
            else:
                cte = 0.0
                heading_error = 0.0
                curvature = 0.0
                confidence = 0.0
                geometry_valid = False
            
            return PerceptionResult(
                cte=cte,
                heading_error=heading_error,
                curvature=curvature,
                confidence=confidence,
                geometry_valid=geometry_valid,
                lane_overlay=mask,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Perception result creation error: {e}")
            return PerceptionResult(
                cte=0.0, heading_error=0.0, curvature=0.0,
                confidence=0.0, geometry_valid=False,
                timestamp=time.time()
            )
    
    def add_frame(self, frame: CameraFrame, callback=None) -> bool:
        """Add frame to optimized processing queue"""
        frame_data = {
            'frame': frame,
            'callback': callback,
            'timestamp': time.time()
        }
        
        try:
            self.frame_queue.put_nowait(frame_data)
            return True
        except queue.Full:
            try:
                # Remove oldest frame
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame_data)
                return True
            except queue.Empty:
                return False
    
    def get_latest_result(self) -> Optional[PerceptionResult]:
        """Get latest perception result"""
        return self.latest_result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        base_metrics = {
            'frame_count': self.frame_count,
            'batch_count': self.batch_count,
            'batch_size': self.batch_size,
            'detection_mode': self.detection_mode.value,
            'runtime': time.time() - self.start_time
        }
        
        # Add detector metrics
        if self.detector:
            detector_metrics = self.detector.get_performance_metrics()
            base_metrics.update(detector_metrics)
        
        # Calculate throughput
        if base_metrics['runtime'] > 0:
            base_metrics['overall_fps'] = self.frame_count / base_metrics['runtime']
            base_metrics['batch_fps'] = self.batch_count / base_metrics['runtime']
        
        # Add CUDA metrics
        if self.enable_cuda_optimizations:
            try:
                bandwidth_stats = cuda_optimizer.memory_bandwidth_test()
                base_metrics.update({f'cuda_{k}': v for k, v in bandwidth_stats.items()})
                
                memory_stats = memory_pool_manager.get_memory_stats()
                base_metrics.update({f'memory_{k}': v for k, v in memory_stats.items()})
            except:
                pass
        
        return base_metrics
    
    def reset(self):
        """Reset perception state"""
        self.latest_result = None
        self.frame_count = 0
        self.batch_count = 0
        self.start_time = time.time()
        
        if self.detector:
            self.detector.reset_metrics()
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def cleanup(self):
        """Cleanup optimized perception resources"""
        try:
            self.stop_processing()
            
            if self.detector:
                self.detector.cleanup()
            
            if self.batch_manager:
                self.batch_manager = None
            
            # Cleanup CUDA resources
            for stream in self.cuda_streams:
                stream.synchronize()
            
            if self.enable_cuda_optimizations:
                memory_pool_manager.cleanup()
            
            torch.cuda.empty_cache()
            
            logger.info("RTX 3080 Ti Perception Manager cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_processing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
