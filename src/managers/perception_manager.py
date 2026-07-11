#!/usr/bin/env python3
"""
Perception Manager - Simple perception pipeline management

Handles:
- UNet and Classical lane detection
- Threading for performance
- Result processing
- Configuration management
"""

import logging
import threading
import time
import queue
from typing import Optional, Any

from utils.type_hints import (
    CameraFrame, PerceptionResult, DetectionMode
)
from config import *  # noqa: F401,F403
import config as _config_module


def get_config():
    return _config_module

logger = logging.getLogger(__name__)


class PerceptionManager:
    """Simple perception management interface"""

    def __init__(self,
                 model_path: Optional[str] = None,
                 detection_mode: DetectionMode = DetectionMode.UNET,
                 use_threading: bool = True,
                 model_type: str = "unet"):
        self.model_path = model_path
        self.detection_mode = detection_mode
        self.use_threading = use_threading
        self.model_type = model_type

        # Configuration
        self.config = get_config()

        # Device setup — supports CUDA (NVIDIA) and ROCm (AMD)
        from utils.device_utils import get_device
        self.device = get_device()

        # Pipeline components
        self.pipeline: Optional[Any] = None
        self.perception_thread: Optional[threading.Thread] = None

        # Threading
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.result_queue: queue.Queue = queue.Queue(maxsize=5)
        self.running = False

        # State
        self.latest_result: Optional[PerceptionResult] = None
        self.frame_count = 0

        # Initialize pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize perception pipeline based on mode"""
        try:
            if self.detection_mode == DetectionMode.CLASSICAL:
                self._initialize_classical_pipeline()
            else:
                self._initialize_unet_pipeline()
        except Exception as e:
            logger.error(f"Failed to initialize perception pipeline: {e}")

    def _initialize_classical_pipeline(self):
        """Initialize classical lane detection pipeline"""
        try:
            from perception.classical_lane import ClassicalLane

            self.pipeline = ClassicalLane()
            logger.info("Classical lane detection pipeline initialized")
        except ImportError as e:
            logger.error(f"Failed to import classical pipeline: {e}")
            raise

    def _initialize_unet_pipeline(self):
        """Initialize UNet lane detection pipeline"""
        if not self.model_path:
            raise ValueError("Model path required for UNet detection")

        try:
            from pipeline import LKAPipeline

            self.pipeline = LKAPipeline(
                model_path=self.model_path,
                device=self.device,
                target_speed_kmh=self.config.TARGET_SPEED_KMH,
                use_trajectory_pipeline=self.config.USE_TRAJECTORY_PIPELINE,
                model_type=self.model_type,
            )
            logger.info(f"{self.model_type.upper()} lane detection pipeline initialized")
        except ImportError as e:
            logger.error(f"Failed to import UNet pipeline: {e}")
            raise

    def start_threading(self):
        """Start perception thread if enabled"""
        if not self.use_threading or self.perception_thread:
            return

        self.running = True
        self.perception_thread = threading.Thread(
            target=self._perception_loop,
            daemon=True
        )
        self.perception_thread.start()
        logger.info("Perception thread started")

    def stop_threading(self):
        """Stop perception thread"""
        if not self.perception_thread:
            return

        self.running = False
        self.perception_thread.join(timeout=2.0)
        self.perception_thread = None
        logger.info("Perception thread stopped")

    def _perception_loop(self):
        """Main perception thread loop"""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)

                # Process frame
                result = self.process_frame(frame)

                # Put result in queue
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except queue.Empty:
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Perception thread error: {e}")
                continue

    def process_frame(self, frame: CameraFrame) -> PerceptionResult:
        """Process single frame and return perception result"""
        if not self.pipeline:
            raise RuntimeError("Perception pipeline not initialized")

        start_time = time.time()

        try:
            # Run pipeline
            result = self.pipeline.process(frame.rgb)
            process_time = time.time() - start_time

            # Create perception result
            perception_result = PerceptionResult(
                cte=getattr(result, 'cte', 0.0),
                heading_error=getattr(result, 'heading_error', 0.0),
                curvature=getattr(result, 'curvature', 0.0),
                confidence=getattr(result, 'confidence', 0.0),
                geometry_valid=getattr(result, 'geometry_valid', False),
                lane_width=getattr(result, 'lane_width', None),
                left_coeffs=getattr(result, 'left_coeffs', None),
                right_coeffs=getattr(result, 'right_coeffs', None),
                center_coeffs=getattr(result, 'center_coeffs', None),
                timestamp=time.time(),
                processing_time=process_time,
                lane_overlay=getattr(result, 'lane_overlay', None),
                bev_binary=getattr(result, 'bev_binary', None),
                bev_window_vis=getattr(result, 'bev_window_vis', None)
            )

            # Update latest result
            self.latest_result = perception_result
            self.frame_count += 1

            return perception_result

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            # Return fallback result
            return PerceptionResult(
                cte=0.0,
                heading_error=0.0,
                curvature=0.0,
                confidence=0.0,
                geometry_valid=False,
                timestamp=time.time(),
                processing_time=time.time() - start_time
            )

    def add_frame(self, frame: CameraFrame) -> bool:
        """Add frame to processing queue"""
        if self.use_threading:
            try:
                self.frame_queue.put_nowait(frame)
                return True
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                    return True
                except queue.Empty:
                    return False
        else:
            # Process directly
            self.process_frame(frame)
            return True

    def get_latest_result(self) -> Optional[PerceptionResult]:
        """Get latest perception result"""
        if self.use_threading:
            try:
                while not self.result_queue.empty():
                    self.latest_result = self.result_queue.get_nowait()
            except queue.Empty:
                pass

        return self.latest_result

    def is_result_stale(self, max_age_seconds: float = 0.2) -> bool:
        """Check if latest result is stale"""
        if not self.latest_result:
            return True

        age = time.time() - self.latest_result.timestamp
        return age > max_age_seconds

    def get_metrics(self) -> dict:
        """Get perception performance metrics"""
        metrics = {
            'frame_count': self.frame_count,
            'detection_mode': self.detection_mode.value,
            'use_threading': self.use_threading,
            'device': str(self.device)
        }

        if self.latest_result and hasattr(self.latest_result, 'processing_time'):
            metrics['last_processing_time'] = self.latest_result.processing_time

        return metrics

    def reset(self):
        """Reset perception state"""
        self.latest_result = None
        self.frame_count = 0

        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

    def cleanup(self):
        """Cleanup perception resources"""
        self.stop_threading()
        self.reset()
        logger.info("Perception manager cleaned up")

    def __enter__(self):
        """Context manager entry"""
        if self.use_threading:
            self.start_threading()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
