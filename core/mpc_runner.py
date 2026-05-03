#!/usr/bin/env python3
"""
MPC Runner Module - จัดการ main control loop สำหรับ CARLA MPC system

รวบรวมฟังก์ชันที่เกี่ยวข้องกับ:
- Main control loop
- Perception threading
- MPC execution
- Safety systems
- Data logging
"""

import logging
import threading
import time
import queue
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from collections import deque
import numpy as np

# Import unified data structures
from utils.type_hints import (
    ControlCommand, PerceptionResult, SafetyStatus, SystemMetrics,
    CameraFrame, DetectionMode, ControlMode
)

from config_clean import get_config

logger = logging.getLogger(__name__)

# Get configuration instance
config = get_config()


class PerceptionThread(threading.Thread):
    """Thread for running perception pipeline asynchronously"""
    
    def __init__(self, 
                 perception_pipeline: Any,
                 camera_queue: queue.Queue,
                 result_queue: queue.Queue,
                 skip_frames: int = 2):
        super().__init__(daemon=True)
        self.perception_pipeline = perception_pipeline
        self.camera_queue = camera_queue
        self.result_queue = result_queue
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.running = False
        
    def run(self):
        """Main perception thread loop"""
        self.running = True
        logger.info("Perception thread started")
        
        while self.running:
            try:
                # Get camera frame
                try:
                    rgb_frame = self.camera_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Skip frames for performance
                self.frame_counter += 1
                if self.frame_counter % self.skip_frames != 0:
                    continue
                
                # Run perception pipeline
                start_time = time.time()
                result = self.perception_pipeline.process(rgb_frame)
                process_time = time.time() - start_time
                
                # Create perception result
                perception_result = PerceptionResult(
                    cte=getattr(result, 'cte', 0.0),
                    heading_error=getattr(result, 'heading_error', 0.0),
                    curvature=getattr(result, 'curvature', 0.0),
                    confidence=getattr(result, 'confidence', 0.0),
                    geometry_valid=getattr(result, 'geometry_valid', False),
                    timestamp=time.time()
                )
                
                # Add processing time to result
                perception_result.process_time = process_time
                
                # Put result in queue
                try:
                    self.result_queue.put_nowait(perception_result)
                except queue.Full:
                    # Remove oldest result if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(perception_result)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Perception thread error: {e}")
                continue
        
        logger.info("Perception thread stopped")
    
    def stop(self):
        """Stop perception thread"""
        self.running = False


class MPCRunner:
    """Main MPC control loop runner"""
    
    def __init__(self, 
                 mpc_controller: Any,
                 safety_system: Any,
                 perception_pipeline: Any,
                 target_speed_kmh: float = TARGET_SPEED_KMH):
        self.mpc_controller = mpc_controller
        self.safety_system = safety_system
        self.perception_pipeline = perception_pipeline
        self.target_speed_kmh = target_speed_kmh
        
        # Threading components
        self.camera_queue: queue.Queue = queue.Queue(maxsize=2)
        self.perception_queue: queue.Queue = queue.Queue(maxsize=5)
        self.perception_thread: Optional[PerceptionThread] = None
        
        # State tracking
        self.current_state: Optional[ControlState] = None
        self.latest_perception: Optional[PerceptionResult] = None
        self.frame_count = 0
        self.start_time = time.time()
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.control_times = deque(maxlen=100)
        
        # Data logging
        self.log_data = []
        self.enable_logging = False
        
    def start_perception_thread(self) -> bool:
        """Start perception thread if enabled"""
        if not USE_PERCEPTION_THREAD:
            return True
            
        try:
            self.perception_thread = PerceptionThread(
                self.perception_pipeline,
                self.camera_queue,
                self.perception_queue,
                skip_frames=PERCEPTION_SKIP_FRAME
            )
            self.perception_thread.start()
            logger.info("Perception thread started")
            return True
        except Exception as e:
            logger.error(f"Failed to start perception thread: {e}")
            return False
    
    def stop_perception_thread(self):
        """Stop perception thread"""
        if self.perception_thread:
            self.perception_thread.stop()
            self.perception_thread.join(timeout=2.0)
            self.perception_thread = None
            logger.info("Perception thread stopped")
    
    def update_perception(self, rgb_frame: np.ndarray) -> Optional[PerceptionResult]:
        """Update perception result"""
        if USE_PERCEPTION_THREAD:
            # Threaded mode - get result from queue
            try:
                while not self.perception_queue.empty():
                    self.latest_perception = self.perception_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Add frame to camera queue
            try:
                self.camera_queue.put_nowait(rgb_frame)
            except queue.Full:
                # Remove oldest frame if queue is full
                try:
                    self.camera_queue.get_nowait()
                    self.camera_queue.put_nowait(rgb_frame)
                except queue.Empty:
                    pass
        else:
            # Synchronous mode - run perception directly
            try:
                start_time = time.time()
                result = self.perception_pipeline.process(rgb_frame)
                process_time = time.time() - start_time
                
                self.latest_perception = PerceptionResult(
                    cte=getattr(result, 'cte', 0.0),
                    heading_error=getattr(result, 'heading_error', 0.0),
                    curvature=getattr(result, 'curvature', 0.0),
                    confidence=getattr(result, 'confidence', 0.0),
                    geometry_valid=getattr(result, 'geometry_valid', False),
                    timestamp=time.time(),
                    process_time=process_time
                )
            except Exception as e:
                logger.error(f"Perception processing error: {e}")
        
        return self.latest_perception
    
    def is_perception_stale(self) -> bool:
        """Check if perception data is stale"""
        if self.latest_perception is None:
            return True
        
        age = time.time() - self.latest_perception.timestamp
        return age > STALE_PERCEPTION_S
    
    def compute_control(self, 
                       current_speed_ms: float,
                       vehicle_transform: Any) -> ControlState:
        """Compute control output using MPC"""
        start_time = time.time()
        
        # Default control state
        control_state = ControlState(
            steering=0.0,
            throttle=0.0,
            brake=0.0,
            target_speed_kmh=self.target_speed_kmh,
            timestamp=time.time()
        )
        
        try:
            # Check perception validity
            if self.latest_perception is None or self.is_perception_stale():
                logger.warning("Using fallback control - no valid perception")
                return self._compute_fallback_control(control_state, current_speed_ms)
            
            perception = self.latest_perception
            
            # Check geometry validity
            if not perception.geometry_valid:
                logger.warning("Using fallback control - invalid geometry")
                return self._compute_fallback_control(control_state, current_speed_ms)
            
            # Run MPC
            mpc_result = self.mpc_controller.solve(
                cte=perception.cte,
                heading_error=perception.heading_error,
                curvature=perception.curvature,
                current_speed=current_speed_ms,
                target_speed=self.target_speed_kmh / 3.6
            )
            
            if mpc_result is None:
                logger.error("MPC failed to solve")
                return self._compute_fallback_control(control_state, current_speed_ms)
            
            # Extract control values
            control_state.steering = float(mpc_result.get('steering', 0.0))
            control_state.throttle = float(mpc_result.get('throttle', 0.0))
            control_state.brake = float(mpc_result.get('brake', 0.0))
            
            # Apply safety overrides
            control_state = self.safety_system.apply_safety_overrides(
                control_state, current_speed_ms, perception
            )
            
        except Exception as e:
            logger.error(f"Control computation error: {e}")
            control_state = self._compute_fallback_control(control_state, current_speed_ms)
        
        # Record performance
        control_time = time.time() - start_time
        self.control_times.append(control_time)
        
        return control_state
    
    def _compute_fallback_control(self, 
                                control_state: ControlState,
                                current_speed_ms: float) -> ControlState:
        """Compute fallback control when perception is invalid"""
        # Simple speed control
        target_speed_ms = self.target_speed_kmh / 3.6
        speed_error = target_speed_ms - current_speed_ms
        
        if speed_error > 1.0:
            control_state.throttle = 0.3
            control_state.brake = 0.0
        elif speed_error < -2.0:
            control_state.throttle = 0.0
            control_state.brake = 0.2
        else:
            control_state.throttle = 0.1
            control_state.brake = 0.0
        
        # Minimal steering for safety
        control_state.steering = 0.0
        
        return control_state
    
    def step(self, 
            rgb_frame: np.ndarray,
            current_speed_ms: float,
            vehicle_transform: Any) -> Tuple[ControlState, Optional[PerceptionResult]]:
        """Execute one control step"""
        step_start_time = time.time()
        
        # Update perception
        perception_result = self.update_perception(rgb_frame)
        
        # Compute control
        control_state = self.compute_control(current_speed_ms, vehicle_transform)
        
        # Update metrics
        step_time = time.time() - step_start_time
        self.fps_history.append(1.0 / step_time if step_time > 0 else 0.0)
        
        # Log data if enabled
        if self.enable_logging:
            self._log_step(control_state, perception_result, current_speed_ms)
        
        # Update frame counter
        self.frame_count += 1
        
        return control_state, perception_result
    
    def _log_step(self, 
                 control_state: ControlState,
                 perception_result: Optional[PerceptionResult],
                 current_speed_ms: float):
        """Log step data for analysis"""
        log_entry = {
            'timestamp': time.time(),
            'frame': self.frame_count,
            'steering': control_state.steering,
            'throttle': control_state.throttle,
            'brake': control_state.brake,
            'speed_ms': current_speed_ms,
            'speed_kmh': current_speed_ms * 3.6,
        }
        
        if perception_result:
            log_entry.update({
                'cte': perception_result.cte,
                'heading_error': perception_result.heading_error,
                'curvature': perception_result.curvature,
                'confidence': perception_result.confidence,
                'geometry_valid': perception_result.geometry_valid,
            })
        
        self.log_data.append(log_entry)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        metrics = {}
        
        if self.fps_history:
            metrics['fps'] = np.mean(self.fps_history)
            metrics['fps_min'] = np.min(self.fps_history)
            metrics['fps_max'] = np.max(self.fps_history)
        
        if self.control_times:
            metrics['control_time_mean'] = np.mean(self.control_times)
            metrics['control_time_max'] = np.max(self.control_times)
        
        if self.latest_perception and hasattr(self.latest_perception, 'process_time'):
            metrics['perception_time'] = self.latest_perception.process_time
        
        metrics['runtime_s'] = time.time() - self.start_time
        metrics['frame_count'] = self.frame_count
        
        return metrics
    
    def reset(self):
        """Reset runner state"""
        self.current_state = None
        self.latest_perception = None
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history.clear()
        self.control_times.clear()
        self.log_data.clear()
        
        # Clear queues
        while not self.camera_queue.empty():
            try:
                self.camera_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.perception_queue.empty():
            try:
                self.perception_queue.get_nowait()
            except queue.Empty:
                break
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_perception_thread()
        logger.info("MPC runner cleaned up")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_perception_thread()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
