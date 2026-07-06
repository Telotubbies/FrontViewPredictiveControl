#!/usr/bin/env python3
"""
Display Manager - Simple GUI and logging management

Handles:
- Dashboard visualization
- Performance logging
- Data recording
- User interface
"""

import logging
import time
from typing import Optional, Dict, Any
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from utils.type_hints import (
    CameraFrame, PerceptionResult, ControlCommand
)
from config import *  # noqa: F401,F403
import config as _config_module


def get_config():
    return _config_module

logger = logging.getLogger(__name__)


class DisplayManager:
    """Simple display management interface"""

    def __init__(self,
                 enable_gui: bool = True,
                 enable_logging: bool = True,
                 enable_recording: bool = False):
        self.enable_gui = enable_gui
        self.enable_logging = enable_logging
        self.enable_recording = enable_recording

        # Configuration
        self.config = get_config()

        # GUI components
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        self.dashboard_initialized = False

        # Logging
        self.log_data = []
        self.start_time = time.time()

        # Performance metrics
        self.fps_history = []
        self.last_frame_time = time.time()

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize display components"""
        try:
            if self.enable_gui and PYGAME_AVAILABLE:
                self._initialize_gui()

            if self.enable_logging:
                self._initialize_logging()

            logger.info("Display manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize display manager: {e}")
            if self.enable_gui:
                logger.warning("GUI disabled due to initialization error")
                self.enable_gui = False

    def _initialize_gui(self):
        """Initialize pygame GUI"""
        if not PYGAME_AVAILABLE:
            logger.warning("Pygame not available, GUI disabled")
            self.enable_gui = False
            return

        try:
            pygame.init()

            # Create screen
            self.screen = pygame.display.set_mode(
                (self.config.PANEL_W, self.config.PANEL_H)
            )
            pygame.display.set_caption("CARLA MPC Dashboard")

            # Setup fonts
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)

            self.dashboard_initialized = True
            logger.info("GUI initialized")

        except Exception as e:
            logger.error(f"Failed to initialize GUI: {e}")
            self.enable_gui = False

    def _initialize_logging(self):
        """Initialize data logging"""
        self.log_data = []
        self.start_time = time.time()
        logger.info("Logging initialized")

    def update(self,
               frame: Optional[CameraFrame] = None,
               perception: Optional[PerceptionResult] = None,
               control: Optional[ControlCommand] = None,
               vehicle_state: Optional[Dict[str, Any]] = None):
        """Update display with new data"""

        # Update GUI
        if self.enable_gui and self.dashboard_initialized:
            self._update_gui(frame, perception, control, vehicle_state)

        # Update logging
        if self.enable_logging:
            self._update_logging(frame, perception, control, vehicle_state)

        # Update performance metrics
        self._update_metrics()

    def _update_gui(self,
                   frame: Optional[CameraFrame],
                   perception: Optional[PerceptionResult],
                   control: Optional[ControlCommand],
                   vehicle_state: Optional[Dict[str, Any]]):
        """Update GUI display"""
        try:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False

            # Clear screen
            self.screen.fill((20, 20, 30))

            # Render camera frame
            if frame:
                self._render_camera_frame(frame)

            # Render perception data
            if perception:
                self._render_perception_data(perception)

            # Render control data
            if control and vehicle_state:
                self._render_control_data(control, vehicle_state)

            # Render performance metrics
            self._render_performance_metrics()

            # Update display
            pygame.display.flip()

        except Exception as e:
            logger.error(f"GUI update error: {e}")

        return True

    def _render_camera_frame(self, frame: CameraFrame):
        """Render camera frame"""
        try:
            # Resize frame to fit with proper aspect ratio
            import cv2
            target_h = self.config.PANEL_H // 2
            target_w = self.config.PANEL_W // 2
            frame_resized = cv2.resize(frame.rgb, (target_w, target_h))

            # Convert RGB to pygame surface
            frame_surface = pygame.surfarray.make_surface(
                frame_resized.swapaxes(0, 1)
            )

            self.screen.blit(frame_surface, (0, 0))

        except Exception as e:
            logger.error(f"Camera frame render error: {e}")

    def _render_perception_data(self, perception: PerceptionResult):
        """Render perception information"""
        try:
            y_offset = self.config.PANEL_H // 2 + 10
            line_height = 25

            # Perception info
            self._render_text(f"CTE: {perception.cte:.2f}m", (10, y_offset))
            y_offset += line_height

            self._render_text(f"Heading: {perception.heading_error:.3f}rad", (10, y_offset))
            y_offset += line_height

            self._render_text(f"Curvature: {perception.curvature:.3f}", (10, y_offset))
            y_offset += line_height

            # Confidence with color
            conf_color = self._get_confidence_color(perception.confidence)
            self._render_text(f"Confidence: {perception.confidence:.2f}", (10, y_offset), conf_color)
            y_offset += line_height

            # Geometry validity
            geo_color = (0, 255, 0) if perception.geometry_valid else (255, 0, 0)
            self._render_text(f"Valid: {'Yes' if perception.geometry_valid else 'No'}", (10, y_offset), geo_color)

        except Exception as e:
            logger.error(f"Perception data render error: {e}")

    def _render_control_data(self, control: ControlCommand, vehicle_state: Dict[str, Any]):
        """Render control information"""
        try:
            # Control panel position (right side)
            x_offset = self.config.PANEL_W // 2 + 10
            y_offset = self.config.PANEL_H // 2 + 10
            line_height = 25

            # Speed info
            speed_kmh = vehicle_state['speed_kmh']
            self._render_text(f"Speed: {speed_kmh:.1f} km/h", (x_offset, y_offset))
            y_offset += line_height

            # Control values
            self._render_text(f"Steering: {control.steering:.3f}", (x_offset, y_offset))
            y_offset += line_height

            throttle_color = (0, 255, 0) if control.throttle > 0.1 else (200, 200, 200)
            self._render_text(f"Throttle: {control.throttle:.2f}", (x_offset, y_offset), throttle_color)
            y_offset += line_height

            brake_color = (255, 0, 0) if control.brake > 0.1 else (200, 200, 200)
            self._render_text(f"Brake: {control.brake:.2f}", (x_offset, y_offset), brake_color)

        except Exception as e:
            logger.error(f"Control data render error: {e}")

    def _render_performance_metrics(self):
        """Render performance metrics"""
        try:
            if self.fps_history:
                current_fps = self.fps_history[-1]
                avg_fps = np.mean(self.fps_history)

                # Metrics position (bottom)
                y_offset = self.config.PANEL_H - 60

                self._render_text(f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f})",
                                (10, y_offset), self.small_font)

                runtime = time.time() - self.start_time
                self._render_text(f"Runtime: {runtime:.1f}s",
                                (10, y_offset + 20), self.small_font)

        except Exception as e:
            logger.error(f"Performance metrics render error: {e}")

    def _render_text(self, text: str, pos: tuple, color: tuple = (200, 200, 200)):
        """Render text on screen"""
        try:
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, pos)
        except Exception as e:
            logger.error(f"Text render error: {e}")

    def _get_confidence_color(self, confidence: float) -> tuple:
        """Get color based on confidence level"""
        try:
            confidence = float(confidence)
            if confidence > 0.7:
                return (0, 255, 0)  # Green
            elif confidence > 0.4:
                return (255, 255, 0)  # Yellow
            else:
                return (255, 0, 0)  # Red
        except (ValueError, TypeError):
            return (200, 200, 200)  # Gray for invalid confidence

    def _update_logging(self,
                       frame: Optional[CameraFrame],
                       perception: Optional[PerceptionResult],
                       control: Optional[ControlCommand],
                       vehicle_state: Optional[Dict[str, Any]]):
        """Update data logging"""
        try:
            log_entry = {
                'timestamp': time.time(),
                'frame_count': len(self.log_data)
            }

            if vehicle_state:
                log_entry.update({
                    'speed_ms': vehicle_state['speed_ms'],
                    'speed_kmh': vehicle_state['speed_kmh']
                })

            if perception:
                log_entry.update({
                    'cte': perception.cte,
                    'heading_error': perception.heading_error,
                    'curvature': perception.curvature,
                    'confidence': perception.confidence,
                    'geometry_valid': perception.geometry_valid
                })

            if control:
                log_entry.update({
                    'steering': control.steering,
                    'throttle': control.throttle,
                    'brake': control.brake
                })

            self.log_data.append(log_entry)

            # Limit log size
            if len(self.log_data) > 10000:
                self.log_data = self.log_data[-5000:]

        except Exception as e:
            logger.error(f"Logging update error: {e}")

    def _update_metrics(self):
        """Update performance metrics"""
        try:
            current_time = time.time()
            if self.last_frame_time > 0:
                fps = 1.0 / (current_time - self.last_frame_time)
                self.fps_history.append(fps)

                # Limit history size
                if len(self.fps_history) > 100:
                    self.fps_history = self.fps_history[-50:]

            self.last_frame_time = current_time

        except Exception as e:
            logger.error(f"Metrics update error: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get display performance metrics"""
        metrics = {
            'gui_enabled': self.enable_gui,
            'logging_enabled': self.enable_logging,
            'recording_enabled': self.enable_recording,
            'log_entries': len(self.log_data)
        }

        if self.fps_history:
            metrics['fps_current'] = self.fps_history[-1]
            metrics['fps_average'] = np.mean(self.fps_history)
            metrics['fps_min'] = np.min(self.fps_history)
            metrics['fps_max'] = np.max(self.fps_history)

        return metrics

    def save_log(self, filename: str):
        """Save logged data to file"""
        if not self.log_data:
            logger.warning("No data to save")
            return

        try:
            import json

            with open(filename, 'w') as f:
                json.dump(self.log_data, f, indent=2)

            logger.info(f"Log data saved to {filename}")

        except Exception as e:
            logger.error(f"Failed to save log: {e}")

    def cleanup(self):
        """Cleanup display resources"""
        try:
            if self.enable_gui and self.dashboard_initialized:
                pygame.quit()
                logger.info("GUI cleaned up")

            # Save log if recording enabled
            if self.enable_recording and self.log_data:
                timestamp = int(time.time())
                self.save_log(f"carla_mpc_log_{timestamp}.json")

            logger.info("Display manager cleaned up")

        except Exception as e:
            logger.error(f"Display cleanup error: {e}")
