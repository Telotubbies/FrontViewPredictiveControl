#!/usr/bin/env python3
"""
Dashboard GUI Module - จัดการการแสดงผล GUI สำหรับ CARLA MPC system

รวบรวมฟังก์ชันที่เกี่ยวข้องกับ:
- Pygame dashboard
- Real-time visualization
- Lane detection display
- BEV view
- Control status
"""

import logging
from typing import Optional, Tuple, Any
import pygame
import numpy as np
import cv2

from config import (
    BEV_W, BEV_H, BEV_WIN_W, BEV_WIN_H,
    PANEL_W, PANEL_H, VIEW3D_W, VIEW3D_H,
    UXColors
)

logger = logging.getLogger(__name__)


class Dashboard:
    """Dashboard GUI สำหรับแสดงผล real-time data"""
    
    def __init__(self, 
                 panel_w: int = PANEL_W,
                 panel_h: int = PANEL_H,
                 bev_win_w: int = BEV_WIN_W,
                 bev_win_h: int = BEV_WIN_H):
        self.panel_w = panel_w
        self.panel_h = panel_h
        self.bev_win_w = bev_win_w
        self.bev_win_h = bev_win_h
        
        # Pygame components
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        
        # Display surfaces
        self.panel_surface: Optional[pygame.Surface] = None
        self.bev_surface: Optional[pygame.Surface] = None
        self.view3d_surface: Optional[pygame.Surface] = None
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize pygame and dashboard components"""
        try:
            pygame.init()
            
            # Create main screen
            self.screen = pygame.display.set_mode((self.panel_w, self.panel_h))
            pygame.display.set_caption("CARLA MPC Dashboard")
            
            # Setup fonts
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            
            # Create display surfaces
            self.panel_surface = pygame.Surface((self.panel_w, self.panel_h))
            self.bev_surface = pygame.Surface((self.bev_win_w, self.bev_win_h))
            self.view3d_surface = pygame.Surface((VIEW3D_W, VIEW3D_H))
            
            self._initialized = True
            logger.info("Dashboard initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            return False
    
    def render_text(self, 
                   text: str, 
                   pos: Tuple[int, int],
                   color: Tuple[int, int, int] = UXColors.TEXT,
                   use_small_font: bool = False) -> None:
        """Render text on dashboard"""
        if not self._initialized:
            return
            
        font = self.small_font if use_small_font else self.font
        try:
            text_surface = font.render(text, True, color)
            self.panel_surface.blit(text_surface, pos)
        except Exception as e:
            logger.error(f"Failed to render text: {e}")
    
    def render_image(self, 
                    image: np.ndarray,
                    surface: pygame.Surface,
                    pos: Tuple[int, int] = (0, 0)) -> None:
        """Render numpy image to pygame surface"""
        if not self._initialized or image is None:
            return
            
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            # Convert to pygame surface
            image_surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))
            surface.blit(image_surface, pos)
            
        except Exception as e:
            logger.error(f"Failed to render image: {e}")
    
    def render_lane_overlay(self, 
                           rgb_frame: np.ndarray,
                           lane_overlay: Optional[np.ndarray] = None) -> None:
        """Render lane overlay on RGB frame"""
        if not self._initialized or rgb_frame is None:
            return
            
        try:
            # Resize frame to fit panel
            frame_resized = cv2.resize(rgb_frame, (self.panel_w // 2, self.panel_h // 2))
            
            # Apply lane overlay if available
            if lane_overlay is not None:
                overlay_resized = cv2.resize(lane_overlay, (frame_resized.shape[1], frame_resized.shape[0]))
                # Blend overlay with frame
                frame_resized = cv2.addWeighted(frame_resized, 0.7, overlay_resized, 0.3, 0)
            
            # Convert to RGB and render
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            self.render_image(frame_rgb, self.panel_surface, (0, 0))
            
        except Exception as e:
            logger.error(f"Failed to render lane overlay: {e}")
    
    def render_bev_view(self, bev_image: np.ndarray) -> None:
        """Render BEV (Bird's Eye View) image"""
        if not self._initialized or bev_image is None:
            return
            
        try:
            # Resize BEV image to fit window
            bev_resized = cv2.resize(bev_image, (self.bev_win_w, self.bev_win_h))
            
            # Convert to RGB if needed
            if len(bev_resized.shape) == 3 and bev_resized.shape[2] == 3:
                bev_rgb = cv2.cvtColor(bev_resized, cv2.COLOR_BGR2RGB)
            else:
                bev_rgb = bev_resized
                
            self.render_image(bev_rgb, self.bev_surface)
            
        except Exception as e:
            logger.error(f"Failed to render BEV view: {e}")
    
    def render_status_info(self, 
                          speed_kmh: float,
                          steering: float,
                          cte: float,
                          confidence: float,
                          fps: float) -> None:
        """Render vehicle status information"""
        if not self._initialized:
            return
            
        try:
            # Status panel position (right side of dashboard)
            status_x = self.panel_w // 2 + 10
            y_offset = 10
            line_height = 25
            
            # Render status text
            self.render_text(f"Speed: {speed_kmh:.1f} km/h", (status_x, y_offset))
            y_offset += line_height
            
            self.render_text(f"Steering: {steering:.3f}", (status_x, y_offset))
            y_offset += line_height
            
            self.render_text(f"CTE: {cte:.2f} m", (status_x, y_offset))
            y_offset += line_height
            
            # Color code confidence
            conf_color = UXColors.SUCCESS if confidence > 0.7 else UXColors.WARNING if confidence > 0.4 else UXColors.ERROR
            self.render_text(f"Confidence: {confidence:.2f}", (status_x, y_offset), conf_color)
            y_offset += line_height
            
            self.render_text(f"FPS: {fps:.1f}", (status_x, y_offset))
            
        except Exception as e:
            logger.error(f"Failed to render status info: {e}")
    
    def render_control_info(self, 
                          throttle: float,
                          brake: float,
                          target_speed: float) -> None:
        """Render control information"""
        if not self._initialized:
            return
            
        try:
            # Control panel position (bottom right)
            control_x = self.panel_w // 2 + 10
            control_y = self.panel_h - 80
            
            # Render control values with color coding
            throttle_color = UXColors.SUCCESS if throttle > 0.1 else UXColors.TEXT
            brake_color = UXColors.ERROR if brake > 0.1 else UXColors.TEXT
            
            self.render_text(f"Throttle: {throttle:.2f}", (control_x, control_y), throttle_color)
            self.render_text(f"Brake: {brake:.2f}", (control_x, control_y + 25), brake_color)
            self.render_text(f"Target: {target_speed:.1f} km/h", (control_x, control_y + 50))
            
        except Exception as e:
            logger.error(f"Failed to render control info: {e}")
    
    def update(self) -> None:
        """Update display"""
        if not self._initialized:
            return
            
        try:
            # Clear screen
            self.screen.fill(UXColors.BACKGROUND)
            
            # Blit panel surface
            self.screen.blit(self.panel_surface, (0, 0))
            
            # Update display
            pygame.display.flip()
            
        except Exception as e:
            logger.error(f"Failed to update display: {e}")
    
    def handle_events(self) -> bool:
        """Handle pygame events, return False if should quit"""
        if not self._initialized:
            return False
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def cleanup(self) -> None:
        """Cleanup pygame resources"""
        try:
            if self._initialized:
                pygame.quit()
                self._initialized = False
                logger.info("Dashboard cleaned up")
        except Exception as e:
            logger.error(f"Error during dashboard cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        if self.initialize():
            return self
        raise RuntimeError("Failed to initialize dashboard")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


class DashboardRenderer:
    """Helper class for rendering dashboard components"""
    
    @staticmethod
    def create_lane_overlay(rgb_frame: np.ndarray, 
                          lane_mask: np.ndarray,
                          overlay_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Create lane overlay on RGB frame"""
        try:
            overlay = rgb_frame.copy()
            
            # Create colored lane mask
            colored_mask = np.zeros_like(rgb_frame)
            colored_mask[lane_mask > 0] = overlay_color
            
            # Blend with original frame
            overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
            
            return overlay
        except Exception as e:
            logger.error(f"Failed to create lane overlay: {e}")
            return rgb_frame
    
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, 
                               target_size: Tuple[int, int],
                               fill_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h))
            
            # Create target image with fill color
            target = np.full((target_h, target_w) + image.shape[2:], fill_color, dtype=image.dtype)
            
            # Calculate position to center the resized image
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            
            # Place resized image in center
            target[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return target
        except Exception as e:
            logger.error(f"Failed to resize with aspect ratio: {e}")
            return cv2.resize(image, target_size)
