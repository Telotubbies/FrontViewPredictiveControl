#!/usr/bin/env python3
"""
DSUNet Dashboard - GUI สำหรับแสดงผล DSUNet lane detection และ overlay
ปรับ layout ให้เน้น lane marking visualization จาก DSUNet
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


class DSUNetDashboard:
    """Dashboard GUI สำหรับ DSUNet lane detection"""
    
    def __init__(self, 
                 screen_width: int = 1400,
                 screen_height: int = 800):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Layout zones
        self.main_camera_w = 640
        self.main_camera_h = 480
        self.lane_overlay_w = 640
        self.lane_overlay_h = 240
        self.bev_w = 450
        self.bev_h = 320
        self.info_panel_w = 300
        self.info_panel_h = 800
        
        # Pygame components
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        self.title_font: Optional[pygame.font.Font] = None
        
        # Colors
        self.BG_COLOR = (20, 20, 30)
        self.CARD_BG = (30, 30, 40)
        self.CARD_BORDER = (60, 60, 80)
        self.DSUNET_ACCENT = (100, 200, 255)  # Blue accent for DSUNet
        self.LANE_COLOR = (255, 255, 255)    # White lane markings
        self.SUCCESS_COLOR = (100, 255, 100)
        self.WARNING_COLOR = (255, 200, 100)
        
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize pygame and dashboard"""
        try:
            pygame.init()
            
            # Setup display
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("DSUNet Lane Detection Dashboard | CARLA")
            
            # Setup fonts
            self.font = pygame.font.SysFont("monospace", 16)
            self.small_font = pygame.font.SysFont("monospace", 12)
            self.title_font = pygame.font.SysFont("monospace", 20, bold=True)
            
            self._initialized = True
            logger.info("DSUNet Dashboard initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DSUNet dashboard: {e}")
            return False
    
    def clear_screen(self) -> None:
        """Clear screen with background color"""
        if not self._initialized:
            return
        self.screen.fill(self.BG_COLOR)
    
    def draw_card(self, x: int, y: int, w: int, h: int, title: str) -> None:
        """Draw card background with border and title"""
        # Card background
        pygame.draw.rect(self.screen, self.CARD_BG, (x, y, w, h))
        # Card border
        pygame.draw.rect(self.screen, self.CARD_BORDER, (x, y, w, h), 2)
        # Title
        title_surf = self.title_font.render(title, True, self.DSUNET_ACCENT)
        title_rect = title_surf.get_rect()
        title_rect.topleft = (x + 10, y + 5)
        self.screen.blit(title_surf, title_rect)
    
    def render_main_camera(self, rgb_frame: np.ndarray, x: int = 20, y: int = 60) -> None:
        """Render main camera view"""
        if not self._initialized or rgb_frame is None:
            return
            
        try:
            # Resize frame
            frame_resized = cv2.resize(rgb_frame, (self.main_camera_w, self.main_camera_h))
            
            # Convert RGB to pygame surface
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            
            # Draw card
            self.draw_card(x, y - 40, self.main_camera_w, self.main_camera_h + 40, "📷 Main Camera")
            
            # Blit frame
            self.screen.blit(frame_surface, (x, y))
            
        except Exception as e:
            logger.error(f"Failed to render main camera: {e}")
    
    def render_dsunet_overlay(self, lane_overlay: Optional[np.ndarray], 
                            confidence: float = 0.0,
                            x: int = 680, y: int = 60) -> None:
        """Render DSUNet lane overlay with confidence"""
        if not self._initialized:
            return
            
        try:
            # Draw card
            self.draw_card(x, y - 40, self.lane_overlay_w, self.lane_overlay_h + 40, "🎯 DSUNet Lane Overlay")
            
            if lane_overlay is not None:
                # Resize overlay
                overlay_resized = cv2.resize(lane_overlay, (self.lane_overlay_w, self.lane_overlay_h))
                
                # Convert to RGB
                if len(overlay_resized.shape) == 3:
                    overlay_rgb = cv2.cvtColor(overlay_resized, cv2.COLOR_BGR2RGB)
                else:
                    overlay_rgb = cv2.cvtColor(overlay_resized, cv2.COLOR_GRAY2RGB)
                    
                overlay_surface = pygame.surfarray.make_surface(overlay_rgb.swapaxes(0, 1))
                self.screen.blit(overlay_surface, (x, y))
                
                # Confidence indicator
                conf_color = self.SUCCESS_COLOR if confidence > 0.7 else self.WARNING_COLOR
                conf_text = f"Confidence: {confidence:.3f}"
                conf_surf = self.small_font.render(conf_text, True, conf_color)
                self.screen.blit(conf_surf, (x + 10, y + self.lane_overlay_h + 5))
            else:
                # No overlay available
                no_overlay_surf = self.small_font.render("No lane overlay available", True, UXColors.TEXT_DIM)
                self.screen.blit(no_overlay_surf, (x + 10, y + 10))
                
        except Exception as e:
            logger.error(f"Failed to render DSUNet overlay: {e}")
    
    def render_bev_view(self, bev_binary: Optional[np.ndarray],
                       bev_window_vis: Optional[np.ndarray],
                       x: int = 20, y: int = 560) -> None:
        """Render BEV view with lane detection"""
        if not self._initialized:
            return
            
        try:
            # Draw card
            self.draw_card(x, y - 40, self.bev_w, self.bev_h + 40, "🗺️ Bird's Eye View")
            
            # Use BEV window visualization if available (more detailed), otherwise binary
            bev_image = bev_window_vis if bev_window_vis is not None else bev_binary
            
            if bev_image is not None:
                # Resize BEV
                bev_resized = cv2.resize(bev_image, (self.bev_w, self.bev_h))
                
                # Convert to RGB
                if len(bev_resized.shape) == 3:
                    bev_rgb = cv2.cvtColor(bev_resized, cv2.COLOR_BGR2RGB)
                else:
                    bev_rgb = cv2.cvtColor(bev_resized, cv2.COLOR_GRAY2RGB)
                    
                bev_surface = pygame.surfarray.make_surface(bev_rgb.swapaxes(0, 1))
                self.screen.blit(bev_surface, (x, y))
            else:
                # No BEV data
                no_bev_surf = self.small_font.render("No BEV data available", True, UXColors.TEXT_DIM)
                self.screen.blit(no_bev_surf, (x + 10, y + 10))
                
        except Exception as e:
            logger.error(f"Failed to render BEV view: {e}")
    
    def render_info_panel(self, 
                         speed_kmh: float,
                         steering: float, 
                         cte: float,
                         confidence: float,
                         fps: float,
                         phase_info: str = "",
                         x: int = 490, y: int = 560) -> None:
        """Render information panel"""
        if not self._initialized:
            return
            
        try:
            # Draw card
            self.draw_card(x, y - 40, self.info_panel_w, self.info_panel_h - 520, "📊 System Status")
            
            # Status information
            info_lines = [
                ("Speed", f"{speed_kmh:.1f} km/h", self.SUCCESS_COLOR),
                ("Steering", f"{steering:+.3f} rad", self.LANE_COLOR),
                ("CTE", f"{cte:+.3f} m", 
                 self.SUCCESS_COLOR if abs(cte) < 0.5 else self.WARNING_COLOR),
                ("Confidence", f"{confidence:.3f}", 
                 self.SUCCESS_COLOR if confidence > 0.7 else self.WARNING_COLOR),
                ("FPS", f"{fps:.1f}", 
                 self.SUCCESS_COLOR if fps >= 20 else self.WARNING_COLOR),
                ("", "", (0, 0, 0)),  # Separator
                ("DSUNet Status", "ACTIVE", self.DSUNET_ACCENT),
                ("Model", "DSUNet-40ep", self.LANE_COLOR),
                ("Detector", "Deep Learning", self.LANE_COLOR),
                ("", "", (0, 0, 0)),  # Separator
                ("Phase Info", phase_info, UXColors.TEXT_NORMAL),
            ]
            
            y_offset = y + 10
            for label, value, color in info_lines:
                if label:  # Skip empty lines
                    label_surf = self.small_font.render(f"{label}:", True, UXColors.TEXT_DIM)
                    value_surf = self.small_font.render(value, True, color)
                    self.screen.blit(label_surf, (x + 10, y_offset))
                    self.screen.blit(value_surf, (x + 150, y_offset))
                y_offset += 25
                
        except Exception as e:
            logger.error(f"Failed to render info panel: {e}")
    
    def render_lane_markings_info(self, 
                                left_curvature: float = 0.0,
                                right_curvature: float = 0.0,
                                lane_width: float = 3.5,
                                x: int = 680, y: int = 340) -> None:
        """Render lane markings specific information"""
        if not self._initialized:
            return
            
        try:
            # Draw card
            self.draw_card(x, y - 40, self.lane_overlay_w, 200, "🛣️ Lane Analysis")
            
            # Lane information
            lane_lines = [
                ("Lane Width", f"{lane_width:.2f} m", self.LANE_COLOR),
                ("Left Curvature", f"{left_curvature:+.4f}", self.LANE_COLOR),
                ("Right Curvature", f"{right_curvature:+.4f}", self.LANE_COLOR),
                ("Avg Curvature", f"{(left_curvature + right_curvature)/2:+.4f}", self.LANE_COLOR),
                ("", "", (0, 0, 0)),  # Separator
                ("Detection", "DSUNet Model", self.DSUNET_ACCENT),
                ("Training", "40 Epochs", self.DSUNET_ACCENT),
                ("Architecture", "Dual-Scale UNet", self.DSUNET_ACCENT),
            ]
            
            y_offset = y + 10
            for label, value, color in lane_lines:
                if label:
                    label_surf = self.small_font.render(f"{label}:", True, UXColors.TEXT_DIM)
                    value_surf = self.small_font.render(value, True, color)
                    self.screen.blit(label_surf, (x + 10, y_offset))
                    self.screen.blit(value_surf, (x + 150, y_offset))
                y_offset += 22
                
        except Exception as e:
            logger.error(f"Failed to render lane markings info: {e}")
    
    def render_header(self) -> None:
        """Render dashboard header"""
        if not self._initialized:
            return
            
        try:
            # Main title
            title_surf = self.title_font.render("DSUNet Lane Detection System", True, self.DSUNET_ACCENT)
            title_rect = title_surf.get_rect(center=(self.screen_width // 2, 25))
            self.screen.blit(title_surf, title_rect)
            
            # Subtitle
            subtitle_surf = self.small_font.render("Real-time lane detection with Deep Learning", True, UXColors.TEXT_DIM)
            subtitle_rect = subtitle_surf.get_rect(center=(self.screen_width // 2, 45))
            self.screen.blit(subtitle_surf, subtitle_rect)
            
        except Exception as e:
            logger.error(f"Failed to render header: {e}")
    
    def update_display(self) -> None:
        """Update pygame display"""
        if not self._initialized:
            return
        pygame.display.flip()
    
    def cleanup(self) -> None:
        """Cleanup pygame resources"""
        if self._initialized:
            pygame.quit()
            self._initialized = False


def create_dsunet_dashboard() -> DSUNetDashboard:
    """Create and initialize DSUNet dashboard"""
    dashboard = DSUNetDashboard()
    if dashboard.initialize():
        return dashboard
    else:
        raise RuntimeError("Failed to initialize DSUNet dashboard")
