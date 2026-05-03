#!/usr/bin/env python3
"""
BEV Transform Test with GUI Visualization
ทดสอบการแปลงพิกัด Camera → BEV พร้อม interactive GUI
"""

import sys
import numpy as np
import cv2
import pygame
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import time

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CAM_W, CAM_H, BEV_W, BEV_H, BEV_WIN_W, BEV_WIN_H,
    BEV_LANE_SEARCH_TOP_RATIO, ROI_BOTTOM_RATIO
)
from perception.lane_trajectory import _get_bev_transform
from perception.lane_detector import LaneDetector
from perception.lane_trajectory import LaneTrajectoryPipeline

logger = logging.getLogger(__name__)


class BEVTransformTester:
    """BEV Transform Tester with GUI Visualization"""
    
    def __init__(self):
        # Window settings
        self.screen_w = 1400
        self.screen_h = 800
        
        # Viewport areas
        self.camera_view = (20, 60, 640, 480)        # x, y, w, h
        self.bev_view = (680, 60, BEV_W, BEV_H)      # x, y, w, h
        self.transform_view = (20, 560, 640, 200)    # x, y, w, h
        self.info_view = (680, 560, 400, 200)        # x, y, w, h
        
        # BEV parameters (adjustable)
        self.bev_params = {
            'src_top_ratio': 0.45,      # Default from _get_bev_transform
            'src_bot_ratio': 0.97,      # Default from _get_bev_transform
            'src_top_margin': 0.42,     # Default from _get_bev_transform
            'src_bot_margin': 0.10,     # Default from _get_bev_transform
            'lookahead_m': 30.0,
            'half_width_m': 8.0
        }
        
        # Pygame components
        self.screen = None
        self.font = None
        self.small_font = None
        self.clock = None
        
        # Test data
        self.test_image = None
        self.lane_detector = None
        self.trajectory_pipeline = None
        
        # Transform matrices
        self.M = None
        self.M_inv = None
        self.src_pts = None
        self.dst_pts = None
        
        # Mouse tracking
        self.mouse_pos = (0, 0)
        self.camera_coord = None
        self.bev_coord = None
        
        # Performance
        self.fps = 0
        self.last_time = time.time()
        
    def initialize_pygame(self) -> bool:
        """Initialize pygame components"""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("BEV Transform Test - Interactive Visualization")
            
            self.font = pygame.font.SysFont("monospace", 16)
            self.small_font = pygame.font.SysFont("monospace", 12)
            self.clock = pygame.time.Clock()
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}")
            return False
    
    def initialize_perception(self) -> bool:
        """Initialize perception components"""
        try:
            # Create lane detector
            self.lane_detector = LaneDetector(
                model_path="model/lane_unet_final.pth",
                use_carla=False,
                model_type="unet"
            )
            
            # Create trajectory pipeline
            self.trajectory_pipeline = LaneTrajectoryPipeline(detector=self.lane_detector)
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize perception: {e}")
            return False
    
    def create_test_image(self) -> np.ndarray:
        """Create synthetic test image with lane patterns"""
        # Create base image
        img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        img.fill(50)  # Dark gray background
        
        # Add road texture
        img[100:380, :] = [80, 80, 80]  # Lighter gray road
        
        # Draw lane markings
        # Center line (dashed)
        for y in range(100, 380, 20):
            if y % 40 < 20:  # Dashed pattern
                cv2.line(img, (CAM_W//2 - 2, y), (CAM_W//2 + 2, y), (255, 255, 255), 3)
        
        # Left lane boundary
        left_x = CAM_W//2 - 150
        cv2.line(img, (left_x, 100), (left_x - 20, 380), (255, 255, 255), 4)
        
        # Right lane boundary  
        right_x = CAM_W//2 + 150
        cv2.line(img, (right_x, 100), (right_x + 20, 380), (255, 255, 255), 4)
        
        # Add some test points
        test_points = [
            (CAM_W//2, 200),      # Center
            (left_x, 200),        # Left lane
            (right_x, 200),       # Right lane
            (CAM_W//2, 300),      # Center forward
        ]
        
        for i, (x, y) in enumerate(test_points):
            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)][i]
            cv2.circle(img, (x, y), 8, color, -1)
            cv2.circle(img, (x, y), 8, (0, 0, 0), 2)
        
        return img
    
    def update_bev_transform(self):
        """Update BEV transform matrices"""
        self.M, self.M_inv, self.src_pts, self.dst_pts = _get_bev_transform(
            CAM_W, CAM_H,
            self.bev_params['src_top_ratio'],
            self.bev_params['src_bot_ratio'],
            self.bev_params['src_top_margin'],
            self.bev_params['src_bot_margin']
        )
    
    def camera_to_bev(self, cam_point: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """Convert camera coordinates to BEV coordinates"""
        if self.M is None:
            return None
            
        try:
            # Convert to homogeneous coordinates
            pt = np.array([cam_point[0], cam_point[1], 1], dtype=np.float32)
            
            # Apply transform
            bev_pt = self.M @ pt
            bev_pt = bev_pt / bev_pt[2]  # Normalize
            
            # Convert to meters
            bev_x_m = (bev_pt[0] / BEV_W) * 2 * self.bev_params['half_width_m'] - self.bev_params['half_width_m']
            bev_y_m = self.bev_params['lookahead_m'] * (1 - bev_pt[1] / BEV_H)
            
            return (bev_x_m, bev_y_m)
        except Exception as e:
            logger.error(f"Camera to BEV conversion failed: {e}")
            return None
    
    def bev_to_camera(self, bev_point: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Convert BEV coordinates to camera coordinates"""
        if self.M_inv is None:
            return None
            
        try:
            # Convert meters to BEV pixels
            bev_x_px = (bev_point[0] + self.bev_params['half_width_m']) / (2 * self.bev_params['half_width_m']) * BEV_W
            bev_y_px = (1 - bev_point[1] / self.bev_params['lookahead_m']) * BEV_H
            
            # Convert to homogeneous coordinates
            pt = np.array([bev_x_px, bev_y_px, 1], dtype=np.float32)
            
            # Apply inverse transform
            cam_pt = self.M_inv @ pt
            cam_pt = cam_pt / cam_pt[2]  # Normalize
            
            return (int(cam_pt[0]), int(cam_pt[1]))
        except Exception as e:
            logger.error(f"BEV to camera conversion failed: {e}")
            return None
    
    def process_lane_detection(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process lane detection on test image"""
        if self.trajectory_pipeline is None or self.test_image is None:
            return None, None
            
        try:
            result = self.trajectory_pipeline.process(self.test_image)
            return result.bev_binary, result.mask_vis
        except Exception as e:
            logger.error(f"Lane detection failed: {e}")
            return None, None
    
    def draw_viewport(self, x: int, y: int, w: int, h: int, title: str):
        """Draw viewport border and title"""
        # Border
        pygame.draw.rect(self.screen, (100, 100, 100), (x-2, y-2, w+4, h+4), 2)
        
        # Title
        title_surf = self.font.render(title, True, (200, 200, 200))
        self.screen.blit(title_surf, (x, y - 25))
    
    def draw_camera_view(self):
        """Draw camera view with overlay"""
        if self.test_image is None:
            logger.warning("Test image is None")
            return
            
        # Create a copy for drawing overlays
        display_img = self.test_image.copy()
        
        # Draw source trapezoid
        if self.src_pts is not None:
            pts = self.src_pts.astype(int)
            cv2.polylines(display_img, [pts], True, (0, 255, 0), 2)
        
        # Draw mouse coordinate if in camera view
        cam_x, cam_y, cam_w, cam_h = self.camera_view
        mouse_rel_x = self.mouse_pos[0] - cam_x
        mouse_rel_y = self.mouse_pos[1] - cam_y
        
        if 0 <= mouse_rel_x < cam_w and 0 <= mouse_rel_y < cam_h:
            cv2.circle(display_img, (mouse_rel_x, mouse_rel_y), 10, (255, 255, 0), 2)
            
            # Convert to BEV and show
            bev_coord = self.camera_to_bev((mouse_rel_x, mouse_rel_y))
            if bev_coord:
                text = f"BEV: ({bev_coord[0]:.2f}m, {bev_coord[1]:.2f}m)"
                cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Convert image for pygame
        rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        # Create pygame surface and display
        try:
            surf = pygame.surfarray.make_surface(rgb_img.swapaxes(0, 1))
            self.screen.blit(surf, (cam_x, cam_y))
            logger.debug(f"Camera view drawn at ({cam_x}, {cam_y}) size ({cam_w}, {cam_h})")
        except Exception as e:
            logger.error(f"Failed to draw camera view: {e}")
    
    def draw_bev_view(self, bev_binary: Optional[np.ndarray]):
        """Draw BEV view with coordinate grid"""
        cam_x, cam_y, cam_w, cam_h = self.bev_view
        
        # Background
        pygame.draw.rect(self.screen, (20, 20, 30), (cam_x, cam_y, cam_w, cam_h))
        
        # Draw coordinate grid
        grid_color = (60, 60, 80)
        
        # Vertical lines (lateral)
        for i in range(-8, 9, 2):  # -8m to +8m every 2m
            x_px = cam_x + int((i + 8) / 16 * cam_w)
            pygame.draw.line(self.screen, grid_color, (x_px, cam_y), (x_px, cam_y + cam_h), 1)
            
            # Label
            if i != 0:
                label = self.small_font.render(f"{i}m", True, (150, 150, 150))
                self.screen.blit(label, (x_px - 10, cam_y + cam_h - 20))
        
        # Horizontal lines (forward)
        for i in range(0, 31, 5):  # 0m to 30m every 5m
            y_px = cam_y + int(i / 30 * cam_h)
            pygame.draw.line(self.screen, grid_color, (cam_x, y_px), (cam_x + cam_w, y_px), 1)
            
            # Label
            label = self.small_font.render(f"{i}m", True, (150, 150, 150))
            self.screen.blit(label, (cam_x + 5, y_px - 15))
        
        # Draw lane detection result
        if bev_binary is not None:
            # Resize BEV to fit viewport
            bev_resized = cv2.resize(bev_binary, (cam_w, cam_h))
            
            # Create colored overlay
            bev_color = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            bev_color[bev_resized > 0] = [0, 255, 0]  # Green for lane pixels
            
            # Convert to pygame surface
            bev_surf = pygame.surfarray.make_surface(bev_color.swapaxes(0, 1))
            bev_surf.set_alpha(128)  # Semi-transparent
            self.screen.blit(bev_surf, (cam_x, cam_y))
        
        # Draw mouse coordinate in BEV
        mouse_rel_x = self.mouse_pos[0] - cam_x
        mouse_rel_y = self.mouse_pos[1] - cam_y
        
        if 0 <= mouse_rel_x < cam_w and 0 <= mouse_rel_y < cam_h:
            # Convert to meters
            bev_x_m = (mouse_rel_x / cam_w) * 16 - 8  # -8m to +8m
            bev_y_m = (1 - mouse_rel_y / cam_h) * 30   # 0m to 30m
            
            # Draw marker
            pygame.draw.circle(self.screen, (255, 255, 0), (cam_x + mouse_rel_x, cam_y + mouse_rel_y), 8, 2)
            
            # Convert to camera and show
            cam_coord = self.bev_to_camera((bev_x_m, bev_y_m))
            if cam_coord:
                text = f"Camera: ({cam_coord[0]}, {cam_coord[1]})"
                text_surf = self.small_font.render(text, True, (255, 255, 0))
                self.screen.blit(text_surf, (cam_x + 10, cam_y + 10))
    
    def draw_transform_visualization(self):
        """Draw transform matrix visualization"""
        cam_x, cam_y, cam_w, cam_h = self.transform_view
        
        # Background
        pygame.draw.rect(self.screen, (30, 30, 40), (cam_x, cam_y, cam_w, cam_h))
        
        # Draw transform matrix
        if self.M is not None:
            matrix_text = "Transform Matrix M:"
            y_offset = cam_y + 10
            
            text_surf = self.small_font.render(matrix_text, True, (200, 200, 200))
            self.screen.blit(text_surf, (cam_x + 10, y_offset))
            y_offset += 20
            
            for i, row in enumerate(self.M):
                row_text = f"[{row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f}]"
                text_surf = self.small_font.render(row_text, True, (150, 255, 150))
                self.screen.blit(text_surf, (cam_x + 10, y_offset))
                y_offset += 15
        
        # Draw parameters
        y_offset = cam_y + 80
        param_text = "BEV Parameters:"
        text_surf = self.small_font.render(param_text, True, (200, 200, 200))
        self.screen.blit(text_surf, (cam_x + 10, y_offset))
        y_offset += 20
        
        for key, value in self.bev_params.items():
            param_line = f"{key}: {value:.3f}"
            text_surf = self.small_font.render(param_line, True, (150, 150, 255))
            self.screen.blit(text_surf, (cam_x + 10, y_offset))
            y_offset += 15
    
    def draw_info_panel(self):
        """Draw information panel"""
        cam_x, cam_y, cam_w, cam_h = self.info_view
        
        # Background
        pygame.draw.rect(self.screen, (30, 30, 40), (cam_x, cam_y, cam_w, cam_h))
        
        # Title
        title_surf = self.font.render("Information & Controls", True, (200, 200, 200))
        self.screen.blit(title_surf, (cam_x + 10, cam_y + 10))
        
        # FPS
        y_offset = cam_y + 40
        fps_text = f"FPS: {self.fps:.1f}"
        text_surf = self.small_font.render(fps_text, True, (100, 255, 100))
        self.screen.blit(text_surf, (cam_x + 10, y_offset))
        y_offset += 20
        
        # Mouse position
        mouse_text = f"Mouse: ({self.mouse_pos[0]}, {self.mouse_pos[1]})"
        text_surf = self.small_font.render(mouse_text, True, (150, 150, 150))
        self.screen.blit(text_surf, (cam_x + 10, y_offset))
        y_offset += 20
        
        # Controls
        y_offset += 20
        controls_title = "Controls:"
        text_surf = self.small_font.render(controls_title, True, (200, 200, 200))
        self.screen.blit(text_surf, (cam_x + 10, y_offset))
        y_offset += 20
        
        controls = [
            "↑/↓: Adjust top ratio",
            "←/→: Adjust bottom ratio", 
            "W/S: Adjust top margin",
            "A/D: Adjust bottom margin",
            "R: Reset parameters",
            "SPACE: Process lane detection",
            "ESC: Exit"
        ]
        
        for control in controls:
            text_surf = self.small_font.render(control, True, (150, 150, 150))
            self.screen.blit(text_surf, (cam_x + 10, y_offset))
            y_offset += 15
    
    def handle_events(self) -> bool:
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_UP:
                    self.bev_params['src_top_ratio'] = min(0.8, self.bev_params['src_top_ratio'] + 0.05)
                    self.update_bev_transform()
                elif event.key == pygame.K_DOWN:
                    self.bev_params['src_top_ratio'] = max(0.1, self.bev_params['src_top_ratio'] - 0.05)
                    self.update_bev_transform()
                elif event.key == pygame.K_LEFT:
                    self.bev_params['src_bot_ratio'] = max(0.5, self.bev_params['src_bot_ratio'] - 0.05)
                    self.update_bev_transform()
                elif event.key == pygame.K_RIGHT:
                    self.bev_params['src_bot_ratio'] = min(1.0, self.bev_params['src_bot_ratio'] + 0.05)
                    self.update_bev_transform()
                elif event.key == pygame.K_w:
                    self.bev_params['src_top_margin'] = min(0.5, self.bev_params['src_top_margin'] + 0.05)
                    self.update_bev_transform()
                elif event.key == pygame.K_s:
                    self.bev_params['src_top_margin'] = max(0.1, self.bev_params['src_top_margin'] - 0.05)
                    self.update_bev_transform()
                elif event.key == pygame.K_a:
                    self.bev_params['src_bot_margin'] = max(0.0, self.bev_params['src_bot_margin'] - 0.05)
                    self.update_bev_transform()
                elif event.key == pygame.K_d:
                    self.bev_params['src_bot_margin'] = min(0.5, self.bev_params['src_bot_margin'] + 0.05)
                    self.update_bev_transform()
                elif event.key == pygame.K_r:
                    # Reset to defaults
                    self.bev_params = {
                        'src_top_ratio': 0.45,
                        'src_bot_ratio': 0.97,
                        'src_top_margin': 0.42,
                        'src_bot_margin': 0.10,
                        'lookahead_m': 30.0,
                        'half_width_m': 8.0
                    }
                    self.update_bev_transform()
                elif event.key == pygame.K_SPACE:
                    # Process lane detection
                    logger.info("Processing lane detection...")
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
        
        return True
    
    def run(self) -> None:
        """Main run loop"""
        logger.info("Starting BEV Transform Test GUI...")
        
        # Initialize
        if not self.initialize_pygame():
            logger.error("Failed to initialize pygame")
            return
        
        if not self.initialize_perception():
            logger.error("Failed to initialize perception")
            return
        
        # Create test image
        self.test_image = self.create_test_image()
        
        # Initialize BEV transform
        self.update_bev_transform()
        
        # Main loop
        running = True
        lane_bev = None
        frame_count = 0
        
        logger.info("Starting main loop...")
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Clear screen
            self.screen.fill((20, 20, 30))
            
            # Process lane detection periodically
            current_time = time.time()
            if current_time - self.last_time > 2.0:  # Every 2 seconds
                logger.info("Processing lane detection...")
                lane_bev, _ = self.process_lane_detection()
                self.last_time = current_time
                logger.info(f"Lane detection processed, BEV shape: {lane_bev.shape if lane_bev is not None else None}")
            
            # Draw viewports
            self.draw_viewport(*self.camera_view, "Camera View (640x480)")
            self.draw_viewport(*self.bev_view, "BEV View (320x450)")
            self.draw_viewport(*self.transform_view, "Transform Matrix")
            self.draw_viewport(*self.info_view, "Information")
            
            # Draw content
            self.draw_camera_view()
            self.draw_bev_view(lane_bev)
            self.draw_transform_visualization()
            self.draw_info_panel()
            
            # Update display
            pygame.display.flip()
            
            # Calculate FPS
            self.clock.tick(30)
            self.fps = self.clock.get_fps()
            
            frame_count += 1
            if frame_count % 60 == 0:  # Log every 60 frames
                logger.info(f"Frame {frame_count}, FPS: {self.fps:.1f}")
        
        # Cleanup
        pygame.quit()
        logger.info("BEV Transform Test GUI closed")


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    
    tester = BEVTransformTester()
    tester.run()


if __name__ == "__main__":
    main()
