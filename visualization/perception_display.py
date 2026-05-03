"""
Perception Display - 4-Panel Visualization System

Creates multi-panel display showing:
- Top row: Raw camera, Left lane mask, Right lane mask
- Bottom row: Position panel with lane overlay and metrics

Author: Adapted from Peter Moran's visualization concepts
"""
import numpy as np
import cv2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PerceptionDisplay:
    """
    4-panel perception visualization:
    - Top row: Raw camera, Left mask, Right mask
    - Bottom row: Position panel (lane overlay + metrics)
    """
    
    def __init__(self, panel_width: int = 1920, panel_height: int = 1080):
        """
        Initialize perception display.
        
        Args:
            panel_width: Total display width in pixels
            panel_height: Total display height in pixels
        """
        self.panel_w = panel_width
        self.panel_h = panel_height
        
        # Calculate panel sizes - adjusted for 640x480 camera aspect ratio (4:3)
        # Top row: 3 panels at 4:3 aspect ratio
        # Bottom row: Full width for position panel
        self.top_panel_h = int(panel_height * 0.4)  # 40% for top row
        self.bottom_panel_h = int(panel_height * 0.6)  # 60% for bottom row
        self.single_panel_w = panel_width // 3  # Each top panel is 1/3 width
        
        logger.info(f"PerceptionDisplay initialized: {panel_width}x{panel_height}")
    
    def create_display(
        self,
        raw_camera: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        lane_overlay: np.ndarray,
        cte_m: float,
        curvature_m: float,
        confidence: float,
    ) -> np.ndarray:
        """
        Create complete 4-panel visualization.
        
        Args:
            raw_camera: Original RGB camera image
            left_mask: Left lane mask (grayscale or RGB)
            right_mask: Right lane mask (grayscale or RGB)
            lane_overlay: Lane overlay on camera image
            cte_m: Cross-track error in meters
            curvature_m: Radius of curvature in meters
            confidence: Detection confidence (0-1)
        
        Returns:
            Combined visualization image (H x W x 3)
        """
        try:
            # Create top row (3 panels)
            top_row = self._create_top_row(raw_camera, left_mask, right_mask)
            
            # Create bottom row (position panel)
            bottom_row = self._create_position_panel(
                lane_overlay, cte_m, curvature_m, confidence
            )
            
            # Combine vertically
            display = np.vstack([top_row, bottom_row])
            
            return display
            
        except Exception as e:
            logger.error(f"Error creating perception display: {e}")
            # Return blank display on error
            return np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
    
    def _create_top_row(
        self,
        raw: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create top row with 3 panels: Raw, Left Mask, Right Mask.
        
        Args:
            raw: Raw camera image
            left_mask: Left lane mask
            right_mask: Right lane mask
        
        Returns:
            Top row image (H x W x 3)
        """
        h, w = self.top_panel_h, self.single_panel_w
        
        # Resize all to same size
        raw_resized = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
        left_resized = cv2.resize(left_mask, (w, h), interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(right_mask, (w, h), interpolation=cv2.INTER_AREA)
        
        # Convert masks to color (green overlay)
        left_colored = self._colorize_mask(left_resized, (0, 255, 0))
        right_colored = self._colorize_mask(right_resized, (0, 255, 0))
        
        # Add labels
        raw_labeled = self._add_label(raw_resized.copy(), "Raw Camera")
        left_labeled = self._add_label(left_colored, "Left Lane Mask")
        right_labeled = self._add_label(right_colored, "Right Lane Mask")
        
        # Concatenate horizontally
        top_row = np.hstack([raw_labeled, left_labeled, right_labeled])
        
        return top_row
    
    def _create_position_panel(
        self,
        lane_overlay: np.ndarray,
        cte: float,
        curvature: float,
        confidence: float,
    ) -> np.ndarray:
        """
        Create position panel with lane overlay and metrics.
        
        Args:
            lane_overlay: Lane overlay image
            cte: Cross-track error in meters
            curvature: Radius of curvature in meters
            confidence: Detection confidence (0-1)
        
        Returns:
            Position panel image (H x W x 3)
        """
        h, w = self.bottom_panel_h, self.panel_w
        
        # Resize lane overlay to full width
        panel = cv2.resize(lane_overlay, (w, h), interpolation=cv2.INTER_AREA)
        
        # Add metrics overlay
        panel = self._add_metrics_overlay(panel, cte, curvature, confidence)
        
        return panel
    
    def _colorize_mask(self, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert grayscale mask to colored overlay.
        
        Args:
            mask: Grayscale or RGB mask
            color: BGR color tuple
        
        Returns:
            Colored mask (H x W x 3)
        """
        # Ensure mask is grayscale
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        
        # Create colored version (start with black background)
        colored = np.zeros((*mask_gray.shape, 3), dtype=np.uint8)
        
        # Apply color where mask is active
        mask_bool = mask_gray > 127
        colored[mask_bool] = color
        
        return colored
    
    def _add_label(self, img: np.ndarray, text: str) -> np.ndarray:
        """
        Add text label to top-left of image.
        
        Args:
            img: Image to label
            text: Label text
        
        Returns:
            Labeled image
        """
        # Ensure image is RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Add semi-transparent background for text
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (300, 50), (0, 0, 0), -1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Add text
        cv2.putText(
            img, text,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        return img
    
    def _add_metrics_overlay(
        self,
        img: np.ndarray,
        cte: float,
        curvature: float,
        confidence: float,
    ) -> np.ndarray:
        """
        Add metrics text overlay to position panel.
        
        Args:
            img: Position panel image
            cte: Cross-track error
            curvature: Radius of curvature
            confidence: Detection confidence
        
        Returns:
            Image with metrics overlay
        """
        # Semi-transparent background for metrics
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (400, 200), (0, 0, 0), -1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Position panel label
        cv2.putText(
            img, "Position",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # CTE
        cte_color = (0, 255, 0) if abs(cte) < 0.5 else (255, 255, 0) if abs(cte) < 1.0 else (0, 0, 255)
        cte_text = f"CTE: {cte:+.2f}m"
        cv2.putText(
            img, cte_text,
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, cte_color, 2, cv2.LINE_AA
        )
        
        # Curvature
        curv_text = f"Curvature: {curvature:.0f}m"
        cv2.putText(
            img, curv_text,
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Confidence
        conf_color = (0, 255, 0) if confidence > 0.7 else (255, 255, 0) if confidence > 0.5 else (0, 0, 255)
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(
            img, conf_text,
            (10, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, conf_color, 2, cv2.LINE_AA
        )
        
        return img


def create_perception_display(
    raw_camera: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    lane_overlay: np.ndarray,
    cte_m: float,
    curvature_m: float,
    confidence: float,
    panel_width: int = 1920,
    panel_height: int = 1080,
) -> np.ndarray:
    """
    Convenience function to create perception display.
    
    Args:
        raw_camera: Original RGB camera image
        left_mask: Left lane mask
        right_mask: Right lane mask
        lane_overlay: Lane overlay on camera
        cte_m: Cross-track error in meters
        curvature_m: Radius of curvature in meters
        confidence: Detection confidence (0-1)
        panel_width: Display width
        panel_height: Display height
    
    Returns:
        4-panel visualization image
    """
    display = PerceptionDisplay(panel_width, panel_height)
    return display.create_display(
        raw_camera, left_mask, right_mask, lane_overlay,
        cte_m, curvature_m, confidence
    )
