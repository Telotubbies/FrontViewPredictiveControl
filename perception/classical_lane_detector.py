"""
Classical Lane Detection based on Peter Moran's highway-lane-tracker
Uses color thresholding instead of deep learning (no GPU required!)

Adapted for CARLA from: https://github.com/peter-moran/highway-lane-tracker
"""
import numpy as np
import cv2
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class ClassicalLaneDetector:
    """
    Color-based lane detection using multi-channel thresholding.
    No deep learning required - works on CPU!
    
    Based on Peter Moran's approach:
    - LAB B-channel: Detects yellow/white lane markings
    - HSV V-channel: Detects bright pixels
    - HLS L-channel: Detects light pixels
    
    Each channel is:
    1. Normalized with CLAHE (adaptive histogram equalization)
    2. Thresholded to binary
    3. Combined into final score
    """
    
    def __init__(
        self,
        lab_threshold: int = 150,
        hsv_threshold: int = 220,
        hls_threshold: int = 210,
        lab_clip_limit: float = 2.0,
        hsv_clip_limit: float = 6.0,
        hls_clip_limit: float = 2.0,
    ):
        """
        Initialize classical lane detector with color thresholds.
        
        Args:
            lab_threshold: Threshold for LAB B-channel (yellow/white detection)
            hsv_threshold: Threshold for HSV V-channel (brightness)
            hls_threshold: Threshold for HLS L-channel (lightness)
            lab_clip_limit: CLAHE clip limit for LAB
            hsv_clip_limit: CLAHE clip limit for HSV
            hls_clip_limit: CLAHE clip limit for HLS
        """
        self.settings = [
            {
                'name': 'lab_b',
                'cspace': cv2.COLOR_RGB2LAB,
                'channel': 2,  # B channel
                'clipLimit': lab_clip_limit,
                'threshold': lab_threshold,
            },
            {
                'name': 'hsv_v',
                'cspace': cv2.COLOR_RGB2HSV,
                'channel': 2,  # V channel
                'clipLimit': hsv_clip_limit,
                'threshold': hsv_threshold,
            },
            {
                'name': 'hls_l',
                'cspace': cv2.COLOR_RGB2HLS,
                'channel': 1,  # L channel
                'clipLimit': hls_clip_limit,
                'threshold': hls_threshold,
            },
        ]
        
        logger.info(f"Classical lane detector initialized (LAB={lab_threshold}, HSV={hsv_threshold}, HLS={hls_threshold})")
    
    def score_pixels(self, img: np.ndarray) -> np.ndarray:
        """
        Score each pixel's likelihood of being a lane marking.
        
        Uses multi-channel color thresholding with CLAHE normalization.
        Higher pixel intensity = higher confidence of being lane marking.
        
        Args:
            img: RGB image (typically overhead/BEV view)
        
        Returns:
            Pixel scores (0-255), where higher = more likely lane marking
        """
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {img.shape}")
        
        # Accumulate scores from all channels
        scores = np.zeros(img.shape[0:2], dtype=np.uint8)
        
        for params in self.settings:
            # Convert to target color space
            color_img = cv2.cvtColor(img, params['cspace'])
            channel = color_img[:, :, params['channel']]
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This normalizes lighting across different regions of the image
            clahe = cv2.createCLAHE(
                clipLimit=params['clipLimit'],
                tileGridSize=(8, 8)
            )
            normalized = clahe.apply(channel)
            
            # Threshold to binary
            _, binary = cv2.threshold(
                normalized,
                params['threshold'],
                1,  # Binary value (will be summed)
                cv2.THRESH_BINARY
            )
            
            # Add to accumulated scores
            scores += binary
        
        # Normalize to 0-255 range
        scores_normalized = cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX)
        
        return scores_normalized
    
    def detect_lanes(
        self,
        image: np.ndarray,
        world=None,
        vehicle=None,
        score_threshold: int = 127,
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Detect lane markings in image using classical color thresholding.
        
        Compatible with UNet detector interface for easy swapping.
        
        Args:
            image: RGB image (H, W, 3)
            world: CARLA world (unused, for compatibility)
            vehicle: CARLA vehicle (unused, for compatibility)
            score_threshold: Threshold to convert scores to binary mask
        
        Returns:
            mask: Binary lane mask (H, W) uint8
            confidence: Confidence map (H, W) float32
            contours: Empty list (for compatibility)
        """
        # Score pixels
        pixel_scores = self.score_pixels(image)
        
        # Convert to binary mask
        mask = (pixel_scores > score_threshold).astype(np.uint8) * 255
        
        # Confidence is normalized pixel scores
        confidence = pixel_scores.astype(np.float32) / 255.0
        
        # Calculate overall confidence (mean of detected pixels)
        detected_pixels = np.sum(mask > 0)
        if detected_pixels > 0:
            avg_confidence = float(np.mean(confidence[mask > 0]))
        else:
            avg_confidence = 0.0
        
        logger.debug(f"Classical detection: {detected_pixels} pixels, confidence={avg_confidence:.3f}")
        
        return mask, confidence, []
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Callable interface for compatibility.
        """
        return self.detect_lanes(image)


def create_classical_detector(
    carla_optimized: bool = True
) -> ClassicalLaneDetector:
    """
    Factory function to create classical detector with optimal settings.
    
    Args:
        carla_optimized: Use thresholds optimized for CARLA simulator
    
    Returns:
        Configured ClassicalLaneDetector instance
    """
    if carla_optimized:
        # Thresholds tuned for CARLA's lighting and lane markings
        # May need adjustment based on testing
        return ClassicalLaneDetector(
            lab_threshold=140,  # Slightly lower for CARLA
            hsv_threshold=200,  # Lower for CARLA's lighting
            hls_threshold=190,  # Lower for CARLA
            lab_clip_limit=2.5,
            hsv_clip_limit=5.0,
            hls_clip_limit=2.5,
        )
    else:
        # Original Peter Moran settings (for real dashcam footage)
        return ClassicalLaneDetector(
            lab_threshold=150,
            hsv_threshold=220,
            hls_threshold=210,
            lab_clip_limit=2.0,
            hsv_clip_limit=6.0,
            hls_clip_limit=2.0,
        )


# Example usage:
if __name__ == "__main__":
    # Test classical detector
    detector = create_classical_detector(carla_optimized=True)
    
    # Load test image
    test_img = cv2.imread("test_image.png")
    if test_img is not None:
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        
        # Detect lanes
        mask, confidence, _ = detector.detect_lanes(test_img)
        
        print(f"Mask shape: {mask.shape}")
        print(f"Detected pixels: {np.sum(mask > 0)}")
        print(f"Mean confidence: {np.mean(confidence):.3f}")
