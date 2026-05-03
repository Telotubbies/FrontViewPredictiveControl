#!/usr/bin/env python3
"""
Edge detection helpers for classical lane detection pipeline.

Used by classical_lane.py: threshold, blur_gaussian, mag_thresh.
Based on automaticaddison.com / Addison Sears-Collins style pipeline.
"""

import cv2
import numpy as np
from typing import Tuple


def threshold(gray: np.ndarray, thresh: Tuple[int, int] = (120, 255)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary threshold: output is 0 or 255 in range [thresh[0], thresh[1]].

    :param gray: Grayscale image
    :param thresh: (low, high) pixel range to keep as white
    :return: (mask_uint8, binary_255)
    """
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[(gray >= thresh[0]) & (gray <= thresh[1])] = 255
    return binary, binary


def blur_gaussian(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def mag_thresh(
    gray: np.ndarray,
    sobel_kernel: int = 3,
    thresh: Tuple[int, int] = (110, 255),
) -> np.ndarray:
    """
    Sobel magnitude threshold: gradient magnitude in [thresh[0], thresh[1]] -> 255.

    :param gray: Grayscale image
    :param sobel_kernel: Kernel size for Sobel (3, 5, 7)
    :param thresh: Magnitude range (after scaling to 0–255) to keep as white
    :return: Binary image (0 or 255)
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale = np.max(gradmag) if np.max(gradmag) > 0 else 1.0
    scaled = np.uint8(255 * gradmag / scale)
    binary = np.zeros_like(scaled, dtype=np.uint8)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 255
    return binary
