"""
Kalman filter tracker for left/right lane polynomial coefficients.

When UNet misses a frame, predicts 1–2 frames from dynamics (constant curvature)
so overlay and reference path stay smooth instead of resetting.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

POLY_DIM = 3  # [c2, c1, c0] for y = c2*x^2 + c1*x + c0
DEFAULT_Q = 0.15  # เพิ่มจาก 0.08 → 0.15 (faster response)
DEFAULT_R = 0.05  # ลดจาก 0.08 → 0.05 (เชื่อ measurement มากขึ้น)
MAX_FRAMES_STALE = 3


class _KalmanPoly:
    """Single polynomial coefficient Kalman filter."""

    def __init__(
        self,
        q: float = DEFAULT_Q,
        r: float = DEFAULT_R,
        dim: int = POLY_DIM,
        outlier_threshold: float = 3.0,
        adaptive_R: bool = False,
    ):
        self.dim = dim
        self.x = np.zeros(dim, dtype=np.float64)
        self.P = np.eye(dim) * 0.1
        self.F = np.eye(dim)
        self.H = np.eye(dim)
        self.Q = np.eye(dim) * q
        self.R = np.eye(dim) * r
        self._base_r = r
        self._initialized = False
        self._outlier_threshold = outlier_threshold
        self._adaptive_R = adaptive_R
        self._outlier_count = 0
        self._last_residual = 0.0
        self._adaptive_R_value = r

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).ravel()
        if len(z) != self.dim:
            return self.x.copy()
        if not self._initialized:
            self.x = z.copy()
            self._initialized = True
            return self.x.copy()
        y = z - self.H @ self.x
        residual_norm = float(np.linalg.norm(y))
        self._last_residual = residual_norm
        if residual_norm > self._outlier_threshold:
            self._outlier_count += 1
            logger.debug(
                "Outlier rejected: residual=%.3f > threshold=%.3f (count=%d)",
                residual_norm,
                self._outlier_threshold,
                self._outlier_count,
            )
            return self.x.copy()
        if self._adaptive_R:
            if residual_norm < 1.0:
                self._adaptive_R_value = self._base_r * 0.8
            elif residual_norm > 2.0:
                self._adaptive_R_value = self._base_r * 1.5
            else:
                self._adaptive_R_value = self._base_r
            self.R = np.eye(self.dim) * self._adaptive_R_value
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x.copy()
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim) - K @ self.H) @ self.P
        return self.x.copy()

    def get_state(self) -> Optional[np.ndarray]:
        return self.x.copy() if self._initialized else None

    def reset(self) -> None:
        self._initialized = False
        self.x = np.zeros(self.dim, dtype=np.float64)
        self.P = np.eye(self.dim) * 0.1
        self._outlier_count = 0
        self._last_residual = 0.0
        self._adaptive_R_value = self._base_r
        self.R = np.eye(self.dim) * self._base_r

    def get_debug_info(self) -> dict:
        return {
            "last_residual": self._last_residual,
            "outlier_count": self._outlier_count,
            "current_P": float(np.trace(self.P)),
            "adaptive_R_value": self._adaptive_R_value,
        }


class KalmanLaneTracker:
    """
    Tracks left and right lane polynomial coefficients (a, b, c for y = ax^2 + bx + c).
    When observation is missing, predicts from previous state so overlay does not jump.
    """

    def __init__(
        self,
        q: float = DEFAULT_Q,
        r: float = DEFAULT_R,
        max_frames_stale: int = MAX_FRAMES_STALE,
        outlier_threshold: float = 3.0,
        adaptive_R: bool = False,
    ):
        self._left = _KalmanPoly(q=q, r=r, outlier_threshold=outlier_threshold, adaptive_R=adaptive_R)
        self._right = _KalmanPoly(q=q, r=r, outlier_threshold=outlier_threshold, adaptive_R=adaptive_R)
        self._max_frames_stale = max_frames_stale
        self._frames_without_left = 0
        self._frames_without_right = 0

    def update(
        self,
        left_coeffs: Optional[np.ndarray],
        right_coeffs: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Update with new observations. If a side is None, predict from dynamics.
        Returns (left_smoothed, right_smoothed) for overlay / reference path.
        """
        if left_coeffs is not None:
            left_out = self._left.update(np.asarray(left_coeffs, dtype=np.float64).ravel()[:POLY_DIM])
            self._frames_without_left = 0
        else:
            self._frames_without_left += 1
            if self._frames_without_left >= self._max_frames_stale:
                self._left.reset()
                left_out = None
            else:
                left_out = self._left.predict() if self._left._initialized else None

        if right_coeffs is not None:
            right_out = self._right.update(np.asarray(right_coeffs, dtype=np.float64).ravel()[:POLY_DIM])
            self._frames_without_right = 0
        else:
            self._frames_without_right += 1
            if self._frames_without_right >= self._max_frames_stale:
                self._right.reset()
                right_out = None
            else:
                right_out = self._right.predict() if self._right._initialized else None

        return (left_out, right_out)

    def get_left_right(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return current left and right state (after last update/predict)."""
        return (self._left.get_state(), self._right.get_state())

    def reset(self) -> None:
        self._left.reset()
        self._right.reset()
        self._frames_without_left = 0
        self._frames_without_right = 0

    def get_debug_info(self) -> dict:
        """Return debug info for left and right filters."""
        left_info = self._left.get_debug_info()
        right_info = self._right.get_debug_info()
        return {
            "left": left_info,
            "right": right_info,
            "last_residual": max(left_info["last_residual"], right_info["last_residual"]),
            "outlier_count": left_info["outlier_count"] + right_info["outlier_count"],
            "current_P": left_info["current_P"] + right_info["current_P"],
            "adaptive_R_value": max(left_info["adaptive_R_value"], right_info["adaptive_R_value"]),
        }
