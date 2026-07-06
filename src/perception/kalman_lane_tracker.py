"""
Kalman filter tracker for left/right lane polynomial coefficients.

When UNet misses a frame, predicts 1–2 frames from dynamics (constant curvature)
so overlay and reference path stay smooth instead of resetting.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

POLY_DIM = 3  # [c2, c1, c0] for y = c2*x^2 + c1*x + c0
DEFAULT_Q = 0.15  # เพิ่มจาก 0.08 → 0.15 (faster response)
DEFAULT_R = 0.05  # ลดจาก 0.08 → 0.05 (เชื่อ measurement มากขึ้น)
MAX_FRAMES_STALE = 3


class _KalmanPoly:
    """Single polynomial coefficient Kalman filter."""

    def __init__(self, q: float = DEFAULT_Q, r: float = DEFAULT_R, dim: int = POLY_DIM):
        self.dim = dim
        self.x = np.zeros(dim, dtype=np.float64)
        self.P = np.eye(dim) * 0.1
        self.F = np.eye(dim)
        self.H = np.eye(dim)
        self.Q = np.eye(dim) * q
        self.R = np.eye(dim) * r
        self._initialized = False

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
    ):
        self._left = _KalmanPoly(q=q, r=r)
        self._right = _KalmanPoly(q=q, r=r)
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
