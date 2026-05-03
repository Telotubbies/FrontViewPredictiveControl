"""
Temporal lane buffer: aggregate last N frame masks with decay weights for consensus.
Reduces flicker in low light / rain when UNet output is noisy.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

DEFAULT_BUFFER_SIZE = 3  # ลดจาก 6 → 3 frames เพื่อลด lag
DEFAULT_DECAY = 0.7  # ลดจาก 0.85 → responsive ขึ้น
CONSENSUS_THRESHOLD = 0.35  # weighted sum above this → lane pixel


class TemporalLaneBuffer:
    """
    Holds last N binary masks; each pixel consensus = weighted average then threshold.
    Newest frame has weight 1.0, older frames have exponentially lower weight.
    """

    def __init__(
        self,
        size: int = DEFAULT_BUFFER_SIZE,
        decay: float = DEFAULT_DECAY,
        consensus_threshold: float = CONSENSUS_THRESHOLD,
    ):
        self._size = max(1, size)
        self._decay = max(0.01, min(1.0, decay))
        self._threshold = consensus_threshold
        self._masks: deque = deque(maxlen=self._size)
        self._weights: list[float] = []

    def push_and_get(self, mask: np.ndarray) -> np.ndarray:
        """
        Push new mask (binary 0/1 or bool) and return consensus mask (same shape, uint8 0/255).
        If buffer is empty or shape changes, return current mask as-is.
        """
        mask = (np.asarray(mask) > 0).astype(np.float64)
        if mask.ndim != 2:
            return (mask > 0).astype(np.uint8) * 255

        # Shape change: clear buffer and use only current frame
        if self._masks and self._masks[-1].shape != mask.shape:
            self._masks.clear()
        self._masks.append(mask.copy())
        n = len(self._masks)
        if n == 1:
            return (mask > 0).astype(np.uint8) * 255

        # Weights: newest = 1.0, then decay, decay^2, ...
        weights = np.array([self._decay ** (n - 1 - i) for i in range(n)], dtype=np.float64)
        weights /= weights.sum()
        stacked = np.stack(list(self._masks), axis=0)
        weighted = np.tensordot(weights, stacked, axes=(0, 0))
        out = (weighted >= self._threshold).astype(np.uint8) * 255
        return out

    def reset(self) -> None:
        self._masks.clear()
