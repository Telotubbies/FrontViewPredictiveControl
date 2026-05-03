#!/usr/bin/env python3
"""
LSTM Temporal Predictor for Lane Polynomial Coefficients.

Takes sequence of lane polynomial coefficients from UNet mask processing
and outputs temporally smoothed coefficients for MPC.

Input: sequence of (cte, heading_err, curvature) from last N frames
Output: smoothed (cte, heading_err, curvature) + predicted next state

Architecture:
    Input(3) → LSTM(hidden=32, layers=2) → FC(32→16) → FC(16→3)
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Configuration for LSTM temporal predictor."""
    input_dim: int = 3       # (cte, heading_error, curvature)
    hidden_dim: int = 32
    num_layers: int = 2
    output_dim: int = 3      # (cte_smooth, heading_smooth, curvature_smooth)
    seq_len: int = 10        # frames of history
    dropout: float = 0.1


class LaneLSTM(nn.Module):
    """
    LSTM for temporal smoothing of lane state.

    Processes a window of recent lane measurements to produce
    a temporally consistent estimate, reducing frame-to-frame jitter
    from UNet detection noise.
    """

    def __init__(self, cfg: LSTMConfig = None):
        super().__init__()
        self.cfg = cfg or LSTMConfig()
        c = self.cfg

        self.lstm = nn.LSTM(
            input_size=c.input_dim,
            hidden_size=c.hidden_dim,
            num_layers=c.num_layers,
            batch_first=True,
            dropout=c.dropout if c.num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(c.hidden_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, c.output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) tensor of lane state history

        Returns:
            (batch, output_dim) smoothed lane state
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class LaneTemporalSmoother:
    """
    Online temporal smoother using LSTM or exponential moving average.

    Maintains a sliding window of lane states and produces
    smoothed output each frame.

    If no trained LSTM model is available, falls back to EMA smoothing.
    """

    def __init__(self, model_path: str = None, device: torch.device = None,
                 seq_len: int = 10, ema_alpha: float = 0.3):
        self.device = device or torch.device("cpu")
        self.seq_len = seq_len
        self.ema_alpha = ema_alpha
        self._history = deque(maxlen=seq_len)
        self._prev_output = np.zeros(3)

        self.model = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load trained LSTM model."""
        cfg = LSTMConfig()
        self.model = LaneLSTM(cfg).to(self.device)
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        self.model.eval()

    def update(self, cte: float, heading: float, curvature: float):
        """
        Process new lane measurement, return smoothed output.

        Args:
            cte: cross-track error (normalized)
            heading: heading error (radians)
            curvature: road curvature estimate

        Returns:
            (cte_smooth, heading_smooth, curvature_smooth) as numpy array
        """
        state = np.array([cte, heading, curvature], dtype=np.float32)
        self._history.append(state)

        if self.model is not None and len(self._history) >= self.seq_len:
            return self._lstm_smooth()
        else:
            return self._ema_smooth(state)

    @torch.no_grad()
    def _lstm_smooth(self):
        """Use LSTM for temporal smoothing."""
        seq = np.array(list(self._history), dtype=np.float32)
        tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        output = self.model(tensor).squeeze(0).cpu().numpy()
        self._prev_output = output
        return output

    def _ema_smooth(self, state: np.ndarray):
        """Exponential moving average fallback."""
        a = self.ema_alpha
        smoothed = a * self._prev_output + (1 - a) * state
        self._prev_output = smoothed
        return smoothed

    def reset(self):
        """Reset history and state."""
        self._history.clear()
        self._prev_output = np.zeros(3)


def _test_lstm():
    """Basic sanity test for LSTM module."""
    cfg = LSTMConfig()
    model = LaneLSTM(cfg)

    batch = torch.randn(4, cfg.seq_len, cfg.input_dim)
    out = model(batch)
    assert out.shape == (4, cfg.output_dim), f"Expected (4,3), got {out.shape}"

    smoother = LaneTemporalSmoother(ema_alpha=0.3)
    for i in range(20):
        cte = 0.1 * np.sin(i * 0.5) + np.random.normal(0, 0.02)
        heading = 0.05 * np.cos(i * 0.3) + np.random.normal(0, 0.01)
        curv = 0.02 + np.random.normal(0, 0.005)
        result = smoother.update(cte, heading, curv)
        assert result.shape == (3,), f"Expected (3,), got {result.shape}"

    logger.info("LSTM tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _test_lstm()
