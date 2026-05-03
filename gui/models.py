# Data models for dashboard (no Qt dependency)
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_adaptive_weights(confidence: float, curvature: float) -> Dict[str, float]:
    """Compute scale_cte, scale_heading, scale_steer_rate, scale_jerk for display (mirrors control/lane_mpc)."""
    conf = float(np.clip(confidence, 0, 1))
    curv = abs(curvature)
    return {
        "w_e": 1.0,
        "w_psi": 1.0 + 0.8 * min(1.0, curv * 30.0),
        "w_delta": 1.0 + 0.6 * (1.0 - conf),
        "w_jerk": 1.0 + 0.5 * (1.0 - conf),
    }


@dataclass
class FrameData:
    """Single frame telemetry for dashboard updates (worker → UI)."""
    rgb: np.ndarray
    speed_kmh: float
    steer: float
    throttle: float
    brake: float
    cte_m: float
    heading_rad: float
    curvature: float
    mode: str
    lane_conf: float
    reference_path: Optional[List[Tuple[float, float]]] = None
    lane_overlay: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
