"""
ADAS v2: Intelligent Mode Switching Lane Keeping System

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    ADAS Controller                          │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
    │  │   UNet      │    │  Waypoint   │    │   Mode      │     │
    │  │   Lane      │───▶│   Fusion    │───▶│  Selector   │     │
    │  │  Detection  │    │   Module    │    │             │     │
    │  └─────────────┘    └─────────────┘    └─────────────┘     │
    │                                                             │
    │  Modes:                                                     │
    │    - LANE_ONLY: High confidence, straight road              │
    │    - WAYPOINT_ONLY: Turning, low confidence                 │
    │    - FUSION: Weighted blend (default)                       │
    └─────────────────────────────────────────────────────────────┘

Key Features:
    1. Turn Prediction: Use waypoint lookahead to know turns in advance
    2. Confidence-based Switching: Switch mode based on lane detection confidence
    3. Smooth Transition: EMA blend between modes to avoid jerky steering
    4. Google Maps Style: When turning, trust waypoint more than lane detection
"""

from .mode_selector import ModeSelector, PerceptionMode
from .fusion import PerceptionFusion
from .controller import ADASController
from .recorder import ADASRecorder, ADASReplayer
from .dashboard import ADASv2Dashboard, DashboardState
from .lane_change import LaneChangeDetector, LaneChangeState
from .metrics import PerformanceMetrics, MetricsSnapshot
from .multi_lane import MultiLaneDetector, LaneChangePlanner, LanePosition

__all__ = [
    "ModeSelector", "PerceptionMode", "PerceptionFusion", "ADASController",
    "ADASRecorder", "ADASReplayer",
    "ADASv2Dashboard", "DashboardState",
    "LaneChangeDetector", "LaneChangeState",
    "PerformanceMetrics", "MetricsSnapshot",
    "MultiLaneDetector", "LaneChangePlanner", "LanePosition",
]
