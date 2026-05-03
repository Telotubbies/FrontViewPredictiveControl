"""
Perception Fusion: Blend UNet lane detection with waypoint data.

Fuses CTE, heading, and curvature from both sources based on mode weights.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


@dataclass
class LaneState:
    """State from UNet lane detection."""
    cte: float = 0.0           # Cross-track error (m)
    heading: float = 0.0       # Heading error (rad)
    curvature: float = 0.0     # Road curvature (1/m)
    confidence: float = 0.0    # Detection confidence [0, 1]
    valid: bool = False        # Geometry valid


@dataclass
class WaypointState:
    """State from CARLA waypoints."""
    cte: float = 0.0           # Cross-track error (m)
    heading: float = 0.0       # Heading error (rad)
    curvature: float = 0.0     # Road curvature (1/m)
    lookahead_curv: float = 0.0  # Max curvature in lookahead
    path: Optional[List[Tuple[float, float]]] = None  # (s, lat) path


@dataclass
class FusedState:
    """Fused perception state for MPC."""
    cte: float = 0.0
    heading: float = 0.0
    curvature: float = 0.0
    confidence: float = 0.0
    lane_weight: float = 0.5
    mode: str = "FUSION"
    reference_path: Optional[List[Tuple[float, float]]] = None


class PerceptionFusion:
    """
    Fuses UNet lane detection with waypoint data.
    
    Uses weighted blending based on mode selector weights.
    """
    
    def __init__(self, ema_alpha: float = 0.3):
        self._ema_alpha = ema_alpha
        self._prev_fused = FusedState()
        
    def reset(self):
        """Reset fusion state."""
        self._prev_fused = FusedState()
        
    def fuse(
        self,
        lane: LaneState,
        waypoint: WaypointState,
        lane_weight: float,
        mode: str = "FUSION",
    ) -> FusedState:
        """
        Fuse lane and waypoint states.
        
        Args:
            lane: State from UNet lane detection
            waypoint: State from CARLA waypoints
            lane_weight: Weight for lane detection [0, 1]
            mode: Current perception mode string
            
        Returns:
            Fused state for MPC
        """
        w_lane = lane_weight
        w_wp = 1.0 - lane_weight
        
        # Ego-lane sanity check: if lane CTE differs too much from waypoint CTE,
        # the lane detection likely jumped to adjacent lane - reduce trust
        LANE_WIDTH = 3.5  # meters
        cte_diff = abs(lane.cte - waypoint.cte) if lane.valid else 0
        
        # Progressive reduction based on CTE difference
        if cte_diff > LANE_WIDTH:
            # Way off - use waypoint only
            w_lane = 0.0
            w_wp = 1.0
        elif cte_diff > LANE_WIDTH * 0.5:
            # Significantly off - mostly waypoint
            w_lane = min(w_lane, 0.1)
            w_wp = 1.0 - w_lane
        elif cte_diff > LANE_WIDTH * 0.3:
            # Somewhat off - reduce lane weight
            w_lane = min(w_lane, 0.3)
            w_wp = 1.0 - w_lane
        
        # Weighted blend of CTE, heading, curvature
        if lane.valid and lane.confidence > 0.1:
            fused_cte = w_lane * lane.cte + w_wp * waypoint.cte
            fused_heading = w_lane * lane.heading + w_wp * waypoint.heading
            fused_curvature = w_lane * lane.curvature + w_wp * waypoint.curvature
        else:
            # Lane invalid: use waypoint only
            fused_cte = waypoint.cte
            fused_heading = waypoint.heading
            fused_curvature = waypoint.curvature
            
        # Confidence: blend with waypoint always having base confidence
        wp_base_conf = 0.6  # Waypoints are always somewhat reliable
        fused_conf = w_lane * lane.confidence + w_wp * wp_base_conf
        
        # Reference path: use waypoint path when turning, else generate from lane
        if w_wp > 0.5 and waypoint.path:
            ref_path = waypoint.path
        else:
            ref_path = None  # Will be generated from fused state
            
        # EMA smoothing
        a = self._ema_alpha
        fused = FusedState(
            cte=a * fused_cte + (1 - a) * self._prev_fused.cte,
            heading=a * fused_heading + (1 - a) * self._prev_fused.heading,
            curvature=a * fused_curvature + (1 - a) * self._prev_fused.curvature,
            confidence=fused_conf,
            lane_weight=lane_weight,
            mode=mode,
            reference_path=ref_path,
        )
        
        self._prev_fused = fused
        return fused
    
    def generate_reference_path(
        self,
        fused: FusedState,
        lookahead_m: float = 30.0,
        num_pts: int = 50,
    ) -> List[Tuple[float, float]]:
        """
        Generate reference path from fused state.
        
        Uses quintic polynomial for smooth path.
        
        Args:
            fused: Fused perception state
            lookahead_m: Lookahead distance (m)
            num_pts: Number of path points
            
        Returns:
            List of (s, lat) path points
        """
        if fused.reference_path:
            return fused.reference_path
            
        # Generate parabolic path from CTE, heading, curvature
        path = []
        for i in range(num_pts + 1):
            s = lookahead_m * i / num_pts
            lat = (
                fused.cte + 
                fused.heading * s + 
                0.5 * fused.curvature * s * s
            )
            path.append((s, lat))
        return path
