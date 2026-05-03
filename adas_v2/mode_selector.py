"""
Mode Selector: Intelligent switching between perception modes.

Modes:
    - LANE_ONLY: Use UNet lane detection (high confidence, straight road)
    - WAYPOINT_ONLY: Use CARLA waypoints (turning, low confidence)
    - FUSION: Weighted blend of both (default, transitions)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np


class PerceptionMode(Enum):
    """Perception mode for lane keeping."""
    LANE_ONLY = "lane_only"       # Trust UNet lane detection
    WAYPOINT_ONLY = "waypoint_only"  # Trust waypoints (like Google Maps)
    FUSION = "fusion"             # Weighted blend


@dataclass
class ModeConfig:
    """Configuration for mode switching thresholds."""
    # Confidence thresholds
    conf_high: float = 0.75       # Above this: can use LANE_ONLY
    conf_low: float = 0.40        # Below this: switch to WAYPOINT_ONLY
    
    # Curvature thresholds (from waypoint lookahead)
    curv_straight: float = 0.005  # Below this: straight road (lowered)
    curv_turning: float = 0.015   # Above this: definitely turning (lowered to trust WP earlier)
    
    # Transition smoothing
    mode_switch_frames: int = 3   # Frames to confirm mode switch (faster)
    blend_alpha: float = 0.4      # EMA alpha for weight transitions (faster)


class ModeSelector:
    """
    Intelligent mode selector for ADAS.
    
    Decides when to trust UNet lane detection vs waypoints based on:
    1. Lane detection confidence
    2. Upcoming curvature (from waypoint lookahead)
    3. Historical stability
    """
    
    def __init__(self, config: Optional[ModeConfig] = None):
        self.config = config or ModeConfig()
        self._current_mode = PerceptionMode.FUSION
        self._target_mode = PerceptionMode.FUSION
        self._mode_frames = 0
        self._lane_weight = 0.5  # 0 = full waypoint, 1 = full lane
        
    def reset(self):
        """Reset mode selector state."""
        self._current_mode = PerceptionMode.FUSION
        self._target_mode = PerceptionMode.FUSION
        self._mode_frames = 0
        self._lane_weight = 0.5
        
    def update(
        self,
        lane_conf: float,
        wp_lookahead_curv: float,
        current_curv: float = 0.0,
        lane_valid: bool = True,
    ) -> PerceptionMode:
        """
        Update mode based on current perception state.
        
        Args:
            lane_conf: Lane detection confidence [0, 1]
            wp_lookahead_curv: Max curvature from waypoint lookahead
            current_curv: Current road curvature
            lane_valid: Whether lane detection is geometrically valid
            
        Returns:
            Current perception mode
        """
        cfg = self.config
        effective_curv = max(abs(wp_lookahead_curv), abs(current_curv))
        
        # Determine target mode based on conditions
        if not lane_valid or lane_conf < cfg.conf_low:
            # Low confidence or invalid geometry → trust waypoints
            target = PerceptionMode.WAYPOINT_ONLY
        elif effective_curv > cfg.curv_turning:
            # Sharp turn coming → trust waypoints (like Google Maps navigation)
            target = PerceptionMode.WAYPOINT_ONLY
        elif lane_conf > cfg.conf_high and effective_curv < cfg.curv_straight:
            # High confidence + straight road → trust lane detection
            target = PerceptionMode.LANE_ONLY
        else:
            # Default: blend both
            target = PerceptionMode.FUSION
            
        # Hysteresis: require multiple frames to confirm mode switch
        if target == self._target_mode:
            self._mode_frames += 1
        else:
            self._target_mode = target
            self._mode_frames = 1
            
        if self._mode_frames >= cfg.mode_switch_frames:
            self._current_mode = self._target_mode
            
        # Update lane weight with EMA smoothing
        if self._current_mode == PerceptionMode.LANE_ONLY:
            target_weight = 1.0
        elif self._current_mode == PerceptionMode.WAYPOINT_ONLY:
            target_weight = 0.0
        else:
            # FUSION: weight based on confidence and curvature
            # Max lane weight is 0.4 (40% lane, 60% waypoint always)
            conf_factor = np.clip((lane_conf - cfg.conf_low) / (cfg.conf_high - cfg.conf_low), 0, 1)
            curv_factor = 1.0 - np.clip(effective_curv / cfg.curv_turning, 0, 1) ** 0.5  # sqrt for faster decay
            # Base weight: 40% max for lane, always keep 60% waypoint
            raw_weight = 0.3 * conf_factor + 0.7 * curv_factor
            target_weight = 0.4 * raw_weight  # Cap at 40% lane weight
            
        self._lane_weight = (
            cfg.blend_alpha * target_weight + 
            (1 - cfg.blend_alpha) * self._lane_weight
        )
        
        return self._current_mode
    
    @property
    def mode(self) -> PerceptionMode:
        """Current perception mode."""
        return self._current_mode
    
    @property
    def lane_weight(self) -> float:
        """Weight for lane detection [0, 1]. 0 = full waypoint, 1 = full lane."""
        return self._lane_weight
    
    @property
    def waypoint_weight(self) -> float:
        """Weight for waypoint [0, 1]. Complement of lane_weight."""
        return 1.0 - self._lane_weight
    
    def get_mode_string(self) -> str:
        """Get human-readable mode string for display."""
        mode_names = {
            PerceptionMode.LANE_ONLY: "LANE",
            PerceptionMode.WAYPOINT_ONLY: "WP",
            PerceptionMode.FUSION: "FUSION",
        }
        return f"{mode_names[self._current_mode]} (L:{self._lane_weight:.0%})"
