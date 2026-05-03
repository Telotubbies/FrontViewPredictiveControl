"""
Lane Change Detection for ADAS v2

Detects:
    - Intentional lane changes (smooth CTE transition)
    - Unintentional lane departure (sudden CTE spike)
    - Lane change direction (left/right)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import numpy as np


class LaneChangeState(Enum):
    """Lane change state."""
    CENTERED = "centered"           # In lane center
    DRIFTING_LEFT = "drifting_left"
    DRIFTING_RIGHT = "drifting_right"
    CHANGING_LEFT = "changing_left"   # Intentional lane change
    CHANGING_RIGHT = "changing_right"
    DEPARTED = "departed"           # Lane departure warning


@dataclass
class LaneChangeConfig:
    """Configuration for lane change detection."""
    # CTE thresholds (meters)
    cte_centered: float = 0.3       # Within this = centered
    cte_drifting: float = 0.8       # Within this = drifting
    cte_departed: float = 1.5       # Beyond this = departed
    
    # Rate thresholds (m/s)
    cte_rate_slow: float = 0.2      # Slow drift
    cte_rate_fast: float = 0.5      # Fast change (intentional)
    
    # Time thresholds
    min_change_duration_s: float = 1.0  # Min time for lane change
    max_change_duration_s: float = 5.0  # Max time for lane change
    
    # History
    history_size: int = 20          # Frames to track


@dataclass
class LaneChangeEvent:
    """Lane change event."""
    state: LaneChangeState
    direction: str  # "left", "right", "none"
    cte: float
    cte_rate: float
    confidence: float
    warning: bool


class LaneChangeDetector:
    """
    Detects lane changes and departures.
    
    Uses CTE history to distinguish:
    - Intentional lane change (smooth, sustained)
    - Unintentional drift (slow, correctable)
    - Lane departure (sudden, dangerous)
    """
    
    def __init__(self, config: Optional[LaneChangeConfig] = None):
        self.config = config or LaneChangeConfig()
        self._cte_history: List[float] = []
        self._time_history: List[float] = []
        self._state = LaneChangeState.CENTERED
        self._change_start_time: Optional[float] = None
        
    def reset(self):
        """Reset detector state."""
        self._cte_history = []
        self._time_history = []
        self._state = LaneChangeState.CENTERED
        self._change_start_time = None
        
    def update(self, cte: float, timestamp: float) -> LaneChangeEvent:
        """
        Update with new CTE measurement.
        
        Args:
            cte: Cross-track error (positive = right of center)
            timestamp: Current timestamp (seconds)
            
        Returns:
            LaneChangeEvent with current state
        """
        cfg = self.config
        
        # Update history
        self._cte_history.append(cte)
        self._time_history.append(timestamp)
        
        # Trim history
        while len(self._cte_history) > cfg.history_size:
            self._cte_history.pop(0)
            self._time_history.pop(0)
            
        # Compute CTE rate
        cte_rate = self._compute_cte_rate()
        
        # Determine direction
        if cte > cfg.cte_centered:
            direction = "right"
        elif cte < -cfg.cte_centered:
            direction = "left"
        else:
            direction = "none"
            
        # Determine state
        abs_cte = abs(cte)
        abs_rate = abs(cte_rate)
        
        warning = False
        
        if abs_cte < cfg.cte_centered:
            # Centered
            new_state = LaneChangeState.CENTERED
            self._change_start_time = None
            
        elif abs_cte < cfg.cte_drifting:
            # Drifting
            if abs_rate > cfg.cte_rate_fast:
                # Fast change = intentional lane change
                if self._change_start_time is None:
                    self._change_start_time = timestamp
                    
                if cte > 0:
                    new_state = LaneChangeState.CHANGING_RIGHT
                else:
                    new_state = LaneChangeState.CHANGING_LEFT
            else:
                # Slow drift
                if cte > 0:
                    new_state = LaneChangeState.DRIFTING_RIGHT
                else:
                    new_state = LaneChangeState.DRIFTING_LEFT
                    
        elif abs_cte < cfg.cte_departed:
            # Still drifting but getting close to departure
            warning = True
            if cte > 0:
                new_state = LaneChangeState.DRIFTING_RIGHT
            else:
                new_state = LaneChangeState.DRIFTING_LEFT
                
        else:
            # Departed
            new_state = LaneChangeState.DEPARTED
            warning = True
            self._change_start_time = None
            
        # Check if lane change completed
        if self._state in (LaneChangeState.CHANGING_LEFT, LaneChangeState.CHANGING_RIGHT):
            if new_state == LaneChangeState.CENTERED:
                # Lane change completed
                pass
            elif self._change_start_time is not None:
                duration = timestamp - self._change_start_time
                if duration > cfg.max_change_duration_s:
                    # Too long, probably not a lane change
                    self._change_start_time = None
                    
        self._state = new_state
        
        # Compute confidence
        confidence = self._compute_confidence(abs_cte, abs_rate)
        
        return LaneChangeEvent(
            state=new_state,
            direction=direction,
            cte=cte,
            cte_rate=cte_rate,
            confidence=confidence,
            warning=warning,
        )
    
    def _compute_cte_rate(self) -> float:
        """Compute CTE rate of change (m/s)."""
        if len(self._cte_history) < 2:
            return 0.0
            
        # Use last few samples for smoothing
        n = min(5, len(self._cte_history))
        ctes = self._cte_history[-n:]
        times = self._time_history[-n:]
        
        dt = times[-1] - times[0]
        if dt < 0.01:
            return 0.0
            
        dcte = ctes[-1] - ctes[0]
        return dcte / dt
    
    def _compute_confidence(self, abs_cte: float, abs_rate: float) -> float:
        """Compute detection confidence."""
        cfg = self.config
        
        # Higher confidence when clearly in a state
        if abs_cte < cfg.cte_centered * 0.5:
            return 1.0  # Clearly centered
        elif abs_cte > cfg.cte_departed:
            return 1.0  # Clearly departed
        else:
            # In between - lower confidence
            return 0.5 + 0.5 * (1.0 - abs_cte / cfg.cte_departed)
    
    @property
    def state(self) -> LaneChangeState:
        return self._state
    
    @property
    def is_changing_lane(self) -> bool:
        return self._state in (LaneChangeState.CHANGING_LEFT, 
                               LaneChangeState.CHANGING_RIGHT)
    
    @property
    def is_warning(self) -> bool:
        return self._state == LaneChangeState.DEPARTED or \
               (self._state in (LaneChangeState.DRIFTING_LEFT, 
                               LaneChangeState.DRIFTING_RIGHT) and
                len(self._cte_history) > 0 and 
                abs(self._cte_history[-1]) > self.config.cte_drifting)
