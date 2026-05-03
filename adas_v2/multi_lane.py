"""
Multi-lane Support for ADAS v2

Features:
    - Detect multiple lanes from UNet output
    - Select target lane (ego, left, right)
    - Lane change planning
    - Adjacent lane monitoring
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np


class LanePosition(Enum):
    """Lane position relative to ego."""
    LEFT_2 = -2
    LEFT_1 = -1
    EGO = 0
    RIGHT_1 = 1
    RIGHT_2 = 2


@dataclass
class LaneInfo:
    """Information about a detected lane."""
    position: LanePosition
    left_boundary: Optional[np.ndarray] = None   # Polynomial coeffs
    right_boundary: Optional[np.ndarray] = None
    center_line: Optional[np.ndarray] = None
    width_m: float = 3.5
    confidence: float = 0.0
    is_valid: bool = False


@dataclass
class MultiLaneConfig:
    """Configuration for multi-lane detection."""
    # Lane width
    lane_width_min_m: float = 2.5
    lane_width_max_m: float = 4.5
    lane_width_nominal_m: float = 3.5
    
    # Detection
    min_confidence: float = 0.3
    max_lanes: int = 5  # Max lanes to detect
    
    # Lane change
    lane_change_time_s: float = 3.0
    lane_change_min_gap_m: float = 10.0


class MultiLaneDetector:
    """
    Detects and tracks multiple lanes.
    
    Uses peak detection in BEV histogram to find lane boundaries,
    then groups them into lanes.
    """
    
    def __init__(self, config: Optional[MultiLaneConfig] = None):
        self.config = config or MultiLaneConfig()
        self._lanes: List[LaneInfo] = []
        self._target_lane = LanePosition.EGO
        
    def reset(self):
        """Reset detector state."""
        self._lanes = []
        self._target_lane = LanePosition.EGO
        
    def detect_lanes(
        self,
        bev_prob: np.ndarray,
        ego_left_coeffs: Optional[np.ndarray] = None,
        ego_right_coeffs: Optional[np.ndarray] = None,
    ) -> List[LaneInfo]:
        """
        Detect multiple lanes from BEV probability map.
        
        Args:
            bev_prob: BEV lane probability map (H, W)
            ego_left_coeffs: Ego lane left boundary coefficients
            ego_right_coeffs: Ego lane right boundary coefficients
            
        Returns:
            List of detected lanes
        """
        cfg = self.config
        h, w = bev_prob.shape
        
        # Create ego lane
        ego_lane = LaneInfo(
            position=LanePosition.EGO,
            left_boundary=ego_left_coeffs,
            right_boundary=ego_right_coeffs,
            confidence=1.0 if ego_left_coeffs is not None else 0.0,
            is_valid=ego_left_coeffs is not None and ego_right_coeffs is not None,
        )
        
        if ego_lane.is_valid:
            # Estimate ego lane width
            y_eval = h * 0.8  # Near bottom
            left_x = np.polyval(ego_left_coeffs, y_eval)
            right_x = np.polyval(ego_right_coeffs, y_eval)
            ego_lane.width_m = abs(right_x - left_x) * self._px_to_m(w)
            
            # Compute center line
            ego_lane.center_line = (ego_left_coeffs + ego_right_coeffs) / 2
            
        lanes = [ego_lane]
        
        # Try to detect adjacent lanes
        if ego_lane.is_valid:
            # Left lane
            left_lane = self._detect_adjacent_lane(
                bev_prob, ego_left_coeffs, direction=-1
            )
            if left_lane:
                left_lane.position = LanePosition.LEFT_1
                lanes.append(left_lane)
                
            # Right lane
            right_lane = self._detect_adjacent_lane(
                bev_prob, ego_right_coeffs, direction=1
            )
            if right_lane:
                right_lane.position = LanePosition.RIGHT_1
                lanes.append(right_lane)
                
        self._lanes = lanes
        return lanes
    
    def _detect_adjacent_lane(
        self,
        bev_prob: np.ndarray,
        boundary_coeffs: np.ndarray,
        direction: int,  # -1 = left, 1 = right
    ) -> Optional[LaneInfo]:
        """Detect adjacent lane from boundary."""
        cfg = self.config
        h, w = bev_prob.shape
        
        # Search for next boundary
        search_offset_px = int(cfg.lane_width_nominal_m * self._m_to_px(w))
        
        # Sample points along boundary
        ys = np.linspace(h * 0.3, h * 0.9, 10)
        xs = np.polyval(boundary_coeffs, ys)
        
        # Search for peaks in adjacent region
        found_xs = []
        for y, x in zip(ys, xs):
            y_int = int(y)
            search_start = int(x + direction * search_offset_px * 0.5)
            search_end = int(x + direction * search_offset_px * 1.5)
            
            if direction < 0:
                search_start, search_end = search_end, search_start
                
            search_start = max(0, min(w-1, search_start))
            search_end = max(0, min(w-1, search_end))
            
            if search_start >= search_end:
                continue
                
            row = bev_prob[y_int, search_start:search_end]
            if len(row) > 0 and np.max(row) > 0.3:
                peak_idx = np.argmax(row)
                found_xs.append(search_start + peak_idx)
                
        if len(found_xs) < 5:
            return None
            
        # Fit polynomial to found points
        try:
            found_ys = ys[:len(found_xs)]
            new_boundary = np.polyfit(found_ys, found_xs, 2)
            
            # Create lane info
            if direction < 0:
                # Left lane: new boundary is right, existing is left
                return LaneInfo(
                    position=LanePosition.LEFT_1,
                    left_boundary=None,  # Would need another search
                    right_boundary=new_boundary,
                    confidence=0.5,
                    is_valid=True,
                )
            else:
                # Right lane: existing is left, new is right
                return LaneInfo(
                    position=LanePosition.RIGHT_1,
                    left_boundary=new_boundary,
                    right_boundary=None,
                    confidence=0.5,
                    is_valid=True,
                )
        except:
            return None
    
    def _px_to_m(self, width: int) -> float:
        """Convert pixels to meters (approximate)."""
        # Assume BEV covers ~10m width
        return 10.0 / width
    
    def _m_to_px(self, width: int) -> float:
        """Convert meters to pixels."""
        return width / 10.0
    
    def set_target_lane(self, position: LanePosition):
        """Set target lane for navigation."""
        self._target_lane = position
        
    def get_target_lane(self) -> Optional[LaneInfo]:
        """Get target lane info."""
        for lane in self._lanes:
            if lane.position == self._target_lane:
                return lane
        return None
    
    def get_ego_lane(self) -> Optional[LaneInfo]:
        """Get ego lane info."""
        for lane in self._lanes:
            if lane.position == LanePosition.EGO:
                return lane
        return None
    
    @property
    def lanes(self) -> List[LaneInfo]:
        return self._lanes
    
    @property
    def num_lanes(self) -> int:
        return len(self._lanes)
    
    @property
    def has_left_lane(self) -> bool:
        return any(l.position == LanePosition.LEFT_1 for l in self._lanes)
    
    @property
    def has_right_lane(self) -> bool:
        return any(l.position == LanePosition.RIGHT_1 for l in self._lanes)


class LaneChangePlanner:
    """
    Plans lane change maneuvers.
    
    Generates smooth trajectory for lane changes.
    """
    
    def __init__(self, config: Optional[MultiLaneConfig] = None):
        self.config = config or MultiLaneConfig()
        self._is_changing = False
        self._change_start_time: Optional[float] = None
        self._change_direction: int = 0  # -1 = left, 1 = right
        self._change_progress: float = 0.0
        
    def start_lane_change(self, direction: int, current_time: float) -> bool:
        """
        Start lane change maneuver.
        
        Args:
            direction: -1 for left, 1 for right
            current_time: Current timestamp
            
        Returns:
            True if lane change started
        """
        if self._is_changing:
            return False
            
        self._is_changing = True
        self._change_start_time = current_time
        self._change_direction = direction
        self._change_progress = 0.0
        return True
    
    def update(self, current_time: float) -> Tuple[float, bool]:
        """
        Update lane change progress.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            (lateral_offset_m, is_complete)
        """
        if not self._is_changing or self._change_start_time is None:
            return 0.0, False
            
        elapsed = current_time - self._change_start_time
        duration = self.config.lane_change_time_s
        
        if elapsed >= duration:
            self._is_changing = False
            self._change_progress = 1.0
            return self._change_direction * self.config.lane_width_nominal_m, True
            
        # Smooth S-curve trajectory
        t = elapsed / duration
        # Quintic polynomial for smooth acceleration
        self._change_progress = 10 * t**3 - 15 * t**4 + 6 * t**5
        
        lateral_offset = self._change_direction * self.config.lane_width_nominal_m * self._change_progress
        return lateral_offset, False
    
    def cancel(self):
        """Cancel ongoing lane change."""
        self._is_changing = False
        self._change_start_time = None
        self._change_direction = 0
        self._change_progress = 0.0
        
    @property
    def is_changing(self) -> bool:
        return self._is_changing
    
    @property
    def progress(self) -> float:
        return self._change_progress
    
    @property
    def direction(self) -> int:
        return self._change_direction
