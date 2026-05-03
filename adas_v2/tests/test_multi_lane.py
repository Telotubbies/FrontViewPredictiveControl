"""Unit tests for MultiLaneDetector and LaneChangePlanner."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adas_v2.multi_lane import (
    MultiLaneDetector, MultiLaneConfig, LaneInfo, LanePosition,
    LaneChangePlanner
)


class TestLanePosition:
    """Tests for LanePosition enum."""
    
    def test_all_positions(self):
        """Test all position values."""
        assert LanePosition.LEFT_2.value == -2
        assert LanePosition.LEFT_1.value == -1
        assert LanePosition.EGO.value == 0
        assert LanePosition.RIGHT_1.value == 1
        assert LanePosition.RIGHT_2.value == 2


class TestLaneInfo:
    """Tests for LaneInfo dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        info = LaneInfo(position=LanePosition.EGO)
        assert info.position == LanePosition.EGO
        assert info.left_boundary is None
        assert info.right_boundary is None
        assert info.width_m == 3.5
        assert info.confidence == 0.0
        assert not info.is_valid
        
    def test_with_boundaries(self):
        """Test with boundary coefficients."""
        left = np.array([0.001, 0.1, 100])
        right = np.array([0.001, 0.1, 200])
        
        info = LaneInfo(
            position=LanePosition.EGO,
            left_boundary=left,
            right_boundary=right,
            confidence=0.9,
            is_valid=True,
        )
        
        assert info.is_valid
        assert info.confidence == 0.9
        np.testing.assert_array_equal(info.left_boundary, left)


class TestMultiLaneConfig:
    """Tests for MultiLaneConfig dataclass."""
    
    def test_default_values(self):
        """Test default config values."""
        config = MultiLaneConfig()
        assert config.lane_width_min_m == 2.5
        assert config.lane_width_max_m == 4.5
        assert config.lane_width_nominal_m == 3.5
        assert config.min_confidence == 0.3
        assert config.max_lanes == 5
        
    def test_custom_values(self):
        """Test custom config values."""
        config = MultiLaneConfig(
            lane_width_nominal_m=3.7,
            max_lanes=3,
        )
        assert config.lane_width_nominal_m == 3.7
        assert config.max_lanes == 3


class TestMultiLaneDetector:
    """Tests for MultiLaneDetector class."""
    
    def test_init(self):
        """Test initialization."""
        detector = MultiLaneDetector()
        assert detector.num_lanes == 0
        assert not detector.has_left_lane
        assert not detector.has_right_lane
        
    def test_reset(self):
        """Test reset functionality."""
        detector = MultiLaneDetector()
        # Detect some lanes
        bev = np.zeros((480, 640), dtype=np.float32)
        detector.detect_lanes(bev, np.array([0, 0, 100]), np.array([0, 0, 200]))
        # Reset
        detector.reset()
        assert detector.num_lanes == 0
        
    def test_detect_ego_lane_only(self):
        """Test detection with only ego lane coefficients."""
        detector = MultiLaneDetector()
        
        bev = np.zeros((480, 640), dtype=np.float32)
        left_coeffs = np.array([0.0, 0.0, 200.0])
        right_coeffs = np.array([0.0, 0.0, 400.0])
        
        lanes = detector.detect_lanes(bev, left_coeffs, right_coeffs)
        
        assert len(lanes) >= 1
        ego = detector.get_ego_lane()
        assert ego is not None
        assert ego.position == LanePosition.EGO
        assert ego.is_valid
        
    def test_detect_no_coefficients(self):
        """Test detection without coefficients."""
        detector = MultiLaneDetector()
        
        bev = np.zeros((480, 640), dtype=np.float32)
        
        lanes = detector.detect_lanes(bev, None, None)
        
        ego = detector.get_ego_lane()
        assert ego is not None
        assert not ego.is_valid
        
    def test_set_target_lane(self):
        """Test setting target lane."""
        detector = MultiLaneDetector()
        
        bev = np.zeros((480, 640), dtype=np.float32)
        detector.detect_lanes(bev, np.array([0, 0, 200]), np.array([0, 0, 400]))
        
        detector.set_target_lane(LanePosition.EGO)
        target = detector.get_target_lane()
        
        assert target is not None
        assert target.position == LanePosition.EGO
        
    def test_lanes_property(self):
        """Test lanes property."""
        detector = MultiLaneDetector()
        
        bev = np.zeros((480, 640), dtype=np.float32)
        detector.detect_lanes(bev, np.array([0, 0, 200]), np.array([0, 0, 400]))
        
        assert isinstance(detector.lanes, list)
        
    def test_px_to_m_conversion(self):
        """Test pixel to meter conversion."""
        detector = MultiLaneDetector()
        
        # Internal method test
        m_per_px = detector._px_to_m(640)
        assert m_per_px > 0
        
    def test_m_to_px_conversion(self):
        """Test meter to pixel conversion."""
        detector = MultiLaneDetector()
        
        px_per_m = detector._m_to_px(640)
        assert px_per_m > 0


class TestLaneChangePlanner:
    """Tests for LaneChangePlanner class."""
    
    def test_init(self):
        """Test initialization."""
        planner = LaneChangePlanner()
        assert not planner.is_changing
        assert planner.progress == 0.0
        assert planner.direction == 0
        
    def test_start_lane_change_right(self):
        """Test starting lane change to the right."""
        planner = LaneChangePlanner()
        
        result = planner.start_lane_change(direction=1, current_time=0.0)
        
        assert result == True
        assert planner.is_changing
        assert planner.direction == 1
        
    def test_start_lane_change_left(self):
        """Test starting lane change to the left."""
        planner = LaneChangePlanner()
        
        result = planner.start_lane_change(direction=-1, current_time=0.0)
        
        assert result == True
        assert planner.is_changing
        assert planner.direction == -1
        
    def test_cannot_start_while_changing(self):
        """Test that cannot start new lane change while one is in progress."""
        planner = LaneChangePlanner()
        
        planner.start_lane_change(direction=1, current_time=0.0)
        result = planner.start_lane_change(direction=-1, current_time=0.5)
        
        assert result == False
        assert planner.direction == 1  # Still original direction
        
    def test_update_progress(self):
        """Test lane change progress update."""
        config = MultiLaneConfig(lane_change_time_s=2.0)
        planner = LaneChangePlanner(config)
        
        planner.start_lane_change(direction=1, current_time=0.0)
        
        # Halfway through
        offset, complete = planner.update(current_time=1.0)
        
        assert not complete
        assert 0 < planner.progress < 1
        assert offset > 0  # Moving right
        
    def test_lane_change_completion(self):
        """Test lane change completion."""
        config = MultiLaneConfig(lane_change_time_s=2.0, lane_width_nominal_m=3.5)
        planner = LaneChangePlanner(config)
        
        planner.start_lane_change(direction=1, current_time=0.0)
        
        # After completion time
        offset, complete = planner.update(current_time=3.0)
        
        assert complete
        assert not planner.is_changing
        assert offset == pytest.approx(3.5, abs=0.1)  # Full lane width
        
    def test_cancel_lane_change(self):
        """Test canceling lane change."""
        planner = LaneChangePlanner()
        
        planner.start_lane_change(direction=1, current_time=0.0)
        planner.update(current_time=0.5)
        planner.cancel()
        
        assert not planner.is_changing
        assert planner.progress == 0.0
        assert planner.direction == 0
        
    def test_smooth_trajectory(self):
        """Test that trajectory is smooth (S-curve)."""
        config = MultiLaneConfig(lane_change_time_s=2.0)
        planner = LaneChangePlanner(config)
        
        planner.start_lane_change(direction=1, current_time=0.0)
        
        # Sample progress at different times
        offsets = []
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
            offset, _ = planner.update(current_time=t)
            offsets.append(offset)
            # Reset for next test
            planner._is_changing = True
            planner._change_start_time = 0.0
            
        # Progress should be monotonically increasing
        for i in range(len(offsets) - 1):
            assert offsets[i] <= offsets[i + 1]
            
    def test_update_when_not_changing(self):
        """Test update when not changing lanes."""
        planner = LaneChangePlanner()
        
        offset, complete = planner.update(current_time=1.0)
        
        assert offset == 0.0
        assert not complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
