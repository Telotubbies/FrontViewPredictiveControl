"""Unit tests for PerceptionFusion."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adas_v2.fusion import PerceptionFusion, LaneState, WaypointState, FusedState


class TestLaneState:
    """Tests for LaneState dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        state = LaneState()
        assert state.cte == 0.0
        assert state.heading == 0.0
        assert state.curvature == 0.0
        assert state.confidence == 0.0
        assert state.valid == False
        
    def test_custom_values(self):
        """Test custom values."""
        state = LaneState(
            cte=0.5,
            heading=0.1,
            curvature=0.02,
            confidence=0.8,
            valid=True,
        )
        assert state.cte == 0.5
        assert state.heading == 0.1
        assert state.curvature == 0.02
        assert state.confidence == 0.8
        assert state.valid == True


class TestWaypointState:
    """Tests for WaypointState dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        state = WaypointState()
        assert state.cte == 0.0
        assert state.heading == 0.0
        assert state.curvature == 0.0
        assert state.lookahead_curv == 0.0
        assert state.path is None
        
    def test_with_path(self):
        """Test with path data."""
        path = [(0, 0), (1, 0.1), (2, 0.2)]
        state = WaypointState(
            cte=0.3,
            heading=0.05,
            path=path,
        )
        assert state.path == path
        assert len(state.path) == 3


class TestPerceptionFusion:
    """Tests for PerceptionFusion class."""
    
    def test_init(self):
        """Test initialization."""
        fusion = PerceptionFusion()
        assert fusion._ema_alpha == 0.3
        
    def test_init_custom_alpha(self):
        """Test initialization with custom alpha."""
        fusion = PerceptionFusion(ema_alpha=0.5)
        assert fusion._ema_alpha == 0.5
        
    def test_reset(self):
        """Test reset functionality."""
        fusion = PerceptionFusion()
        # Do some fusions
        lane = LaneState(cte=1.0, valid=True, confidence=0.8)
        wp = WaypointState(cte=0.5)
        fusion.fuse(lane, wp, lane_weight=0.5)
        # Reset
        fusion.reset()
        assert fusion._prev_fused.cte == 0.0
        
    def test_fuse_lane_only(self):
        """Test fusion with full lane weight."""
        fusion = PerceptionFusion(ema_alpha=1.0)  # No smoothing
        
        lane = LaneState(cte=1.0, heading=0.1, curvature=0.02, confidence=0.9, valid=True)
        wp = WaypointState(cte=0.5, heading=0.05, curvature=0.01)
        
        fused = fusion.fuse(lane, wp, lane_weight=1.0, mode="LANE")
        
        assert fused.cte == pytest.approx(1.0, abs=0.01)
        assert fused.heading == pytest.approx(0.1, abs=0.01)
        assert fused.curvature == pytest.approx(0.02, abs=0.01)
        assert fused.mode == "LANE"
        
    def test_fuse_waypoint_only(self):
        """Test fusion with full waypoint weight."""
        fusion = PerceptionFusion(ema_alpha=1.0)
        
        lane = LaneState(cte=1.0, heading=0.1, curvature=0.02, confidence=0.9, valid=True)
        wp = WaypointState(cte=0.5, heading=0.05, curvature=0.01)
        
        fused = fusion.fuse(lane, wp, lane_weight=0.0, mode="WP")
        
        assert fused.cte == pytest.approx(0.5, abs=0.01)
        assert fused.heading == pytest.approx(0.05, abs=0.01)
        assert fused.curvature == pytest.approx(0.01, abs=0.01)
        
    def test_fuse_50_50(self):
        """Test fusion with equal weights."""
        fusion = PerceptionFusion(ema_alpha=1.0)
        
        lane = LaneState(cte=1.0, heading=0.1, curvature=0.02, confidence=0.9, valid=True)
        wp = WaypointState(cte=0.0, heading=0.0, curvature=0.0)
        
        fused = fusion.fuse(lane, wp, lane_weight=0.5, mode="FUSION")
        
        assert fused.cte == pytest.approx(0.5, abs=0.01)
        assert fused.heading == pytest.approx(0.05, abs=0.01)
        assert fused.curvature == pytest.approx(0.01, abs=0.01)
        
    def test_fuse_invalid_lane(self):
        """Test fusion when lane is invalid."""
        fusion = PerceptionFusion(ema_alpha=1.0)
        
        lane = LaneState(cte=1.0, heading=0.1, curvature=0.02, confidence=0.0, valid=False)
        wp = WaypointState(cte=0.5, heading=0.05, curvature=0.01)
        
        fused = fusion.fuse(lane, wp, lane_weight=0.8, mode="FUSION")
        
        # Should use waypoint only when lane is invalid
        assert fused.cte == pytest.approx(0.5, abs=0.01)
        
    def test_fuse_ema_smoothing(self):
        """Test EMA smoothing across frames."""
        fusion = PerceptionFusion(ema_alpha=0.5)
        
        lane = LaneState(cte=0.0, valid=True, confidence=0.8)
        wp = WaypointState(cte=0.0)
        
        # First frame
        fused1 = fusion.fuse(lane, wp, lane_weight=0.5)
        
        # Second frame with different CTE
        lane2 = LaneState(cte=1.0, valid=True, confidence=0.8)
        wp2 = WaypointState(cte=1.0)
        fused2 = fusion.fuse(lane2, wp2, lane_weight=0.5)
        
        # Should be smoothed
        assert fused2.cte < 1.0  # Not fully jumped to 1.0
        
    def test_generate_reference_path_from_state(self):
        """Test reference path generation."""
        fusion = PerceptionFusion()
        
        fused = FusedState(cte=0.5, heading=0.1, curvature=0.01)
        
        path = fusion.generate_reference_path(fused, lookahead_m=30.0, num_pts=10)
        
        assert len(path) == 11  # num_pts + 1
        assert path[0][0] == 0  # First point at s=0
        assert path[-1][0] == 30.0  # Last point at lookahead
        
    def test_generate_reference_path_uses_existing(self):
        """Test that existing reference path is used if available."""
        fusion = PerceptionFusion()
        
        existing_path = [(0, 0), (10, 0.5), (20, 1.0)]
        fused = FusedState(cte=0.5, heading=0.1, curvature=0.01, reference_path=existing_path)
        
        path = fusion.generate_reference_path(fused)
        
        assert path == existing_path


class TestFusedState:
    """Tests for FusedState dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        state = FusedState()
        assert state.cte == 0.0
        assert state.heading == 0.0
        assert state.curvature == 0.0
        assert state.confidence == 0.0
        assert state.lane_weight == 0.5
        assert state.mode == "FUSION"
        assert state.reference_path is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
