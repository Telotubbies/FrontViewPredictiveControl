"""Unit tests for ModeSelector."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adas_v2.mode_selector import ModeSelector, ModeConfig, PerceptionMode


class TestModeSelector:
    """Tests for ModeSelector class."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        selector = ModeSelector()
        assert selector.mode == PerceptionMode.FUSION
        assert selector.lane_weight == 0.5
        assert selector.waypoint_weight == 0.5
        
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ModeConfig(conf_high=0.9, conf_low=0.3)
        selector = ModeSelector(config)
        assert selector.config.conf_high == 0.9
        assert selector.config.conf_low == 0.3
        
    def test_reset(self):
        """Test reset functionality."""
        selector = ModeSelector()
        # Change state
        selector.update(lane_conf=0.9, wp_lookahead_curv=0.0)
        selector.update(lane_conf=0.9, wp_lookahead_curv=0.0)
        selector.update(lane_conf=0.9, wp_lookahead_curv=0.0)
        selector.update(lane_conf=0.9, wp_lookahead_curv=0.0)
        selector.update(lane_conf=0.9, wp_lookahead_curv=0.0)
        # Reset
        selector.reset()
        assert selector.mode == PerceptionMode.FUSION
        assert selector.lane_weight == 0.5
        
    def test_high_confidence_straight_road(self):
        """Test LANE_ONLY mode with high confidence on straight road."""
        selector = ModeSelector()
        # Need multiple frames to confirm mode switch
        for _ in range(10):
            mode = selector.update(
                lane_conf=0.9,
                wp_lookahead_curv=0.001,
                current_curv=0.001,
                lane_valid=True,
            )
        assert mode == PerceptionMode.LANE_ONLY
        assert selector.lane_weight > 0.9
        
    def test_low_confidence(self):
        """Test WAYPOINT_ONLY mode with low confidence."""
        selector = ModeSelector()
        for _ in range(10):
            mode = selector.update(
                lane_conf=0.2,
                wp_lookahead_curv=0.01,
                current_curv=0.01,
                lane_valid=True,
            )
        assert mode == PerceptionMode.WAYPOINT_ONLY
        assert selector.lane_weight < 0.2
        
    def test_invalid_lane(self):
        """Test WAYPOINT_ONLY mode when lane is invalid."""
        selector = ModeSelector()
        for _ in range(10):
            mode = selector.update(
                lane_conf=0.9,
                wp_lookahead_curv=0.01,
                current_curv=0.01,
                lane_valid=False,
            )
        assert mode == PerceptionMode.WAYPOINT_ONLY
        
    def test_sharp_turn(self):
        """Test WAYPOINT_ONLY mode during sharp turn."""
        selector = ModeSelector()
        for _ in range(10):
            mode = selector.update(
                lane_conf=0.8,
                wp_lookahead_curv=0.05,  # Sharp turn
                current_curv=0.01,
                lane_valid=True,
            )
        assert mode == PerceptionMode.WAYPOINT_ONLY
        
    def test_fusion_mode(self):
        """Test FUSION mode with medium confidence."""
        selector = ModeSelector()
        for _ in range(10):
            mode = selector.update(
                lane_conf=0.6,
                wp_lookahead_curv=0.01,
                current_curv=0.01,
                lane_valid=True,
            )
        assert mode == PerceptionMode.FUSION
        
    def test_hysteresis(self):
        """Test mode switching hysteresis."""
        config = ModeConfig(mode_switch_frames=5)
        selector = ModeSelector(config)
        
        # First few frames shouldn't switch
        for i in range(3):
            mode = selector.update(lane_conf=0.9, wp_lookahead_curv=0.001)
        assert mode != PerceptionMode.LANE_ONLY  # Not enough frames
        
        # After 5+ frames, should switch
        for i in range(5):
            mode = selector.update(lane_conf=0.9, wp_lookahead_curv=0.001)
        assert mode == PerceptionMode.LANE_ONLY
        
    def test_get_mode_string(self):
        """Test mode string formatting."""
        selector = ModeSelector()
        mode_str = selector.get_mode_string()
        assert "FUSION" in mode_str
        assert "L:" in mode_str
        
    def test_lane_weight_bounds(self):
        """Test lane weight stays in [0, 1]."""
        selector = ModeSelector()
        
        # Extreme high confidence
        for _ in range(20):
            selector.update(lane_conf=1.0, wp_lookahead_curv=0.0)
        assert 0 <= selector.lane_weight <= 1
        
        # Extreme low confidence
        for _ in range(20):
            selector.update(lane_conf=0.0, wp_lookahead_curv=0.1)
        assert 0 <= selector.lane_weight <= 1


class TestModeConfig:
    """Tests for ModeConfig dataclass."""
    
    def test_default_values(self):
        """Test default config values."""
        config = ModeConfig()
        assert config.conf_high == 0.75
        assert config.conf_low == 0.40
        assert config.curv_straight == 0.008
        assert config.curv_turning == 0.025
        assert config.mode_switch_frames == 5
        
    def test_custom_values(self):
        """Test custom config values."""
        config = ModeConfig(
            conf_high=0.85,
            conf_low=0.35,
            curv_turning=0.03,
        )
        assert config.conf_high == 0.85
        assert config.conf_low == 0.35
        assert config.curv_turning == 0.03


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
