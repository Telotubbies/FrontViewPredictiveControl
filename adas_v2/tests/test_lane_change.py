"""Unit tests for LaneChangeDetector."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adas_v2.lane_change import (
    LaneChangeDetector, LaneChangeConfig, LaneChangeState, LaneChangeEvent
)


class TestLaneChangeConfig:
    """Tests for LaneChangeConfig dataclass."""
    
    def test_default_values(self):
        """Test default config values."""
        config = LaneChangeConfig()
        assert config.cte_centered == 0.3
        assert config.cte_drifting == 0.8
        assert config.cte_departed == 1.5
        assert config.cte_rate_slow == 0.2
        assert config.cte_rate_fast == 0.5
        
    def test_custom_values(self):
        """Test custom config values."""
        config = LaneChangeConfig(
            cte_centered=0.2,
            cte_departed=2.0,
        )
        assert config.cte_centered == 0.2
        assert config.cte_departed == 2.0


class TestLaneChangeDetector:
    """Tests for LaneChangeDetector class."""
    
    def test_init(self):
        """Test initialization."""
        detector = LaneChangeDetector()
        assert detector.state == LaneChangeState.CENTERED
        assert not detector.is_changing_lane
        assert not detector.is_warning
        
    def test_reset(self):
        """Test reset functionality."""
        detector = LaneChangeDetector()
        # Change state
        detector.update(cte=1.0, timestamp=0.0)
        detector.update(cte=1.2, timestamp=0.1)
        # Reset
        detector.reset()
        assert detector.state == LaneChangeState.CENTERED
        
    def test_centered_state(self):
        """Test centered state detection."""
        detector = LaneChangeDetector()
        
        event = detector.update(cte=0.1, timestamp=0.0)
        
        assert event.state == LaneChangeState.CENTERED
        assert event.direction == "none"
        assert not event.warning
        
    def test_drifting_right(self):
        """Test drifting right detection."""
        config = LaneChangeConfig(cte_rate_fast=2.0)  # High threshold to avoid CHANGING
        detector = LaneChangeDetector(config)
        
        # Gradual drift right (slow rate)
        for i in range(5):
            event = detector.update(cte=0.4 + i * 0.05, timestamp=i * 0.5)
            
        assert event.state == LaneChangeState.DRIFTING_RIGHT
        assert event.direction == "right"
        
    def test_drifting_left(self):
        """Test drifting left detection."""
        config = LaneChangeConfig(cte_rate_fast=2.0)  # High threshold to avoid CHANGING
        detector = LaneChangeDetector(config)
        
        # Gradual drift left (slow rate)
        for i in range(5):
            event = detector.update(cte=-0.4 - i * 0.05, timestamp=i * 0.5)
            
        assert event.state == LaneChangeState.DRIFTING_LEFT
        assert event.direction == "left"
        
    def test_lane_departure_warning(self):
        """Test lane departure warning."""
        detector = LaneChangeDetector()
        
        # Large CTE = departure
        event = detector.update(cte=1.6, timestamp=0.0)
        
        assert event.state == LaneChangeState.DEPARTED
        assert event.warning == True
        
    def test_lane_departure_warning_left(self):
        """Test lane departure warning on left side."""
        detector = LaneChangeDetector()
        
        event = detector.update(cte=-1.6, timestamp=0.0)
        
        assert event.state == LaneChangeState.DEPARTED
        assert event.warning == True
        assert event.direction == "left"
        
    def test_intentional_lane_change_right(self):
        """Test intentional lane change detection (fast CTE change)."""
        config = LaneChangeConfig(cte_rate_fast=0.3)
        detector = LaneChangeDetector(config)
        
        # Fast change to the right
        detector.update(cte=0.0, timestamp=0.0)
        detector.update(cte=0.2, timestamp=0.1)
        detector.update(cte=0.4, timestamp=0.2)
        event = detector.update(cte=0.6, timestamp=0.3)
        
        # Should detect as changing lane (fast rate)
        assert event.cte_rate > 0
        
    def test_cte_rate_calculation(self):
        """Test CTE rate calculation."""
        detector = LaneChangeDetector()
        
        # Linear increase
        detector.update(cte=0.0, timestamp=0.0)
        detector.update(cte=0.1, timestamp=0.1)
        event = detector.update(cte=0.2, timestamp=0.2)
        
        # Rate should be ~1.0 m/s
        assert event.cte_rate == pytest.approx(1.0, abs=0.2)
        
    def test_confidence_centered(self):
        """Test confidence when clearly centered."""
        detector = LaneChangeDetector()
        
        event = detector.update(cte=0.0, timestamp=0.0)
        
        assert event.confidence == 1.0
        
    def test_confidence_departed(self):
        """Test confidence when clearly departed."""
        detector = LaneChangeDetector()
        
        event = detector.update(cte=2.0, timestamp=0.0)
        
        assert event.confidence == 1.0
        
    def test_is_changing_lane_property(self):
        """Test is_changing_lane property."""
        detector = LaneChangeDetector()
        
        # Initially not changing
        assert not detector.is_changing_lane
        
    def test_is_warning_property(self):
        """Test is_warning property."""
        detector = LaneChangeDetector()
        
        # Initially no warning
        assert not detector.is_warning
        
        # After departure
        detector.update(cte=2.0, timestamp=0.0)
        assert detector.is_warning
        
    def test_history_trimming(self):
        """Test that history is trimmed to window size."""
        config = LaneChangeConfig(history_size=10)
        detector = LaneChangeDetector(config)
        
        # Add more than history_size updates
        for i in range(20):
            detector.update(cte=0.1 * i, timestamp=i * 0.1)
            
        assert len(detector._cte_history) <= config.history_size


class TestLaneChangeEvent:
    """Tests for LaneChangeEvent dataclass."""
    
    def test_event_creation(self):
        """Test event creation."""
        event = LaneChangeEvent(
            state=LaneChangeState.CENTERED,
            direction="none",
            cte=0.0,
            cte_rate=0.0,
            confidence=1.0,
            warning=False,
        )
        assert event.state == LaneChangeState.CENTERED
        assert event.direction == "none"
        assert not event.warning


class TestLaneChangeState:
    """Tests for LaneChangeState enum."""
    
    def test_all_states(self):
        """Test all state values."""
        assert LaneChangeState.CENTERED.value == "centered"
        assert LaneChangeState.DRIFTING_LEFT.value == "drifting_left"
        assert LaneChangeState.DRIFTING_RIGHT.value == "drifting_right"
        assert LaneChangeState.CHANGING_LEFT.value == "changing_left"
        assert LaneChangeState.CHANGING_RIGHT.value == "changing_right"
        assert LaneChangeState.DEPARTED.value == "departed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
