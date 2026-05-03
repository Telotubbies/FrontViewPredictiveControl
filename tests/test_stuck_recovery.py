"""Unit tests for StuckRecovery should_recover method."""
import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety.stuck_recovery import StuckRecovery


class TestStuckRecovery:
    """Test cases for StuckRecovery class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.recovery = StuckRecovery()
        self.mock_transform = Mock()
    
    def test_should_recover_method_exists(self):
        """Test that should_recover method exists and is callable."""
        assert hasattr(self.recovery, 'should_recover')
        assert callable(self.recovery.should_recover)
    
    def test_should_recover_idle_normal_conditions(self):
        """Test should_recover returns False in idle with normal speed."""
        # Normal driving conditions
        speed_ms = 10.0  # Normal speed
        result = self.recovery.should_recover(speed_ms, self.mock_transform, {})
        assert result is False
    
    def test_should_recover_idle_stuck_conditions(self):
        """Test should_recover returns True when stuck in idle phase."""
        # Stuck conditions: low speed, high throttle count
        speed_ms = 0.1  # Very low speed
        
        # Simulate stuck condition by setting internal counter
        self.recovery._n = self.recovery.STUCK_THRESHOLD + 10
        
        result = self.recovery.should_recover(speed_ms, self.mock_transform, {})
        assert result is True
    
    def test_should_recover_during_recovery(self):
        """Test should_recover returns True during any recovery phase."""
        # Set to recovery phase
        self.recovery._phase = "brake"
        
        # Even with normal speed, should return True during recovery
        speed_ms = 10.0
        result = self.recovery.should_recover(speed_ms, self.mock_transform, {})
        assert result is True
        
        # Test other recovery phases
        self.recovery._phase = "reverse"
        result = self.recovery.should_recover(speed_ms, self.mock_transform, {})
        assert result is True
        
        self.recovery._phase = "forward"
        result = self.recovery.should_recover(speed_ms, self.mock_transform, {})
        assert result is True
    
    def test_should_recover_threshold_not_met(self):
        """Test should_recover returns False when stuck threshold not met."""
        speed_ms = 0.1  # Low speed
        self.recovery._n = self.recovery.STUCK_THRESHOLD - 10  # Below threshold
        
        result = self.recovery.should_recover(speed_ms, self.mock_transform, {})
        assert result is False
    
    def test_should_recover_with_cooldown(self):
        """Test should_recover behavior during cooldown."""
        # Set cooldown period
        self.recovery._cooldown = 50
        
        # In idle phase with cooldown, should return False
        speed_ms = 10.0
        result = self.recovery.should_recover(speed_ms, self.mock_transform, {})
        assert result is False
        
        # But if in recovery phase, should still return True
        self.recovery._phase = "brake"
        result = self.recovery.should_recover(speed_ms, self.mock_transform, {})
        assert result is True
    
    def test_should_recover_edge_cases(self):
        """Test should_recover with edge case inputs."""
        # Zero speed
        result = self.recovery.should_recover(0.0, self.mock_transform, {})
        assert isinstance(result, bool)
        
        # Negative speed (shouldn't happen but test robustness)
        result = self.recovery.should_recover(-1.0, self.mock_transform, {})
        assert isinstance(result, bool)
        
        # Very high speed
        result = self.recovery.should_recover(50.0, self.mock_transform, {})
        assert isinstance(result, bool)
    
    def test_should_recover_ignores_control_state(self):
        """Test that should_recover ignores control_state parameter."""
        speed_ms = 0.1
        self.recovery._n = self.recovery.STUCK_THRESHOLD + 10
        
        # Different control states should not affect result
        for control_state in [{}, {'throttle': 0.5}, {'brake': 1.0}, None]:
            result = self.recovery.should_recover(speed_ms, self.mock_transform, control_state)
            assert result is True
    
    def test_should_recover_ignores_transform(self):
        """Test that should_recover ignores transform parameter."""
        speed_ms = 0.1
        self.recovery._n = self.recovery.STUCK_THRESHOLD + 10
        
        # Different transforms should not affect result
        for transform in [self.mock_transform, Mock(), None]:
            result = self.recovery.should_recover(speed_ms, transform, {})
            assert result is True
