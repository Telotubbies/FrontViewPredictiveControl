"""Production tests for StuckRecovery (safety/stuck_recovery.py)."""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from safety.stuck_recovery import StuckRecovery


# ── Constructor ──────────────────────────────────────────────────────────────


class TestStuckRecoveryInit:
    """Test StuckRecovery constructor."""

    def test_initial_state(self):
        r = StuckRecovery()
        assert r._phase == "idle"
        assert r._n == 0
        assert r._confirm == 0
        assert r._counter == 0
        assert r.just_recovered is False
        assert r._cooldown == 0

    def test_thresholds_are_class_constants(self):
        """All thresholds should be class-level constants."""
        assert isinstance(StuckRecovery.STUCK_THRESHOLD, int)
        assert isinstance(StuckRecovery.STUCK_CONFIRM_FRAMES, int)
        assert isinstance(StuckRecovery.STUCK_SPEED_MS, float)
        assert isinstance(StuckRecovery.STUCK_THROTTLE_MIN, float)
        assert isinstance(StuckRecovery.COOLDOWN_AFTER_RECOVERY, int)
        assert isinstance(StuckRecovery.BRAKE_FRAMES, int)
        assert isinstance(StuckRecovery.REVERSE_FRAMES, int)

    def test_thresholds_positive(self):
        assert StuckRecovery.STUCK_THRESHOLD > 0
        assert StuckRecovery.STUCK_CONFIRM_FRAMES > 0
        assert StuckRecovery.STUCK_SPEED_MS > 0
        assert StuckRecovery.STUCK_THROTTLE_MIN > 0
        assert StuckRecovery.COOLDOWN_AFTER_RECOVERY > 0
        assert StuckRecovery.BRAKE_FRAMES > 0
        assert StuckRecovery.REVERSE_FRAMES > 0


# ── update() — idle phase ────────────────────────────────────────────────────


class TestUpdateIdle:
    """Test update() in idle phase."""

    def test_normal_driving_returns_none(self):
        """Normal speed + normal throttle → None (no recovery)."""
        r = StuckRecovery()
        result = r.update(speed=10.0, throttle=0.3)
        assert result is None

    def test_high_speed_returns_none(self):
        """High speed → not stuck → None."""
        r = StuckRecovery()
        result = r.update(speed=20.0, throttle=0.8)
        assert result is None

    def test_low_throttle_returns_none(self):
        """Low speed but low throttle → not stuck → None."""
        r = StuckRecovery()
        result = r.update(speed=0.01, throttle=0.1)
        assert result is None

    def test_stuck_condition_increments_n(self):
        """Low speed + high throttle → increment stuck counter."""
        r = StuckRecovery()
        for _ in range(5):
            r.update(speed=0.01, throttle=0.5)
        assert r._n > 0

    def test_non_stuck_decrements_n(self):
        """Normal driving should decrement stuck counter."""
        r = StuckRecovery()
        r._n = 10
        r.update(speed=10.0, throttle=0.3)
        assert r._n < 10

    def test_just_recovered_cleared_each_call(self):
        """just_recovered should be False at start of each update."""
        r = StuckRecovery()
        r.just_recovered = True
        r.update(speed=10.0, throttle=0.3)
        assert r.just_recovered is False


# ── update() — stuck detection ───────────────────────────────────────────────


class TestStuckDetection:
    """Test stuck condition detection and confirmation."""

    def test_triggers_after_threshold_frames(self):
        """After enough stuck frames, recovery should trigger."""
        r = StuckRecovery()
        # Simulate stuck condition for enough frames
        for _ in range(r.STUCK_THRESHOLD + r.STUCK_CONFIRM_FRAMES + 1):
            result = r.update(speed=0.01, throttle=0.5)
            if result is not None:
                # Recovery triggered
                assert r._phase != "idle"
                return
        # If we get here, recovery didn't trigger
        pytest.fail("Recovery should have triggered")

    def test_cooldown_prevents_immediate_retrigger(self):
        """After recovery, cooldown should prevent immediate re-trigger."""
        r = StuckRecovery()
        r._cooldown = 50
        # Even with stuck conditions, should not increment n during cooldown
        r.update(speed=0.01, throttle=0.5)
        # n should not increment during cooldown
        assert r._n == 0

    def test_confirm_resets_on_non_stuck(self):
        """Confirm counter should reset when not stuck."""
        r = StuckRecovery()
        r._confirm = 3
        r.update(speed=10.0, throttle=0.3)
        assert r._confirm == 0


# ── update() — brake phase ───────────────────────────────────────────────────


class TestBrakePhase:
    """Test brake phase of recovery."""

    def test_brake_phase_returns_full_brake(self):
        """Brake phase should return (0.0, 0.0, 1.0, False)."""
        r = StuckRecovery()
        r._phase = "brake"
        r._counter = 0
        result = r.update(speed=0.01, throttle=0.5)
        assert result is not None
        steer, throttle, brake, reverse = result
        assert steer == pytest.approx(0.0)
        assert throttle == pytest.approx(0.0)
        assert brake == pytest.approx(1.0)
        assert reverse is False

    def test_brake_transitions_to_reverse(self):
        """After BRAKE_FRAMES, should transition to reverse."""
        r = StuckRecovery()
        r._phase = "brake"
        r._counter = r.BRAKE_FRAMES - 1
        r.update(speed=0.01, throttle=0.5)
        assert r._phase == "reverse"


# ── update() — reverse phase ─────────────────────────────────────────────────


class TestReversePhase:
    """Test reverse phase of recovery."""

    def test_reverse_returns_reverse_control(self):
        """Reverse phase should return reverse=True with negative throttle."""
        r = StuckRecovery()
        r._phase = "reverse"
        r._counter = 0
        result = r.update(speed=0.01, throttle=0.5)
        assert result is not None
        steer, throttle, brake, reverse = result
        assert reverse is True
        assert throttle < 0  # negative throttle = reverse in CARLA
        assert brake == pytest.approx(0.0)

    def test_reverse_transitions_to_forward(self):
        """After REVERSE_FRAMES, should transition to forward."""
        r = StuckRecovery()
        r._phase = "reverse"
        r._counter = r.REVERSE_FRAMES - 1
        r.update(speed=0.01, throttle=0.5)
        assert r._phase == "forward"


# ── update() — forward phase ─────────────────────────────────────────────────


class TestForwardPhase:
    """Test forward phase of recovery."""

    def test_forward_returns_forward_control(self):
        """Forward phase should return throttle with reverse=False."""
        r = StuckRecovery()
        r._phase = "forward"
        r._counter = 0
        result = r.update(speed=0.01, throttle=0.5)
        assert result is not None
        steer, throttle, brake, reverse = result
        assert reverse is False
        assert throttle > 0

    def test_forward_transitions_to_idle(self):
        """After forward phase, should return to idle."""
        r = StuckRecovery()
        r._phase = "forward"
        r._counter = 29  # forward phase is 30 frames
        result = r.update(speed=0.01, throttle=0.5)
        assert r._phase == "idle"
        assert r.just_recovered is True
        assert result is None  # Returns None when transitioning to idle


# ── should_recover() ─────────────────────────────────────────────────────────


class TestShouldRecover:
    """Test should_recover method."""

    def setup_method(self):
        self.recovery = StuckRecovery()
        self.mock_transform = Mock()

    def test_method_exists(self):
        assert hasattr(self.recovery, "should_recover")
        assert callable(self.recovery.should_recover)

    def test_normal_conditions_returns_false(self):
        result = self.recovery.should_recover(10.0, self.mock_transform, {})
        assert result is False

    def test_stuck_conditions_returns_true(self):
        self.recovery._n = self.recovery.STUCK_THRESHOLD + 10
        result = self.recovery.should_recover(0.1, self.mock_transform, {})
        assert result is True

    def test_during_recovery_returns_true(self):
        for phase in ["brake", "reverse", "forward"]:
            self.recovery._phase = phase
            result = self.recovery.should_recover(10.0, self.mock_transform, {})
            assert result is True

    def test_below_threshold_returns_false(self):
        self.recovery._n = self.recovery.STUCK_THRESHOLD - 10
        result = self.recovery.should_recover(0.1, self.mock_transform, {})
        assert result is False

    def test_returns_bool(self):
        """should_recover should always return a bool."""
        for speed in [0.0, -1.0, 0.1, 10.0, 50.0]:
            result = self.recovery.should_recover(speed, self.mock_transform, {})
            assert isinstance(result, bool)

    def test_ignores_control_state(self):
        self.recovery._n = self.recovery.STUCK_THRESHOLD + 10
        for cs in [{}, {"throttle": 0.5}, None]:
            result = self.recovery.should_recover(0.1, self.mock_transform, cs)
            assert result is True

    def test_ignores_transform(self):
        self.recovery._n = self.recovery.STUCK_THRESHOLD + 10
        for t in [self.mock_transform, Mock(), None]:
            result = self.recovery.should_recover(0.1, t, {})
            assert result is True


# ── Full recovery cycle ──────────────────────────────────────────────────────


class TestFullRecoveryCycle:
    """Test complete brake → reverse → forward → idle cycle."""

    def test_complete_cycle(self):
        """Run through the complete recovery cycle and verify return to idle."""
        r = StuckRecovery()
        # Force into brake phase
        r._phase = "brake"
        r._counter = 0

        total_frames = 0
        max_frames = r.BRAKE_FRAMES + r.REVERSE_FRAMES + 30 + 10

        while r._phase != "idle" and total_frames < max_frames:
            result = r.update(speed=0.01, throttle=0.5)
            total_frames += 1
            if result is not None:
                steer, throttle, brake, reverse = result
                # All outputs should be valid
                assert isinstance(steer, float)
                assert isinstance(throttle, float)
                assert isinstance(brake, float)
                assert isinstance(reverse, bool)

        assert r._phase == "idle"
        assert r.just_recovered is True
        assert total_frames < max_frames  # Did not timeout

    def test_cooldown_set_after_recovery(self):
        """After recovery completes, cooldown should be set."""
        r = StuckRecovery()
        r._phase = "forward"
        r._counter = 29
        r.update(speed=0.01, throttle=0.5)
        assert r._cooldown == r.COOLDOWN_AFTER_RECOVERY
