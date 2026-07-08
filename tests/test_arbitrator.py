"""Unit tests for ControlArbitrator (control/arbitrator.py).

Tests follow TDD principles: DAMP over DRY, Arrange-Act-Assert,
test STATE not interactions, one assertion per concept.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import MagicMock, Mock

# Try to import ControlArbitrator; skip module if not yet created.
try:
    from control.arbitrator import ControlArbitrator, ControlArbitratorResult, ArbitrationLogEntry
    _ARBITRATOR_AVAILABLE = True
except Exception:
    _ARBITRATOR_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _ARBITRATOR_AVAILABLE,
    reason="control.arbitrator.ControlArbitrator not yet implemented",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_safety():
    """Mock SafetyOverride that passes values through unchanged."""
    s = MagicMock()
    s.apply_safety_override = Mock(side_effect=lambda vs, st, th, br: (st, th, br))
    return s


@pytest.fixture
def mock_adas():
    """Mock ADASManager that passes values through unchanged."""
    a = MagicMock()
    a.apply_to_control = Mock(side_effect=lambda st, th, br, out: (st, th, br, None))
    return a


@pytest.fixture
def mock_stuck():
    """Mock StuckRecovery that returns None (no recovery active)."""
    s = MagicMock()
    s.update = Mock(return_value=None)
    return s


@pytest.fixture
def arbitrator(mock_safety, mock_adas, mock_stuck):
    """A ControlArbitrator with mocked sub-components."""
    return ControlArbitrator(mock_safety, mock_adas, mock_stuck)


@pytest.fixture
def mock_frame_state():
    """Minimal FrameState mock."""
    fs = MagicMock()
    fs.cte_m = 0.0
    fs.heading_rad = 0.0
    return fs


@pytest.fixture
def mock_adas_out():
    """Minimal ADAS output mock (no AEB)."""
    out = MagicMock()
    out.aeb_active = False
    out.active_features = []
    return out


# ── No override ───────────────────────────────────────────────────────────────


class TestArbitratorNoOverride:
    """When no safety/ADAS/stuck override is active, MPC values pass through."""

    def test_arbitrator_no_override(self, arbitrator, mock_frame_state, mock_adas_out):
        # Arrange
        mpc_steer, mpc_throttle, mpc_brake = 0.3, 0.6, 0.0

        # Act
        result = arbitrator.arbitrate(
            mpc_steer=mpc_steer, mpc_throttle=mpc_throttle, mpc_brake=mpc_brake,
            speed_ms=5.0, frame_state=mock_frame_state, adas_out=mock_adas_out,
        )

        # Assert — values unchanged
        assert result.steer == pytest.approx(0.3)
        assert result.throttle == pytest.approx(0.6)
        assert result.brake == pytest.approx(0.0)
        assert result.winning_source == "mpc"
        assert not result.any_override_active


# ── Safety clamps steering ────────────────────────────────────────────────────


class TestArbitratorSafetyClamp:
    """Safety override modifies steering."""

    def test_arbitrator_safety_clamps_steering(self, arbitrator, mock_safety,
                                                mock_frame_state, mock_adas_out):
        # Arrange — safety clamps steering to 0.1
        mock_safety.apply_safety_override = Mock(return_value=(0.1, 0.5, 0.0))

        # Act
        result = arbitrator.arbitrate(
            mpc_steer=0.9, mpc_throttle=0.5, mpc_brake=0.0,
            speed_ms=5.0, frame_state=mock_frame_state, adas_out=mock_adas_out,
        )

        # Assert — steering clamped down
        assert result.steer == pytest.approx(0.1)
        assert abs(result.steer) < abs(0.9)
        assert result.winning_source == "safety"
        assert result.any_override_active


# ── AEB overrides all ─────────────────────────────────────────────────────────


class TestArbitratorAEB:
    """AEB must force brake=1.0 and throttle=0.0."""

    def test_arbitrator_aeb_overrides_all(self, arbitrator, mock_adas,
                                           mock_frame_state, mock_adas_out):
        # Arrange — AEB active
        mock_adas_out.aeb_active = True

        # Act
        result = arbitrator.arbitrate(
            mpc_steer=0.0, mpc_throttle=0.8, mpc_brake=0.0,
            speed_ms=10.0, frame_state=mock_frame_state, adas_out=mock_adas_out,
        )

        # Assert — full brake, no throttle
        assert result.brake == pytest.approx(1.0)
        assert result.throttle == pytest.approx(0.0)
        assert result.winning_source == "aeb"


# ── Log populated ─────────────────────────────────────────────────────────────


class TestArbitratorLog:
    """The arbitration log must contain entries when an override is active."""

    def test_arbitrator_log_populated(self, arbitrator, mock_safety,
                                       mock_frame_state, mock_adas_out):
        # Arrange — safety changes steering
        mock_safety.apply_safety_override = Mock(return_value=(0.1, 0.5, 0.0))

        # Act
        result = arbitrator.arbitrate(
            mpc_steer=0.9, mpc_throttle=0.5, mpc_brake=0.0,
            speed_ms=5.0, frame_state=mock_frame_state, adas_out=mock_adas_out,
        )

        # Assert — log has at least one entry
        assert len(result.log) >= 1
        assert any(entry.source == "safety" for entry in result.log)


# ── Winning source ────────────────────────────────────────────────────────────


class TestArbitratorWinningSource:
    """winning_source must identify the highest-priority active source."""

    def test_arbitrator_winning_source(self, arbitrator, mock_adas,
                                        mock_frame_state, mock_adas_out):
        # Arrange — AEB active (highest priority)
        mock_adas_out.aeb_active = True

        # Act
        result = arbitrator.arbitrate(
            mpc_steer=0.0, mpc_throttle=0.5, mpc_brake=0.0,
            speed_ms=10.0, frame_state=mock_frame_state, adas_out=mock_adas_out,
        )

        # Assert — winning source is AEB (highest priority)
        assert result.winning_source == "aeb"


# ── any_override_active property ──────────────────────────────────────────────


class TestArbitratorAnyOverride:
    """any_override_active property must reflect whether any override occurred."""

    def test_arbitrator_result_any_override(self, arbitrator, mock_safety,
                                             mock_frame_state, mock_adas_out):
        # Arrange — no override
        result_no = arbitrator.arbitrate(
            mpc_steer=0.3, mpc_throttle=0.5, mpc_brake=0.0,
            speed_ms=5.0, frame_state=mock_frame_state, adas_out=mock_adas_out,
        )
        assert not result_no.any_override_active

        # Arrange — safety override active
        mock_safety.apply_safety_override = Mock(return_value=(0.1, 0.5, 0.0))
        result_yes = arbitrator.arbitrate(
            mpc_steer=0.9, mpc_throttle=0.5, mpc_brake=0.0,
            speed_ms=5.0, frame_state=mock_frame_state, adas_out=mock_adas_out,
        )
        assert result_yes.any_override_active
