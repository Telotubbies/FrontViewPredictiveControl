"""Unit tests for MPC controller (control/lane_mpc.py).

Tests follow TDD principles: DAMP over DRY, Arrange-Act-Assert,
test STATE not interactions, one assertion per concept.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from control.lane_mpc import LaneMPC, MPCConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mpc():
    """A small-horizon MPC instance for fast unit tests."""
    cfg = MPCConfig(N=5, dt=0.1)
    return LaneMPC(cfg)


# ── solve() return contract ───────────────────────────────────────────────────


class TestMPCSolveReturnsTrajectory:
    """solve() must return a 4-tuple with a trajectory of shape (4, N+1)."""

    def test_mpc_solve_returns_trajectory(self, mpc):
        # Arrange
        N = mpc.cfg.N

        # Act
        result = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=0.5, heading_err=0.1,
        )

        # Assert — 4-tuple
        assert isinstance(result, tuple)
        assert len(result) == 4

        steer, accel, status, trajectory = result
        assert isinstance(steer, float)
        assert isinstance(accel, float)
        assert isinstance(status, str)

        # Assert — trajectory shape (4, N+1) when solver succeeds
        if trajectory is not None:
            assert isinstance(trajectory, np.ndarray)
            assert trajectory.shape == (4, N + 1)


# ── Steering limits ───────────────────────────────────────────────────────────


class TestMPCSteerLimits:
    """Steering output must always respect max_steer."""

    def test_mpc_steer_within_limits(self, mpc):
        # Arrange
        max_steer = mpc.cfg.max_steer

        # Act
        steer, accel, status, trajectory = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=2.0, heading_err=0.3,
        )

        # Assert
        assert abs(steer) <= max_steer


# ── Acceleration limits ───────────────────────────────────────────────────────


class TestMPCAccelLimits:
    """Acceleration output must always respect [min_accel, max_accel]."""

    def test_mpc_accel_within_limits(self, mpc):
        # Arrange
        min_accel = mpc.cfg.min_accel
        max_accel = mpc.cfg.max_accel

        # Act
        steer, accel, status, trajectory = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=20.0, cte=0.5, heading_err=0.1,
        )

        # Assert
        assert min_accel <= accel <= max_accel


# ── Fallback behavior ─────────────────────────────────────────────────────────


class TestMPCFallback:
    """When the solver fails, trajectory must be None (fallback path)."""

    def test_mpc_fallback_returns_none_trajectory(self, mpc):
        # Arrange — force solver failure by replacing _solver with a stub
        class _FailingSolver:
            def __call__(self, **kwargs):
                raise RuntimeError("forced solver failure")

        mpc._solver = _FailingSolver()

        # Act
        steer, accel, status, trajectory = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=0.5, heading_err=0.1,
        )

        # Assert — fallback path returns None trajectory
        assert status == "Fallback_PP"
        assert trajectory is None


# ── CARLA conversion ──────────────────────────────────────────────────────────


class TestMPCSteerToCarla:
    """steer_to_carla converts radians to [-1, 1]."""

    def test_mpc_steer_to_carla(self, mpc):
        # Arrange
        max_steer = mpc.cfg.max_steer

        # Act — convert max steering
        carla_full = mpc.steer_to_carla(max_steer)
        carla_zero = mpc.steer_to_carla(0.0)
        carla_half = mpc.steer_to_carla(max_steer * 0.5)

        # Assert — normalized to [-1, 1]
        assert -1.0 <= carla_full <= 1.0
        assert carla_full == pytest.approx(1.0)
        assert carla_zero == pytest.approx(0.0)
        assert carla_half == pytest.approx(0.5)

    def test_mpc_steer_to_carla_clamps(self, mpc):
        # Arrange — value exceeding max_steer
        over = mpc.cfg.max_steer * 2.0

        # Act
        carla_val = mpc.steer_to_carla(over)

        # Assert — clamped to 1.0
        assert carla_val == pytest.approx(1.0)


class TestMPCAccelToCarla:
    """accel_to_carla: positive accel → throttle, negative → brake."""

    def test_mpc_accel_to_carla_positive_throttle(self, mpc):
        # Arrange
        accel = mpc.cfg.max_accel

        # Act
        throttle, brake = mpc.accel_to_carla(accel)

        # Assert — positive accel maps to throttle, no brake
        assert throttle > 0.0
        assert brake == pytest.approx(0.0)

    def test_mpc_accel_to_carla_negative_brake(self, mpc):
        # Arrange
        accel = mpc.cfg.min_accel  # negative

        # Act
        throttle, brake = mpc.accel_to_carla(accel)

        # Assert — negative accel maps to brake, no throttle
        assert throttle == pytest.approx(0.0)
        assert brake > 0.0

    def test_mpc_accel_to_carla_zero(self, mpc):
        # Arrange
        accel = 0.0

        # Act
        throttle, brake = mpc.accel_to_carla(accel)

        # Assert — zero accel → zero throttle, zero brake
        assert throttle == pytest.approx(0.0)
        assert brake == pytest.approx(0.0)


# ── Zero-CTE behavior ─────────────────────────────────────────────────────────


class TestMPCZeroCTE:
    """When CTE=0 and heading=0, steering should be near zero."""

    def test_mpc_zero_cte_minimal_steer(self, mpc):
        # Arrange — perfectly centered, no heading error
        cte = 0.0
        heading_err = 0.0

        # Act
        steer, accel, status, trajectory = mpc.solve(
            x0=0, y0=0, psi0=0, v0=8.0,
            v_ref=8.0, cte=cte, heading_err=heading_err,
        )

        # Assert — no lateral error → minimal steering
        assert abs(steer) < 0.1, f"Expected near-zero steer, got {steer}"
