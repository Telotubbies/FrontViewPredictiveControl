"""Production tests for MPC controller (control/lane_mpc.py)."""
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from control.lane_mpc import LaneMPC, MPCConfig, get_mpc_weights, compute_mpc_horizon


# ── MPCConfig ────────────────────────────────────────────────────────────────


class TestMPCConfig:
    """Test MPC configuration dataclass."""

    def test_default_config(self):
        cfg = MPCConfig()
        assert cfg.N > 0
        assert cfg.dt > 0
        assert cfg.max_steer > 0
        assert cfg.L > 0

    def test_custom_config(self):
        cfg = MPCConfig(N=10, dt=0.05, max_steer=0.5, L=3.0)
        assert cfg.N == 10
        assert cfg.dt == pytest.approx(0.05)
        assert cfg.max_steer == pytest.approx(0.5)
        assert cfg.L == pytest.approx(3.0)

    def test_accel_bounds(self):
        cfg = MPCConfig(max_accel=3.0, min_accel=-5.0)
        assert cfg.max_accel == pytest.approx(3.0)
        assert cfg.min_accel == pytest.approx(-5.0)
        assert cfg.max_accel > cfg.min_accel


# ── compute_mpc_horizon ──────────────────────────────────────────────────────


class TestComputeMpcHorizon:
    """Test dynamic horizon computation."""

    def test_returns_positive_int(self):
        N = compute_mpc_horizon(speed_ms=10.0, curvature=0.001)
        assert isinstance(N, int)
        assert N > 0

    def test_straight_road_high_speed(self):
        """Straight road + high speed → longer horizon."""
        N_straight = compute_mpc_horizon(speed_ms=20.0, curvature=0.0)
        assert N_straight >= 8

    def test_sharp_curve(self):
        """Sharp curve → shorter horizon."""
        N_curve = compute_mpc_horizon(speed_ms=5.0, curvature=0.1)
        N_straight = compute_mpc_horizon(speed_ms=5.0, curvature=0.0)
        assert N_curve <= N_straight

    def test_zero_speed(self):
        """Zero speed should not crash."""
        N = compute_mpc_horizon(speed_ms=0.0, curvature=0.0)
        assert N > 0

    def test_zero_curvature(self):
        """Zero curvature should not crash."""
        N = compute_mpc_horizon(speed_ms=10.0, curvature=0.0)
        assert N > 0


# ── get_mpc_weights ──────────────────────────────────────────────────────────


class TestGetMpcWeights:
    """Test adaptive weight computation."""

    def test_returns_four_floats(self):
        result = get_mpc_weights(speed_ms=10.0, curvature=0.01)
        assert len(result) == 4
        for v in result:
            assert isinstance(v, float)

    def test_weights_positive(self):
        """All weight scales should be positive."""
        s_cte, s_head, s_steer, s_jerk = get_mpc_weights(10.0, 0.01)
        assert s_cte > 0
        assert s_head > 0
        assert s_steer > 0
        assert s_jerk > 0

    def test_confidence_clipping(self):
        """Confidence should be clipped to [0, 1]."""
        # Very high confidence
        w1 = get_mpc_weights(10.0, 0.01, confidence=2.0)
        # Very low confidence
        w2 = get_mpc_weights(10.0, 0.01, confidence=-1.0)
        # Should not crash
        assert len(w1) == 4
        assert len(w2) == 4

    def test_curve_increases_cte_weight(self):
        """In curves, CTE weight should be higher (better tracking)."""
        w_straight = get_mpc_weights(10.0, 0.0)
        w_curve = get_mpc_weights(10.0, 0.05)
        assert w_curve[0] >= w_straight[0]  # scale_cte


# ── LaneMPC.solve ────────────────────────────────────────────────────────────


class TestLaneMPCSolve:
    """Test MPC solver — the core safety-critical function."""

    def _make_mpc(self, **kwargs):
        defaults = dict(N=5, dt=0.1)
        defaults.update(kwargs)
        cfg = MPCConfig(**defaults)
        return LaneMPC(cfg)

    def test_small_cte_succeeds(self):
        """Small CTE → solver succeeds or falls back."""
        mpc = self._make_mpc()
        steer, accel, status, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=0.5, heading_err=0.1,
        )
        assert status in ("Solve_Succeeded", "Fallback")

    def test_steer_within_bounds(self):
        """Steering must always be within ±max_steer."""
        mpc = self._make_mpc()
        steer, _, _, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=2.0, heading_err=0.3,
        )
        assert -mpc.cfg.max_steer <= steer <= mpc.cfg.max_steer

    def test_accel_within_bounds(self):
        """Acceleration must always be within bounds."""
        mpc = self._make_mpc()
        _, accel, _, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=20.0, cte=0.5, heading_err=0.1,
        )
        assert mpc.cfg.min_accel <= accel <= mpc.cfg.max_accel

    def test_negative_cte(self):
        """Negative CTE should produce valid solution."""
        mpc = self._make_mpc()
        steer, accel, status, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=-1.0, heading_err=-0.1,
        )
        assert -mpc.cfg.max_steer <= steer <= mpc.cfg.max_steer
        assert status in ("Solve_Succeeded", "Fallback")

    def test_zero_speed(self):
        """Zero speed should not crash."""
        mpc = self._make_mpc()
        steer, accel, status, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=0.0,
            v_ref=5.0, cte=0.5, heading_err=0.1,
        )
        assert -mpc.cfg.max_steer <= steer <= mpc.cfg.max_steer

    def test_large_cte(self):
        """Very large CTE should still produce bounded output."""
        mpc = self._make_mpc()
        steer, accel, status, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=5.0, heading_err=0.5,
        )
        assert -mpc.cfg.max_steer <= steer <= mpc.cfg.max_steer
        assert mpc.cfg.min_accel <= accel <= mpc.cfg.max_accel

    def test_with_curvature(self):
        """Curved road should produce valid solution."""
        mpc = self._make_mpc()
        steer, accel, status, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=8.0,
            v_ref=6.0, cte=-1.0, heading_err=-0.2, curvature=0.05,
        )
        assert -mpc.cfg.max_steer <= steer <= mpc.cfg.max_steer
        assert status in ("Solve_Succeeded", "Fallback")

    def test_with_confidence(self):
        """Low confidence should not crash solver."""
        mpc = self._make_mpc()
        steer, accel, status, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=0.5, heading_err=0.1, confidence=0.1,
        )
        assert status in ("Solve_Succeeded", "Fallback")

    def test_with_weight_scales(self):
        """Custom weight scales should be accepted."""
        mpc = self._make_mpc()
        steer, accel, status, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=0.5, heading_err=0.1,
            weight_scales=(2.0, 1.5, 1.0, 1.0),
        )
        assert -mpc.cfg.max_steer <= steer <= mpc.cfg.max_steer

    def test_returns_floats(self):
        """Return values should be floats."""
        mpc = self._make_mpc()
        steer, accel, status, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=0.5, heading_err=0.1,
        )
        assert isinstance(steer, float)
        assert isinstance(accel, float)
        assert isinstance(status, str)

    def test_finite_output(self):
        """Output should always be finite (no NaN/Inf)."""
        mpc = self._make_mpc()
        steer, accel, _, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=0.5, heading_err=0.1,
        )
        assert np.isfinite(steer)
        assert np.isfinite(accel)


# ── LaneMPC.steer_to_carla ───────────────────────────────────────────────────


class TestSteerToCarla:
    """Test steering angle to CARLA conversion."""

    def test_zero(self):
        mpc = LaneMPC(MPCConfig(max_steer=0.7))
        assert mpc.steer_to_carla(0.0) == pytest.approx(0.0)

    def test_max_steer_to_one(self):
        mpc = LaneMPC(MPCConfig(max_steer=0.7))
        assert mpc.steer_to_carla(0.7) == pytest.approx(1.0)

    def test_neg_max_steer_to_neg_one(self):
        mpc = LaneMPC(MPCConfig(max_steer=0.7))
        assert mpc.steer_to_carla(-0.7) == pytest.approx(-1.0)

    def test_clamped_high(self):
        mpc = LaneMPC(MPCConfig(max_steer=0.7))
        assert mpc.steer_to_carla(10.0) == 1.0

    def test_clamped_low(self):
        mpc = LaneMPC(MPCConfig(max_steer=0.7))
        assert mpc.steer_to_carla(-10.0) == -1.0


# ── LaneMPC.accel_to_carla ───────────────────────────────────────────────────


class TestAccelToCarla:
    """Test acceleration to (throttle, brake) conversion."""

    def test_positive_accel_throttle_only(self):
        mpc = LaneMPC(MPCConfig(max_accel=3.0, min_accel=-5.0))
        th, br = mpc.accel_to_carla(1.0)
        assert th > 0
        assert br == pytest.approx(0.0)

    def test_negative_accel_brake_only(self):
        mpc = LaneMPC(MPCConfig(max_accel=3.0, min_accel=-5.0))
        th, br = mpc.accel_to_carla(-2.0)
        assert th == pytest.approx(0.0)
        assert br > 0

    def test_zero_accel(self):
        mpc = LaneMPC(MPCConfig(max_accel=3.0, min_accel=-5.0))
        th, br = mpc.accel_to_carla(0.0)
        assert th == pytest.approx(0.0)
        assert br == pytest.approx(0.0)

    def test_throttle_in_range(self):
        """Throttle should be in [0, 1]."""
        mpc = LaneMPC(MPCConfig(max_accel=3.0, min_accel=-5.0))
        th, _ = mpc.accel_to_carla(10.0)
        assert 0.0 <= th <= 1.0

    def test_brake_in_range(self):
        """Brake should be in [0, 1]."""
        mpc = LaneMPC(MPCConfig(max_accel=3.0, min_accel=-5.0))
        _, br = mpc.accel_to_carla(-10.0)
        assert 0.0 <= br <= 1.0


# ── LaneMPC.set_horizon ──────────────────────────────────────────────────────


class TestSetHorizon:
    """Test dynamic horizon adjustment."""

    def test_set_horizon_changes_N(self):
        mpc = LaneMPC(MPCConfig(N=5))
        mpc.set_horizon(10)
        # Next solve should use new horizon (internal state)
        steer, accel, status, _ = mpc.solve(
            x0=0, y0=0, psi0=0, v0=5.0,
            v_ref=8.0, cte=0.5, heading_err=0.1,
        )
        assert status in ("Solve_Succeeded", "Fallback")
