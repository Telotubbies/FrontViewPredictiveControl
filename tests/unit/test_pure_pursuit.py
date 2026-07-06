"""Production tests for Pure Pursuit controller and Hybrid controller."""
import math
import pytest
from unittest.mock import Mock, MagicMock

from control.pure_pursuit import PurePursuitController, HybridController


# ── PurePursuitController ────────────────────────────────────────────────────


class TestPurePursuitInit:
    """Test constructor and default parameters."""

    def test_default_params(self):
        c = PurePursuitController()
        assert c.L == pytest.approx(2.875)
        assert c.min_ld == pytest.approx(4.0)
        assert c.max_ld == pytest.approx(20.0)
        assert c.ld_gain == pytest.approx(0.6)
        assert c.max_steer == pytest.approx(0.7)

    def test_custom_params(self):
        c = PurePursuitController(
            wheelbase=3.0, min_lookahead=5.0, max_lookahead=25.0,
            lookahead_gain=0.8, max_steer=0.5,
        )
        assert c.L == pytest.approx(3.0)
        assert c.min_ld == pytest.approx(5.0)
        assert c.max_ld == pytest.approx(25.0)
        assert c.ld_gain == pytest.approx(0.8)
        assert c.max_steer == pytest.approx(0.5)

    def test_prev_steer_initialised(self):
        c = PurePursuitController()
        assert c._prev_steer == 0.0


class TestComputeLookahead:
    """Test speed-dependent lookahead distance."""

    @pytest.mark.parametrize("speed,expected", [
        (0.0, 4.0),       # zero speed → min_lookahead
        (1.0, 4.0),       # low speed → min_lookahead (0.6*1=0.6 < 4)
        (10.0, 6.0),      # 0.6*10=6.0
        (20.0, 12.0),     # 0.6*20=12.0
        (50.0, 20.0),     # high speed → max_lookahead
        (100.0, 20.0),    # very high → clamped to max
    ])
    def test_lookahead_values(self, speed, expected):
        c = PurePursuitController()
        assert c.compute_lookahead(speed) == pytest.approx(expected)

    def test_lookahead_monotonic(self):
        """Lookahead should be non-decreasing with speed."""
        c = PurePursuitController()
        prev = 0.0
        for v in range(0, 60, 5):
            ld = c.compute_lookahead(float(v))
            assert ld >= prev
            prev = ld


class TestComputeSteering:
    """Test steering computation — the core safety-critical function."""

    def test_zero_cte_zero_heading_straight(self):
        """No error on straight road → near-zero steering."""
        c = PurePursuitController()
        steer = c.compute_steering(cte=0.0, heading_err=0.0, speed_ms=10.0)
        assert abs(steer) < 0.01

    def test_positive_cte_steers_right(self):
        """Positive CTE → positive steer (turn toward target point)."""
        c = PurePursuitController()
        steer = c.compute_steering(cte=1.0, heading_err=0.0, speed_ms=10.0)
        assert steer > 0.0

    def test_negative_cte_steers_left(self):
        """Negative CTE → negative steer (turn toward target point)."""
        c = PurePursuitController()
        steer = c.compute_steering(cte=-1.0, heading_err=0.0, speed_ms=10.0)
        assert steer < 0.0

    def test_steering_clamped_to_max(self):
        """Very large CTE → steering clamped to max_steer."""
        c = PurePursuitController()
        steer = c.compute_steering(cte=10.0, heading_err=0.5, speed_ms=2.0)
        assert abs(steer) <= c.max_steer + 0.01  # +0.01 for smoothing

    def test_steering_clamped_to_neg_max(self):
        """Very large negative CTE → steering clamped to -max_steer."""
        c = PurePursuitController()
        steer = c.compute_steering(cte=-10.0, heading_err=-0.5, speed_ms=2.0)
        assert steer >= -c.max_steer - 0.01

    def test_heading_error_contributes(self):
        """Heading error should affect steering direction."""
        c = PurePursuitController()
        s_no_head = c.compute_steering(cte=0.0, heading_err=0.0, speed_ms=10.0)
        c.reset()
        s_with_head = c.compute_steering(cte=0.0, heading_err=0.1, speed_ms=10.0)
        assert s_with_head > s_no_head  # positive heading → positive steer

    def test_curvature_feedforward(self):
        """Curvature should shift target point and affect steering."""
        c = PurePursuitController()
        s_no_curv = c.compute_steering(cte=0.0, heading_err=0.0, speed_ms=10.0, curvature=0.0)
        c.reset()
        s_with_curv = c.compute_steering(cte=0.0, heading_err=0.0, speed_ms=10.0, curvature=0.05)
        assert s_with_curv != pytest.approx(s_no_curv, abs=0.001)

    def test_low_speed_no_crash(self):
        """Very low speed should not cause division by zero."""
        c = PurePursuitController()
        steer = c.compute_steering(cte=1.0, heading_err=0.1, speed_ms=0.01)
        assert math.isfinite(steer)

    def test_zero_speed_no_crash(self):
        """Zero speed should not crash."""
        c = PurePursuitController()
        steer = c.compute_steering(cte=1.0, heading_err=0.1, speed_ms=0.0)
        assert math.isfinite(steer)

    def test_smoothing_reduces_jitter(self):
        """Successive calls should smooth steering transitions."""
        c = PurePursuitController()
        s1 = c.compute_steering(cte=1.0, heading_err=0.0, speed_ms=10.0)
        s2 = c.compute_steering(cte=-1.0, heading_err=0.0, speed_ms=10.0)
        # s2 should be less extreme than raw due to smoothing
        raw_s2 = math.atan(2.0 * c.L * (-1.0) / (c.compute_lookahead(10.0) ** 2))
        assert abs(s2) < abs(raw_s2) + 0.01


class TestSteerToCarla:
    """Test CARLA steering conversion."""

    def test_zero(self):
        c = PurePursuitController()
        assert c.steer_to_carla(0.0) == pytest.approx(0.0)

    def test_max_steer_maps_to_one(self):
        c = PurePursuitController()
        assert c.steer_to_carla(c.max_steer) == pytest.approx(1.0)

    def test_neg_max_steer_maps_to_neg_one(self):
        c = PurePursuitController()
        assert c.steer_to_carla(-c.max_steer) == pytest.approx(-1.0)

    def test_clamped_to_range(self):
        c = PurePursuitController()
        assert c.steer_to_carla(10.0) == 1.0
        assert c.steer_to_carla(-10.0) == -1.0

    def test_proportional(self):
        c = PurePursuitController()
        half = c.steer_to_carla(c.max_steer / 2)
        assert half == pytest.approx(0.5, abs=0.01)


class TestReset:
    """Test controller reset."""

    def test_reset_clears_prev_steer(self):
        c = PurePursuitController()
        c.compute_steering(cte=1.0, heading_err=0.0, speed_ms=10.0)
        assert c._prev_steer != 0.0
        c.reset()
        assert c._prev_steer == 0.0

    def test_reset_restores_zero_steering(self):
        """After reset, first call should produce raw steering (no smoothing)."""
        c = PurePursuitController()
        c.compute_steering(cte=1.0, heading_err=0.0, speed_ms=10.0)
        c.reset()
        steer = c.compute_steering(cte=0.0, heading_err=0.0, speed_ms=10.0)
        # Without smoothing from prev, should be closer to raw
        assert abs(steer) < 0.01


# ── HybridController ─────────────────────────────────────────────────────────


class TestHybridController:
    """Test hybrid PP/MPC controller."""

    def _make_mock_mpc(self, steer_rad=0.1):
        mpc = Mock()
        mpc.solve = Mock(return_value=(steer_rad, 0.5, 0.0))
        mpc.steer_to_carla = Mock(return_value=0.15)
        return mpc

    def test_init(self):
        mpc = self._make_mock_mpc()
        hc = HybridController(mpc)
        assert hc.mpc is mpc
        assert hc.pp is not None
        assert hc._mode == "PP"

    def test_low_curvature_uses_pp(self):
        """Straight road → Pure Pursuit mode."""
        mpc = self._make_mock_mpc()
        hc = HybridController(mpc)
        steer, mode = hc.compute_steering(
            cte=0.5, heading_err=0.0, speed_ms=10.0, curvature=0.001,
        )
        assert mode == "PP"
        mpc.solve.assert_not_called()

    def test_high_curvature_uses_mpc(self):
        """Sharp curve → MPC mode."""
        mpc = self._make_mock_mpc()
        hc = HybridController(mpc)
        steer, mode = hc.compute_steering(
            cte=0.5, heading_err=0.0, speed_ms=10.0, curvature=0.05,
        )
        assert mode == "MPC"
        mpc.solve.assert_called_once()

    def test_transition_zone_blends(self):
        """Mid curvature → HYBRID mode with blending."""
        mpc = self._make_mock_mpc()
        hc = HybridController(mpc)
        steer, mode = hc.compute_steering(
            cte=0.5, heading_err=0.0, speed_ms=10.0, curvature=0.02,
        )
        assert mode == "HYBRID"
        mpc.solve.assert_called_once()

    def test_mpc_failure_falls_back_to_pp(self):
        """If MPC raises exception, fall back to Pure Pursuit."""
        mpc = self._make_mock_mpc()
        mpc.solve.side_effect = RuntimeError("IPOPT failed")
        hc = HybridController(mpc)
        steer, mode = hc.compute_steering(
            cte=0.5, heading_err=0.0, speed_ms=10.0, curvature=0.05,
        )
        assert mode == "PP"
        assert math.isfinite(steer)

    def test_reset(self):
        mpc = self._make_mock_mpc()
        hc = HybridController(mpc)
        hc.compute_steering(cte=0.5, heading_err=0.0, speed_ms=10.0, curvature=0.02)
        hc.reset()
        assert hc._prev_steer == 0.0
        assert hc._mode == "PP"
