"""Production tests for SafetyOverride (safety/override.py)."""
import sys
import math
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from safety.override import SafetyOverride, SafetyConfig


# ── SafetyConfig ─────────────────────────────────────────────────────────────


class TestSafetyConfig:
    """Test SafetyConfig dataclass."""

    def test_defaults(self):
        cfg = SafetyConfig()
        assert cfg.emergency_brake_enabled is True
        assert cfg.max_speed_kmh == pytest.approx(30.0)
        assert cfg.min_speed_kmh == pytest.approx(0.0)
        assert cfg.max_steering_angle == pytest.approx(0.5)
        assert cfg.collision_timeout == pytest.approx(5.0)
        assert cfg.emergency_deceleration == pytest.approx(-5.0)

    def test_custom_values(self):
        cfg = SafetyConfig(
            emergency_brake_enabled=False,
            max_speed_kmh=50.0,
            max_steering_angle=0.45,
        )
        assert cfg.emergency_brake_enabled is False
        assert cfg.max_speed_kmh == pytest.approx(50.0)
        assert cfg.max_steering_angle == pytest.approx(0.45)


# ── SafetyOverride init ──────────────────────────────────────────────────────


class TestSafetyOverrideInit:
    """Test SafetyOverride constructor."""

    def test_default_config(self):
        safety = SafetyOverride({})
        assert safety.config.emergency_brake_enabled is True
        assert safety.config.max_speed_kmh == pytest.approx(30.0)

    def test_custom_config(self):
        safety = SafetyOverride({
            "max_speed_kmh": 50.0,
            "max_steering_angle": 0.45,
            "emergency_brake_enabled": False,
        })
        assert safety.config.max_speed_kmh == pytest.approx(50.0)
        assert safety.config.max_steering_angle == pytest.approx(0.45)
        assert safety.config.emergency_brake_enabled is False

    def test_override_count_starts_zero(self):
        safety = SafetyOverride({})
        assert safety.override_count == 0

    def test_last_override_none(self):
        safety = SafetyOverride({})
        assert safety.last_safety_override is None


# ── check_speed_limit ────────────────────────────────────────────────────────


class TestCheckSpeedLimit:
    """Test speed limit enforcement."""

    def test_under_limit_no_override(self):
        safety = SafetyOverride({"max_speed_kmh": 30.0, "emergency_brake_enabled": True})
        th, br = safety.check_speed_limit({"velocity": 5.0}, 0.5, 0.0)
        assert th == pytest.approx(0.5)
        assert br == pytest.approx(0.0)

    def test_over_limit_emergency_brake(self):
        """Over speed limit → full brake, no throttle."""
        safety = SafetyOverride({"max_speed_kmh": 30.0, "emergency_brake_enabled": True})
        th, br = safety.check_speed_limit({"velocity": 100.0 / 3.6}, 1.0, 0.0)
        assert th == pytest.approx(0.0)
        assert br == pytest.approx(1.0)

    def test_over_limit_no_emergency_brake(self):
        """Over limit but emergency brake disabled → no override."""
        safety = SafetyOverride({"max_speed_kmh": 30.0, "emergency_brake_enabled": False})
        th, br = safety.check_speed_limit({"velocity": 100.0 / 3.6}, 1.0, 0.0)
        # Without emergency brake, should not force brake
        assert th == pytest.approx(1.0)

    def test_negative_speed(self):
        """Negative speed should be handled."""
        safety = SafetyOverride({"max_speed_kmh": 30.0})
        th, br = safety.check_speed_limit({"velocity": -1.0}, 0.5, 0.3)
        # Should not crash, may reduce brake
        assert isinstance(th, float)
        assert isinstance(br, float)

    def test_missing_velocity_defaults_zero(self):
        """Missing velocity key should default to 0."""
        safety = SafetyOverride({"max_speed_kmh": 30.0})
        th, br = safety.check_speed_limit({}, 0.5, 0.0)
        assert th == pytest.approx(0.5)

    def test_override_count_incremented(self):
        """Speed limit violation should increment override_count."""
        safety = SafetyOverride({"max_speed_kmh": 30.0, "emergency_brake_enabled": True})
        safety.check_speed_limit({"velocity": 100.0 / 3.6}, 1.0, 0.0)
        assert safety.override_count == 1
        assert safety.last_safety_override == "speed_limit"


# ── check_steering_limit ─────────────────────────────────────────────────────


class TestCheckSteeringLimit:
    """Test steering angle limit enforcement."""

    def test_within_limit(self):
        safety = SafetyOverride({"max_steering_angle": 0.5})
        assert safety.check_steering_limit(0.3) == pytest.approx(0.3)

    def test_exceeds_positive(self):
        """Steering above max → clamped."""
        safety = SafetyOverride({"max_steering_angle": 0.5})
        result = safety.check_steering_limit(1.0)
        assert result == pytest.approx(0.5)

    def test_exceeds_negative(self):
        """Steering below -max → clamped."""
        safety = SafetyOverride({"max_steering_angle": 0.5})
        result = safety.check_steering_limit(-1.0)
        assert result == pytest.approx(-0.5)

    def test_zero_steering(self):
        safety = SafetyOverride({"max_steering_angle": 0.5})
        assert safety.check_steering_limit(0.0) == pytest.approx(0.0)

    def test_override_count_incremented(self):
        safety = SafetyOverride({"max_steering_angle": 0.5})
        safety.check_steering_limit(1.0)
        assert safety.override_count == 1
        assert safety.last_safety_override == "steering_limit"


# ── check_control_validity ───────────────────────────────────────────────────


class TestCheckControlValidity:
    """Test NaN/Inf detection and range clamping."""

    def test_valid_controls_pass_through(self):
        safety = SafetyOverride({})
        st, th, br = safety.check_control_validity(0.1, 0.5, 0.0)
        assert st == pytest.approx(0.1)
        assert th == pytest.approx(0.5)
        assert br == pytest.approx(0.0)

    def test_nan_steering_replaced(self):
        safety = SafetyOverride({})
        st, _, _ = safety.check_control_validity(float("nan"), 0.5, 0.0)
        assert st == pytest.approx(0.0)

    def test_nan_throttle_replaced(self):
        safety = SafetyOverride({})
        _, th, _ = safety.check_control_validity(0.1, float("nan"), 0.0)
        assert th == pytest.approx(0.0)

    def test_nan_brake_replaced(self):
        safety = SafetyOverride({})
        _, _, br = safety.check_control_validity(0.1, 0.5, float("nan"))
        assert br == pytest.approx(0.0)

    def test_inf_steering_replaced(self):
        safety = SafetyOverride({})
        st, _, _ = safety.check_control_validity(float("inf"), 0.5, 0.0)
        assert st == pytest.approx(0.0)

    def test_neg_inf_throttle_replaced(self):
        safety = SafetyOverride({})
        _, th, _ = safety.check_control_validity(0.1, float("-inf"), 0.0)
        assert th == pytest.approx(0.0)

    def test_steering_clamped_to_range(self):
        safety = SafetyOverride({})
        st, _, _ = safety.check_control_validity(5.0, 0.5, 0.0)
        assert st <= 1.1

    def test_throttle_clamped_to_range(self):
        safety = SafetyOverride({})
        _, th, _ = safety.check_control_validity(0.1, 5.0, 0.0)
        assert th <= 1.1

    def test_brake_clamped_to_range(self):
        safety = SafetyOverride({})
        _, _, br = safety.check_control_validity(0.1, 0.5, 5.0)
        assert br <= 1.1

    def test_all_nan(self):
        """All NaN → all replaced with 0."""
        safety = SafetyOverride({})
        st, th, br = safety.check_control_validity(float("nan"), float("nan"), float("nan"))
        assert st == pytest.approx(0.0)
        assert th == pytest.approx(0.0)
        assert br == pytest.approx(0.0)


# ── check_cte_intervention ───────────────────────────────────────────────────


class TestCTEIntervention:
    """Test CTE-based safety intervention."""

    def test_low_cte_no_intervention(self):
        """Low CTE → no speed reduction."""
        safety = SafetyOverride({})
        th, br = safety.check_cte_intervention({"cte": 0.5}, 0.8, 0.0)
        assert th == pytest.approx(0.8)

    def test_high_cte_reduces_throttle(self):
        """High CTE → throttle reduced."""
        safety = SafetyOverride({})
        th, br = safety.check_cte_intervention({"cte": 3.0}, 0.8, 0.0)
        assert th < 0.8

    def test_very_high_cte(self):
        """Very high CTE → significant throttle reduction."""
        safety = SafetyOverride({})
        th, br = safety.check_cte_intervention({"cte": 5.0}, 1.0, 0.0)
        assert th < 1.0

    def test_missing_cte_defaults_zero(self):
        """Missing CTE → no intervention."""
        safety = SafetyOverride({})
        th, br = safety.check_cte_intervention({}, 0.8, 0.0)
        assert th == pytest.approx(0.8)

    def test_negative_cte_treated_as_abs(self):
        """Negative CTE should be treated by absolute value."""
        safety = SafetyOverride({})
        th_pos, _ = safety.check_cte_intervention({"cte": 3.0}, 0.8, 0.0)
        th_neg, _ = safety.check_cte_intervention({"cte": -3.0}, 0.8, 0.0)
        assert th_pos == pytest.approx(th_neg)


# ── apply_safety_override (main entry point) ─────────────────────────────────


class TestApplySafetyOverride:
    """Test the main safety override entry point."""

    def test_normal_operation(self):
        """Normal controls should pass through mostly unchanged."""
        safety = SafetyOverride({"max_speed_kmh": 50.0, "max_steering_angle": 0.5})
        st, th, br = safety.apply_safety_override(
            {"velocity": 10.0 / 3.6, "cte": 0.1},
            steering=0.1, throttle=0.5, brake=0.0,
        )
        assert st == pytest.approx(0.1)
        assert th == pytest.approx(0.5)

    def test_clamps_steering(self):
        safety = SafetyOverride({"max_steering_angle": 0.45})
        st, _, _ = safety.apply_safety_override(
            {"velocity": 5.0}, steering=1.0, throttle=0.3, brake=0.0,
        )
        assert abs(st) <= 0.45

    def test_nan_steering_replaced(self):
        safety = SafetyOverride({})
        st, _, _ = safety.apply_safety_override(
            {"velocity": 0.0}, steering=float("nan"), throttle=0.0, brake=0.0,
        )
        assert not math.isnan(st)

    def test_speed_limit_enforced(self):
        safety = SafetyOverride({"max_speed_kmh": 30.0, "emergency_brake_enabled": True})
        _, th, br = safety.apply_safety_override(
            {"velocity": 100.0 / 3.6}, steering=0.0, throttle=1.0, brake=0.0,
        )
        # Speed limit should trigger brake
        assert br > 0 or th < 1.0

    def test_returns_three_floats(self):
        safety = SafetyOverride({})
        result = safety.apply_safety_override(
            {"velocity": 5.0}, steering=0.1, throttle=0.5, brake=0.0,
        )
        assert len(result) == 3
        for v in result:
            assert isinstance(v, (float, np.floating))

    def test_all_finite(self):
        """Output should always be finite."""
        safety = SafetyOverride({})
        st, th, br = safety.apply_safety_override(
            {"velocity": 5.0}, steering=0.1, throttle=0.5, brake=0.0,
        )
        assert np.isfinite(st)
        assert np.isfinite(th)
        assert np.isfinite(br)

    def test_chained_overrides(self):
        """Multiple violations should all be caught."""
        safety = SafetyOverride({
            "max_speed_kmh": 30.0,
            "max_steering_angle": 0.5,
            "emergency_brake_enabled": True,
        })
        st, th, br = safety.apply_safety_override(
            {"velocity": 100.0 / 3.6},
            steering=2.0, throttle=1.0, brake=0.0,
        )
        # Steering should be clamped
        assert abs(st) <= 0.5
        # Speed should be addressed
        assert br > 0 or th < 1.0
