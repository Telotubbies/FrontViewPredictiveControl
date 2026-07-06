"""Unit tests for safety.safety_override.SafetyOverride.apply_safety_override."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety.override import SafetyOverride


def test_safety_clamps_steering():
    """Steering above max_steering_angle is clamped."""
    safety = SafetyOverride({
        "max_speed_kmh": 50.0,
        "max_steering_angle": 0.45,
        "emergency_brake_enabled": True,
    })
    st, th, br = safety.apply_safety_override(
        {"velocity": 5.0 / 3.6},
        steering=1.0,
        throttle=0.3,
        brake=0.0,
    )
    assert abs(st) <= 0.45
    assert -0.1 <= st <= 1.1
    assert 0 <= th <= 1.1
    assert 0 <= br <= 1.1


def test_safety_speed_limit():
    """High velocity + throttle: throttle reduced to respect max_speed_kmh."""
    safety = SafetyOverride({
        "max_speed_kmh": 30.0,
        "max_steering_angle": 0.5,
        "emergency_brake_enabled": True,
    })
    # Vehicle at 100 km/h (over limit), throttle 1.0 -> should be reduced
    st, th, br = safety.apply_safety_override(
        {"velocity": 100.0 / 3.6},
        steering=0.0,
        throttle=1.0,
        brake=0.0,
    )
    assert th <= 1.1
    assert st == 0.0


def test_safety_nan_steering_replaced():
    """NaN steering is replaced with 0."""
    import math
    safety = SafetyOverride({
        "max_speed_kmh": 50.0,
        "max_steering_angle": 0.5,
        "emergency_brake_enabled": True,
    })
    st, th, br = safety.apply_safety_override(
        {"velocity": 0.0},
        steering=float("nan"),
        throttle=0.0,
        brake=0.0,
    )
    assert not math.isnan(st)
    assert st == 0.0


def test_safety_returns_valid_triple():
    """apply_safety_override always returns (steer, throttle, brake) in valid range."""
    safety = SafetyOverride({
        "max_speed_kmh": 50.0,
        "max_steering_angle": 0.45,
        "emergency_brake_enabled": True,
    })
    st, th, br = safety.apply_safety_override(
        {"velocity": 10.0 / 3.6},
        steering=0.1,
        throttle=0.5,
        brake=0.0,
    )
    assert -1.1 <= st <= 1.1
    assert -0.1 <= th <= 1.1
    assert -0.1 <= br <= 1.1
