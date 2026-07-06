"""Unit tests for carla_io.waypoints_to_cte_heading with mocked waypoints."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_io import waypoints_to_cte_heading


def _make_transform(x: float, y: float, yaw_deg: float):
    """Fake carla.Transform with .location and .rotation."""
    loc = SimpleNamespace(x=x, y=y, z=0.0)
    rot = SimpleNamespace(roll=0.0, pitch=0.0, yaw=yaw_deg)
    return SimpleNamespace(location=loc, rotation=rot)


def _make_waypoint(x: float, y: float, yaw_deg: float):
    """Fake carla.Waypoint with .transform."""
    return SimpleNamespace(
        transform=SimpleNamespace(
            location=SimpleNamespace(x=x, y=y, z=0.0),
            rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=yaw_deg),
        )
    )


def test_waypoints_to_cte_heading_empty_returns_none():
    """Empty or single waypoint returns None."""
    t = _make_transform(0.0, 0.0, 0.0)
    assert waypoints_to_cte_heading(t, []) is None
    assert waypoints_to_cte_heading(t, [_make_waypoint(0, 0, 0)]) is None


def test_waypoints_to_cte_heading_straight_on_center():
    """Vehicle at (0,0) heading 0, road along x: cte≈0, heading≈0."""
    # Road: (0,0) -> (10,0) -> (20,0)
    t = _make_transform(0.0, 0.0, 0.0)
    wps = [
        _make_waypoint(0, 0, 0),
        _make_waypoint(10, 0, 0),
        _make_waypoint(20, 0, 0),
    ]
    out = waypoints_to_cte_heading(t, wps)
    assert out is not None
    cte, head, curv = out
    assert abs(cte) < 0.01
    assert abs(head) < 0.01
    assert -0.08 <= curv <= 0.08


def test_waypoints_to_cte_heading_lateral_offset():
    """Vehicle to the right of road center has negative CTE (or positive depending on sign convention)."""
    # Road along x at y=0; vehicle at (5, 0.3) heading along road
    t = _make_transform(5.0, 0.3, 0.0)
    wps = [
        _make_waypoint(0, 0, 0),
        _make_waypoint(10, 0, 0),
        _make_waypoint(20, 0, 0),
    ]
    out = waypoints_to_cte_heading(t, wps)
    assert out is not None
    cte, head, curv = out
    # Vehicle at y=0.3, road at y=0 -> CTE ~ 0.3 (or -0.3 by convention)
    assert abs(cte) == pytest.approx(0.3, abs=0.05)
    assert -5.0 <= cte <= 5.0
    assert -0.4 <= head <= 0.4
    assert -0.08 <= curv <= 0.08


def test_waypoints_to_cte_heading_output_clamped():
    """Output (cte, heading, curvature) is within documented bounds."""
    t = _make_transform(100.0, 100.0, 90.0)
    wps = [
        _make_waypoint(0, 0, 0),
        _make_waypoint(2, 0, 0),
        _make_waypoint(4, 0, 0),
    ]
    out = waypoints_to_cte_heading(t, wps)
    if out is None:
        return
    cte, head, curv = out
    assert -5.0 <= cte <= 5.0
    assert -0.4 <= head <= 0.4
    assert -0.08 <= curv <= 0.08
