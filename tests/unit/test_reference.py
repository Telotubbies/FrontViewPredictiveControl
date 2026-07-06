"""Unit tests for alg.reference: dynamic_lookahead_m, get_reference_path."""
import sys
from pathlib import Path

import pytest

# Ensure project root on path when running from repo root or from carla_mpc_classical
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alg.reference import dynamic_lookahead_m, get_reference_path


def test_dynamic_lookahead_m_min():
    """When speed is 0, lookahead is LOOKAHEAD_MIN_M (8.0)."""
    assert dynamic_lookahead_m(0.0) >= 8.0
    assert dynamic_lookahead_m(1.0) >= 8.0


def test_dynamic_lookahead_m_increases_with_speed():
    """Lookahead increases with speed (speed * LOOKAHEAD_TIME_S when above min)."""
    low = dynamic_lookahead_m(5.0)
    high = dynamic_lookahead_m(20.0)
    assert high >= low
    assert high > 8.0


def test_get_reference_path_shape():
    """Reference path has (num_pts + 1) points, each (s_m, lat_m)."""
    path = get_reference_path(cte_m=0.0, head_rad=0.0, curv=0.0, lookahead_m=10.0, num_pts=5)
    assert len(path) == 6
    for s, lat in path:
        assert isinstance(s, (int, float))
        assert isinstance(lat, (int, float))


def test_get_reference_path_straight_road():
    """Zero cte, zero heading, zero curvature: lateral stays 0 along path."""
    path = get_reference_path(cte_m=0.0, head_rad=0.0, curv=0.0, lookahead_m=10.0, num_pts=10)
    for s, lat in path:
        assert abs(lat) < 1e-9


def test_get_reference_path_cte_offset():
    """Non-zero CTE shifts entire path laterally."""
    path_zero = get_reference_path(cte_m=0.0, head_rad=0.0, curv=0.0, lookahead_m=10.0, num_pts=5)
    path_off = get_reference_path(cte_m=0.5, head_rad=0.0, curv=0.0, lookahead_m=10.0, num_pts=5)
    assert path_off[0][1] == pytest.approx(0.5, abs=1e-6)
    assert path_zero[0][1] == pytest.approx(0.0, abs=1e-6)


def test_get_reference_path_s_range():
    """First point s=0, last point s=lookahead_m."""
    path = get_reference_path(cte_m=0.0, head_rad=0.0, curv=0.0, lookahead_m=20.0, num_pts=4)
    assert path[0][0] == pytest.approx(0.0, abs=1e-6)
    assert path[-1][0] == pytest.approx(20.0, abs=1e-6)
