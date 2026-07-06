"""Production tests for reference path generation (alg/reference.py)."""
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from alg.reference import (
    dynamic_lookahead_m,
    smooth_path_lat,
    resample_path,
    get_quintic_coefficients,
    get_reference_path,
    get_reference_path_quintic,
)
import config


# ── dynamic_lookahead_m ──────────────────────────────────────────────────────


class TestDynamicLookahead:
    """Test speed-dependent lookahead distance."""

    def test_zero_speed_returns_min(self):
        """Zero speed → minimum lookahead."""
        assert dynamic_lookahead_m(0.0) >= config.LOOKAHEAD_MIN_M

    def test_low_speed_returns_min(self):
        """Low speed → minimum lookahead (clamped)."""
        assert dynamic_lookahead_m(1.0) >= config.LOOKAHEAD_MIN_M

    def test_high_speed_exceeds_min(self):
        """High speed → lookahead > minimum."""
        assert dynamic_lookahead_m(30.0) > config.LOOKAHEAD_MIN_M

    def test_monotonic_with_speed(self):
        """Lookahead should be non-decreasing with speed."""
        prev = 0.0
        for v in range(0, 50, 5):
            la = dynamic_lookahead_m(float(v))
            assert la >= prev
            prev = la

    def test_negative_speed_returns_min(self):
        """Negative speed should not crash, returns min."""
        la = dynamic_lookahead_m(-5.0)
        assert la >= config.LOOKAHEAD_MIN_M

    def test_returns_float(self):
        assert isinstance(dynamic_lookahead_m(10.0), float)


# ── smooth_path_lat ──────────────────────────────────────────────────────────


class TestSmoothPathLat:
    """Test EMA smoothing on lateral coordinate."""

    def test_empty_path_returns_empty(self):
        assert smooth_path_lat([]) == []

    def test_single_point_returns_as_is(self):
        assert smooth_path_lat([(0.0, 1.0)]) == [(0.0, 1.0)]

    def test_two_points_smoothed(self):
        path = [(0.0, 0.0), (5.0, 1.0)]
        smoothed = smooth_path_lat(path, alpha=0.5)
        assert len(smoothed) == 2
        # First point: alpha*prev + (1-alpha)*lat = 0.5*0 + 0.5*0 = 0
        assert smoothed[0][1] == pytest.approx(0.0)
        # Second point: 0.5*0 + 0.5*1 = 0.5
        assert smoothed[1][1] == pytest.approx(0.5)

    def test_alpha_one_freezes_at_first(self):
        """alpha=1 → lat_s = prev_lat (first lat propagated to all)."""
        path = [(0.0, 0.0), (5.0, 1.0), (10.0, -1.0)]
        smoothed = smooth_path_lat(path, alpha=1.0)
        for _, lat in smoothed:
            assert lat == pytest.approx(0.0)

    def test_alpha_zero_returns_original(self):
        """alpha=0 → lat_s = lat (original path returned)."""
        path = [(0.0, 0.5), (5.0, 1.0), (10.0, -1.0)]
        smoothed = smooth_path_lat(path, alpha=0.0)
        for i, (_, lat) in enumerate(smoothed):
            assert lat == pytest.approx(path[i][1])

    def test_preserves_s_coordinate(self):
        """S coordinate should not be modified by smoothing."""
        path = [(0.0, 0.0), (5.0, 1.0), (10.0, 2.0)]
        smoothed = smooth_path_lat(path, alpha=0.5)
        for i, (s, _) in enumerate(smoothed):
            assert s == pytest.approx(path[i][0])

    def test_reduces_jitter(self):
        """Smoothing should reduce variance of lateral values."""
        path = [(float(i), float((-1) ** i)) for i in range(20)]
        smoothed = smooth_path_lat(path, alpha=0.6)
        lat_orig = [p[1] for p in path]
        lat_smooth = [p[1] for p in smoothed]
        assert np.std(lat_smooth) < np.std(lat_orig)


# ── resample_path ────────────────────────────────────────────────────────────


class TestResamplePath:
    """Test path resampling to uniform s-spacing."""

    def test_empty_path_returns_empty(self):
        assert resample_path([], 10.0, 5) == []

    def test_single_point_returns_single(self):
        result = resample_path([(5.0, 1.0)], 10.0, 5)
        # Single point has s=5 which is within [0, 10], but only 1 unique s
        # so interpolation can't happen — returns single point or empty
        assert len(result) <= 1

    def test_two_points_resampled(self):
        path = [(0.0, 0.0), (10.0, 1.0)]
        result = resample_path(path, 10.0, 5)
        assert len(result) == 6  # num_pts + 1
        # First s = 0
        assert result[0][0] == pytest.approx(0.0)
        # Last s = 10
        assert result[-1][0] == pytest.approx(10.0)

    def test_filters_outside_range(self):
        """Points with s > lookahead should be filtered out."""
        path = [(0.0, 0.0), (5.0, 1.0), (15.0, 2.0)]
        result = resample_path(path, 10.0, 3)
        # Should only use points with s in [0, 10]
        for s, _ in result:
            assert 0.0 <= s <= 10.0

    def test_unsorted_path_handled(self):
        """Unsorted path should be sorted by s before resampling."""
        path = [(10.0, 1.0), (0.0, 0.0), (5.0, 0.5)]
        result = resample_path(path, 10.0, 3)
        # Result should be sorted by s
        s_vals = [s for s, _ in result]
        assert s_vals == sorted(s_vals)

    def test_duplicate_s_handled(self):
        """Duplicate s values should not crash interpolation."""
        path = [(0.0, 0.0), (5.0, 1.0), (5.0, 1.1), (10.0, 2.0)]
        result = resample_path(path, 10.0, 3)
        assert len(result) >= 1


# ── get_quintic_coefficients ─────────────────────────────────────────────────


class TestQuinticCoefficients:
    """Test quintic polynomial coefficient solving."""

    def test_zero_boundary_conditions(self):
        """All zero boundary conditions → all coefficients zero."""
        coeffs = get_quintic_coefficients(0, 0, 0, 0, 0, 0, T=10.0)
        assert np.allclose(coeffs, 0.0)

    def test_returns_six_coefficients(self):
        """Quintic has 6 coefficients [a5, a4, a3, a2, a1, a0]."""
        coeffs = get_quintic_coefficients(0, 0, 0, 1, 0, 0, T=10.0)
        assert len(coeffs) == 6

    def test_boundary_y0(self):
        """y(0) should equal y0."""
        y0 = 0.5
        coeffs = get_quintic_coefficients(y0, 0, 0, 0, 0, 0, T=10.0)
        # y(0) = a0 = coeffs[5]
        assert coeffs[5] == pytest.approx(y0)

    def test_boundary_yT(self):
        """y(T) should equal yT."""
        yT = 1.0
        T = 10.0
        coeffs = get_quintic_coefficients(0, 0, 0, yT, 0, 0, T=T)
        # Reconstruct y(T) = a5*T^5 + a4*T^4 + ... + a0
        y_reconstructed = sum(c * T ** (5 - i) for i, c in enumerate(coeffs))
        assert y_reconstructed == pytest.approx(yT, abs=1e-6)

    def test_boundary_dy0(self):
        """y'(0) should equal dy0."""
        dy0 = 0.1
        coeffs = get_quintic_coefficients(0, dy0, 0, 0, 0, 0, T=10.0)
        # y'(0) = a1 = coeffs[4]
        assert coeffs[4] == pytest.approx(dy0)

    def test_large_T(self):
        """Large T should not cause numerical issues."""
        coeffs = get_quintic_coefficients(0, 0, 0, 1, 0, 0, T=1000.0)
        assert np.all(np.isfinite(coeffs))


# ── get_reference_path / get_reference_path_quintic ──────────────────────────


class TestGetReferencePath:
    """Test reference path generation."""

    def test_shape_num_pts_plus_one(self):
        """Path should have num_pts + 1 points."""
        path = get_reference_path(0.0, 0.0, 0.0, lookahead_m=10.0, num_pts=5)
        assert len(path) == 6

    def test_straight_road_lateral_zero(self):
        """Zero errors → lateral stays 0 along path."""
        path = get_reference_path(0.0, 0.0, 0.0, lookahead_m=10.0, num_pts=10)
        for _, lat in path:
            assert abs(lat) < 1e-6

    def test_cte_offset_first_point(self):
        """CTE should appear as lateral offset at first point."""
        path = get_reference_path(0.5, 0.0, 0.0, lookahead_m=10.0, num_pts=5)
        assert path[0][1] == pytest.approx(0.5, abs=1e-4)

    def test_s_starts_at_zero(self):
        """First point s should be 0."""
        path = get_reference_path(0.0, 0.0, 0.0, lookahead_m=15.0, num_pts=5)
        assert path[0][0] == pytest.approx(0.0, abs=1e-6)

    def test_s_ends_at_lookahead(self):
        """Last point s should equal lookahead_m."""
        path = get_reference_path(0.0, 0.0, 0.0, lookahead_m=20.0, num_pts=5)
        assert path[-1][0] == pytest.approx(20.0, abs=1e-6)

    def test_heading_error_affects_lateral(self):
        """Non-zero heading should change lateral along path."""
        path_no_head = get_reference_path(0.0, 0.0, 0.0, lookahead_m=10.0, num_pts=10)
        path_with_head = get_reference_path(0.0, 0.1, 0.0, lookahead_m=10.0, num_pts=10)
        # Lateral at end should differ
        assert abs(path_with_head[-1][1]) > abs(path_no_head[-1][1])

    def test_curvature_affects_lateral(self):
        """Non-zero curvature should change lateral along path (parabolic mode)."""
        path_no_curv = get_reference_path(0.0, 0.0, 0.0, lookahead_m=10.0, num_pts=10, use_quintic=False)
        path_with_curv = get_reference_path(0.0, 0.0, 0.01, lookahead_m=10.0, num_pts=10, use_quintic=False)
        # Lateral at end should differ: 0.5*curv*lookahead^2
        assert abs(path_with_curv[-1][1]) > abs(path_no_curv[-1][1])

    def test_negative_cte(self):
        """Negative CTE should produce negative lateral offset."""
        path = get_reference_path(-0.5, 0.0, 0.0, lookahead_m=10.0, num_pts=5)
        assert path[0][1] < 0

    def test_returns_list_of_tuples(self):
        path = get_reference_path(0.0, 0.0, 0.0, lookahead_m=10.0, num_pts=5)
        assert isinstance(path, list)
        for point in path:
            assert isinstance(point, tuple)
            assert len(point) == 2

    def test_quintic_mode(self):
        """Quintic mode should produce valid path."""
        path = get_reference_path(0.3, 0.05, 0.002, lookahead_m=15.0, num_pts=8, use_quintic=True)
        assert len(path) == 9
        for s, lat in path:
            assert isinstance(s, float)
            assert isinstance(lat, float)
            assert np.isfinite(s)
            assert np.isfinite(lat)


class TestGetReferencePathQuintic:
    """Test quintic reference path generation directly."""

    def test_shape(self):
        path = get_reference_path_quintic(0.0, 0.0, 0.0, lookahead_m=10.0, num_pts=5)
        assert len(path) == 6

    def test_straight_road(self):
        path = get_reference_path_quintic(0.0, 0.0, 0.0, lookahead_m=10.0, num_pts=10)
        for _, lat in path:
            assert abs(lat) < 1e-6

    def test_cte_offset(self):
        path = get_reference_path_quintic(0.5, 0.0, 0.0, lookahead_m=10.0, num_pts=5)
        assert path[0][1] == pytest.approx(0.5, abs=1e-4)

    def test_smooth_transitions(self):
        """Quintic path should be smooth (no discontinuities)."""
        path = get_reference_path_quintic(0.5, 0.1, 0.01, lookahead_m=20.0, num_pts=20)
        lats = [lat for _, lat in path]
        # Check no large jumps between consecutive points
        for i in range(1, len(lats)):
            assert abs(lats[i] - lats[i - 1]) < 1.0
