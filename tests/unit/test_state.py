"""Production tests for state types (state.py)."""
import pytest
import numpy as np
from dataclasses import fields as dataclass_fields

from state import FrameState


def _make_minimal_state(**overrides):
    defaults = dict(
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        speed_kmh=30.0,
        steer=0.0,
        throttle=0.5,
        brake=0.0,
        cte_m=0.1,
        heading_rad=0.02,
        curvature=0.001,
        mode="MPC",
    )
    defaults.update(overrides)
    return FrameState(**defaults)


class TestFrameStateConstruction:
    """Test FrameState dataclass construction."""

    def test_minimal_construction(self):
        fs = _make_minimal_state()
        assert fs.speed_kmh == pytest.approx(30.0)
        assert fs.steer == pytest.approx(0.0)
        assert fs.mode == "MPC"

    def test_rgb_is_ndarray(self):
        fs = _make_minimal_state()
        assert isinstance(fs.rgb, np.ndarray)

    def test_default_optional_fields_none(self):
        """Optional fields should default to None."""
        fs = _make_minimal_state()
        assert fs.reference_path is None
        assert fs.lane_overlay is None
        assert fs.mask is None
        assert fs.bev_window_vis is None
        assert fs.left_px_img is None
        assert fs.right_px_img is None

    def test_default_bool_fields_false(self):
        """Default bool fields should be False."""
        fs = _make_minimal_state()
        assert fs.geometry_valid is False
        assert fs.tracked is False
        assert fs.used_completion is False
        assert fs.phase_p1_ok is False
        assert fs.phase_p2_ok is False
        assert fs.phase_p3_ok is False
        assert fs.phase_p4_ok is False
        assert fs.phase_p5_ok is False

    def test_default_lane_conf_zero(self):
        fs = _make_minimal_state()
        assert fs.lane_conf == pytest.approx(0.0)

    def test_default_curvatures_large(self):
        """Default curvature should be very large (effectively no curvature)."""
        fs = _make_minimal_state()
        assert fs.left_curvature_m == pytest.approx(9999.0)
        assert fs.right_curvature_m == pytest.approx(9999.0)

    def test_default_solver_status(self):
        fs = _make_minimal_state()
        assert fs.solver_status == "Solve_Succeeded"

    def test_default_phase_cases(self):
        fs = _make_minimal_state()
        assert fs.phase_p2_case == "none"
        assert fs.phase_p3_case == "none"


class TestFrameStateCustomValues:
    """Test FrameState with custom values."""

    def test_custom_mode(self):
        fs = _make_minimal_state(mode="PP")
        assert fs.mode == "PP"

    def test_custom_reference_path(self):
        path = [(0.0, 0.0), (5.0, 0.1), (10.0, 0.2)]
        fs = _make_minimal_state(reference_path=path)
        assert fs.reference_path == path
        assert len(fs.reference_path) == 3

    def test_custom_lane_overlay(self):
        overlay = np.ones((480, 640, 3), dtype=np.uint8) * 128
        fs = _make_minimal_state(lane_overlay=overlay)
        assert np.array_equal(fs.lane_overlay, overlay)

    def test_custom_phase_flags(self):
        fs = _make_minimal_state(
            phase_p1_ok=True, phase_p2_ok=True,
            phase_p3_ok=True, phase_p4_ok=True, phase_p5_ok=True,
        )
        assert all([fs.phase_p1_ok, fs.phase_p2_ok, fs.phase_p3_ok,
                     fs.phase_p4_ok, fs.phase_p5_ok])

    def test_custom_solver_status(self):
        fs = _make_minimal_state(solver_status="Fallback")
        assert fs.solver_status == "Fallback"

    def test_negative_cte(self):
        """Negative CTE (left of center) should be stored correctly."""
        fs = _make_minimal_state(cte_m=-0.5)
        assert fs.cte_m == pytest.approx(-0.5)

    def test_negative_heading(self):
        fs = _make_minimal_state(heading_rad=-0.1)
        assert fs.heading_rad == pytest.approx(-0.1)


class TestFrameStateFields:
    """Test that FrameState has all required fields."""

    @pytest.mark.parametrize("field", [
        "rgb", "speed_kmh", "steer", "throttle", "brake",
        "cte_m", "heading_rad", "curvature", "mode",
        "lane_conf", "reference_path", "geometry_valid",
        "lane_overlay", "mask", "bev_window_vis",
        "left_px_img", "right_px_img",
        "road_symbol_result", "tracked", "used_completion",
        "phase_p1_ok", "phase_p2_ok", "phase_p3_ok",
        "phase_p4_ok", "phase_p5_ok",
        "phase_p2_case", "phase_p3_case",
        "solver_status",
        "bev_binary", "raw_windows_left", "raw_windows_right",
        "filt_windows_left", "filt_windows_right",
        "left_px_bev", "right_px_bev",
        "left_curvature_m", "right_curvature_m",
    ])
    def test_field_exists(self, field):
        field_names = [f.name for f in dataclass_fields(FrameState)]
        assert field in field_names, f"FrameState missing field: {field}"
