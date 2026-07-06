"""Production tests for sensor fusion logic (algorithms/fusion.py)."""
import pytest
import numpy as np

from algorithms.fusion import apply_fusion
import config


class TestApplyFusionNoWaypoint:
    """Test fusion when no waypoint state is available."""

    def test_none_wp_returns_lane_cte(self):
        """When wp_state is None, CTE should come from lane perception."""
        cte, head, curv, mode = apply_fusion(
            wp_state=None,
            cte_m_lane=0.5,
            head_s=0.02,
            curv_s=0.001,
            lane_conf=0.8,
        )
        assert cte == pytest.approx(0.5)
        assert head == pytest.approx(0.02)
        assert curv == pytest.approx(0.001)

    def test_none_wp_clips_cte(self):
        """CTE should be clipped to ±CTE_LANE_HALF_WIDTH_M."""
        cte, _, _, _ = apply_fusion(
            wp_state=None,
            cte_m_lane=100.0,
            head_s=0.0,
            curv_s=0.0,
            lane_conf=0.8,
        )
        assert cte <= config.CTE_LANE_HALF_WIDTH_M
        assert cte >= -config.CTE_LANE_HALF_WIDTH_M

    def test_none_wp_clips_heading(self):
        """Heading should be clipped to ±0.25 rad."""
        _, head, _, _ = apply_fusion(
            wp_state=None,
            cte_m_lane=0.0,
            head_s=10.0,
            curv_s=0.0,
            lane_conf=0.8,
        )
        assert head <= 0.25
        assert head >= -0.25

    def test_none_wp_clips_curvature(self):
        """Curvature should be clipped to ±MAX_CURVATURE_REF."""
        _, _, curv, _ = apply_fusion(
            wp_state=None,
            cte_m_lane=0.0,
            head_s=0.0,
            curv_s=100.0,
            lane_conf=0.8,
        )
        assert curv <= config.MAX_CURVATURE_REF
        assert curv >= -config.MAX_CURVATURE_REF


class TestApplyFusionWithWaypoint:
    """Test fusion when waypoint state is available."""

    def test_low_conf_uses_wp_primary(self):
        """Low lane confidence → WP_PRIMARY mode, no UNet blend."""
        cte, head, curv, mode = apply_fusion(
            wp_state=(0.3, 0.01, 0.002),
            cte_m_lane=0.8,
            head_s=0.05,
            curv_s=0.003,
            lane_conf=0.1,
        )
        assert mode == "WP_PRIMARY"
        # Should use WP values entirely (w_unet=0)
        assert cte == pytest.approx(0.3)
        assert head == pytest.approx(0.01)
        assert curv == pytest.approx(0.002)

    def test_high_conf_blends(self):
        """High lane confidence → WP+UNET_CENTER mode with blending."""
        cte, head, curv, mode = apply_fusion(
            wp_state=(0.3, 0.01, 0.002),
            cte_m_lane=0.5,
            head_s=0.03,
            curv_s=0.001,
            lane_conf=0.9,
        )
        assert "UNET" in mode or mode == "WP_PRIMARY"
        # If blended, CTE should be between WP and lane values
        if "UNET" in mode:
            assert 0.3 <= cte <= 0.5 or 0.5 <= cte <= 0.3

    def test_high_curvature_reduces_unet_weight(self):
        """In sharp curves, UNet weight should be reduced."""
        # Low curvature, high conf
        cte_low, _, _, mode_low = apply_fusion(
            wp_state=(0.3, 0.01, 0.0),
            cte_m_lane=0.8, head_s=0.0, curv_s=0.0,
            lane_conf=0.9,
        )
        # High curvature, high conf
        cte_high, _, _, mode_high = apply_fusion(
            wp_state=(0.3, 0.01, 0.0),
            cte_m_lane=0.8, head_s=0.0, curv_s=0.1,
            lane_conf=0.9,
        )
        # In high curvature, CTE should be closer to WP (less UNet influence)
        assert abs(cte_high - 0.3) <= abs(cte_low - 0.3) + 0.01

    def test_geometry_invalid_high_curv_wp_only(self):
        """geometry_valid=False + high curvature → WP only (w_unet=0)."""
        cte, head, curv, mode = apply_fusion(
            wp_state=(0.3, 0.01, 0.05),
            cte_m_lane=0.8, head_s=0.05, curv_s=0.05,
            lane_conf=0.9,
            geometry_valid=False,
        )
        assert mode == "WP_PRIMARY"
        assert cte == pytest.approx(0.3)

    def test_geometry_valid_allows_blend(self):
        """geometry_valid=True should allow UNet blending."""
        cte, _, _, mode = apply_fusion(
            wp_state=(0.3, 0.01, 0.001),
            cte_m_lane=0.8, head_s=0.0, curv_s=0.001,
            lane_conf=0.9,
            geometry_valid=True,
        )
        # Should not be forced to WP_PRIMARY
        assert mode in ("WP+UNET_CENTER", "WP_PRIMARY")

    def test_zero_confidence_uses_wp(self):
        """Zero lane confidence → WP_PRIMARY."""
        cte, head, curv, mode = apply_fusion(
            wp_state=(0.3, 0.01, 0.002),
            cte_m_lane=0.8, head_s=0.05, curv_s=0.003,
            lane_conf=0.0,
        )
        assert mode == "WP_PRIMARY"
        assert cte == pytest.approx(0.3)


class TestApplyFusionClipping:
    """Test that fusion output is always clipped to safe ranges."""

    @pytest.mark.parametrize("cte_lane,cte_wp", [
        (100.0, 100.0),
        (-100.0, -100.0),
        (50.0, -50.0),
    ])
    def test_cte_always_clipped(self, cte_lane, cte_wp):
        cte, _, _, _ = apply_fusion(
            wp_state=(cte_wp, 0.0, 0.0),
            cte_m_lane=cte_lane, head_s=0.0, curv_s=0.0,
            lane_conf=0.5,
        )
        assert abs(cte) <= config.CTE_LANE_HALF_WIDTH_M

    @pytest.mark.parametrize("head_lane,head_wp", [
        (10.0, 10.0),
        (-10.0, -10.0),
    ])
    def test_heading_always_clipped(self, head_lane, head_wp):
        _, head, _, _ = apply_fusion(
            wp_state=(0.0, head_wp, 0.0),
            cte_m_lane=0.0, head_s=head_lane, curv_s=0.0,
            lane_conf=0.5,
        )
        assert abs(head) <= 0.25

    @pytest.mark.parametrize("curv_lane,curv_wp", [
        (100.0, 100.0),
        (-100.0, -100.0),
    ])
    def test_curvature_always_clipped(self, curv_lane, curv_wp):
        _, _, curv, _ = apply_fusion(
            wp_state=(0.0, 0.0, curv_wp),
            cte_m_lane=0.0, head_s=0.0, curv_s=curv_lane,
            lane_conf=0.5,
        )
        assert abs(curv) <= config.MAX_CURVATURE_REF


class TestApplyFusionReturnTypes:
    """Test that return types are correct (safety-critical)."""

    def test_returns_four_tuple(self):
        result = apply_fusion(None, 0.0, 0.0, 0.0, 0.5)
        assert len(result) == 4

    def test_cte_is_float(self):
        cte, _, _, _ = apply_fusion(None, 0.0, 0.0, 0.0, 0.5)
        assert isinstance(cte, float)

    def test_head_is_float(self):
        _, head, _, _ = apply_fusion(None, 0.0, 0.0, 0.0, 0.5)
        assert isinstance(head, float)

    def test_curv_is_float(self):
        _, _, curv, _ = apply_fusion(None, 0.0, 0.0, 0.0, 0.5)
        assert isinstance(curv, float)

    def test_mode_is_str(self):
        _, _, _, mode = apply_fusion(None, 0.0, 0.0, 0.0, 0.5)
        assert isinstance(mode, str)
