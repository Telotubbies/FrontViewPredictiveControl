"""Unit tests for ADASv2Dashboard."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adas_v2.dashboard import ADASv2Dashboard, DashboardState


class TestDashboardState:
    """Tests for DashboardState dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        state = DashboardState()
        assert state.rgb is None
        assert state.speed_kmh == 0.0
        assert state.target_speed_kmh == 0.0
        assert state.steer == 0.0
        assert state.cte == 0.0
        assert state.mode == "INIT"
        assert state.frame_id == 0
        
    def test_custom_values(self):
        """Test custom values."""
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        state = DashboardState(
            rgb=rgb,
            speed_kmh=30.0,
            target_speed_kmh=35.0,
            steer=0.1,
            cte=0.5,
            mode="LANE",
            frame_id=100,
        )
        
        assert state.speed_kmh == 30.0
        assert state.mode == "LANE"
        assert state.frame_id == 100


class TestADASv2Dashboard:
    """Tests for ADASv2Dashboard class."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        dashboard = ADASv2Dashboard()
        assert dashboard.width == 800
        assert dashboard.height == 600
        assert dashboard.camera_height == 400
        
    def test_init_custom(self):
        """Test initialization with custom values."""
        dashboard = ADASv2Dashboard(width=1024, height=768, camera_height=500)
        assert dashboard.width == 1024
        assert dashboard.height == 768
        assert dashboard.camera_height == 500
        
    def test_render_empty_state(self):
        """Test rendering with empty state."""
        dashboard = ADASv2Dashboard()
        state = DashboardState()
        
        frame = dashboard.render(state)
        
        assert frame.shape == (600, 800, 3)
        assert frame.dtype == np.uint8
        
    def test_render_with_camera(self):
        """Test rendering with camera image."""
        dashboard = ADASv2Dashboard()
        
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[100:200, 100:200] = [255, 0, 0]  # Red square
        
        state = DashboardState(
            rgb=rgb,
            speed_kmh=30.0,
            target_speed_kmh=35.0,
            mode="LANE",
        )
        
        frame = dashboard.render(state)
        
        assert frame.shape == (600, 800, 3)
        
    def test_render_all_modes(self):
        """Test rendering with all modes."""
        dashboard = ADASv2Dashboard()
        
        for mode in ["LANE", "WP", "FUSION", "INIT"]:
            state = DashboardState(mode=mode)
            frame = dashboard.render(state)
            assert frame.shape == (600, 800, 3)
            
    def test_render_extreme_values(self):
        """Test rendering with extreme values."""
        dashboard = ADASv2Dashboard()
        
        state = DashboardState(
            speed_kmh=200.0,
            target_speed_kmh=250.0,
            steer=1.0,
            throttle=1.0,
            brake=1.0,
            cte=5.0,
            heading_deg=90.0,
            curvature=0.5,
            lane_confidence=1.0,
            lane_weight=1.0,
            wp_lookahead_curv=0.5,
        )
        
        frame = dashboard.render(state)
        assert frame.shape == (600, 800, 3)
        
    def test_render_negative_values(self):
        """Test rendering with negative values."""
        dashboard = ADASv2Dashboard()
        
        state = DashboardState(
            steer=-1.0,
            cte=-2.0,
            heading_deg=-45.0,
        )
        
        frame = dashboard.render(state)
        assert frame.shape == (600, 800, 3)
        
    def test_mode_colors(self):
        """Test that mode colors are defined."""
        dashboard = ADASv2Dashboard()
        
        assert "LANE" in dashboard.MODE_COLORS
        assert "WP" in dashboard.MODE_COLORS
        assert "FUSION" in dashboard.MODE_COLORS
        assert "INIT" in dashboard.MODE_COLORS
        
    def test_draw_bar_center(self):
        """Test center-based bar drawing."""
        dashboard = ADASv2Dashboard()
        
        # This should not raise
        dashboard._draw_bar(10, 10, 100, 20, 0.5, -1.0, 1.0, center=True)
        dashboard._draw_bar(10, 10, 100, 20, -0.5, -1.0, 1.0, center=True)
        
    def test_draw_bar_left_to_right(self):
        """Test left-to-right bar drawing."""
        dashboard = ADASv2Dashboard()
        
        # This should not raise
        dashboard._draw_bar(10, 10, 100, 20, 0.5, 0.0, 1.0, center=False)
        dashboard._draw_bar(10, 10, 100, 20, 0.0, 0.0, 1.0, center=False)
        dashboard._draw_bar(10, 10, 100, 20, 1.0, 0.0, 1.0, center=False)
        
    def test_draw_bar_out_of_range(self):
        """Test bar drawing with out-of-range values."""
        dashboard = ADASv2Dashboard()
        
        # Values outside range should be clamped
        dashboard._draw_bar(10, 10, 100, 20, 2.0, 0.0, 1.0, center=False)
        dashboard._draw_bar(10, 10, 100, 20, -1.0, 0.0, 1.0, center=False)
        
    def test_camera_resize(self):
        """Test camera image resizing."""
        dashboard = ADASv2Dashboard(width=800, height=600, camera_height=400)
        
        # Small image
        small_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        state = DashboardState(rgb=small_rgb)
        frame = dashboard.render(state)
        assert frame.shape == (600, 800, 3)
        
        # Large image
        large_rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        state = DashboardState(rgb=large_rgb)
        frame = dashboard.render(state)
        assert frame.shape == (600, 800, 3)
        
    def test_panel_height(self):
        """Test panel height calculation."""
        dashboard = ADASv2Dashboard(width=800, height=600, camera_height=400)
        assert dashboard.panel_height == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
