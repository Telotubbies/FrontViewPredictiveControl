"""Unit tests for Dashboard (gui/dashboard.py).

Tests follow TDD principles: DAMP over DRY, Arrange-Act-Assert,
test STATE not interactions, one assertion per concept.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
import pytest

from gui.dashboard import Dashboard
from state import FrameState


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_frame_state(**kwargs):
    """Build a minimal FrameState with sensible defaults for dashboard tests."""
    defaults = dict(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        speed_kmh=50.0,
        steer=0.0,
        throttle=0.5,
        brake=0.0,
        cte_m=0.3,
        heading_rad=0.0,
        curvature=0.0,
        mode="normal",
    )
    defaults.update(kwargs)
    return FrameState(**defaults)


@pytest.fixture
def dashboard():
    """A Dashboard instance (initialized) for each test."""
    d = Dashboard()
    d.initialize()
    yield d
    d.cleanup()


# ── initialize ────────────────────────────────────────────────────────────────


class TestDashboardInitialize:
    """initialize() must return True and set _initialized."""

    def test_dashboard_initialize(self):
        # Arrange
        d = Dashboard()

        # Act
        ok = d.initialize()

        # Assert
        assert ok is True
        assert d._initialized is True

        # Cleanup
        d.cleanup()


# ── render_text ───────────────────────────────────────────────────────────────


class TestDashboardRenderText:
    """render_text must not crash with valid input."""

    def test_dashboard_render_text(self, dashboard):
        # Arrange — valid text and position
        text = "Hello"
        pos = (10, 10)

        # Act — should not raise
        dashboard.render_text(text, pos)

        # Assert — no exception means success (state-based: surface still valid)
        assert dashboard.panel_surface is not None


# ── handle_events ─────────────────────────────────────────────────────────────


class TestDashboardHandleEvents:
    """handle_events must return True when no quit event is pending."""

    def test_dashboard_handle_events(self, dashboard):
        # Arrange — no events injected (dummy SDL driver => empty queue)

        # Act
        keep_running = dashboard.handle_events()

        # Assert — no quit event => True
        assert keep_running is True


# ── cleanup ───────────────────────────────────────────────────────────────────


class TestDashboardCleanup:
    """cleanup() must set _initialized to False."""

    def test_dashboard_cleanup(self):
        # Arrange — initialized dashboard
        d = Dashboard()
        d.initialize()
        assert d._initialized is True

        # Act
        d.cleanup()

        # Assert
        assert d._initialized is False


# ── render_adas_panel ─────────────────────────────────────────────────────────


class TestDashboardRenderADASPanel:
    """render_adas_panel must not crash with a valid FrameState."""

    def test_dashboard_render_adas_panel(self, dashboard):
        # Arrange — FrameState with ADAS fields populated
        fs = _make_frame_state(
            aeb_active=True,
            aeb_ttc=1.2,
            acc_active=True,
            acc_target_speed_ms=10.0,
            acc_distance_m=5.0,
            ldw_state="in_lane",
            bsw_left_alert="clear",
            bsw_right_alert="clear",
            tsr_speed_limit_kmh=60.0,
            tsr_traffic_light="green",
            tja_active=True,
        )

        # Act — should not raise
        dashboard.render_adas_panel(fs)

        # Assert — ADAS surface still valid
        assert dashboard.adas_surface is not None
