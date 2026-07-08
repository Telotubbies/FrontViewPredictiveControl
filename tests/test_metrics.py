"""Unit tests for metrics modules (lane_compliance, waypoint_following,
run_logger, realtime_stats).

Tests follow TDD principles: DAMP over DRY, Arrange-Act-Assert,
test STATE not interactions, one assertion per concept.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from metrics.lane_compliance import LaneComplianceMetrics
from metrics.waypoint_following import WaypointFollowingMetrics
from metrics.run_logger import RunLogger
from metrics.realtime_stats import RealTimeStats
from state import FrameState


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_frame_state(**kwargs):
    """Build a minimal FrameState with sensible defaults for metric tests."""
    defaults = dict(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        speed_kmh=0.0,
        steer=0.0,
        throttle=0.0,
        brake=0.0,
        cte_m=0.0,
        heading_rad=0.0,
        curvature=0.0,
        mode="normal",
    )
    defaults.update(kwargs)
    return FrameState(**defaults)


# ── LaneComplianceMetrics ─────────────────────────────────────────────────────


class TestLaneComplianceUpdate:
    """update() must store CTE and return current-frame metrics."""

    def test_lane_compliance_update_stores_cte(self):
        # Arrange
        m = LaneComplianceMetrics()

        # Act
        result = m.update(cte_m=0.5, heading_rad=0.0, timestamp=1.0)

        # Assert — returned dict reflects the CTE
        assert result["cte_m"] == pytest.approx(0.5)
        # Assert — summary reflects the stored CTE
        summary = m.get_summary()
        assert summary.cte_current == pytest.approx(0.5)


class TestLaneComplianceDeparture:
    """CTE exceeding the departure threshold must trigger a departure event."""

    def test_lane_compliance_departure_detection(self):
        # Arrange — threshold 1.75m
        m = LaneComplianceMetrics(departure_threshold_m=1.75)

        # Act — first frame in-lane, second frame departed
        m.update(cte_m=0.5, heading_rad=0.0, timestamp=1.0)
        departed = m.update(cte_m=2.0, heading_rad=0.0, timestamp=2.0)

        # Assert — departure flagged in returned dict
        assert departed["departure_active"] is True
        assert departed["in_lane"] is False
        # Assert — summary counts one departure event
        summary = m.get_summary()
        assert summary.departure_count == 1


class TestLaneComplianceSummary:
    """get_summary() must compute mean/std/rms over collected CTE samples."""

    def test_lane_compliance_summary_stats(self):
        # Arrange
        m = LaneComplianceMetrics()
        ctes = [0.0, 1.0, -1.0, 0.5, -0.5]
        for i, c in enumerate(ctes):
            m.update(cte_m=c, heading_rad=0.0, timestamp=float(i))

        # Act
        summary = m.get_summary()

        # Assert
        assert summary.cte_mean == pytest.approx(np.mean(ctes))
        assert summary.cte_std == pytest.approx(np.std(ctes))
        assert summary.cte_rms == pytest.approx(np.sqrt(np.mean(np.square(ctes))))
        assert summary.total_frames == len(ctes)


class TestLaneComplianceCSVExport:
    """export_csv must create a file with the expected columns and rows."""

    def test_lane_compliance_csv_export(self, tmp_path):
        # Arrange
        m = LaneComplianceMetrics()
        m.update(cte_m=0.3, heading_rad=0.1, timestamp=1.0)
        m.update(cte_m=0.6, heading_rad=0.2, timestamp=2.0)
        csv_file = tmp_path / "lane.csv"

        # Act
        m.export_csv(str(csv_file))

        # Assert — file exists
        assert csv_file.exists()
        # Assert — header has expected columns
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        assert "cte_m" in header
        assert "timestamp" in header
        # Assert — one data row per update
        assert len(rows) == 2


class TestLaneComplianceJSONExport:
    """export_json must create a valid JSON file with summary fields."""

    def test_lane_compliance_json_export(self, tmp_path):
        # Arrange
        m = LaneComplianceMetrics()
        m.update(cte_m=0.4, heading_rad=0.0, timestamp=1.0)
        json_file = tmp_path / "lane.json"

        # Act
        m.export_json(str(json_file))

        # Assert — file exists and is valid JSON
        assert json_file.exists()
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        # Assert — contains expected summary keys
        assert "cte_mean" in data
        assert "cte_rms" in data
        assert "departure_count" in data


# ── WaypointFollowingMetrics ──────────────────────────────────────────────────


class TestWaypointFollowingCTE:
    """CTE vs waypoints must be computed from the nearest segment."""

    def test_waypoint_following_cte(self):
        # Arrange — straight horizontal path along y=0
        m = WaypointFollowingMetrics(reach_radius_m=3.0)
        wps = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        m.set_waypoints(wps)

        # Act — vehicle offset 1m in +y (lateral)
        result = m.update(
            vehicle_x=5.0, vehicle_y=1.0,
            vehicle_heading_rad=0.0,
            speed_actual_ms=5.0, speed_target_ms=5.0,
            timestamp=1.0,
        )

        # Assert — signed CTE from cross product with segment dir (1,0)
        # rel=(5,1), seg_dir=(1,0) => cte = rel[0]*seg_dir[1] - rel[1]*seg_dir[0] = -1
        assert result["cte_m"] == pytest.approx(-1.0, abs=1e-6)


class TestWaypointFollowingDistance:
    """Distance traveled must accumulate between updates."""

    def test_waypoint_following_distance(self):
        # Arrange
        m = WaypointFollowingMetrics()
        m.set_waypoints([(0.0, 0.0), (10.0, 0.0)])

        # Act — move 3m along x
        m.update(vehicle_x=0.0, vehicle_y=0.0, vehicle_heading_rad=0.0,
                 speed_actual_ms=1.0, speed_target_ms=1.0, timestamp=1.0)
        result = m.update(vehicle_x=3.0, vehicle_y=0.0, vehicle_heading_rad=0.0,
                          speed_actual_ms=1.0, speed_target_ms=1.0, timestamp=2.0)

        # Assert
        assert result["distance_traveled_m"] == pytest.approx(3.0)


class TestWaypointFollowingReach:
    """A waypoint within reach_radius must be detected as reached."""

    def test_waypoint_following_reach(self):
        # Arrange — single waypoint at origin, reach radius 3m
        m = WaypointFollowingMetrics(reach_radius_m=3.0)
        m.set_waypoints([(0.0, 0.0), (10.0, 0.0)])

        # Act — vehicle near first waypoint
        result = m.update(
            vehicle_x=1.0, vehicle_y=0.0,
            vehicle_heading_rad=0.0,
            speed_actual_ms=2.0, speed_target_ms=2.0,
            timestamp=1.0,
        )

        # Assert — at least one waypoint reached
        assert result["waypoints_reached"] >= 1


# ── RealTimeStats ─────────────────────────────────────────────────────────────


class TestRealTimeStatsUpdate:
    """update() must accumulate counters from FrameState + context."""

    def test_realtime_stats_update(self):
        # Arrange
        stats = RealTimeStats(print_interval=1000)  # avoid console printing
        fs = _make_frame_state(aeb_active=True, cte_m=0.5, final_steer=0.1)

        # Act
        stats.update(fs, context={"frame_idx": 1, "speed_ms": 5.0, "loop_time_ms": 10.0})

        # Assert — one frame counted, AEB counter incremented
        assert stats._total_frames == 1
        assert stats._aeb_count == 1


class TestRealTimeStatsCurrentStats:
    """get_current_stats() must return a dict with expected keys."""

    def test_realtime_stats_current_stats(self):
        # Arrange
        stats = RealTimeStats(print_interval=1000)
        fs = _make_frame_state(cte_m=0.3)
        stats.update(fs, context={"frame_idx": 1, "speed_ms": 5.0, "loop_time_ms": 10.0})

        # Act
        current = stats.get_current_stats()

        # Assert — expected keys present
        assert "frame_count" in current
        assert "cte_mean" in current
        assert "cte_rmse" in current
        assert "speed_current_kmh" in current
        assert "aeb_count" in current
        assert current["frame_count"] == 1


# ── RunLogger ─────────────────────────────────────────────────────────────────


class TestRunLoggerStartStop:
    """start() + stop() must create a CSV file with a header row."""

    def test_run_logger_start_stop(self, tmp_path):
        # Arrange
        run_dir = tmp_path / "run1"
        logger = RunLogger(str(run_dir), map_name="test", vehicle_type="car")

        # Act
        logger.start()
        logger.stop()

        # Assert — CSV file created with header
        csv_path = run_dir / "frame_log.csv"
        assert csv_path.exists()
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "frame_idx" in header
        assert "cte_m" in header


class TestRunLoggerLogFrame:
    """log_frame() must write a row to the CSV file."""

    def test_run_logger_log_frame(self, tmp_path):
        # Arrange
        run_dir = tmp_path / "run2"
        logger = RunLogger(str(run_dir))
        fs = _make_frame_state(cte_m=0.7, heading_rad=0.05)
        ctx = {"frame_idx": 1, "speed_ms": 5.0, "loop_time_ms": 12.0, "sim_time": 0.1}

        # Act
        logger.start()
        logger.log_frame(fs, ctx)
        logger.stop()

        # Assert — one data row present
        csv_path = run_dir / "frame_log.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert float(rows[0]["cte_m"]) == pytest.approx(0.7)
