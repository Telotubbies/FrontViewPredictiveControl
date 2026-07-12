"""Tests for route planner and lane-map fusion.

Tests use mocked CARLA objects so they run without a live CARLA server.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import math
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (SRC, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ──────────────────────────────────────────────────────────────────────────────
# Mock CARLA objects
# ──────────────────────────────────────────────────────────────────────────────

class MockLocation:
    """Mock carla.Location."""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


class MockRotation:
    """Mock carla.Rotation."""
    def __init__(self, yaw=0.0, pitch=0.0, roll=0.0):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll


class MockTransform:
    """Mock carla.Transform."""
    def __init__(self, location=None, rotation=None):
        self.location = location or MockLocation()
        self.rotation = rotation or MockRotation()


class MockWaypoint:
    """Mock carla.Waypoint."""
    def __init__(self, road_id=0, lane_id=-1, x=0.0, y=0.0, yaw=0.0, lane_width=3.5):
        self.road_id = road_id
        self.lane_id = lane_id
        self.lane_width = lane_width
        self.transform = MockTransform(
            MockLocation(x, y), MockRotation(yaw)
        )

    def next(self, distance):
        """Return next waypoint(s)."""
        return [MockWaypoint(
            self.road_id, self.lane_id,
            self.transform.location.x + distance,
            self.transform.location.y,
            self.transform.rotation.yaw,
        )]

    def next_until_lane_end(self, distance):
        """Return waypoints until lane ends."""
        return [MockWaypoint(
            self.road_id, self.lane_id,
            self.transform.location.x + d,
            self.transform.location.y,
            self.transform.rotation.yaw,
        ) for d in [distance, distance*2, distance*3]]

    def get_left_lane(self):
        return MockWaypoint(self.road_id, self.lane_id - 1,
                           self.transform.location.x,
                           self.transform.location.y + 3.5)

    def get_right_lane(self):
        return MockWaypoint(self.road_id, self.lane_id + 1,
                           self.transform.location.x,
                           self.transform.location.y - 3.5)


class MockMap:
    """Mock carla.Map with a simple topology."""
    def __init__(self, topology=None, spawn_points=None):
        self._topology = topology or []
        self._spawn_points = spawn_points or []
        self.name = "MockMap"
        # Build waypoint index from topology for nearest-waypoint lookup
        self._all_wps = []
        for entry, exit in self._topology:
            self._all_wps.append(entry)
            self._all_wps.append(exit)

    def get_topology(self):
        return self._topology

    def get_spawn_points(self):
        return self._spawn_points

    def get_waypoint(self, location):
        """Return nearest waypoint by location."""
        if not self._all_wps:
            return MockWaypoint(x=location.x, y=location.y)
        nearest = min(self._all_wps,
                      key=lambda wp: (wp.transform.location.x - location.x)**2
                                   + (wp.transform.location.y - location.y)**2)
        return MockWaypoint(
            road_id=nearest.road_id, lane_id=nearest.lane_id,
            x=location.x, y=location.y,
            yaw=nearest.transform.rotation.yaw,
        )


def _make_simple_map():
    """Create a mock map with a simple 3-lane road topology + intersection."""
    # Lane segments: (entry_wp, exit_wp) pairs
    # Road 0, lanes -1, -2, -3 (right to left)
    topo = []
    for lane_id in [-1, -2, -3]:
        entry = MockWaypoint(road_id=0, lane_id=lane_id, x=0, y=lane_id * 3.5)
        exit = MockWaypoint(road_id=1, lane_id=lane_id, x=100, y=lane_id * 3.5)
        topo.append((entry, exit))

    # Road 1 → Road 2 (continuation)
    for lane_id in [-1, -2, -3]:
        entry = MockWaypoint(road_id=1, lane_id=lane_id, x=100, y=lane_id * 3.5)
        exit = MockWaypoint(road_id=2, lane_id=lane_id, x=200, y=lane_id * 3.5)
        topo.append((entry, exit))

    # Road 2 → Road 3 (turn right for lane -1) — intersection
    entry = MockWaypoint(road_id=2, lane_id=-1, x=200, y=-3.5)
    exit = MockWaypoint(road_id=3, lane_id=-1, x=250, y=-53.5, yaw=90)
    topo.append((entry, exit))

    # Road 2 → Road 5 (straight for lane -2)
    entry = MockWaypoint(road_id=2, lane_id=-2, x=200, y=-7.0)
    exit = MockWaypoint(road_id=5, lane_id=-2, x=300, y=-7.0)
    topo.append((entry, exit))

    # Road 3 → Road 4 (continuation after turn)
    entry = MockWaypoint(road_id=3, lane_id=-1, x=250, y=-53.5, yaw=90)
    exit = MockWaypoint(road_id=4, lane_id=-1, x=250, y=-153.5, yaw=90)
    topo.append((entry, exit))

    # Road 5 → Road 6 (continuation straight)
    entry = MockWaypoint(road_id=5, lane_id=-2, x=300, y=-7.0)
    exit = MockWaypoint(road_id=6, lane_id=-2, x=400, y=-7.0)
    topo.append((entry, exit))

    spawn = [
        MockTransform(MockLocation(0, -3.5)),
        MockTransform(MockLocation(250, -153.5)),  # goal at end of turn
    ]

    return MockMap(topology=topo, spawn_points=spawn)


# ──────────────────────────────────────────────────────────────────────────────
# Route Planner tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGlobalRoutePlanner:
    """Test GlobalRoutePlanner with mock map."""

    def setup_method(self):
        from planning.route_planner import GlobalRoutePlanner
        self.mock_map = _make_simple_map()
        self.planner = GlobalRoutePlanner(self.mock_map)

    def test_graph_built(self):
        """Graph should have nodes and edges."""
        assert len(self.planner._node_lookup) > 0
        assert sum(len(e) for e in self.planner._graph.values()) > 0

    def test_plan_route_returns_entries(self):
        """plan_route should return a list of RouteEntry."""
        from planning.route_planner import RouteEntry
        start = MockTransform(MockLocation(0, -3.5))
        goal = MockTransform(MockLocation(250, -153.5))  # after turn
        route = self.planner.plan_route(start, goal)
        assert len(route) > 0
        assert all(isinstance(r, RouteEntry) for r in route)

    def test_route_starts_with_lane_keep(self):
        """First entry should be LANE_KEEP."""
        from planning.route_planner import LANE_KEEP
        start = MockTransform(MockLocation(0, -3.5))
        goal = MockTransform(MockLocation(250, -153.5))
        route = self.planner.plan_route(start, goal)
        assert route[0].action == LANE_KEEP

    def test_route_ends_with_arrive(self):
        """Last entry should be ARRIVE."""
        from planning.route_planner import ARRIVE
        start = MockTransform(MockLocation(0, -3.5))
        goal = MockTransform(MockLocation(250, -153.5))
        route = self.planner.plan_route(start, goal)
        assert route[-1].action == ARRIVE

    def test_route_contains_maneuvers(self):
        """Route should contain at least one maneuver (non LANE_KEEP)."""
        from planning.route_planner import LANE_KEEP
        start = MockTransform(MockLocation(0, -3.5))
        goal = MockTransform(MockLocation(250, -153.5))
        route = self.planner.plan_route(start, goal)
        maneuvers = [r for r in route if r.action != LANE_KEEP]
        assert len(maneuvers) > 0

    def test_lane_change_edges_exist(self):
        """Graph should have lane-change edges between adjacent lanes."""
        from planning.route_planner import LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT
        has_lc = False
        for edges in self.planner._graph.values():
            for e in edges:
                if e.action in (LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT):
                    has_lc = True
                    break
        assert has_lc

    def test_get_next_maneuver(self):
        """get_next_maneuver should find the next significant action."""
        from planning.route_planner import GlobalRoutePlanner
        start = MockTransform(MockLocation(0, -3.5))
        goal = MockTransform(MockLocation(250, -153.5))
        route = self.planner.plan_route(start, goal)
        maneuver = self.planner.get_next_maneuver(start, route, lookahead_m=500)
        assert maneuver is not None

    def test_get_next_maneuver_none_when_far(self):
        """get_next_maneuver should return None when no maneuver within lookahead."""
        start = MockTransform(MockLocation(0, -3.5))
        goal = MockTransform(MockLocation(250, -153.5))
        route = self.planner.plan_route(start, goal)
        # Very short lookahead from a position far from any maneuver
        maneuver = self.planner.get_next_maneuver(
            MockTransform(MockLocation(1000, 1000)), route, lookahead_m=1.0
        )
        assert maneuver is None

    def test_fallback_route_on_no_path(self):
        """Should fallback to waypoint route when no path found."""
        # Create map with no topology
        empty_map = MockMap(topology=[], spawn_points=[])
        from planning.route_planner import GlobalRoutePlanner
        planner = GlobalRoutePlanner(empty_map)
        start = MockTransform(MockLocation(0, 0))
        goal = MockTransform(MockLocation(100, 0))
        route = planner.plan_route(start, goal)
        assert len(route) >= 1  # At least start entry


# ──────────────────────────────────────────────────────────────────────────────
# Lane-Map Fusion tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLaneMapFusion:
    """Test LaneMapFusion with mock map."""

    def setup_method(self):
        from planning.lane_map_fusion import LaneMapFusion
        self.mock_map = _make_simple_map()
        self.fusion = LaneMapFusion(self.mock_map, cam_w=640, cam_h=480)

    def test_get_lane_selection_info(self):
        """get_lane_selection_info should return a dict with expected keys."""
        from planning.route_planner import GlobalRoutePlanner
        planner = GlobalRoutePlanner(self.mock_map)
        start = MockTransform(MockLocation(0, -3.5))
        goal = MockTransform(MockLocation(200, -3.5))
        route = planner.plan_route(start, goal)

        info = self.fusion.get_lane_selection_info(start, route)
        assert "current_lane" in info
        assert "target_lane" in info
        assert "mode" in info
        assert "next_maneuver" in info
        assert "distance_to_maneuver" in info

    def test_determine_mode_lane_keep(self):
        """Mode should be LANE_KEEP when no maneuver nearby."""
        from planning.route_planner import GlobalRoutePlanner, LANE_KEEP, RouteEntry
        # Create a route with only LANE_KEEP entries
        route = [RouteEntry(
            waypoint=MockWaypoint(x=100, y=0),
            action=LANE_KEEP, road_id=0, lane_id=-1,
            distance_from_start_m=0,
        )]
        vt = MockTransform(MockLocation(0, 0))
        mode = self.fusion._determine_mode(vt, route, lookahead_m=50)
        assert mode == "LANE_KEEP"

    def test_determine_mode_lane_change(self):
        """Mode should be LANE_CHANGE when a lane change is nearby."""
        from planning.route_planner import LANE_CHANGE_LEFT, RouteEntry
        route = [RouteEntry(
            waypoint=MockWaypoint(x=10, y=0),
            action=LANE_CHANGE_LEFT, road_id=0, lane_id=-1,
            distance_from_start_m=0,
        )]
        vt = MockTransform(MockLocation(0, 0))
        mode = self.fusion._determine_mode(vt, route, lookahead_m=50)
        assert mode == "LANE_CHANGE"

    def test_determine_mode_intersection(self):
        """Mode should be INTERSECTION when a turn is nearby."""
        from planning.route_planner import TURN_RIGHT, RouteEntry
        route = [RouteEntry(
            waypoint=MockWaypoint(x=20, y=0),
            action=TURN_RIGHT, road_id=0, lane_id=-1,
            distance_from_start_m=0,
        )]
        vt = MockTransform(MockLocation(0, 0))
        mode = self.fusion._determine_mode(vt, route, lookahead_m=50)
        assert mode == "INTERSECTION"

    def test_generate_route_aware_mask_lane_keep(self):
        """In LANE_KEEP mode, should return DSUNet mask unchanged."""
        dsunet_mask = np.zeros((480, 640), dtype=np.uint8)
        dsunet_mask[200:, 200:440] = 255
        from planning.route_planner import LANE_KEEP, RouteEntry
        route = [RouteEntry(
            waypoint=MockWaypoint(x=100, y=0),
            action=LANE_KEEP, road_id=0, lane_id=-1,
            distance_from_start_m=0,
        )]
        vt = MockTransform(MockLocation(0, 0))
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        mask, mode = self.fusion.generate_route_aware_mask(
            rgb, dsunet_mask, vt, route,
        )
        assert mode == "LANE_KEEP"
        np.testing.assert_array_equal(mask, dsunet_mask)

    def test_waypoints_to_mask_shape(self):
        """_waypoints_to_mask should produce correct shape."""
        wps = [MockWaypoint(x=10, y=0), MockWaypoint(x=20, y=0)]
        vt = MockTransform(MockLocation(0, 0))
        mask = self.fusion._waypoints_to_mask(wps, vt, (480, 640))
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8

    def test_blend_masks(self):
        """_blend_masks should take union of two masks."""
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m1[:50, :] = 255
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m2[50:, :] = 255
        result = self.fusion._blend_masks(m1, m2)
        assert np.all(result == 255)  # All pixels should be 255


# ──────────────────────────────────────────────────────────────────────────────
# DSUNet Driving Evaluator tests (mocked)
# ──────────────────────────────────────────────────────────────────────────────

class TestDSUNetDrivingEvaluator:
    """Test DSUNetDrivingEvaluator with mocked CARLA."""

    def test_frame_metrics_dataclass(self):
        """FrameMetrics should be a proper dataclass."""
        from evaluation.dsunet_driving_eval import FrameMetrics
        m = FrameMetrics(frame_idx=0, timestamp=0.0)
        assert m.frame_idx == 0
        assert m.dsunet_iou == 0.0
        assert m.mode == "unknown"

    def test_evaluation_report_dataclass(self):
        """EvaluationReport should be a proper dataclass."""
        from evaluation.dsunet_driving_eval import EvaluationReport
        r = EvaluationReport()
        assert r.total_frames == 0
        assert r.mean_iou == 0.0

    def test_compute_iou_perfect_match(self):
        """IoU should be 1.0 for identical masks."""
        from evaluation.dsunet_driving_eval import DSUNetDrivingEvaluator
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50:, :] = 255
        # Create evaluator with mocked dependencies
        with patch.object(DSUNetDrivingEvaluator, '__init__', return_value=None):
            ev = DSUNetDrivingEvaluator.__new__(DSUNetDrivingEvaluator)
            iou, prec, rec = ev._compute_iou(mask, mask)
            assert iou == 1.0
            assert prec == 1.0
            assert rec == 1.0

    def test_compute_iou_no_overlap(self):
        """IoU should be 0.0 for non-overlapping masks."""
        from evaluation.dsunet_driving_eval import DSUNetDrivingEvaluator
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m1[:50, :] = 255
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m2[50:, :] = 255
        with patch.object(DSUNetDrivingEvaluator, '__init__', return_value=None):
            ev = DSUNetDrivingEvaluator.__new__(DSUNetDrivingEvaluator)
            iou, prec, rec = ev._compute_iou(m1, m2)
            assert iou == 0.0

    def test_compute_iou_partial_overlap(self):
        """IoU should be between 0 and 1 for partial overlap."""
        from evaluation.dsunet_driving_eval import DSUNetDrivingEvaluator
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m1[:, :60] = 255
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m2[:, 40:] = 255
        with patch.object(DSUNetDrivingEvaluator, '__init__', return_value=None):
            ev = DSUNetDrivingEvaluator.__new__(DSUNetDrivingEvaluator)
            iou, prec, rec = ev._compute_iou(m1, m2)
            assert 0 < iou < 1

    def test_generate_report_empty(self):
        """generate_report with no frames should return empty report."""
        from evaluation.dsunet_driving_eval import DSUNetDrivingEvaluator, EvaluationReport
        with patch.object(DSUNetDrivingEvaluator, '__init__', return_value=None):
            ev = DSUNetDrivingEvaluator.__new__(DSUNetDrivingEvaluator)
            ev.frame_metrics = []
            ev._start_location = None
            ev._prev_location = None
            report = ev.generate_report()
            assert report.total_frames == 0

    def test_generate_report_with_frames(self):
        """generate_report should aggregate metrics correctly."""
        from evaluation.dsunet_driving_eval import (
            DSUNetDrivingEvaluator, FrameMetrics,
        )
        with patch.object(DSUNetDrivingEvaluator, '__init__', return_value=None):
            ev = DSUNetDrivingEvaluator.__new__(DSUNetDrivingEvaluator)
            ev.target_speed_ms = 10.0
            ev.frame_metrics = [
                FrameMetrics(frame_idx=0, timestamp=0.0, dsunet_iou=0.8,
                             dsunet_precision=0.9, dsunet_recall=0.85,
                             dsunet_used=True, cte_m=0.1, heading_error_rad=0.01,
                             speed_ms=9.0),
                FrameMetrics(frame_idx=1, timestamp=0.1, dsunet_iou=0.9,
                             dsunet_precision=0.95, dsunet_recall=0.88,
                             dsunet_used=True, cte_m=0.2, heading_error_rad=0.02,
                             speed_ms=11.0),
            ]
            ev._start_location = None
            ev._prev_location = None
            report = ev.generate_report()
            assert report.total_frames == 2
            assert abs(report.mean_iou - 0.85) < 0.01
            assert abs(report.mean_cte - 0.15) < 0.01
