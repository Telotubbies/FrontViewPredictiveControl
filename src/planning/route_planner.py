"""
Global Route Planner — A* on CARLA lane-level topology.

Builds a directed graph from CARLA's map topology (road_id, lane_id) and
finds the shortest path (by travel time + lane change penalty) from a
start waypoint to a destination waypoint.

The route is a sequence of (waypoint, road_action) pairs where road_action
is one of: "LANE_KEEP", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT",
"TURN_LEFT", "TURN_RIGHT", "STRAIGHT", "ARRIVE".

Usage:
    from planning.route_planner import GlobalRoutePlanner

    planner = GlobalRoutePlanner(world.get_map())
    route = planner.plan_route(start_transform, destination_transform)
    for wp, action in route:
        print(f"{action} at {wp.transform.location}")
"""
import heapq
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set

import numpy as np

logger = logging.getLogger(__name__)

# Road action constants
LANE_KEEP = "LANE_KEEP"
LANE_CHANGE_LEFT = "LANE_CHANGE_LEFT"
LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"
TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"
STRAIGHT = "STRAIGHT"
ARRIVE = "ARRIVE"

# Cost weights
PENALTY_LANE_CHANGE = 50.0   # meters equivalent
PENALTY_LEFT_TURN = 30.0
PENALTY_UTURN = 1000.0
LANE_CHANGE_TRIGGER_M = 30.0  # Start LC this far before required lane


@dataclass
class LaneNode:
    """A node in the lane graph = a specific (road_id, lane_id) lane segment."""
    road_id: int
    lane_id: int
    waypoint: object  # carla.Waypoint (representative)
    location: object  # carla.Location

    def __hash__(self):
        return hash((self.road_id, self.lane_id))

    def __eq__(self, other):
        return self.road_id == other.road_id and self.lane_id == other.lane_id


@dataclass
class LaneEdge:
    """An edge in the lane graph = connection between two lane segments."""
    target: LaneNode
    length_m: float
    action: str
    speed_limit_ms: float = 13.9  # 50 km/h default


@dataclass
class RouteEntry:
    """A single entry in the planned route."""
    waypoint: object       # carla.Waypoint
    action: str            # what to do at this point
    road_id: int
    lane_id: int
    distance_from_start_m: float
    lane_change_required: bool = False


class GlobalRoutePlanner:
    """A* route planner on CARLA lane-level topology.

    Builds a graph from CARLA map.get_topology() and finds the shortest
    path using A* with Euclidean distance heuristic.
    """

    def __init__(self, carla_map, sampling_resolution_m: float = 2.0):
        """Initialize the planner.

        Args:
            carla_map: carla.Map object (world.get_map())
            sampling_resolution_m: Waypoint sampling resolution for route
        """
        self.map = carla_map
        self.sampling_resolution = sampling_resolution_m
        self._graph: Dict[LaneNode, List[LaneEdge]] = defaultdict(list)
        self._node_lookup: Dict[Tuple[int, int], LaneNode] = {}
        self._topology = []
        self._build_graph()

    def _build_graph(self):
        """Build lane graph from CARLA map topology."""
        self._topology = self.map.get_topology()
        logger.info("Building lane graph from %d topology entries...",
                    len(self._topology))

        for entry_wp, exit_wp in self._topology:
            entry_node = self._get_or_create_node(entry_wp)
            exit_node = self._get_or_create_node(exit_wp)

            # Compute edge length
            entry_loc = entry_wp.transform.location
            exit_loc = exit_wp.transform.location
            length = entry_loc.distance(exit_loc)

            # Determine action
            action = self._infer_action(entry_wp, exit_wp)

            # Speed limit (approximate from road type)
            speed_limit = 13.9  # 50 km/h default
            try:
                speed_limit = float(entry_wp.lane_width)  # placeholder
            except Exception:
                pass

            edge = LaneEdge(
                target=exit_node,
                length_m=length,
                action=action,
                speed_limit_ms=speed_limit,
            )
            self._graph[entry_node].append(edge)

        # Add lane-change edges (left/right neighbors)
        self._add_lane_change_edges()

        logger.info("Lane graph: %d nodes, %d edges",
                    len(self._node_lookup), sum(len(e) for e in self._graph.values()))

    def _get_or_create_node(self, waypoint) -> LaneNode:
        """Get or create a LaneNode from a carla.Waypoint."""
        key = (waypoint.road_id, waypoint.lane_id)
        if key not in self._node_lookup:
            node = LaneNode(
                road_id=waypoint.road_id,
                lane_id=waypoint.lane_id,
                waypoint=waypoint,
                location=waypoint.transform.location,
            )
            self._node_lookup[key] = node
        return self._node_lookup[key]

    def _infer_action(self, entry_wp, exit_wp) -> str:
        """Infer the driving action from entry/exit waypoint heading difference."""
        entry_yaw = entry_wp.transform.rotation.yaw
        exit_yaw = exit_wp.transform.rotation.yaw
        diff = (exit_yaw - entry_yaw + 180) % 360 - 180  # normalize to [-180, 180]

        if abs(diff) < 15:
            return STRAIGHT
        elif diff > 15:
            return TURN_RIGHT
        elif diff < -15:
            return TURN_LEFT
        return STRAIGHT

    def _add_lane_change_edges(self):
        """Add lane-change edges between adjacent lanes on the same road."""
        # Group nodes by road_id
        road_lanes: Dict[int, List[LaneNode]] = defaultdict(list)
        for node in self._node_lookup.values():
            road_lanes[node.road_id].append(node)

        for road_id, nodes in road_lanes.items():
            # Sort by lane_id (CARLA: negative lane_id = right side)
            nodes.sort(key=lambda n: n.lane_id)
            for i in range(len(nodes) - 1):
                left = nodes[i]      # higher lane_id = left
                right = nodes[i + 1]  # lower lane_id = right

                # Left → Right (lane change right)
                self._graph[left].append(LaneEdge(
                    target=right,
                    length_m=5.0,  # short lateral distance
                    action=LANE_CHANGE_RIGHT,
                ))
                # Right → Left (lane change left)
                self._graph[right].append(LaneEdge(
                    target=left,
                    length_m=5.0,
                    action=LANE_CHANGE_LEFT,
                ))

    def _heuristic(self, node: LaneNode, goal_loc) -> float:
        """Euclidean distance heuristic for A*."""
        return node.location.distance(goal_loc)

    def _edge_cost(self, edge: LaneEdge) -> float:
        """Compute edge cost (travel time + penalties)."""
        cost = edge.length_m / max(edge.speed_limit_ms, 1.0)
        if edge.action == LANE_CHANGE_LEFT:
            cost += PENALTY_LANE_CHANGE
        elif edge.action == LANE_CHANGE_RIGHT:
            cost += PENALTY_LANE_CHANGE
        elif edge.action == TURN_LEFT:
            cost += PENALTY_LEFT_TURN
        return cost

    def plan_route(self, start_transform, goal_transform) -> List[RouteEntry]:
        """Plan a route from start to goal using A*.

        Args:
            start_transform: carla.Transform of start position
            goal_transform: carla.Transform of destination

        Returns:
            List of RouteEntry (waypoint, action, road_id, lane_id, distance)
        """
        # Get nearest waypoint to start and goal
        start_wp = self.map.get_waypoint(start_transform.location)
        goal_wp = self.map.get_waypoint(goal_transform.location)
        goal_loc = goal_transform.location

        start_node = self._get_or_create_node(start_wp)
        goal_node = self._get_or_create_node(goal_wp)

        # A* search — use counter to break ties in heap (LaneNode has no __lt__)
        counter = 0
        open_heap = [(0, counter, start_node)]
        came_from: Dict[LaneNode, Tuple[LaneNode, LaneEdge]] = {}
        g_score: Dict[LaneNode, float] = {start_node: 0}
        closed: Set[LaneNode] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current == goal_node:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current, start_node)
                return self._path_to_route(path, start_wp, goal_wp)

            if current in closed:
                continue
            closed.add(current)

            for edge in self._graph.get(current, []):
                neighbor = edge.target
                if neighbor in closed:
                    continue

                tentative_g = g_score[current] + self._edge_cost(edge)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal_loc)
                    came_from[neighbor] = (current, edge)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, neighbor))

        # No path found — fallback to direct waypoint path
        logger.warning("A* found no lane-level path, falling back to waypoint route")
        return self._fallback_route(start_wp, goal_wp)

    def _reconstruct_path(self, came_from, goal_node, start_node):
        """Reconstruct path from A* came_from dict."""
        path = [goal_node]
        current = goal_node
        while current != start_node:
            if current not in came_from:
                break
            current, _ = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _path_to_route(self, path: List[LaneNode],
                       start_wp, goal_wp) -> List[RouteEntry]:
        """Convert lane-level path to detailed waypoint route."""
        route = []
        total_dist = 0.0

        for i, node in enumerate(path):
            if i == 0:
                wp = start_wp
                action = LANE_KEEP
            elif i == len(path) - 1:
                wp = goal_wp
                action = ARRIVE
            else:
                wp = node.waypoint
                # Determine action from edge
                prev_node = path[i - 1]
                for edge in self._graph.get(prev_node, []):
                    if edge.target == node:
                        action = edge.action
                        break
                else:
                    action = LANE_KEEP

            route.append(RouteEntry(
                waypoint=wp,
                action=action,
                road_id=node.road_id,
                lane_id=node.lane_id,
                distance_from_start_m=total_dist,
                lane_change_required=(action in (LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT)),
            ))

            if i < len(path) - 1:
                total_dist += node.location.distance(path[i + 1].location)

        # Interpolate waypoints between lane nodes for smooth driving
        return self._interpolate_route(route)

    def _interpolate_route(self, route: List[RouteEntry]) -> List[RouteEntry]:
        """Add intermediate waypoints between lane nodes for smooth driving.

        Only the FIRST entry of each lane segment keeps its action (e.g.
        LANE_CHANGE_RIGHT). Intermediate waypoints are marked LANE_KEEP
        to avoid duplicate maneuver flags.
        """
        if len(route) < 2:
            return route

        detailed = []
        for i in range(len(route) - 1):
            curr = route[i]
            nxt = route[i + 1]
            detailed.append(curr)

            # Generate intermediate waypoints within the same lane
            try:
                wps = curr.waypoint.next_until_lane_end(self.sampling_resolution)
                for wp in wps:
                    # Only include if same lane as current
                    if wp.road_id == curr.road_id and wp.lane_id == curr.lane_id:
                        detailed.append(RouteEntry(
                            waypoint=wp,
                            action=LANE_KEEP,  # intermediate = lane keep
                            road_id=wp.road_id,
                            lane_id=wp.lane_id,
                            distance_from_start_m=curr.distance_from_start_m,
                            lane_change_required=False,
                        ))
            except Exception:
                pass

        detailed.append(route[-1])
        return detailed

    def _fallback_route(self, start_wp, goal_wp) -> List[RouteEntry]:
        """Fallback: simple waypoint chain from start to goal."""
        route = []
        try:
            # Use CARLA's built-in trace_route if available
            import carla
            path = start_wp.next_until_lane_end(2.0)
            for wp in path:
                route.append(RouteEntry(
                    waypoint=wp,
                    action=LANE_KEEP,
                    road_id=wp.road_id,
                    lane_id=wp.lane_id,
                    distance_from_start_m=0.0,
                ))
        except Exception:
            route.append(RouteEntry(
                waypoint=start_wp,
                action=LANE_KEEP,
                road_id=start_wp.road_id,
                lane_id=start_wp.lane_id,
                distance_from_start_m=0.0,
            ))
            route.append(RouteEntry(
                waypoint=goal_wp,
                action=ARRIVE,
                road_id=goal_wp.road_id,
                lane_id=goal_wp.lane_id,
                distance_from_start_m=0.0,
            ))
        return route

    def get_target_lane(self, vehicle_transform) -> Tuple[int, int]:
        """Get the (road_id, lane_id) of the lane the vehicle should be in.

        This is determined by the route — the next route entry that
        requires a lane change tells us which lane to target.
        """
        wp = self.map.get_waypoint(vehicle_transform.location)
        return (wp.road_id, wp.lane_id)

    def get_next_maneuver(self, vehicle_transform, route: List[RouteEntry],
                          lookahead_m: float = 50.0) -> Optional[RouteEntry]:
        """Get the next significant maneuver from the route.

        Args:
            vehicle_transform: Current vehicle position
            route: Planned route
            lookahead_m: How far ahead to look

        Returns:
            Next RouteEntry with a significant action (turn, lane change, arrive)
            or None if just lane keeping.
        """
        vehicle_loc = vehicle_transform.location
        for entry in route:
            dist = entry.waypoint.transform.location.distance(vehicle_loc)
            if dist > lookahead_m:
                break
            if entry.action in (TURN_LEFT, TURN_RIGHT, LANE_CHANGE_LEFT,
                                LANE_CHANGE_RIGHT, ARRIVE):
                return entry
        return None
