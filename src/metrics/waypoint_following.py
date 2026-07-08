"""
Waypoint Following Metrics — วัดว่ารถตาม waypoints หรือไม่
Tracks CTE vs waypoints, along-track progress, path deviation, speed tracking.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WaypointFollowingSummary:
    """Summary statistics for waypoint following."""
    # CTE vs waypoints (meters)
    cte_current: float = 0.0
    cte_min: float = 0.0
    cte_max: float = 0.0
    cte_mean: float = 0.0
    cte_std: float = 0.0
    cte_rms: float = 0.0
    # Along-track progress
    distance_traveled_m: float = 0.0
    current_waypoint_idx: int = 0
    # Path deviation
    path_deviation_mean: float = 0.0
    path_deviation_max: float = 0.0
    # Waypoint reach rate
    waypoints_reached: int = 0
    waypoint_reach_rate: float = 0.0  # waypoints per second
    # Speed tracking
    speed_actual_ms: float = 0.0
    speed_target_ms: float = 0.0
    speed_error_rms: float = 0.0
    speed_error_mean: float = 0.0
    # Heading alignment
    heading_alignment_deg: float = 0.0
    heading_alignment_mean_deg: float = 0.0
    # Position accuracy
    position_error_m: float = 0.0
    position_error_rms: float = 0.0
    # Timing
    total_frames: int = 0
    total_duration_s: float = 0.0


class WaypointFollowingMetrics:
    """
    Real-time waypoint following measurement.

    Tracks how well the vehicle follows a sequence of waypoints
    using position, speed, and heading data from CARLA.
    """

    def __init__(
        self,
        waypoint_spacing_m: float = 2.0,
        reach_radius_m: float = 3.0,
        history_size: int = 10000,
    ):
        """
        Args:
            waypoint_spacing_m: expected spacing between waypoints (meters)
            reach_radius_m: radius within which a waypoint is considered "reached"
            history_size: max frames to store
        """
        self.waypoint_spacing_m = waypoint_spacing_m
        self.reach_radius_m = reach_radius_m

        # Time series storage
        self._cte_history: deque = deque(maxlen=history_size)
        self._position_history: deque = deque(maxlen=history_size)  # (x, y)
        self._speed_actual_history: deque = deque(maxlen=history_size)
        self._speed_target_history: deque = deque(maxlen=history_size)
        self._heading_alignment_history: deque = deque(maxlen=history_size)
        self._position_error_history: deque = deque(maxlen=history_size)
        self._timestamps: deque = deque(maxlen=history_size)

        # Waypoint tracking
        self._waypoints: List[Tuple[float, float]] = []
        self._current_wp_idx = 0
        self._waypoints_reached = 0
        self._last_reach_time: Optional[float] = None

        # Distance tracking
        self._last_position: Optional[Tuple[float, float]] = None
        self._total_distance = 0.0

        # Frame counter
        self._frame_count = 0
        self._start_time: Optional[float] = None

    def set_waypoints(self, waypoints: List[Tuple[float, float]]) -> None:
        """Set the reference waypoint path (world coordinates)."""
        self._waypoints = list(waypoints)
        self._current_wp_idx = 0
        self._waypoints_reached = 0
        self._last_reach_time = None

    def update(
        self,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_heading_rad: float,
        speed_actual_ms: float,
        speed_target_ms: float,
        waypoints: Optional[List[Tuple[float, float]]] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update metrics with current frame data.

        Args:
            vehicle_x, vehicle_y: vehicle position in world frame
            vehicle_heading_rad: vehicle heading in radians
            speed_actual_ms: actual vehicle speed (m/s)
            speed_target_ms: target speed (m/s)
            waypoints: optional updated waypoint list
            timestamp: optional timestamp

        Returns:
            dict with current frame metrics
        """
        if timestamp is None:
            timestamp = time.time()
        if self._start_time is None:
            self._start_time = timestamp
        if waypoints is not None:
            self.set_waypoints(waypoints)

        # Distance traveled
        if self._last_position is not None:
            dx = vehicle_x - self._last_position[0]
            dy = vehicle_y - self._last_position[1]
            self._total_distance += float(np.hypot(dx, dy))
        self._last_position = (vehicle_x, vehicle_y)

        # Compute CTE and path deviation
        cte_m = 0.0
        position_error = 0.0
        path_tangent = vehicle_heading_rad
        nearest_idx = 0

        if self._waypoints:
            wp_arr = np.array(self._waypoints)
            veh_pos = np.array([vehicle_x, vehicle_y])
            dists = np.linalg.norm(wp_arr - veh_pos, axis=1)
            nearest_idx = int(np.argmin(dists))
            position_error = float(dists[nearest_idx])

            # Compute CTE using nearest waypoint and next waypoint
            if nearest_idx < len(self._waypoints) - 1:
                p1 = np.array(self._waypoints[nearest_idx])
                p2 = np.array(self._waypoints[nearest_idx + 1])
                seg = p2 - p1
                seg_len = np.linalg.norm(seg)
                if seg_len > 1e-6:
                    seg_dir = seg / seg_len
                    rel = veh_pos - p1
                    # Lateral distance (signed)
                    cte_m = float(rel[0] * seg_dir[1] - rel[1] * seg_dir[0])
                    path_tangent = float(np.arctan2(seg_dir[1], seg_dir[0]))
            elif nearest_idx > 0:
                p1 = np.array(self._waypoints[nearest_idx - 1])
                p2 = np.array(self._waypoints[nearest_idx])
                seg = p2 - p1
                seg_len = np.linalg.norm(seg)
                if seg_len > 1e-6:
                    seg_dir = seg / seg_len
                    rel = veh_pos - p1
                    cte_m = float(rel[0] * seg_dir[1] - rel[1] * seg_dir[0])
                    path_tangent = float(np.arctan2(seg_dir[1], seg_dir[0]))

            # Check waypoint reach
            if dists[nearest_idx] < self.reach_radius_m and nearest_idx >= self._current_wp_idx:
                if self._current_wp_idx < nearest_idx or self._last_reach_time is None:
                    self._waypoints_reached += (nearest_idx - self._current_wp_idx + 1)
                    self._current_wp_idx = nearest_idx + 1
                    self._last_reach_time = timestamp

        # Heading alignment
        heading_diff = float(np.arctan2(
            np.sin(path_tangent - vehicle_heading_rad),
            np.cos(path_tangent - vehicle_heading_rad)
        ))

        # Speed error
        speed_error = speed_actual_ms - speed_target_ms

        # Store in history
        self._cte_history.append(cte_m)
        self._position_history.append((vehicle_x, vehicle_y))
        self._speed_actual_history.append(speed_actual_ms)
        self._speed_target_history.append(speed_target_ms)
        self._heading_alignment_history.append(heading_diff)
        self._position_error_history.append(position_error)
        self._timestamps.append(timestamp)
        self._frame_count += 1

        return {
            "cte_m": cte_m,
            "position_error_m": position_error,
            "heading_alignment_deg": float(np.degrees(heading_diff)),
            "speed_error_ms": speed_error,
            "current_waypoint_idx": self._current_wp_idx,
            "waypoints_reached": self._waypoints_reached,
            "distance_traveled_m": self._total_distance,
        }

    def get_summary(self) -> WaypointFollowingSummary:
        """Compute summary statistics from collected data."""
        if not self._cte_history:
            return WaypointFollowingSummary()

        cte_arr = np.array(self._cte_history)
        pos_err_arr = np.array(self._position_error_history)
        speed_actual_arr = np.array(self._speed_actual_history)
        speed_target_arr = np.array(self._speed_target_history)
        head_align_arr = np.array(self._heading_alignment_history)
        speed_error_arr = speed_actual_arr - speed_target_arr

        total_duration = 0.0
        if len(self._timestamps) > 1:
            total_duration = self._timestamps[-1] - self._timestamps[0]

        reach_rate = self._waypoints_reached / total_duration if total_duration > 0 else 0.0

        return WaypointFollowingSummary(
            cte_current=float(cte_arr[-1]),
            cte_min=float(np.min(cte_arr)),
            cte_max=float(np.max(cte_arr)),
            cte_mean=float(np.mean(cte_arr)),
            cte_std=float(np.std(cte_arr)),
            cte_rms=float(np.sqrt(np.mean(cte_arr ** 2))),
            distance_traveled_m=self._total_distance,
            current_waypoint_idx=self._current_wp_idx,
            path_deviation_mean=float(np.mean(np.abs(cte_arr))),
            path_deviation_max=float(np.max(np.abs(cte_arr))),
            waypoints_reached=self._waypoints_reached,
            waypoint_reach_rate=reach_rate,
            speed_actual_ms=float(speed_actual_arr[-1]),
            speed_target_ms=float(speed_target_arr[-1]),
            speed_error_rms=float(np.sqrt(np.mean(speed_error_arr ** 2))),
            speed_error_mean=float(np.mean(speed_error_arr)),
            heading_alignment_deg=float(np.degrees(head_align_arr[-1])),
            heading_alignment_mean_deg=float(np.degrees(np.mean(head_align_arr))),
            position_error_m=float(pos_err_arr[-1]),
            position_error_rms=float(np.sqrt(np.mean(pos_err_arr ** 2))),
            total_frames=self._frame_count,
            total_duration_s=total_duration,
        )

    def export_csv(self, filepath: str) -> None:
        """Export time series data to CSV."""
        if not self._cte_history:
            logger.warning("No data to export")
            return

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "vehicle_x", "vehicle_y", "cte_m",
                "position_error_m", "heading_alignment_deg",
                "speed_actual_ms", "speed_target_ms", "speed_error_ms",
                "current_wp_idx", "waypoints_reached", "distance_traveled_m"
            ])
            for i in range(len(self._cte_history)):
                ts = self._timestamps[i]
                px, py = self._position_history[i]
                cte = self._cte_history[i]
                pe = self._position_error_history[i]
                ha = np.degrees(self._heading_alignment_history[i])
                sa = self._speed_actual_history[i]
                st = self._speed_target_history[i]
                writer.writerow([
                    f"{ts:.6f}", f"{px:.6f}", f"{py:.6f}", f"{cte:.6f}",
                    f"{pe:.6f}", f"{ha:.4f}", f"{sa:.6f}", f"{st:.6f}",
                    f"{sa - st:.6f}", self._current_wp_idx,
                    self._waypoints_reached, f"{self._total_distance:.6f}"
                ])
        logger.info(f"Waypoint following CSV exported to {path}")

    def export_json(self, filepath: str) -> None:
        """Export summary statistics to JSON."""
        import json
        summary = self.get_summary()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary.__dict__, f, indent=2)
        logger.info(f"Waypoint following JSON exported to {path}")

    def reset(self) -> None:
        """Reset all collected data."""
        self._cte_history.clear()
        self._position_history.clear()
        self._speed_actual_history.clear()
        self._speed_target_history.clear()
        self._heading_alignment_history.clear()
        self._position_error_history.clear()
        self._timestamps.clear()
        self._waypoints.clear()
        self._current_wp_idx = 0
        self._waypoints_reached = 0
        self._last_reach_time = None
        self._last_position = None
        self._total_distance = 0.0
        self._frame_count = 0
        self._start_time = None
