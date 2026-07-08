"""
Lane Compliance Metrics — วัดว่ารถอยู่ใน lane หรือไม่
Tracks CTE, heading error, lane boundary distance, in-lane percentage, departure events.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LaneComplianceSummary:
    """Summary statistics for lane compliance."""
    # CTE (meters)
    cte_current: float = 0.0
    cte_min: float = 0.0
    cte_max: float = 0.0
    cte_mean: float = 0.0
    cte_std: float = 0.0
    cte_rms: float = 0.0
    # Heading error
    heading_current_rad: float = 0.0
    heading_current_deg: float = 0.0
    heading_min_deg: float = 0.0
    heading_max_deg: float = 0.0
    heading_mean_deg: float = 0.0
    heading_std_deg: float = 0.0
    # Lane boundary distances (meters)
    dist_left_edge: float = 0.0
    dist_right_edge: float = 0.0
    # In-lane statistics
    in_lane_percentage: float = 0.0
    lane_center_deviation_mean: float = 0.0
    lane_center_deviation_max: float = 0.0
    # Departure events
    departure_count: int = 0
    departure_total_duration_s: float = 0.0
    departure_max_duration_s: float = 0.0
    # Timing
    total_frames: int = 0
    total_duration_s: float = 0.0


class LaneComplianceMetrics:
    """
    Real-time lane compliance measurement.

    Tracks whether the vehicle stays within lane boundaries using CTE
    (Cross-Track Error) and heading error from the control pipeline.
    """

    def __init__(
        self,
        lane_half_width_m: float = 1.75,
        departure_threshold_m: float = 1.75,
        history_size: int = 10000,
    ):
        """
        Args:
            lane_half_width_m: half-width of lane (meters). Default 1.75 (3.5m lane)
            departure_threshold_m: CTE threshold beyond which vehicle is "departed"
            history_size: max frames to store
        """
        self.lane_half_width_m = lane_half_width_m
        self.departure_threshold_m = departure_threshold_m

        # Time series storage
        self._cte_history: deque = deque(maxlen=history_size)
        self._heading_history: deque = deque(maxlen=history_size)
        self._dist_left_history: deque = deque(maxlen=history_size)
        self._dist_right_history: deque = deque(maxlen=history_size)
        self._timestamps: deque = deque(maxlen=history_size)

        # Departure tracking
        self._in_departure = False
        self._departure_start_time: Optional[float] = None
        self._departure_count = 0
        self._departure_durations: List[float] = []
        self._departure_max_duration = 0.0

        # Frame counter
        self._frame_count = 0
        self._start_time: Optional[float] = None

    def update(
        self,
        cte_m: float,
        heading_rad: float,
        lane_half_width_m: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update metrics with current frame data.

        Args:
            cte_m: cross-track error in meters (positive = right of center)
            heading_rad: heading error in radians
            lane_half_width_m: optional override for lane half-width
            timestamp: optional timestamp (defaults to time.time())

        Returns:
            dict with current frame metrics
        """
        if timestamp is None:
            timestamp = time.time()
        if self._start_time is None:
            self._start_time = timestamp
        if lane_half_width_m is not None:
            self.lane_half_width_m = lane_half_width_m

        # Compute lane boundary distances
        # CTE positive = right of center, so:
        #   dist_to_right_edge = lane_half_width - cte
        #   dist_to_left_edge = lane_half_width + cte
        dist_right = self.lane_half_width_m - cte_m
        dist_left = self.lane_half_width_m + cte_m

        # Store in history
        self._cte_history.append(cte_m)
        self._heading_history.append(heading_rad)
        self._dist_left_history.append(dist_left)
        self._dist_right_history.append(dist_right)
        self._timestamps.append(timestamp)
        self._frame_count += 1

        # Check departure
        is_departed = abs(cte_m) > self.departure_threshold_m
        if is_departed and not self._in_departure:
            # Start of departure
            self._in_departure = True
            self._departure_start_time = timestamp
            self._departure_count += 1
        elif not is_departed and self._in_departure:
            # End of departure
            self._in_departure = False
            if self._departure_start_time is not None:
                duration = timestamp - self._departure_start_time
                self._departure_durations.append(duration)
                self._departure_max_duration = max(self._departure_max_duration, duration)

        return {
            "cte_m": cte_m,
            "heading_deg": np.degrees(heading_rad),
            "dist_left_edge_m": dist_left,
            "dist_right_edge_m": dist_right,
            "in_lane": not is_departed,
            "departure_active": self._in_departure,
        }

    def get_summary(self) -> LaneComplianceSummary:
        """Compute summary statistics from collected data."""
        if not self._cte_history:
            return LaneComplianceSummary()

        cte_arr = np.array(self._cte_history)
        head_arr = np.array(self._heading_history)
        dist_left_arr = np.array(self._dist_left_history)
        dist_right_arr = np.array(self._dist_right_history)

        in_lane_mask = np.abs(cte_arr) <= self.departure_threshold_m
        in_lane_pct = float(np.mean(in_lane_mask) * 100.0)

        total_duration = 0.0
        if len(self._timestamps) > 1:
            total_duration = self._timestamps[-1] - self._timestamps[0]

        departure_total = sum(self._departure_durations)

        return LaneComplianceSummary(
            cte_current=float(cte_arr[-1]),
            cte_min=float(np.min(cte_arr)),
            cte_max=float(np.max(cte_arr)),
            cte_mean=float(np.mean(cte_arr)),
            cte_std=float(np.std(cte_arr)),
            cte_rms=float(np.sqrt(np.mean(cte_arr ** 2))),
            heading_current_rad=float(head_arr[-1]),
            heading_current_deg=float(np.degrees(head_arr[-1])),
            heading_min_deg=float(np.degrees(np.min(head_arr))),
            heading_max_deg=float(np.degrees(np.max(head_arr))),
            heading_mean_deg=float(np.degrees(np.mean(head_arr))),
            heading_std_deg=float(np.degrees(np.std(head_arr))),
            dist_left_edge=float(dist_left_arr[-1]),
            dist_right_edge=float(dist_right_arr[-1]),
            in_lane_percentage=in_lane_pct,
            lane_center_deviation_mean=float(np.mean(np.abs(cte_arr))),
            lane_center_deviation_max=float(np.max(np.abs(cte_arr))),
            departure_count=self._departure_count,
            departure_total_duration_s=departure_total,
            departure_max_duration_s=self._departure_max_duration,
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
                "timestamp", "cte_m", "heading_rad", "heading_deg",
                "dist_left_edge_m", "dist_right_edge_m", "in_lane"
            ])
            for i in range(len(self._cte_history)):
                cte = self._cte_history[i]
                head = self._heading_history[i]
                dl = self._dist_left_history[i]
                dr = self._dist_right_history[i]
                ts = self._timestamps[i]
                in_lane = abs(cte) <= self.departure_threshold_m
                writer.writerow([
                    f"{ts:.6f}", f"{cte:.6f}", f"{head:.6f}",
                    f"{np.degrees(head):.4f}", f"{dl:.6f}", f"{dr:.6f}",
                    int(in_lane)
                ])
        logger.info(f"Lane compliance CSV exported to {path}")

    def export_json(self, filepath: str) -> None:
        """Export summary statistics to JSON."""
        import json
        summary = self.get_summary()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary.__dict__, f, indent=2)
        logger.info(f"Lane compliance JSON exported to {path}")

    def reset(self) -> None:
        """Reset all collected data."""
        self._cte_history.clear()
        self._heading_history.clear()
        self._dist_left_history.clear()
        self._dist_right_history.clear()
        self._timestamps.clear()
        self._in_departure = False
        self._departure_start_time = None
        self._departure_count = 0
        self._departure_durations.clear()
        self._departure_max_duration = 0.0
        self._frame_count = 0
        self._start_time = None
