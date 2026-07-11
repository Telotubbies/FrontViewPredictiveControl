"""
Obstacle Avoidance — Potential Field / Artificial Potential Field (APF) method.

Adds repulsive forces from obstacles to generate a steering offset that steers
the vehicle around obstacles instead of just braking.  Designed to integrate
with the existing MPC + arbitrator stack:

    MPC output → APF steering offset → Arbitrator → CARLA

The algorithm:
  1. For each obstacle within detection range, compute a repulsive "force"
     that pushes the ego vehicle away laterally.
  2. Sum all repulsive forces → desired lateral shift (delta_y).
  3. Convert delta_y to a steering offset via a simple bicycle-model gain.
  4. Also reduce target speed proportionally to the closest obstacle distance.

Key parameters:
  - DETECTION_RANGE_M: how far ahead to look for obstacles (m)
  - LANE_HALF_WIDTH:   lateral boundary for avoidance (m)
  - REPULSIVE_GAIN:    strength of repulsive force
  - MIN_SAFE_DIST:     distance at which full avoidance is applied (m)
  - MAX_STEER_OFFSET:  cap on steering offset (CARLA [-1,1] units)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from bridge.obstacles import Obstacle, ObstacleType

logger = logging.getLogger(__name__)


# ── Tuning constants ────────────────────────────────────────────────────────
DETECTION_RANGE_M = 40.0       # look ahead up to 40 m
DETECTION_LATERAL_M = 6.0      # look ±6 m laterally
LANE_HALF_WIDTH = 1.75         # half lane width (m) — stay within ±1.75 m
REPULSIVE_GAIN = 3.0           # repulsive force gain
MIN_SAFE_DIST = 8.0            # full avoidance within 8 m
MAX_STEER_OFFSET = 0.3         # cap steering offset (CARLA units)
SPEED_REDUCTION_FACTOR = 0.6   # reduce target speed to 60% when avoiding
MIN_OBSTACLE_DIST_FOR_SLOW = 20.0  # slow down within 20 m


@dataclass
class AvoidanceResult:
    """Output of the obstacle avoidance planner."""
    active: bool = False
    steer_offset: float = 0.0       # additive steering offset (CARLA [-1,1])
    lateral_shift_m: float = 0.0    # desired lateral shift (m)
    target_speed_factor: float = 1.0  # multiplier for target speed
    closest_obstacle_dist: float = float('inf')
    avoidance_side: str = "none"    # "left", "right", or "none"
    num_obstacles: int = 0
    reason: str = ""


class ObstacleAvoidancePlanner:
    """
    Potential-field-based obstacle avoidance.

    Computes a steering offset and speed reduction to steer around obstacles
    detected ahead of the vehicle.  The offset is additive to the MPC steering
    output and is applied by the arbitrator.
    """

    def __init__(
        self,
        detection_range: float = DETECTION_RANGE_M,
        lateral_range: float = DETECTION_LATERAL_M,
        repulsive_gain: float = REPULSIVE_GAIN,
        min_safe_dist: float = MIN_SAFE_DIST,
        max_steer_offset: float = MAX_STEER_OFFSET,
    ) -> None:
        self.detection_range = detection_range
        self.lateral_range = lateral_range
        self.repulsive_gain = repulsive_gain
        self.min_safe_dist = min_safe_dist
        self.max_steer_offset = max_steer_offset
        self._last_result = AvoidanceResult()

    def compute_avoidance(
        self,
        obstacles: List[Obstacle],
        ego_x: float,
        ego_y: float,
        ego_heading: float,
        ego_speed: float,
        current_cte: float = 0.0,
    ) -> AvoidanceResult:
        """
        Compute obstacle avoidance steering offset and speed factor.

        Args:
            obstacles: list of detected obstacles (world-frame positions)
            ego_x, ego_y: ego vehicle position (world frame)
            ego_heading: ego heading (rad, 0 = +x)
            ego_speed: current speed (m/s)
            current_cte: current cross-track error (m) — used to bias avoidance

        Returns:
            AvoidanceResult with steer_offset, speed factor, etc.
        """
        result = AvoidanceResult()

        if not obstacles:
            self._last_result = result
            return result

        # Transform obstacles to ego frame (forward = +x, left = +y)
        cos_h = math.cos(ego_heading)
        sin_h = math.sin(ego_heading)

        # Collect relevant obstacles (in front, within range)
        relevant: List[Tuple[float, float, float, Obstacle]] = []  # (fwd, lat, dist, obs)
        for obs in obstacles:
            if obs.type not in (ObstacleType.VEHICLE, ObstacleType.BICYCLE,
                                ObstacleType.PEDESTRIAN, ObstacleType.UNKNOWN_MOVABLE):
                continue

            dx = obs.position[0] - ego_x
            dy = obs.position[1] - ego_y
            # Rotate to ego frame
            fwd = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            dist = math.sqrt(fwd * fwd + lat * lat)

            if fwd < 0 or fwd > self.detection_range:
                continue
            if abs(lat) > self.lateral_range:
                continue
            relevant.append((fwd, lat, dist, obs))

        result.num_obstacles = len(relevant)
        if not relevant:
            self._last_result = result
            return result

        # Find closest obstacle
        closest = min(relevant, key=lambda r: r[2])
        result.closest_obstacle_dist = closest[2]

        # ── Compute repulsive forces ────────────────────────────────────
        # Each obstacle pushes the ego vehicle laterally away.
        # Force magnitude decreases with distance (1/r² potential).
        total_lateral_force = 0.0
        for fwd, lat, dist, obs in relevant:
            # Distance-based scaling: stronger when closer
            if dist < 0.5:
                dist = 0.5  # avoid singularity
            # Forward attenuation: obstacles far ahead have less influence
            fwd_factor = max(0.0, 1.0 - fwd / self.detection_range)
            # Repulsive force: push away laterally
            # sign: if obstacle is on left (lat > 0), push right (negative)
            sign = -1.0 if lat > 0 else 1.0
            # If obstacle is directly ahead (lat ≈ 0), bias to the side with more room
            if abs(lat) < 0.5:
                # Choose side based on current CTE — steer away from lane edge
                sign = 1.0 if current_cte > 0 else -1.0

            force = self.repulsive_gain * fwd_factor * (self.min_safe_dist / dist) ** 2
            total_lateral_force += sign * force

        # Convert force to desired lateral shift (m)
        lateral_shift = total_lateral_force * 0.5  # scale factor
        # Clamp to lane boundary
        max_shift = LANE_HALF_WIDTH - 0.3  # keep 30 cm margin
        lateral_shift = max(-max_shift, min(max_shift, lateral_shift))

        result.lateral_shift_m = lateral_shift

        # ── Convert lateral shift to steering offset ─────────────────────
        # Use a simple proportional gain based on speed
        # At low speed, more steering is needed for the same lateral shift
        if ego_speed > 0.5:
            # Bicycle model: steer ≈ lateral_shift / (L * lookahead)
            lookahead = max(5.0, ego_speed * 1.0)  # lookahead ~ 1s
            steer_rad = lateral_shift / lookahead
            # Convert to CARLA [-1, 1] (max_steer ≈ 0.57 rad)
            steer_offset = steer_rad / 0.57
        else:
            # At very low speed, use direct mapping
            steer_offset = lateral_shift / max_shift * self.max_steer_offset

        steer_offset = max(-self.max_steer_offset,
                          min(self.max_steer_offset, steer_offset))
        result.steer_offset = steer_offset

        # ── Speed reduction ──────────────────────────────────────────────
        if result.closest_obstacle_dist < MIN_OBSTACLE_DIST_FOR_SLOW:
            # Proportional speed reduction: closer → slower
            ratio = result.closest_obstacle_dist / MIN_OBSTACLE_DIST_FOR_SLOW
            result.target_speed_factor = max(0.2, SPEED_REDUCTION_FACTOR * ratio + 0.3)
        else:
            result.target_speed_factor = 1.0

        # ── Determine avoidance side ─────────────────────────────────────
        if lateral_shift > 0.1:
            result.avoidance_side = "left"
        elif lateral_shift < -0.1:
            result.avoidance_side = "right"

        result.active = abs(steer_offset) > 0.02 or result.target_speed_factor < 1.0
        result.reason = (
            f"APF: {result.num_obstacles} obs, closest={result.closest_obstacle_dist:.1f}m, "
            f"shift={lateral_shift:.2f}m, steer_off={steer_offset:.3f}, "
            f"speed_factor={result.target_speed_factor:.2f}"
        )

        if result.active:
            logger.info(result.reason)

        self._last_result = result
        return result

    def get_status(self) -> dict:
        """Return current avoidance status for dashboard."""
        r = self._last_result
        return {
            "active": r.active,
            "steer_offset": r.steer_offset,
            "lateral_shift_m": r.lateral_shift_m,
            "target_speed_factor": r.target_speed_factor,
            "closest_obstacle_dist": r.closest_obstacle_dist,
            "avoidance_side": r.avoidance_side,
            "num_obstacles": r.num_obstacles,
        }
