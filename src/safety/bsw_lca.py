"""
Blind Spot Warning (BSW) + Lane Change Assist (LCA).

BSW monitors the area behind the ego vehicle (left and right blind spots) and
raises an alert when another vehicle occupies a blind spot zone.

LCA extends this by predicting the future positions of surrounding vehicles
and warning when a fast-approaching vehicle will enter the blind spot within a
look-ahead window, making a lane change unsafe.

All geometry is performed in the ego frame (forward / lateral) using the same
convention as ``bridge.obstacles.classify_obstacle_risk``:
    forward  =  dx * cos_h + dy * sin_h
    lateral  = -dx * sin_h + dy * cos_h
where (dx, dy) is the obstacle position minus ego position in ENU and
``ego_heading`` is the ego heading in ENU radians.  Lateral positive = left.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from bridge.obstacles import Obstacle, ObstacleType

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class BSWConfig:
    """Configuration for Blind Spot Warning and Lane Change Assist."""

    blind_spot_range_m: float = 3.0
    """Distance behind the ego vehicle to monitor for the blind spot zone."""

    blind_spot_lateral_m: float = 1.5
    """Lateral half-width of the blind spot zone (applied to each side)."""

    lca_lookahead_s: float = 3.0
    """Look-ahead time used when predicting approaching vehicles for LCA."""

    warning_speed_threshold_ms: float = 5.0
    """BSW is only active when ego speed exceeds this value."""

    approach_speed_threshold_ms: float = 2.0
    """Closing speed (m/s) above which an approaching vehicle triggers LCA."""


# ── Alert levels ──────────────────────────────────────────────────────────────


class BSWAlert(IntEnum):
    """Blind spot / lane change alert levels."""

    CLEAR = 0
    BLIND_SPOT = 1  # vehicle currently in the blind spot
    APPROACHING = 2  # vehicle approaching fast from behind


# ── Helpers ───────────────────────────────────────────────────────────────────


def _transform_obstacle(
    obs: Obstacle,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    ego_speed: float,
) -> tuple[float, float, float, float]:
    """
    Transform an obstacle into the ego frame including ego motion.

    Returns ``(forward, lateral, rel_forward, rel_lateral)``.
    """
    ox, oy, _ = obs.position
    ovx, ovy, _ = obs.velocity

    dx = ox - ego_x
    dy = oy - ego_y
    cos_h = math.cos(ego_heading)
    sin_h = math.sin(ego_heading)

    forward = dx * cos_h + dy * sin_h
    lateral = -dx * sin_h + dy * cos_h

    # Ego velocity in ENU axes.
    ego_vx = ego_speed * cos_h
    ego_vy = ego_speed * sin_h
    rel_vx = ovx - ego_vx
    rel_vy = ovy - ego_vy
    rel_forward = rel_vx * cos_h + rel_vy * sin_h
    rel_lateral = -rel_vx * sin_h + rel_vy * cos_h

    return forward, lateral, rel_forward, rel_lateral


def _in_blind_spot(forward: float, lateral: float, cfg: BSWConfig) -> bool:
    """True if (forward, lateral) lies inside the blind spot zone."""
    return -cfg.blind_spot_range_m < forward < 0.0 and abs(lateral) < cfg.blind_spot_lateral_m


# ── Blind Spot Warning ────────────────────────────────────────────────────────


class BlindSpotWarning:
    """Monitor left/right blind spots and raise alerts."""

    def __init__(self, config: Optional[BSWConfig] = None) -> None:
        self.config = config if config is not None else BSWConfig()
        self._status: dict = {
            "left_alert": BSWAlert.CLEAR,
            "right_alert": BSWAlert.CLEAR,
            "left_obstacle_id": None,
            "right_obstacle_id": None,
            "safe_to_change_left": True,
            "safe_to_change_right": True,
        }

    def update(
        self,
        obstacles: list,
        ego_x: float,
        ego_y: float,
        ego_heading: float,
        ego_speed: float,
        timestamp: float,
    ) -> dict:
        """
        Evaluate the blind spot state given the latest obstacle list.

        Returns a dict with ``left_alert``, ``right_alert``,
        ``left_obstacle_id``, ``right_obstacle_id``, ``safe_to_change_left``
        and ``safe_to_change_right``.
        """
        left_alert = BSWAlert.CLEAR
        right_alert = BSWAlert.CLEAR
        left_id: Optional[int] = None
        right_id: Optional[int] = None

        # Inactive below the warning speed threshold.
        if ego_speed > self.config.warning_speed_threshold_ms:
            for obs in obstacles:
                if obs.type not in (ObstacleType.VEHICLE, ObstacleType.BICYCLE):
                    continue
                forward, lateral, _, _ = _transform_obstacle(
                    obs, ego_x, ego_y, ego_heading, ego_speed
                )
                if not _in_blind_spot(forward, lateral, self.config):
                    continue
                if lateral > 0.0:
                    # Left side.
                    if left_alert == BSWAlert.CLEAR:
                        left_alert = BSWAlert.BLIND_SPOT
                        left_id = obs.id
                else:
                    # Right side.
                    if right_alert == BSWAlert.CLEAR:
                        right_alert = BSWAlert.BLIND_SPOT
                        right_id = obs.id

        self._status = {
            "left_alert": left_alert,
            "right_alert": right_alert,
            "left_obstacle_id": left_id,
            "right_obstacle_id": right_id,
            "safe_to_change_left": left_alert == BSWAlert.CLEAR,
            "safe_to_change_right": right_alert == BSWAlert.CLEAR,
        }
        return self._status

    def get_status(self) -> dict:
        """Return the most recent alert state."""
        return self._status

    def reset(self) -> None:
        """Clear all alerts."""
        self._status = {
            "left_alert": BSWAlert.CLEAR,
            "right_alert": BSWAlert.CLEAR,
            "left_obstacle_id": None,
            "right_obstacle_id": None,
            "safe_to_change_left": True,
            "safe_to_change_right": True,
        }


# ── Lane Change Assist ────────────────────────────────────────────────────────


class LaneChangeAssist:
    """Evaluate the safety of a lane change manoeuvre."""

    def __init__(self, config: Optional[BSWConfig] = None) -> None:
        self.config = config if config is not None else BSWConfig()
        self._debug: dict = {
            "last_target_lane": None,
            "last_risk_level": "low",
            "last_ttc": float("inf"),
        }

    def evaluate_lane_change(
        self,
        obstacles: list,
        ego_x: float,
        ego_y: float,
        ego_heading: float,
        ego_speed: float,
        target_lane: str,
    ) -> dict:
        """
        Evaluate whether a lane change toward ``target_lane`` is safe.

        ``target_lane`` must be ``"left"`` or ``"right"``.

        Returns a dict with ``safe`` (bool), ``risk_level``
        (``"low"``/``"medium"``/``"high"``), ``reason`` (str) and
        ``time_to_collision`` (float, ``inf`` if none).
        """
        if target_lane not in ("left", "right"):
            raise ValueError(f"target_lane must be 'left' or 'right', got {target_lane!r}")

        side_sign = 1.0 if target_lane == "left" else -1.0
        cfg = self.config

        risk_level = "low"
        reason = "clear"
        ttc = float("inf")
        in_blind_now = False
        approaching = False

        for obs in obstacles:
            if obs.type not in (ObstacleType.VEHICLE, ObstacleType.BICYCLE):
                continue
            forward, lateral, rel_forward, rel_lateral = _transform_obstacle(
                obs, ego_x, ego_y, ego_heading, ego_speed
            )

            # Only consider obstacles on the target side.
            if side_sign > 0.0 and lateral <= 0.0:
                continue
            if side_sign < 0.0 and lateral >= 0.0:
                continue

            # High risk: vehicle currently in the blind spot.
            if _in_blind_spot(forward, lateral, cfg):
                in_blind_now = True
                # TTC for a vehicle already in the blind spot is effectively 0.
                ttc = 0.0
                reason = f"vehicle {obs.id} in {target_lane} blind spot"
                break

            # Medium risk: vehicle approaching from behind that will enter the
            # blind spot within the look-ahead window.
            if rel_forward > cfg.approach_speed_threshold_ms and forward < 0.0:
                pred_forward = forward + rel_forward * cfg.lca_lookahead_s
                pred_lateral = lateral + rel_lateral * cfg.lca_lookahead_s
                if _in_blind_spot(pred_forward, pred_lateral, cfg) or (
                    pred_forward >= -cfg.blind_spot_range_m
                ):
                    # Time to reach the trailing edge of the blind spot
                    # (forward == -blind_spot_range_m) or the ego position.
                    if rel_forward > 1e-6:
                        t_enter = (-cfg.blind_spot_range_m - forward) / rel_forward
                        t_ego = (0.0 - forward) / rel_forward
                        cand = t_enter if t_enter > 0.0 else t_ego
                        if cand < ttc:
                            ttc = cand
                    approaching = True
                    reason = f"vehicle {obs.id} approaching from {target_lane} rear"

        if in_blind_now:
            risk_level = "high"
        elif approaching:
            risk_level = "medium"
        else:
            risk_level = "low"
            reason = "clear"
            ttc = float("inf")

        safe = risk_level == "low"

        self._debug = {
            "last_target_lane": target_lane,
            "last_risk_level": risk_level,
            "last_ttc": ttc,
        }

        return {
            "safe": safe,
            "risk_level": risk_level,
            "reason": reason,
            "time_to_collision": ttc,
        }

    def get_debug_info(self) -> dict:
        """Return debug info from the last evaluation."""
        return self._debug
