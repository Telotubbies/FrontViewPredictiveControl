"""
Traffic Sign Recognition (TSR) + Traffic Light Detection.

Detects traffic signs and traffic lights from the CARLA world, maps them to
high-level enums/dataclasses, and provides a controller that decides throttle
/ brake / target-speed overrides when approaching lights.

The CARLA-dependent helpers (``detect_from_carla``, ``get_stop_line_distance``)
are wrapped in try/except so the module remains importable without CARLA.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional


# ── Enums ───────────────────────────────────────────────────────────────────


class TrafficLightState(IntEnum):
    """Discrete state of a traffic light."""

    UNKNOWN = 0
    RED = 1
    YELLOW = 2
    GREEN = 3


class TrafficSignType(IntEnum):
    """High-level traffic sign category."""

    UNKNOWN = 0
    SPEED_LIMIT = 1
    STOP = 2
    YIELD = 3
    NO_ENTRY = 4


# ── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class TrafficSign:
    """A single detected traffic sign."""

    sign_type: TrafficSignType
    value: float = 0.0  # speed limit value (km/h) for SPEED_LIMIT
    distance_m: float = 0.0  # distance to sign
    confidence: float = 0.0


@dataclass
class TrafficLight:
    """A single detected traffic light."""

    state: TrafficLightState
    distance_m: float
    confidence: float


# ── Recognizer ──────────────────────────────────────────────────────────────


# Mapping from CARLA TrafficSign type_id substrings to our sign type.
# CARLA sign ids look like "road.traffic.speed_limit.30", "road.traffic.stop",
# "road.traffic.yield", "road.traffic.noentry".
_CARLA_SIGN_TYPE_MAP = {
    "speed_limit": TrafficSignType.SPEED_LIMIT,
    "stop": TrafficSignType.STOP,
    "yield": TrafficSignType.YIELD,
    "noentry": TrafficSignType.NO_ENTRY,
}


class TrafficSignRecognizer:
    """Detect traffic signs / lights from the CARLA world."""

    def __init__(
        self,
        min_confidence: float = 0.5,
        max_detection_range_m: float = 80.0,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_detection_range_m = max_detection_range_m

    # ── CARLA queries ────────────────────────────────────────────────────

    def detect_from_carla(self, world, vehicle_transform) -> list:
        """
        Query the CARLA world for traffic signs and lights within range.

        Returns a list of :class:`TrafficSign` / :class:`TrafficLight` objects.
        Returns an empty list on any error (e.g. CARLA not available).
        """
        try:
            results: list = []

            # Vehicle location
            v_loc = vehicle_transform.location
            vx, vy, vz = v_loc.x, v_loc.y, v_loc.z

            # --- Traffic lights ---
            for actor in world.get_actors():
                tl = self._as_traffic_light(actor)
                if tl is None:
                    continue
                t_loc = tl.get_location()
                dist = math.sqrt(
                    (t_loc.x - vx) ** 2
                    + (t_loc.y - vy) ** 2
                    + (t_loc.z - vz) ** 2
                )
                if dist >= self.max_detection_range_m:
                    continue
                state = self._carla_light_state(tl.get_state())
                results.append(
                    TrafficLight(
                        state=state,
                        distance_m=dist,
                        confidence=1.0,
                    )
                )

            # --- Traffic signs ---
            for actor in world.get_actors():
                ts = self._as_traffic_sign(actor)
                if ts is None:
                    continue
                t_loc = ts.get_transform().location
                dist = math.sqrt(
                    (t_loc.x - vx) ** 2
                    + (t_loc.y - vy) ** 2
                    + (t_loc.z - vz) ** 2
                )
                if dist >= self.max_detection_range_m:
                    continue
                sign_type, value = self._parse_carla_sign(ts.type_id)
                results.append(
                    TrafficSign(
                        sign_type=sign_type,
                        value=value,
                        distance_m=dist,
                        confidence=1.0,
                    )
                )

            return results
        except Exception:
            return []

    # ── Pure helpers (no CARLA needed) ───────────────────────────────────

    def detect_speed_limit(self, signs: List[TrafficSign]) -> Optional[float]:
        """Return the closest SPEED_LIMIT sign's value (km/h) or None."""
        closest: Optional[TrafficSign] = None
        for s in signs:
            if s.sign_type != TrafficSignType.SPEED_LIMIT:
                continue
            if s.confidence < self.min_confidence:
                continue
            if closest is None or s.distance_m < closest.distance_m:
                closest = s
        if closest is None:
            return None
        return closest.value

    def detect_traffic_light(
        self, lights: List[TrafficLight]
    ) -> Optional[TrafficLight]:
        """Return the closest traffic light or None."""
        closest: Optional[TrafficLight] = None
        for tl in lights:
            if tl.confidence < self.min_confidence:
                continue
            if closest is None or tl.distance_m < closest.distance_m:
                closest = tl
        return closest

    def get_stop_line_distance(
        self, world, vehicle_transform
    ) -> Optional[float]:
        """
        Distance (m) to the stop line of the closest upcoming red light.

        Returns None if there is no red light within range. Wrapped in
        try/except since it depends on CARLA.
        """
        try:
            v_loc = vehicle_transform.location
            vx, vy, vz = v_loc.x, v_loc.y, v_loc.z
            best: Optional[float] = None
            for actor in world.get_actors():
                tl = self._as_traffic_light(actor)
                if tl is None:
                    continue
                if self._carla_light_state(tl.get_state()) != TrafficLightState.RED:
                    continue
                stop_dist = tl.get_stop_line_distance()
                # get_stop_line_distance may be relative to the light; compute
                # straight-line distance to the light as a fallback estimate.
                t_loc = tl.get_location()
                dist = math.sqrt(
                    (t_loc.x - vx) ** 2
                    + (t_loc.y - vy) ** 2
                    + (t_loc.z - vz) ** 2
                )
                if dist >= self.max_detection_range_m:
                    continue
                # Prefer the CARLA-reported stop-line distance when available,
                # otherwise fall back to the light distance.
                candidate = stop_dist if stop_dist and stop_dist > 0 else dist
                if best is None or candidate < best:
                    best = candidate
            return best
        except Exception:
            return None

    # ── Internal CARLA adapters ──────────────────────────────────────────

    @staticmethod
    def _as_traffic_light(actor):
        """Return the actor as a CARLA TrafficLight if it is one, else None."""
        try:
            # CARLA TrafficLight exposes get_state(); duck-typed check.
            if hasattr(actor, "get_state") and hasattr(actor, "get_location"):
                return actor
        except Exception:
            pass
        return None

    @staticmethod
    def _as_traffic_sign(actor):
        """Return the actor as a CARLA TrafficSign if it is one, else None."""
        try:
            if hasattr(actor, "type_id") and "traffic" in str(
                getattr(actor, "type_id", "")
            ):
                # Exclude traffic lights (they also contain "traffic").
                if hasattr(actor, "get_state"):
                    return None
                if hasattr(actor, "get_transform"):
                    return actor
        except Exception:
            pass
        return None

    @staticmethod
    def _carla_light_state(carla_state) -> TrafficLightState:
        """Map a CARLA TrafficLightState to our :class:`TrafficLightState`."""
        try:
            # CARLA enum names: Red / Yellow / Green
            name = str(carla_state).split(".")[-1].lower()
        except Exception:
            return TrafficLightState.UNKNOWN
        if "red" in name:
            return TrafficLightState.RED
        if "yellow" in name:
            return TrafficLightState.YELLOW
        if "green" in name:
            return TrafficLightState.GREEN
        return TrafficLightState.UNKNOWN

    @staticmethod
    def _parse_carla_sign(type_id: str):
        """
        Map a CARLA sign type_id to (TrafficSignType, value).

        e.g. "road.traffic.speed_limit.30" -> (SPEED_LIMIT, 30.0)
        """
        tid = str(type_id).lower()
        for key, sign_type in _CARLA_SIGN_TYPE_MAP.items():
            if key in tid:
                value = 0.0
                if sign_type == TrafficSignType.SPEED_LIMIT:
                    # trailing number is the limit
                    parts = tid.split(".")
                    for part in reversed(parts):
                        try:
                            value = float(part)
                            break
                        except ValueError:
                            continue
                return sign_type, value
        return TrafficSignType.UNKNOWN, 0.0


# ── Controller ──────────────────────────────────────────────────────────────


class TrafficLightController:
    """Decide throttle/brake/target-speed overrides for traffic lights."""

    def __init__(
        self,
        stop_distance_m: float = 5.0,
        slow_distance_m: float = 30.0,
    ) -> None:
        self.stop_distance_m = stop_distance_m
        self.slow_distance_m = slow_distance_m

        # Last computed status for diagnostics.
        self._last_state: Optional[TrafficLightState] = None
        self._last_reason: Optional[str] = None
        self._last_distance: Optional[float] = None

    def compute_control(
        self,
        traffic_light: Optional[TrafficLight],
        current_speed_ms: float,
        stop_line_distance: Optional[float],
    ) -> dict:
        """
        Compute a control override dict.

        Returns a dict with keys:
          - throttle: float (-1 means "no override")
          - brake: float (-1 means "no override")
          - target_speed_ms: float (-1 means "no override")
          - reason: str
          - active: bool  (True when an override is applied)
        """
        # Resolve the effective distance to the stop line / light.
        if stop_line_distance is not None:
            distance = stop_line_distance
        elif traffic_light is not None:
            distance = traffic_light.distance_m
        else:
            distance = None

        # No light -> no action.
        if traffic_light is None:
            return self._no_action(TrafficLightState.UNKNOWN, distance)

        state = traffic_light.state

        if state == TrafficLightState.GREEN:
            self._last_state = state
            self._last_reason = "green_go"
            self._last_distance = distance
            return {
                "throttle": -1.0,
                "brake": -1.0,
                "target_speed_ms": -1.0,
                "reason": "green_go",
                "active": False,
            }

        if state == TrafficLightState.UNKNOWN:
            return self._no_action(state, distance)

        # If we somehow have a state but no distance, fall back to "far".
        close = bool(distance is not None and distance <= self.stop_distance_m)

        if state == TrafficLightState.RED:
            if close:
                result = {
                    "throttle": 0.0,
                    "brake": 0.8,
                    "target_speed_ms": 0.0,
                    "reason": "red_light_stop",
                    "active": True,
                }
            else:
                result = {
                    "throttle": 0.0,
                    "brake": 0.3,
                    "target_speed_ms": 0.0,
                    "reason": "red_light_approaching",
                    "active": True,
                }
        elif state == TrafficLightState.YELLOW:
            if close:
                result = {
                    "throttle": 0.0,
                    "brake": 0.1,
                    "target_speed_ms": current_speed_ms,
                    "reason": "yellow_proceed",
                    "active": True,
                }
            else:
                result = {
                    "throttle": 0.0,
                    "brake": 0.3,
                    "target_speed_ms": current_speed_ms * 0.5,
                    "reason": "yellow_slow",
                    "active": True,
                }
        else:  # pragma: no cover - defensive
            return self._no_action(state, distance)

        self._last_state = state
        self._last_reason = result["reason"]
        self._last_distance = distance
        return result

    def get_status(self) -> dict:
        """Return the last computed state / reason / distance."""
        return {
            "last_state": self._last_state,
            "last_reason": self._last_reason,
            "last_distance": self._last_distance,
        }

    # ── helpers ──────────────────────────────────────────────────────────

    def _no_action(
        self, state: TrafficLightState, distance: Optional[float]
    ) -> dict:
        self._last_state = state
        self._last_reason = "no_action"
        self._last_distance = distance
        return {
            "throttle": -1.0,
            "brake": -1.0,
            "target_speed_ms": -1.0,
            "reason": "no_action",
            "active": False,
        }
