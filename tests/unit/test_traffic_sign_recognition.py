"""Production tests for Traffic Sign Recognition + Traffic Light Detection."""
import pytest

from perception.traffic_sign_recognition import (
    TrafficLight,
    TrafficLightController,
    TrafficLightState,
    TrafficSign,
    TrafficSignRecognizer,
    TrafficSignType,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_sign(
    sign_type: TrafficSignType = TrafficSignType.SPEED_LIMIT,
    value: float = 50.0,
    distance_m: float = 20.0,
    confidence: float = 1.0,
) -> TrafficSign:
    return TrafficSign(
        sign_type=sign_type,
        value=value,
        distance_m=distance_m,
        confidence=confidence,
    )


def _make_light(
    state: TrafficLightState = TrafficLightState.RED,
    distance_m: float = 10.0,
    confidence: float = 1.0,
) -> TrafficLight:
    return TrafficLight(
        state=state,
        distance_m=distance_m,
        confidence=confidence,
    )


# ── Enum tests ──────────────────────────────────────────────────────────────


class TestTrafficLightState:
    def test_values(self):
        assert TrafficLightState.UNKNOWN == 0
        assert TrafficLightState.RED == 1
        assert TrafficLightState.YELLOW == 2
        assert TrafficLightState.GREEN == 3

    def test_is_intenum(self):
        from enum import IntEnum

        assert issubclass(TrafficLightState, IntEnum)

    def test_ordering(self):
        # IntEnum supports comparison
        assert TrafficLightState.UNKNOWN < TrafficLightState.RED
        assert TrafficLightState.GREEN > TrafficLightState.YELLOW


class TestTrafficSignType:
    def test_values(self):
        assert TrafficSignType.UNKNOWN == 0
        assert TrafficSignType.SPEED_LIMIT == 1
        assert TrafficSignType.STOP == 2
        assert TrafficSignType.YIELD == 3
        assert TrafficSignType.NO_ENTRY == 4

    def test_is_intenum(self):
        from enum import IntEnum

        assert issubclass(TrafficSignType, IntEnum)


# ── Dataclass tests ─────────────────────────────────────────────────────────


class TestTrafficSign:
    def test_defaults(self):
        s = TrafficSign(sign_type=TrafficSignType.STOP)
        assert s.sign_type == TrafficSignType.STOP
        assert s.value == 0.0
        assert s.distance_m == 0.0
        assert s.confidence == 0.0

    def test_full(self):
        s = _make_sign(TrafficSignType.SPEED_LIMIT, value=60.0, distance_m=15.0)
        assert s.sign_type == TrafficSignType.SPEED_LIMIT
        assert s.value == 60.0
        assert s.distance_m == 15.0
        assert s.confidence == 1.0

    def test_equality(self):
        a = _make_sign(TrafficSignType.STOP, value=0.0, distance_m=5.0)
        b = _make_sign(TrafficSignType.STOP, value=0.0, distance_m=5.0)
        assert a == b


class TestTrafficLight:
    def test_fields(self):
        tl = _make_light(TrafficLightState.RED, distance_m=12.0, confidence=0.9)
        assert tl.state == TrafficLightState.RED
        assert tl.distance_m == 12.0
        assert tl.confidence == 0.9

    def test_equality(self):
        a = _make_light(TrafficLightState.GREEN, distance_m=3.0, confidence=1.0)
        b = _make_light(TrafficLightState.GREEN, distance_m=3.0, confidence=1.0)
        assert a == b


# ── Recognizer (pure helpers) ───────────────────────────────────────────────


class TestDetectSpeedLimit:
    def test_single_speed_limit(self):
        rec = TrafficSignRecognizer()
        signs = [_make_sign(TrafficSignType.SPEED_LIMIT, value=50.0, distance_m=20.0)]
        assert rec.detect_speed_limit(signs) == 50.0

    def test_closest_speed_limit(self):
        rec = TrafficSignRecognizer()
        signs = [
            _make_sign(TrafficSignType.SPEED_LIMIT, value=30.0, distance_m=40.0),
            _make_sign(TrafficSignType.SPEED_LIMIT, value=60.0, distance_m=10.0),
            _make_sign(TrafficSignType.SPEED_LIMIT, value=90.0, distance_m=70.0),
        ]
        # closest is the 60 km/h sign at 10 m
        assert rec.detect_speed_limit(signs) == 60.0

    def test_no_speed_limit_signs(self):
        rec = TrafficSignRecognizer()
        signs = [
            _make_sign(TrafficSignType.STOP, value=0.0, distance_m=5.0),
            _make_sign(TrafficSignType.YIELD, value=0.0, distance_m=8.0),
        ]
        assert rec.detect_speed_limit(signs) is None

    def test_empty_list(self):
        rec = TrafficSignRecognizer()
        assert rec.detect_speed_limit([]) is None

    def test_below_min_confidence_ignored(self):
        rec = TrafficSignRecognizer(min_confidence=0.8)
        signs = [
            _make_sign(
                TrafficSignType.SPEED_LIMIT,
                value=40.0,
                distance_m=10.0,
                confidence=0.3,
            ),
        ]
        assert rec.detect_speed_limit(signs) is None

    def test_mixed_with_low_confidence(self):
        rec = TrafficSignRecognizer(min_confidence=0.5)
        signs = [
            _make_sign(
                TrafficSignType.SPEED_LIMIT,
                value=30.0,
                distance_m=5.0,
                confidence=0.2,
            ),
            _make_sign(
                TrafficSignType.SPEED_LIMIT,
                value=80.0,
                distance_m=25.0,
                confidence=1.0,
            ),
        ]
        # only the 80 km/h sign passes the confidence filter
        assert rec.detect_speed_limit(signs) == 80.0


class TestDetectTrafficLight:
    def test_single(self):
        rec = TrafficSignRecognizer()
        lights = [_make_light(TrafficLightState.RED, distance_m=15.0)]
        tl = rec.detect_traffic_light(lights)
        assert tl is not None
        assert tl.state == TrafficLightState.RED
        assert tl.distance_m == 15.0

    def test_closest(self):
        rec = TrafficSignRecognizer()
        lights = [
            _make_light(TrafficLightState.GREEN, distance_m=50.0),
            _make_light(TrafficLightState.RED, distance_m=8.0),
            _make_light(TrafficLightState.YELLOW, distance_m=30.0),
        ]
        tl = rec.detect_traffic_light(lights)
        assert tl is not None
        assert tl.distance_m == 8.0

    def test_empty(self):
        rec = TrafficSignRecognizer()
        assert rec.detect_traffic_light([]) is None

    def test_below_min_confidence(self):
        rec = TrafficSignRecognizer(min_confidence=0.7)
        lights = [
            _make_light(
                TrafficLightState.RED, distance_m=5.0, confidence=0.4
            ),
        ]
        assert rec.detect_traffic_light(lights) is None


# ── detect_from_carla: gracefully degrades without CARLA ────────────────────


class TestDetectFromCarlaNoCarla:
    def test_returns_empty_on_error(self):
        rec = TrafficSignRecognizer()
        # Passing a bogus world/transform triggers an exception internally,
        # which must be swallowed and yield an empty list.
        assert rec.detect_from_carla(None, None) == []

    def test_get_stop_line_distance_none_on_error(self):
        rec = TrafficSignRecognizer()
        assert rec.get_stop_line_distance(None, None) is None


# ── Controller ──────────────────────────────────────────────────────────────


class TestTrafficLightControllerInit:
    def test_defaults(self):
        c = TrafficLightController()
        assert c.stop_distance_m == 5.0
        assert c.slow_distance_m == 30.0

    def test_custom(self):
        c = TrafficLightController(stop_distance_m=7.0, slow_distance_m=40.0)
        assert c.stop_distance_m == 7.0
        assert c.slow_distance_m == 40.0


class TestComputeControl:
    def setup_method(self):
        self.ctrl = TrafficLightController(
            stop_distance_m=5.0, slow_distance_m=30.0
        )

    def test_red_close_brakes_hard(self):
        tl = _make_light(TrafficLightState.RED, distance_m=3.0)
        r = self.ctrl.compute_control(tl, current_speed_ms=10.0, stop_line_distance=3.0)
        assert r["active"] is True
        assert r["throttle"] == 0.0
        assert r["brake"] == 0.8
        assert r["target_speed_ms"] == 0.0
        assert r["reason"] == "red_light_stop"

    def test_red_far_decelerates(self):
        tl = _make_light(TrafficLightState.RED, distance_m=25.0)
        r = self.ctrl.compute_control(tl, current_speed_ms=12.0, stop_line_distance=25.0)
        assert r["active"] is True
        assert r["throttle"] == 0.0
        assert r["brake"] == 0.3
        assert r["target_speed_ms"] == 0.0
        assert r["reason"] == "red_light_approaching"

    def test_red_uses_stop_line_distance_over_light_distance(self):
        # Light is "far" by its own distance, but stop line is close.
        tl = _make_light(TrafficLightState.RED, distance_m=25.0)
        r = self.ctrl.compute_control(tl, current_speed_ms=10.0, stop_line_distance=2.0)
        assert r["reason"] == "red_light_stop"
        assert r["brake"] == 0.8

    def test_red_no_stop_line_uses_light_distance(self):
        tl = _make_light(TrafficLightState.RED, distance_m=3.0)
        r = self.ctrl.compute_control(tl, current_speed_ms=10.0, stop_line_distance=None)
        assert r["reason"] == "red_light_stop"
        assert r["brake"] == 0.8

    def test_yellow_close_proceeds(self):
        tl = _make_light(TrafficLightState.YELLOW, distance_m=3.0)
        r = self.ctrl.compute_control(tl, current_speed_ms=14.0, stop_line_distance=3.0)
        assert r["active"] is True
        assert r["throttle"] == 0.0
        assert r["brake"] == 0.1
        assert r["target_speed_ms"] == 14.0
        assert r["reason"] == "yellow_proceed"

    def test_yellow_far_slows(self):
        tl = _make_light(TrafficLightState.YELLOW, distance_m=25.0)
        r = self.ctrl.compute_control(tl, current_speed_ms=20.0, stop_line_distance=25.0)
        assert r["active"] is True
        assert r["throttle"] == 0.0
        assert r["brake"] == 0.3
        assert r["target_speed_ms"] == pytest.approx(10.0)
        assert r["reason"] == "yellow_slow"

    def test_green_no_action(self):
        tl = _make_light(TrafficLightState.GREEN, distance_m=5.0)
        r = self.ctrl.compute_control(tl, current_speed_ms=15.0, stop_line_distance=5.0)
        assert r["active"] is False
        assert r["throttle"] == -1.0
        assert r["brake"] == -1.0
        assert r["target_speed_ms"] == -1.0
        assert r["reason"] == "green_go"

    def test_unknown_no_action(self):
        tl = _make_light(TrafficLightState.UNKNOWN, distance_m=5.0)
        r = self.ctrl.compute_control(tl, current_speed_ms=15.0, stop_line_distance=5.0)
        assert r["active"] is False
        assert r["throttle"] == -1.0
        assert r["brake"] == -1.0
        assert r["target_speed_ms"] == -1.0
        assert r["reason"] == "no_action"

    def test_none_light_no_action(self):
        r = self.ctrl.compute_control(None, current_speed_ms=15.0, stop_line_distance=None)
        assert r["active"] is False
        assert r["throttle"] == -1.0
        assert r["brake"] == -1.0
        assert r["target_speed_ms"] == -1.0
        assert r["reason"] == "no_action"

    def test_boundary_stop_distance(self):
        # Exactly at stop_distance_m counts as "close".
        tl = _make_light(TrafficLightState.RED, distance_m=5.0)
        r = self.ctrl.compute_control(tl, current_speed_ms=10.0, stop_line_distance=5.0)
        assert r["reason"] == "red_light_stop"

    def test_just_beyond_stop_distance(self):
        tl = _make_light(TrafficLightState.RED, distance_m=5.1)
        r = self.ctrl.compute_control(tl, current_speed_ms=10.0, stop_line_distance=5.1)
        assert r["reason"] == "red_light_approaching"


class TestGetStatus:
    def test_initial_status(self):
        c = TrafficLightController()
        s = c.get_status()
        assert s["last_state"] is None
        assert s["last_reason"] is None
        assert s["last_distance"] is None

    def test_status_after_red_close(self):
        c = TrafficLightController()
        tl = _make_light(TrafficLightState.RED, distance_m=3.0)
        c.compute_control(tl, current_speed_ms=10.0, stop_line_distance=3.0)
        s = c.get_status()
        assert s["last_state"] == TrafficLightState.RED
        assert s["last_reason"] == "red_light_stop"
        assert s["last_distance"] == 3.0

    def test_status_after_green(self):
        c = TrafficLightController()
        tl = _make_light(TrafficLightState.GREEN, distance_m=10.0)
        c.compute_control(tl, current_speed_ms=10.0, stop_line_distance=10.0)
        s = c.get_status()
        assert s["last_state"] == TrafficLightState.GREEN
        assert s["last_reason"] == "green_go"
        assert s["last_distance"] == 10.0

    def test_status_after_none(self):
        c = TrafficLightController()
        c.compute_control(None, current_speed_ms=10.0, stop_line_distance=None)
        s = c.get_status()
        assert s["last_state"] == TrafficLightState.UNKNOWN
        assert s["last_reason"] == "no_action"
        assert s["last_distance"] is None
