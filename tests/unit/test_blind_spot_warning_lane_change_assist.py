"""Production tests for Blind Spot Warning + Lane Change Assist (safety/blind_spot_warning_lane_change_assist.py)."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bridge.obstacles import Obstacle, ObstacleType
from safety.blind_spot_warning_lane_change_assist import (
    BSWAlert,
    BSWConfig,
    BlindSpotWarning,
    LaneChangeAssist,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_NEXT_ID = 1000


def _make_obstacle(
    x: float,
    y: float,
    vx: float,
    vy: float,
    otype: ObstacleType = ObstacleType.VEHICLE,
) -> Obstacle:
    """
    Build an Obstacle from CARLA-style coordinates.

    CARLA: x forward, y right.  Bridge/ENU: x forward, y left (y_enu = -y_carla).
    So we negate y (and vy) when storing the ENU position/velocity.
    """
    global _NEXT_ID
    obs = Obstacle(
        id=_NEXT_ID,
        position=(x, -y, 0.0),
        velocity=(vx, -vy, 0.0),
        theta=0.0,
        length=4.0,
        width=2.0,
        height=1.5,
        type=otype,
    )
    _NEXT_ID += 1
    return obs


# Ego at the origin, heading 0 (ENU: x forward, y left).
EGO_X = 0.0
EGO_Y = 0.0
EGO_HEADING = 0.0
EGO_SPEED = 10.0  # above default warning_speed_threshold (5.0)


# ── BSWConfig ─────────────────────────────────────────────────────────────────


class TestBSWConfig:
    def test_defaults(self):
        cfg = BSWConfig()
        assert cfg.blind_spot_range_m == 3.0
        assert cfg.blind_spot_lateral_m == 1.5
        assert cfg.lca_lookahead_s == 3.0
        assert cfg.warning_speed_threshold_ms == 5.0
        assert cfg.approach_speed_threshold_ms == 2.0


class TestBSWAlert:
    def test_values(self):
        assert BSWAlert.CLEAR == 0
        assert BSWAlert.BLIND_SPOT == 1
        assert BSWAlert.APPROACHING == 2


# ── BlindSpotWarning ──────────────────────────────────────────────────────────


class TestBlindSpotWarning:
    def test_vehicle_behind_left_triggers_left_alert(self):
        bsw = BlindSpotWarning()
        # CARLA: behind = -x, left = -y  →  ENU (-2, +1): forward=-2, lateral=+1
        obs = _make_obstacle(-2.0, -1.0, 0.0, 0.0)
        status = bsw.update([obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, 0.0)
        assert status["left_alert"] == BSWAlert.BLIND_SPOT
        assert status["right_alert"] == BSWAlert.CLEAR
        assert status["left_obstacle_id"] == obs.id
        assert status["right_obstacle_id"] is None
        assert status["safe_to_change_left"] is False
        assert status["safe_to_change_right"] is True

    def test_vehicle_behind_right_triggers_right_alert(self):
        bsw = BlindSpotWarning()
        # CARLA: behind = -x, right = +y  →  ENU (-2, -1): forward=-2, lateral=-1
        obs = _make_obstacle(-2.0, 1.0, 0.0, 0.0)
        status = bsw.update([obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, 0.0)
        assert status["right_alert"] == BSWAlert.BLIND_SPOT
        assert status["left_alert"] == BSWAlert.CLEAR
        assert status["right_obstacle_id"] == obs.id
        assert status["left_obstacle_id"] is None
        assert status["safe_to_change_right"] is False
        assert status["safe_to_change_left"] is True

    def test_clear_when_no_obstacles(self):
        bsw = BlindSpotWarning()
        status = bsw.update([], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, 0.0)
        assert status["left_alert"] == BSWAlert.CLEAR
        assert status["right_alert"] == BSWAlert.CLEAR
        assert status["safe_to_change_left"] is True
        assert status["safe_to_change_right"] is True

    def test_clear_when_obstacle_ahead(self):
        bsw = BlindSpotWarning()
        # CARLA ahead-left: (5, -1) → ENU (5, 1): forward=+5 (ahead), not blind spot
        obs = _make_obstacle(5.0, -1.0, 0.0, 0.0)
        status = bsw.update([obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, 0.0)
        assert status["left_alert"] == BSWAlert.CLEAR
        assert status["right_alert"] == BSWAlert.CLEAR

    def test_clear_when_obstacle_too_far_behind(self):
        bsw = BlindSpotWarning()
        # forward = -5 (< -blind_spot_range_m=3.0) → outside zone
        obs = _make_obstacle(-5.0, -1.0, 0.0, 0.0)
        status = bsw.update([obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, 0.0)
        assert status["left_alert"] == BSWAlert.CLEAR

    def test_clear_when_obstacle_too_far_laterally(self):
        bsw = BlindSpotWarning()
        # |lateral| = 2.0 > blind_spot_lateral_m (1.5)
        obs = _make_obstacle(-2.0, -2.0, 0.0, 0.0)
        status = bsw.update([obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, 0.0)
        assert status["left_alert"] == BSWAlert.CLEAR
        assert status["right_alert"] == BSWAlert.CLEAR

    def test_no_alert_at_low_speed(self):
        bsw = BlindSpotWarning()
        obs = _make_obstacle(-2.0, -1.0, 0.0, 0.0)
        status = bsw.update([obs], EGO_X, EGO_Y, EGO_HEADING, ego_speed=3.0, timestamp=0.0)
        assert status["left_alert"] == BSWAlert.CLEAR
        assert status["right_alert"] == BSWAlert.CLEAR
        assert status["safe_to_change_left"] is True
        assert status["safe_to_change_right"] is True

    def test_no_alert_at_threshold_speed(self):
        # Exactly at threshold (5.0) should NOT trigger (strictly greater than).
        bsw = BlindSpotWarning()
        obs = _make_obstacle(-2.0, -1.0, 0.0, 0.0)
        status = bsw.update([obs], EGO_X, EGO_Y, EGO_HEADING, ego_speed=5.0, timestamp=0.0)
        assert status["left_alert"] == BSWAlert.CLEAR

    def test_both_sides_simultaneously(self):
        bsw = BlindSpotWarning()
        left = _make_obstacle(-2.0, -1.0, 0.0, 0.0)
        right = _make_obstacle(-2.0, 1.0, 0.0, 0.0)
        status = bsw.update([left, right], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, 0.0)
        assert status["left_alert"] == BSWAlert.BLIND_SPOT
        assert status["right_alert"] == BSWAlert.BLIND_SPOT
        assert status["safe_to_change_left"] is False
        assert status["safe_to_change_right"] is False

    def test_get_status_returns_last_update(self):
        bsw = BlindSpotWarning()
        assert bsw.get_status()["left_alert"] == BSWAlert.CLEAR
        obs = _make_obstacle(-2.0, -1.0, 0.0, 0.0)
        bsw.update([obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, 0.0)
        assert bsw.get_status()["left_alert"] == BSWAlert.BLIND_SPOT

    def test_reset_clears_alerts(self):
        bsw = BlindSpotWarning()
        obs = _make_obstacle(-2.0, -1.0, 0.0, 0.0)
        bsw.update([obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, 0.0)
        assert bsw.get_status()["left_alert"] == BSWAlert.BLIND_SPOT
        bsw.reset()
        status = bsw.get_status()
        assert status["left_alert"] == BSWAlert.CLEAR
        assert status["right_alert"] == BSWAlert.CLEAR
        assert status["left_obstacle_id"] is None
        assert status["right_obstacle_id"] is None
        assert status["safe_to_change_left"] is True
        assert status["safe_to_change_right"] is True

    def test_ignores_pedestrians(self):
        bsw = BlindSpotWarning()
        obs = _make_obstacle(-2.0, -1.0, 0.0, 0.0, otype=ObstacleType.PEDESTRIAN)
        status = bsw.update([obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, 0.0)
        assert status["left_alert"] == BSWAlert.CLEAR


# ── LaneChangeAssist ──────────────────────────────────────────────────────────


class TestLaneChangeAssist:
    def test_low_risk_when_clear(self):
        lca = LaneChangeAssist()
        res = lca.evaluate_lane_change(
            [], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, target_lane="left"
        )
        assert res["safe"] is True
        assert res["risk_level"] == "low"
        assert res["reason"] == "clear"
        assert math.isinf(res["time_to_collision"])

    def test_high_risk_vehicle_in_blind_spot_left(self):
        lca = LaneChangeAssist()
        obs = _make_obstacle(-2.0, -1.0, 0.0, 0.0)  # behind-left
        res = lca.evaluate_lane_change(
            [obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, target_lane="left"
        )
        assert res["safe"] is False
        assert res["risk_level"] == "high"
        assert res["time_to_collision"] == 0.0

    def test_high_risk_vehicle_in_blind_spot_right(self):
        lca = LaneChangeAssist()
        obs = _make_obstacle(-2.0, 1.0, 0.0, 0.0)  # behind-right
        res = lca.evaluate_lane_change(
            [obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, target_lane="right"
        )
        assert res["safe"] is False
        assert res["risk_level"] == "high"

    def test_medium_risk_approaching_from_left(self):
        lca = LaneChangeAssist()
        # CARLA: behind-left, far, closing fast.
        # ENU position (-25, +1): forward=-25, lateral=+1 (left).
        # ENU velocity (9, 0): rel_forward = 9 - 10 = -1 ... not approaching.
        # Make ego slower so the obstacle closes from behind.
        ego_speed = 5.0
        obs = _make_obstacle(-25.0, -1.0, 9.0, 0.0)
        # ENU pos (-25, 1), ENU vel (9, 0). rel_forward = 9 - 5 = 4 > 2.0.
        # Predicted forward = -25 + 4*3 = -13 ... still outside blind spot.
        # Use a faster obstacle so it enters the blind spot within 3 s.
        obs = _make_obstacle(-13.0, -1.0, 9.0, 0.0)
        # forward=-13, rel_forward=4, predicted=-13+12=-1 → inside [-3,0]. lateral=1.
        res = lca.evaluate_lane_change(
            [obs], EGO_X, EGO_Y, EGO_HEADING, ego_speed, target_lane="left"
        )
        assert res["safe"] is False
        assert res["risk_level"] == "medium"
        assert res["time_to_collision"] >= 0.0
        assert not math.isinf(res["time_to_collision"])

    def test_medium_risk_approaching_from_right(self):
        lca = LaneChangeAssist()
        ego_speed = 5.0
        # behind-right: CARLA (-13, +1) → ENU (-13, -1): lateral=-1 (right)
        obs = _make_obstacle(-13.0, 1.0, 9.0, 0.0)
        res = lca.evaluate_lane_change(
            [obs], EGO_X, EGO_Y, EGO_HEADING, ego_speed, target_lane="right"
        )
        assert res["safe"] is False
        assert res["risk_level"] == "medium"

    def test_low_risk_when_approaching_not_on_target_side(self):
        lca = LaneChangeAssist()
        ego_speed = 5.0
        # Approaching from left, but evaluating right lane change.
        obs = _make_obstacle(-13.0, -1.0, 9.0, 0.0)
        res = lca.evaluate_lane_change(
            [obs], EGO_X, EGO_Y, EGO_HEADING, ego_speed, target_lane="right"
        )
        assert res["safe"] is True
        assert res["risk_level"] == "low"

    def test_low_risk_when_closing_speed_below_threshold(self):
        lca = LaneChangeAssist()
        ego_speed = 8.0
        # rel_forward = 9 - 8 = 1 < approach_speed_threshold (2.0)
        obs = _make_obstacle(-13.0, -1.0, 9.0, 0.0)
        res = lca.evaluate_lane_change(
            [obs], EGO_X, EGO_Y, EGO_HEADING, ego_speed, target_lane="left"
        )
        assert res["risk_level"] == "low"
        assert res["safe"] is True

    def test_low_risk_when_approaching_wont_enter_blind_spot(self):
        lca = LaneChangeAssist()
        ego_speed = 5.0
        # Far behind and slow closing: predicted forward stays < -blind_spot_range.
        obs = _make_obstacle(-30.0, -1.0, 6.0, 0.0)  # rel_forward=1 < 2 → filtered
        res = lca.evaluate_lane_change(
            [obs], EGO_X, EGO_Y, EGO_HEADING, ego_speed, target_lane="left"
        )
        assert res["risk_level"] == "low"

    def test_invalid_target_lane_raises(self):
        lca = LaneChangeAssist()
        with pytest.raises(ValueError):
            lca.evaluate_lane_change(
                [], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, target_lane="straight"
            )

    def test_debug_info_updated(self):
        lca = LaneChangeAssist()
        lca.evaluate_lane_change(
            [], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, target_lane="left"
        )
        info = lca.get_debug_info()
        assert info["last_target_lane"] == "left"
        assert info["last_risk_level"] == "low"
        assert math.isinf(info["last_ttc"])

    def test_debug_info_high_risk(self):
        lca = LaneChangeAssist()
        obs = _make_obstacle(-2.0, -1.0, 0.0, 0.0)
        lca.evaluate_lane_change(
            [obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, target_lane="left"
        )
        info = lca.get_debug_info()
        assert info["last_target_lane"] == "left"
        assert info["last_risk_level"] == "high"
        assert info["last_ttc"] == 0.0

    def test_safe_lane_change_left_when_clear(self):
        lca = LaneChangeAssist()
        # An obstacle ahead on the left does not block a left lane change.
        obs = _make_obstacle(10.0, -1.0, 0.0, 0.0)
        res = lca.evaluate_lane_change(
            [obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, target_lane="left"
        )
        assert res["safe"] is True
        assert res["risk_level"] == "low"

    def test_safe_lane_change_right_when_clear(self):
        lca = LaneChangeAssist()
        obs = _make_obstacle(10.0, 1.0, 0.0, 0.0)
        res = lca.evaluate_lane_change(
            [obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, target_lane="right"
        )
        assert res["safe"] is True
        assert res["risk_level"] == "low"

    def test_custom_config(self):
        cfg = BSWConfig(blind_spot_range_m=5.0, blind_spot_lateral_m=2.0)
        lca = LaneChangeAssist(cfg)
        # forward=-4 now inside the wider blind spot (-5, 0).
        obs = _make_obstacle(-4.0, -1.5, 0.0, 0.0)
        res = lca.evaluate_lane_change(
            [obs], EGO_X, EGO_Y, EGO_HEADING, EGO_SPEED, target_lane="left"
        )
        assert res["risk_level"] == "high"
