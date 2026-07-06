"""Production tests for AEB (Automatic Emergency Braking) + ACC (Adaptive Cruise Control)."""
import math
import pytest

from safety.aeb_acc import AEBACC, AEBResult, ACCResult
from bridge.obstacles import Obstacle, ObstacleType


def _make_obstacle(
    obs_id: int = 1,
    x: float = 10.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    otype: ObstacleType = ObstacleType.VEHICLE,
) -> Obstacle:
    """Create obstacle at (x, y) with velocity (vx, vy) in CARLA frame."""
    return Obstacle(
        id=obs_id,
        position=(x, -y, 0.0),  # ENU: y_enu = -y_carla
        velocity=(vx, -vy, 0.0),
        theta=0.0,
        length=4.0,
        width=2.0,
        height=1.5,
        type=otype,
    )


class TestAEBInit:
    """Test AEBACC constructor."""

    def test_default_params(self):
        aeb = AEBACC()
        assert aeb.ttc_threshold == pytest.approx(2.5)
        assert aeb.ttc_critical == pytest.approx(1.0)
        assert aeb.time_gap == pytest.approx(1.8)
        assert aeb.min_distance == pytest.approx(5.0)

    def test_custom_params(self):
        aeb = AEBACC(ttc_threshold=3.0, ttc_critical=1.5, time_gap=2.0, min_distance=7.0)
        assert aeb.ttc_threshold == pytest.approx(3.0)
        assert aeb.ttc_critical == pytest.approx(1.5)
        assert aeb.time_gap == pytest.approx(2.0)
        assert aeb.min_distance == pytest.approx(7.0)

    def test_initial_state_idle(self):
        aeb = AEBACC()
        status = aeb.get_status()
        assert status['aeb_active'] is False
        assert status['acc_active'] is False


class TestAEB:
    """Test Automatic Emergency Braking."""

    def test_no_obstacles_no_brake(self):
        aeb = AEBACC()
        result = aeb.check_aeb([], 0, 0, 0, 10.0)
        assert result.active is False
        assert result.brake_override == pytest.approx(0.0)

    def test_obstacle_far_no_brake(self):
        aeb = AEBACC()
        obs = _make_obstacle(x=100.0)
        result = aeb.check_aeb([obs], 0, 0, 0, 10.0)
        assert result.active is False
        assert result.brake_override == pytest.approx(0.0)

    def test_obstacle_outside_lane_no_brake(self):
        aeb = AEBACC()
        obs = _make_obstacle(x=10.0, y=5.0)  # 5m lateral = outside lane
        result = aeb.check_aeb([obs], 0, 0, 0, 10.0)
        assert result.active is False

    def test_obstacle_behind_no_brake(self):
        aeb = AEBACC()
        obs = _make_obstacle(x=-10.0)  # behind ego
        result = aeb.check_aeb([obs], 0, 0, 0, 10.0)
        assert result.active is False

    def test_critical_ttc_full_brake(self):
        """TTC < 1.0s → full brake."""
        aeb = AEBACC()
        # Obstacle 5m ahead, stationary, ego at 10 m/s → TTC = 0.5s
        obs = _make_obstacle(x=5.0, vx=0.0)
        result = aeb.check_aeb([obs], 0, 0, 0, 10.0)
        assert result.active is True
        assert result.brake_override == pytest.approx(1.0)
        assert result.ttc < 1.0

    def test_moderate_ttc_proportional_brake(self):
        """TTC between 1.0s and 2.5s → proportional brake."""
        aeb = AEBACC()
        # Obstacle 20m ahead, stationary, ego at 10 m/s → TTC = 2.0s
        obs = _make_obstacle(x=20.0, vx=0.0)
        result = aeb.check_aeb([obs], 0, 0, 0, 10.0)
        assert result.active is True
        assert 0.0 < result.brake_override < 1.0
        assert 1.0 < result.ttc < 2.5

    def test_moving_obstacle_away_no_brake(self):
        """Obstacle moving away at same speed → no AEB."""
        aeb = AEBACC()
        # Obstacle 10m ahead, moving at 10 m/s same as ego → closing rate = 0
        obs = _make_obstacle(x=10.0, vx=10.0)
        result = aeb.check_aeb([obs], 0, 0, 0, 10.0)
        assert result.active is False
        assert result.ttc == float('inf')

    def test_moving_obstacle_closing_brake(self):
        """Obstacle ahead moving slower → closing → AEB if TTC low."""
        aeb = AEBACC()
        # Obstacle 10m ahead at 5 m/s, ego at 10 m/s → closing rate = 5, TTC = 2.0s
        obs = _make_obstacle(x=10.0, vx=5.0)
        result = aeb.check_aeb([obs], 0, 0, 0, 10.0)
        assert result.active is True
        assert result.ttc == pytest.approx(2.0, abs=0.1)

    def test_pedestrian_triggers_aeb(self):
        aeb = AEBACC()
        obs = _make_obstacle(x=5.0, otype=ObstacleType.PEDESTRIAN)
        result = aeb.check_aeb([obs], 0, 0, 0, 10.0)
        assert result.active is True

    def test_static_obstacle_ignored(self):
        """UNKNOWN_UNMOVABLE should not trigger AEB (handled differently)."""
        aeb = AEBACC()
        obs = _make_obstacle(x=5.0, otype=ObstacleType.UNKNOWN_UNMOVABLE)
        result = aeb.check_aeb([obs], 0, 0, 0, 10.0)
        assert result.active is False

    def test_closest_obstacle_selected(self):
        """Multiple obstacles → AEB picks closest with lowest TTC."""
        aeb = AEBACC()
        obs_far = _make_obstacle(obs_id=1, x=30.0, vx=0.0)
        obs_near = _make_obstacle(obs_id=2, x=10.0, vx=0.0)
        result = aeb.check_aeb([obs_far, obs_near], 0, 0, 0, 10.0)
        assert result.obstacle_id == 2


class TestACC:
    """Test Adaptive Cruise Control."""

    def test_no_obstacles_nominal_speed(self):
        aeb = AEBACC()
        result = aeb.check_acc([], 0, 0, 0, 10.0, nominal_target_ms=15.0)
        assert result.active is False
        assert result.target_speed_ms == pytest.approx(15.0)

    def test_obstacle_far_nominal_speed(self):
        aeb = AEBACC()
        obs = _make_obstacle(x=100.0, vx=15.0)
        result = aeb.check_acc([obs], 0, 0, 0, 10.0, nominal_target_ms=15.0)
        assert result.active is True
        # 100m > safe_distance → no speed reduction
        assert result.target_speed_ms == pytest.approx(15.0)

    def test_obstacle_close_reduces_speed(self):
        """Lead vehicle close + slow → reduce target speed."""
        aeb = AEBACC()
        # Lead at 10m, speed 5 m/s, ego at 10 m/s, nominal 15 m/s
        obs = _make_obstacle(x=10.0, vx=5.0)
        result = aeb.check_acc([obs], 0, 0, 0, 10.0, nominal_target_ms=15.0)
        assert result.active is True
        assert result.target_speed_ms < 15.0
        assert result.distance_m == pytest.approx(10.0, abs=1.0)

    def test_obstacle_same_speed_no_reduction(self):
        """Lead at same speed → no speed reduction needed."""
        aeb = AEBACC()
        obs = _make_obstacle(x=20.0, vx=15.0)
        result = aeb.check_acc([obs], 0, 0, 0, 15.0, nominal_target_ms=15.0)
        assert result.active is True
        # Lead speed = ego speed → target = nominal
        assert result.target_speed_ms == pytest.approx(15.0)

    def test_obstacle_outside_lane_ignored(self):
        aeb = AEBACC()
        obs = _make_obstacle(x=10.0, y=5.0, vx=5.0)
        result = aeb.check_acc([obs], 0, 0, 0, 10.0, nominal_target_ms=15.0)
        assert result.active is False

    def test_pedestrian_ignored_by_acc(self):
        """ACC only tracks vehicles and bicycles, not pedestrians."""
        aeb = AEBACC()
        obs = _make_obstacle(x=10.0, otype=ObstacleType.PEDESTRIAN)
        result = aeb.check_acc([obs], 0, 0, 0, 10.0, nominal_target_ms=15.0)
        assert result.active is False

    def test_min_distance_enforced(self):
        """Very close obstacle → target speed near 0."""
        aeb = AEBACC(min_distance=5.0)
        obs = _make_obstacle(x=3.0, vx=0.0)  # 3m < min_distance
        result = aeb.check_acc([obs], 0, 0, 0, 2.0, nominal_target_ms=15.0)
        assert result.active is True
        assert result.target_speed_ms < 5.0  # Should be very low


class TestAEBACCStatus:
    """Test status reporting and reset."""

    def test_status_after_aeb_trigger(self):
        aeb = AEBACC()
        obs = _make_obstacle(x=5.0, vx=0.0)
        aeb.check_aeb([obs], 0, 0, 0, 10.0)
        status = aeb.get_status()
        assert status['aeb_active'] is True
        assert status['aeb_brake'] > 0.0

    def test_status_after_acc_trigger(self):
        aeb = AEBACC()
        obs = _make_obstacle(x=10.0, vx=5.0)
        aeb.check_acc([obs], 0, 0, 0, 10.0, nominal_target_ms=15.0)
        status = aeb.get_status()
        assert status['acc_active'] is True

    def test_reset_clears_state(self):
        aeb = AEBACC()
        obs = _make_obstacle(x=5.0, vx=0.0)
        aeb.check_aeb([obs], 0, 0, 0, 10.0)
        aeb.reset()
        status = aeb.get_status()
        assert status['aeb_active'] is False
        assert status['acc_active'] is False
