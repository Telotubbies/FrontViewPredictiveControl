"""Production tests for type hints (utils/type_hints.py)."""
import pytest
import numpy as np
from dataclasses import fields, is_dataclass
from enum import Enum

from utils.type_hints import (
    DetectionMode, ControlMode,
    VehicleState, ControlCommand, ControlState, PerceptionResult,
    MPCResult, SafetyStatus, SystemMetrics, CameraFrame,
)


# ── Enums ────────────────────────────────────────────────────────────────────


class TestDetectionMode:
    def test_is_enum(self):
        assert issubclass(DetectionMode, Enum)

    def test_has_unet(self):
        assert hasattr(DetectionMode, "UNET")

    def test_has_classical(self):
        assert hasattr(DetectionMode, "CLASSICAL")

    def test_has_hybrid(self):
        assert hasattr(DetectionMode, "HYBRID")

    def test_distinct_values(self):
        values = [m.value for m in DetectionMode]
        assert len(values) == len(set(values))


class TestControlMode:
    def test_is_enum(self):
        assert issubclass(ControlMode, Enum)

    def test_has_mpc(self):
        assert hasattr(ControlMode, "MPC")

    def test_has_pure_pursuit(self):
        assert hasattr(ControlMode, "PURE_PURSUIT")


# ── Dataclasses ──────────────────────────────────────────────────────────────


class TestVehicleState:
    def test_is_dataclass(self):
        assert is_dataclass(VehicleState)

    @pytest.mark.parametrize("field", [
        "transform", "velocity", "acceleration",
        "steering_angle", "throttle", "brake", "timestamp",
    ])
    def test_field_exists(self, field):
        assert field in [f.name for f in fields(VehicleState)]


class TestControlCommand:
    def test_is_dataclass(self):
        assert is_dataclass(ControlCommand)

    def test_required_fields(self):
        fs = [f.name for f in fields(ControlCommand)]
        assert "steering" in fs
        assert "throttle" in fs
        assert "brake" in fs

    def test_optional_target_speed(self):
        cmd = ControlCommand(steering=0.1, throttle=0.5, brake=0.0)
        assert cmd.target_speed is None

    def test_optional_timestamp(self):
        cmd = ControlCommand(steering=0.1, throttle=0.5, brake=0.0)
        assert cmd.timestamp is None

    def test_construction_with_all_fields(self):
        cmd = ControlCommand(
            steering=0.2, throttle=0.6, brake=0.1,
            target_speed=8.0, timestamp=12345.0,
        )
        assert cmd.steering == pytest.approx(0.2)
        assert cmd.throttle == pytest.approx(0.6)
        assert cmd.brake == pytest.approx(0.1)
        assert cmd.target_speed == pytest.approx(8.0)
        assert cmd.timestamp == pytest.approx(12345.0)


class TestControlState:
    def test_is_dataclass(self):
        assert is_dataclass(ControlState)

    def test_default_values(self):
        cs = ControlState()
        assert cs.steering == pytest.approx(0.0)
        assert cs.throttle == pytest.approx(0.0)
        assert cs.brake == pytest.approx(0.0)
        assert cs.target_speed_kmh is None
        assert cs.timestamp is None

    def test_custom_values(self):
        cs = ControlState(
            steering=0.15, throttle=0.5, brake=0.0,
            target_speed_kmh=30.0, timestamp=999.0,
        )
        assert cs.steering == pytest.approx(0.15)
        assert cs.target_speed_kmh == pytest.approx(30.0)


class TestPerceptionResult:
    def test_is_dataclass(self):
        assert is_dataclass(PerceptionResult)

    @pytest.mark.parametrize("field", [
        "cte", "heading_error", "curvature", "confidence",
    ])
    def test_core_field_exists(self, field):
        assert field in [f.name for f in fields(PerceptionResult)]


class TestMPCResult:
    def test_is_dataclass(self):
        assert is_dataclass(MPCResult)

    @pytest.mark.parametrize("field", [
        "steering", "throttle", "brake",
    ])
    def test_control_field_exists(self, field):
        assert field in [f.name for f in fields(MPCResult)]


class TestSafetyStatus:
    def test_is_dataclass(self):
        assert is_dataclass(SafetyStatus)

    def test_has_active_field(self):
        fs = [f.name for f in fields(SafetyStatus)]
        assert "active" in fs


class TestSystemMetrics:
    def test_is_dataclass(self):
        assert is_dataclass(SystemMetrics)

    @pytest.mark.parametrize("field", ["fps", "perception_time", "control_time"])
    def test_metric_field_exists(self, field):
        assert field in [f.name for f in fields(SystemMetrics)]


class TestCameraFrame:
    def test_is_dataclass(self):
        assert is_dataclass(CameraFrame)

    @pytest.mark.parametrize("field", ["rgb", "timestamp", "width", "height"])
    def test_field_exists(self, field):
        assert field in [f.name for f in fields(CameraFrame)]
