"""Utility modules."""

from .device_utils import get_device, is_rocm_available, is_cuda_available, get_device_info
from .type_hints import (
    VehicleState, ControlCommand, PerceptionResult, MPCResult, 
    SafetyStatus, SystemMetrics, CameraFrame, DetectionMode, ControlMode,
    ImageArray, PolynomialCoeffs, Point2D, Point3D, BoundingBox, ColorRGB, ColorRGBA
)

__all__ = [
    'get_device', 'is_rocm_available', 'is_cuda_available', 'get_device_info',
    'VehicleState', 'ControlCommand', 'PerceptionResult', 'MPCResult',
    'SafetyStatus', 'SystemMetrics', 'CameraFrame', 'DetectionMode', 'ControlMode',
    'ImageArray', 'PolynomialCoeffs', 'Point2D', 'Point3D', 'BoundingBox', 'ColorRGB', 'ColorRGBA'
]

