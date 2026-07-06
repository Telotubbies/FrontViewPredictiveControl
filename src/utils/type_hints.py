"""
Type hints and common data structures for CARLA MPC system

Provides standardized type definitions for:
- Control states
- Perception results  
- Vehicle states
- Configuration types
"""

from typing import Optional, Tuple, List, Dict, Any, Union, Literal, Callable, TypeVar, TypeAlias

T = TypeVar('T')
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Forward declarations for CARLA types
try:
    import carla
    CarlaTransform = carla.Transform
    CarlaVehicleControl = carla.VehicleControl
    CarlaSensor = carla.Sensor
    CarlaVehicle = carla.Vehicle
except ImportError:
    # Fallback types when CARLA is not available
    CarlaTransform = Any
    CarlaVehicleControl = Any  
    CarlaSensor = Any
    CarlaVehicle = Any


class DetectionMode(Enum):
    """Lane detection mode"""
    UNET = "unet"
    CLASSICAL = "classical"
    HYBRID = "hybrid"


class ControlMode(Enum):
    """Control mode"""
    MPC = "mpc"
    PID = "pid"
    PURE_PURSUIT = "pure_pursuit"


@dataclass
class VehicleState:
    """Vehicle state information"""
    transform: CarlaTransform
    velocity: float  # m/s
    acceleration: float  # m/s²
    steering_angle: float  # rad
    throttle: float  # [0, 1]
    brake: float  # [0, 1]
    timestamp: float


@dataclass
class ControlCommand:
    """Control command for vehicle"""
    steering: float  # [-1, 1] or rad
    throttle: float  # [0, 1]
    brake: float  # [0, 1]
    target_speed: Optional[float] = None  # m/s
    timestamp: Optional[float] = None


@dataclass
class PerceptionResult:
    """Result from perception pipeline"""
    cte: float  # Cross-track error in meters
    heading_error: float  # Heading error in radians
    curvature: float  # Road curvature
    confidence: float  # Detection confidence [0, 1]
    geometry_valid: bool  # Geometry validation result
    lane_width: Optional[float] = None  # Lane width in meters
    left_coeffs: Optional[np.ndarray] = None  # Left lane polynomial coefficients
    right_coeffs: Optional[np.ndarray] = None  # Right lane polynomial coefficients
    center_coeffs: Optional[np.ndarray] = None  # Center lane polynomial coefficients
    timestamp: float = 0.0
    processing_time: Optional[float] = None  # Processing time in seconds
    
    # Visualization data
    lane_overlay: Optional[np.ndarray] = None
    bev_binary: Optional[np.ndarray] = None
    bev_window_vis: Optional[np.ndarray] = None


@dataclass
class MPCResult:
    """Result from MPC optimization"""
    steering: float  # Steering command
    throttle: float  # Throttle command  
    brake: float  # Brake command
    cost: float  # Optimization cost
    success: bool  # Optimization success
    solve_time: float  # Solve time in seconds
    horizon: int  # Prediction horizon used
    
    # Optional detailed results
    state_trajectory: Optional[np.ndarray] = None
    control_trajectory: Optional[np.ndarray] = None


@dataclass
class SafetyStatus:
    """Safety system status"""
    active: bool
    reason: Optional[str] = None
    override_applied: bool = False
    recovery_active: bool = False
    timestamp: float = 0.0


@dataclass
class SystemMetrics:
    """System performance metrics"""
    fps: float
    perception_time: float  # seconds
    control_time: float  # seconds
    total_time: float  # seconds
    memory_usage: Optional[float] = None  # MB
    cpu_usage: Optional[float] = None  # percentage


@dataclass
class CameraFrame:
    """Camera frame data"""
    rgb: np.ndarray
    timestamp: float
    width: int
    height: int
    fov: float  # degrees


# Type aliases for common types
ImageArray = np.ndarray
PolynomialCoeffs = np.ndarray
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
BoundingBox = Tuple[Point2D, Point2D]
ColorRGB = Tuple[int, int, int]
ColorRGBA = Tuple[int, int, int, int]

# Configuration type aliases
ConfigValue = Union[str, int, float, bool, Tuple, List, Dict]
ConfigSection = Dict[str, ConfigValue]

# Function type hints
PerceptionPipeline = Any
Controller = Any
SafetySystem = Any
Dashboard = Any

# Callback types
CameraCallback = Callable[[ImageArray], None]
ControlCallback = Callable[[ControlCommand], None]
ErrorCallback = Callable[[Exception], None]

# Result types
Result: TypeAlias = Union[T, Exception]
OptionalResult: TypeAlias = Optional[T]

# Dictionary types with specific key-value types
VehicleConfig = Dict[str, Union[str, float, int]]
CameraConfig = Dict[str, Union[int, float, str]]
MPCConfig = Dict[str, Union[float, int, bool]]
PerceptionConfig = Dict[str, ConfigValue]
