"""
ADAS Controller v2: Main controller with intelligent mode switching.

Integrates:
    - UNet lane detection (BEV pipeline)
    - CARLA waypoint navigation
    - Mode selector for intelligent switching
    - Perception fusion
    - MPC control
"""

from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
import numpy as np
import logging

from .mode_selector import ModeSelector, ModeConfig, PerceptionMode
from .fusion import PerceptionFusion, LaneState, WaypointState, FusedState

logger = logging.getLogger(__name__)


@dataclass
class ADASConfig:
    """ADAS v2 configuration."""
    # Target speed
    target_speed_kmh: float = 30.0
    
    # Curve braking - more aggressive for high speed
    curve_brake_mild: float = 0.003      # lower threshold to start braking earlier
    curve_brake_medium: float = 0.008
    curve_brake_sharp: float = 0.015
    curve_brake_extreme: float = 0.025
    speed_factor_mild: float = 0.75      # reduce speed more
    speed_factor_medium: float = 0.55
    speed_factor_sharp: float = 0.40
    speed_factor_extreme: float = 0.25
    
    # Lookahead
    lookahead_base_m: float = 25.0
    lookahead_time_s: float = 2.0
    lookahead_min_m: float = 10.0
    
    # Mode switching
    mode_config: Optional[ModeConfig] = None


@dataclass
class ControlOutput:
    """Control output from ADAS."""
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    target_speed_ms: float = 0.0
    mode: str = "FUSION"
    lane_weight: float = 0.5
    fused_cte: float = 0.0
    fused_heading: float = 0.0
    fused_curvature: float = 0.0
    reference_path: Optional[List[Tuple[float, float]]] = None


class ADASController:
    """
    ADAS v2 Controller with intelligent mode switching.
    
    Flow:
        1. Get UNet lane detection → LaneState
        2. Get waypoint data → WaypointState
        3. Mode selector decides: LANE_ONLY / WAYPOINT_ONLY / FUSION
        4. Fusion blends states based on mode weights
        5. Curve anticipation braking adjusts target speed
        6. MPC computes control
    """
    
    def __init__(
        self,
        model_path: str,
        device: Any,
        config: Optional[ADASConfig] = None,
    ):
        self.config = config or ADASConfig()
        self._target_speed_ms = self.config.target_speed_kmh / 3.6
        
        # Initialize components
        self._mode_selector = ModeSelector(self.config.mode_config)
        self._fusion = PerceptionFusion()
        
        # Lane detector (lazy import to avoid circular deps)
        self._detector = None
        self._model_path = model_path
        self._device = device
        
        # MPC (lazy import)
        self._mpc = None
        
        # Hybrid controller (Pure Pursuit + MPC)
        self._hybrid = None
        
        # State
        self._prev_steer = 0.0
        self._prev_throttle = 0.0
        
        logger.info("ADAS v2 Controller initialized")
        
    def _init_detector(self):
        """Lazy init lane detector."""
        if self._detector is None:
            from pathlib import Path
            from perception.lane_detector import LaneDetector
            
            # Ensure absolute path
            model_path = Path(self._model_path)
            if not model_path.is_absolute():
                model_path = Path(__file__).parent.parent / model_path
            
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}, using CARLA detection")
                self._detector = LaneDetector(model_path=None, use_carla=True)
            else:
                self._detector = LaneDetector(
                    model_path=str(model_path),
                    use_carla=False,
                    model_type="unet"
                )
                logger.info(f"Lane detector initialized with {model_path.name}")
            
    def _init_mpc(self):
        """Lazy init MPC and Hybrid controller."""
        if self._mpc is None:
            from control.lane_mpc import LaneMPC, MPCConfig
            from control.pure_pursuit import HybridController
            self._mpc = LaneMPC(MPCConfig())
            self._hybrid = HybridController(self._mpc)
            logger.info("Hybrid controller (Pure Pursuit + MPC) initialized")
            
    def reset(self):
        """Reset controller state."""
        self._mode_selector.reset()
        self._fusion.reset()
        self._prev_steer = 0.0
        self._prev_throttle = 0.0
        if self._mpc:
            self._mpc.reset()
            
    def compute_target_speed(
        self,
        effective_curv: float,
        current_speed_ms: float,
    ) -> float:
        """
        Compute target speed with curve anticipation braking.
        
        Args:
            effective_curv: Max of current and lookahead curvature
            current_speed_ms: Current vehicle speed (m/s)
            
        Returns:
            Target speed (m/s)
        """
        cfg = self.config
        
        if effective_curv >= cfg.curve_brake_extreme:
            factor = cfg.speed_factor_extreme
        elif effective_curv >= cfg.curve_brake_sharp:
            factor = cfg.speed_factor_sharp
        elif effective_curv >= cfg.curve_brake_medium:
            factor = cfg.speed_factor_medium
        elif effective_curv >= cfg.curve_brake_mild:
            factor = cfg.speed_factor_mild
        else:
            factor = 1.0
            
        return self._target_speed_ms * factor
    
    def compute_lookahead(
        self,
        speed_ms: float,
        curvature: float,
    ) -> float:
        """
        Compute dynamic lookahead distance.
        
        Args:
            speed_ms: Current speed (m/s)
            curvature: Current road curvature
            
        Returns:
            Lookahead distance (m)
        """
        cfg = self.config
        
        # Base: time-based lookahead
        lookahead = max(cfg.lookahead_base_m, speed_ms * cfg.lookahead_time_s)
        
        # Reduce in curves
        if abs(curvature) > 0.02:
            lookahead = max(cfg.lookahead_min_m, lookahead * 0.6)
            
        return lookahead
    
    def step(
        self,
        rgb: np.ndarray,
        speed_ms: float,
        waypoint_cte: float,
        waypoint_heading: float,
        waypoint_curv: float,
        wp_lookahead_curv: float,
        waypoint_path: Optional[List[Tuple[float, float]]] = None,
        lane_id: Optional[int] = None,
        waypoints: Optional[List] = None,
        vehicle_transform = None,
    ) -> ControlOutput:
        """
        Single step of ADAS control.
        
        Args:
            rgb: Camera image (H, W, 3)
            speed_ms: Current vehicle speed (m/s)
            waypoint_cte: CTE from waypoints (m)
            waypoint_heading: Heading error from waypoints (rad)
            waypoint_curv: Current curvature from waypoints
            wp_lookahead_curv: Max curvature in lookahead
            waypoint_path: Optional (s, lat) path from waypoints
            lane_id: CARLA lane_id for validation ("cheating")
            waypoints: CARLA waypoints for ego-lane mask
            vehicle_transform: Vehicle transform for ego-lane mask
            
        Returns:
            ControlOutput with steer, throttle, brake, and debug info
        """
        self._lane_id = lane_id
        self._waypoints = waypoints
        self._vehicle_transform = vehicle_transform
        self._init_detector()
        self._init_mpc()
        
        # 1. UNet lane detection
        lane_state = self._detect_lanes(rgb)
        
        # 2. Waypoint state
        wp_state = WaypointState(
            cte=waypoint_cte,
            heading=waypoint_heading,
            curvature=waypoint_curv,
            lookahead_curv=wp_lookahead_curv,
            path=waypoint_path,
        )
        
        # 3. Mode selection
        mode = self._mode_selector.update(
            lane_conf=lane_state.confidence,
            wp_lookahead_curv=wp_lookahead_curv,
            current_curv=lane_state.curvature if lane_state.valid else waypoint_curv,
            lane_valid=lane_state.valid,
        )
        
        # 4. Fusion
        fused = self._fusion.fuse(
            lane=lane_state,
            waypoint=wp_state,
            lane_weight=self._mode_selector.lane_weight,
            mode=self._mode_selector.get_mode_string(),
        )
        
        # 5. Curve anticipation braking
        effective_curv = max(abs(fused.curvature), abs(wp_lookahead_curv))
        target_speed = self.compute_target_speed(effective_curv, speed_ms)
        
        # 6. Reference path
        lookahead = self.compute_lookahead(speed_ms, fused.curvature)
        ref_path = self._fusion.generate_reference_path(fused, lookahead)
        
        # 7. Hybrid control (Pure Pursuit for straight, MPC for curves)
        steer, ctrl_mode = self._hybrid.compute_steering(
            cte=fused.cte,
            heading_err=fused.heading,
            speed_ms=speed_ms,
            curvature=fused.curvature,
            confidence=fused.confidence,
            v_ref=target_speed,
        )
        
        # Get accel from MPC for speed control
        _, accel, _ = self._mpc.solve(
            x0=0, y0=fused.cte, psi0=fused.heading, v0=speed_ms,
            v_ref=target_speed, cte=fused.cte, heading_err=fused.heading,
            curvature=fused.curvature, confidence=fused.confidence,
        )
        
        # Smooth steering with rate limiting
        max_steer_rate = 0.06
        steer = float(np.clip(steer, self._prev_steer - max_steer_rate, self._prev_steer + max_steer_rate))
        self._prev_steer = steer
        
        # Throttle/brake from MPC
        throttle, brake = self._mpc.accel_to_carla(accel)
        
        # CRITICAL: Override MPC throttle/brake for reliable speed control
        # MPC sometimes gives wrong accel, so use simple P-controller for throttle
        speed_error = target_speed - speed_ms
        
        if speed_error > 0.5:
            # Need to accelerate
            throttle = min(1.0, 0.3 + speed_error * 0.15)
            brake = 0.0
        elif speed_error < -1.0:
            # Need to brake
            throttle = 0.0
            brake = min(1.0, abs(speed_error) * 0.2)
        else:
            # Maintain speed
            throttle = max(0.1, throttle)
            brake = 0.0
        
        # Extra boost when nearly stopped or slow
        if speed_ms < 3.0:
            # Force throttle when car is slow
            throttle = max(throttle, 0.7)
            brake = 0.0
        if speed_ms < 1.0:
            throttle = 1.0
            brake = 0.0
            
        return ControlOutput(
            steer=steer,
            throttle=throttle,
            brake=brake,
            target_speed_ms=target_speed,
            mode=fused.mode,
            lane_weight=fused.lane_weight,
            fused_cte=fused.cte,
            fused_heading=fused.heading,
            fused_curvature=fused.curvature,
            reference_path=ref_path,
        )
    
    def _detect_lanes(self, rgb: np.ndarray) -> LaneState:
        """
        Detect lanes using UNet + BEV pipeline with ego-lane filtering.
        
        Returns:
            LaneState with CTE, heading, curvature, confidence
        """
        if self._detector is None:
            return LaneState(valid=False)
            
        try:
            import cv2
            img = cv2.resize(rgb, (640, 480))
            
            # Create ego-lane mask from waypoints (if available)
            ego_mask = None
            if hasattr(self, '_waypoints') and self._waypoints and hasattr(self, '_vehicle_transform') and self._vehicle_transform:
                try:
                    from perception.ego_lane_mask import create_ego_lane_mask_from_waypoints
                    ego_mask = create_ego_lane_mask_from_waypoints(
                        image_shape=(480, 640),
                        waypoints=self._waypoints,
                        vehicle_transform=self._vehicle_transform,
                        fov=90.0,
                        lane_half_width=2.0,
                    )
                except Exception as e:
                    logger.debug(f"Ego-lane mask failed: {e}")
            
            # BEV lane detection with ego-lane mask
            result = self._detector.detect_lanes_bev(img, return_vis=False, ego_mask=ego_mask)
            if result is None or len(result) < 7:
                return LaneState(valid=False)
                
            left_c, right_c, cte, heading, curv, conf, _ = result
            
            valid = left_c is not None and right_c is not None and conf > 0.1
            
            return LaneState(
                cte=float(cte) if cte is not None else 0.0,
                heading=float(heading) if heading is not None else 0.0,
                curvature=float(curv) if curv is not None else 0.0,
                confidence=float(conf) if conf is not None else 0.0,
                valid=valid,
            )
        except Exception as e:
            logger.debug(f"Lane detection failed: {e}")
            return LaneState(valid=False)
