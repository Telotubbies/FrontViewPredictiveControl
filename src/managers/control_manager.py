#!/usr/bin/env python3
"""
Control Manager - Simple control system management

Handles:
- MPC control computation
- Safety systems
- Stuck recovery
- Control command generation
"""

import logging
import time
from typing import Optional, Dict, Any
import numpy as np

from utils.type_hints import (
    ControlCommand, PerceptionResult, SafetyStatus, VehicleState
)
import config as _config_module
from config import *  # noqa: F401,F403 — module-level constants from config.py

logger = logging.getLogger(__name__)


class ControlManager:
    """Simple control management interface"""

    def __init__(self,
                 target_speed_kmh: float = 25.0,
                 enable_safety: bool = True,
                 enable_stuck_recovery: bool = True):
        self.target_speed_kmh = target_speed_kmh
        self.enable_safety = enable_safety
        self.enable_stuck_recovery = enable_stuck_recovery

        # Configuration
        self.config = _config_module

        # Control components
        self.mpc_controller: Optional[Any] = None
        self.safety_system: Optional[Any] = None
        self.stuck_recovery: Optional[Any] = None

        # State
        self.last_control: Optional[ControlCommand] = None
        self.control_count = 0
        self.safety_status = SafetyStatus(active=False)

        # Performance metrics
        self.control_times = []

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize control components"""
        try:
            # Initialize MPC controller
            self._initialize_mpc_controller()

            # Initialize safety system
            if self.enable_safety:
                self._initialize_safety_system()

            # Initialize stuck recovery
            if self.enable_stuck_recovery:
                self._initialize_stuck_recovery()

            logger.info("Control manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize control manager: {e}")
            raise

    def _initialize_mpc_controller(self):
        """Initialize MPC controller"""
        try:
            from control.lane_mpc import LaneMPC, MPCConfig

            mpc_config = MPCConfig(
                N=self.config.MPC_N,
                dt=self.config.MPC_DT,
                L=self.config.MPC_L,
                max_steer=self.config.MPC_MAX_STEER,
                max_steer_rate=self.config.MPC_MAX_STEER_RATE,
                max_accel=self.config.MPC_MAX_ACCEL,
                min_accel=self.config.MPC_MIN_ACCEL,
                w_cte=self.config.MPC_W_CTE,
                w_heading=self.config.MPC_W_HEADING,
                w_vel=self.config.MPC_W_VEL,
                w_steer=self.config.MPC_W_STEER,
                w_accel=self.config.MPC_W_ACCEL,
                w_steer_rate=self.config.MPC_W_STEER_RATE,
                w_accel_rate=self.config.MPC_W_ACCEL_RATE,
                w_steer_jerk=self.config.MPC_W_STEER_JERK,
                use_adaptive_weights=self.config.MPC_USE_ADAPTIVE_WEIGHTS
            )

            self.mpc_controller = LaneMPC(mpc_config)
            logger.info("MPC controller initialized")

        except ImportError as e:
            logger.error(f"Failed to import MPC controller: {e}")
            raise

    def _initialize_safety_system(self):
        """Initialize safety system"""
        try:
            from safety.override import SafetyOverride

            # Create safety config
            safety_config = {
                'emergency_brake_enabled': True,
                'max_speed_kmh': self.target_speed_kmh,
                'min_speed_kmh': 0.0,
                'max_steering_angle': self.config.SAFETY_MAX_STEER_RAD,
                'collision_timeout': 5.0,
                'emergency_deceleration': -5.0
            }

            self.safety_system = SafetyOverride(safety_config)
            logger.info("Safety system initialized")

        except ImportError as e:
            logger.error(f"Failed to import safety system: {e}")
            raise

    def _initialize_stuck_recovery(self):
        """Initialize stuck recovery system"""
        try:
            from safety.stuck_recovery import StuckRecovery

            self.stuck_recovery = StuckRecovery()
            logger.info("Stuck recovery initialized")

        except ImportError as e:
            logger.error(f"Failed to import stuck recovery: {e}")
            raise

    def compute_control(self,
                       perception: Optional[PerceptionResult],
                       vehicle_state: VehicleState) -> ControlCommand:
        """Compute control command"""
        start_time = time.time()

        # Default control
        control = ControlCommand(
            steering=0.0,
            throttle=0.0,
            brake=0.0,
            target_speed=self.target_speed_kmh / 3.6,  # Convert to m/s
            timestamp=time.time()
        )

        try:
            # Check for stuck recovery
            if self.enable_stuck_recovery and self.stuck_recovery:
                speed_ms = vehicle_state['speed_ms']
                throttle = self.last_control.throttle if self.last_control else 0.0
                recovery = self.stuck_recovery.update(speed_ms, throttle)
                if recovery is not None:
                    steer_r, throttle_r, brake_r, reverse_r = recovery
                    control.steering = steer_r
                    control.throttle = throttle_r
                    control.brake = brake_r
                    control.reverse = reverse_r
                    self.safety_status = SafetyStatus(
                        active=True,
                        reason="Stuck recovery",
                        recovery_active=True,
                        timestamp=time.time()
                    )
                    logger.info("Applying stuck recovery (steer=%.2f thr=%.2f brake=%.2f reverse=%s)",
                                steer_r, throttle_r, brake_r, reverse_r)
                    return control

            # Check perception validity
            if not perception or not perception.geometry_valid:
                control = self._compute_fallback_control(control, vehicle_state['speed_ms'])
                self.safety_status = SafetyStatus(
                    active=True,
                    reason="Invalid perception",
                    timestamp=time.time()
                )
                return control

            # Run MPC
            if self.mpc_controller:
                mpc_result = self.mpc_controller.solve(
                    cte=perception.cte,
                    heading_error=perception.heading_error,
                    curvature=perception.curvature,
                    current_speed=vehicle_state['speed_ms'],
                    target_speed=control.target_speed
                )

                if mpc_result:
                    control.steering = float(mpc_result.get('steering', 0.0))
                    control.throttle = float(mpc_result.get('throttle', 0.0))
                    control.brake = float(mpc_result.get('brake', 0.0))
                else:
                    logger.warning("MPC failed to solve, using fallback")
                    control = self._compute_fallback_control(control, vehicle_state['speed_ms'])

            # Apply safety overrides
            if self.enable_safety and self.safety_system:
                control = self._apply_safety_overrides(control, vehicle_state, perception)

            self.safety_status = SafetyStatus(active=False)

        except Exception as e:
            logger.error(f"Control computation error: {e}")
            control = self._compute_fallback_control(control, vehicle_state['speed_ms'])
            self.safety_status = SafetyStatus(
                active=True,
                reason=f"Control error: {e}",
                timestamp=time.time()
            )

        # Record metrics
        control_time = time.time() - start_time
        self.control_times.append(control_time)
        self.last_control = control
        self.control_count += 1

        return control

    def _compute_fallback_control(self,
                                control: ControlCommand,
                                current_speed_ms: float) -> ControlCommand:
        """Compute fallback control when main control fails"""
        target_speed_ms = self.target_speed_kmh / 3.6
        speed_error = target_speed_ms - current_speed_ms

        # Simple speed control
        if speed_error > 1.0:
            control.throttle = 0.3
            control.brake = 0.0
        elif speed_error < -2.0:
            control.throttle = 0.0
            control.brake = 0.2
        else:
            control.throttle = 0.1
            control.brake = 0.0

        # Minimal steering
        control.steering = 0.0

        return control

    def _apply_safety_overrides(self,
                               control: ControlCommand,
                               vehicle_state: VehicleState,
                               perception: PerceptionResult) -> ControlCommand:
        """Apply safety system overrides"""
        try:
            # Steering rate limiting
            if self.last_control:
                max_delta = self.config.STEER_MAX_DELTA_PER_FRAME
                delta = control.steering - self.last_control.steering
                if abs(delta) > max_delta:
                    control.steering = self.last_control.steering + np.sign(delta) * max_delta

            # Steering limits
            max_steer = self.config.SAFETY_MAX_STEER_RAD
            control.steering = np.clip(control.steering, -max_steer, max_steer)

            # Speed limits
            max_speed_ms = self.target_speed_kmh / 3.6 * 1.2  # 20% over target
            if vehicle_state['speed_ms'] > max_speed_ms:
                control.throttle = 0.0
                control.brake = min(0.5, (vehicle_state['speed_ms'] - max_speed_ms) / 10.0)

        except Exception as e:
            logger.error(f"Safety override error: {e}")

        return control

    def get_safety_status(self) -> SafetyStatus:
        """Get current safety status"""
        return self.safety_status

    def get_metrics(self) -> Dict[str, Any]:
        """Get control performance metrics"""
        metrics = {
            'control_count': self.control_count,
            'target_speed_kmh': self.target_speed_kmh,
            'safety_enabled': self.enable_safety,
            'stuck_recovery_enabled': self.enable_stuck_recovery
        }

        if self.control_times:
            metrics['avg_control_time'] = np.mean(self.control_times)
            metrics['max_control_time'] = np.max(self.control_times)
            metrics['last_control_time'] = self.control_times[-1]

        return metrics

    def reset(self):
        """Reset control state"""
        self.last_control = None
        self.control_count = 0
        self.control_times.clear()
        self.safety_status = SafetyStatus(active=False)

        # Reset components if available
        if self.stuck_recovery:
            try:
                self.stuck_recovery.reset()
            except Exception:
                pass

    def cleanup(self):
        """Cleanup control resources"""
        self.reset()
        logger.info("Control manager cleaned up")
