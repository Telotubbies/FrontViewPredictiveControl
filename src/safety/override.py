"""
Independent Safety Override System.

This module provides safety checks that must NEVER be bypassed by AI output.
Safety logic is completely independent from control module.
"""

import logging
import math
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from config import SAFETY_MAX_STEER_RAD, SAFETY_CTE_LIMIT_M, SAFETY_CTE_MAX_SPEED_REDUCTION
except ImportError:
    SAFETY_MAX_STEER_RAD = 0.45
    SAFETY_CTE_LIMIT_M = 2.0
    SAFETY_CTE_MAX_SPEED_REDUCTION = 0.5


@dataclass
class SafetyConfig:
    """Safety configuration."""
    emergency_brake_enabled: bool = True
    max_speed_kmh: float = 30.0
    min_speed_kmh: float = 0.0
    max_steering_angle: float = 0.5
    collision_timeout: float = 5.0
    emergency_deceleration: float = -5.0  # m/s²
    max_steer_rate: float = 0.1


class SafetyOverride:
    """
    Independent safety override system.

    This class provides safety checks that override control commands
    when safety limits are violated. It is completely independent
    from the control module to ensure safety cannot be bypassed.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety override system.

        Args:
            config: Safety configuration dictionary
        """
        self.config = SafetyConfig(
            emergency_brake_enabled=config.get('emergency_brake_enabled', True),
            max_speed_kmh=config.get('max_speed_kmh', 30.0),
            min_speed_kmh=config.get('min_speed_kmh', 0.0),
            max_steering_angle=config.get('max_steering_angle', SAFETY_MAX_STEER_RAD),
            collision_timeout=config.get('collision_timeout', 5.0),
            emergency_deceleration=config.get('emergency_deceleration', -5.0),
            max_steer_rate=config.get('max_steer_rate', 0.1)
        )

        # State tracking
        self.last_safety_override = None
        self.override_count = 0
        self._prev_steer = 0.0

        logger.info("✅ SafetyOverride initialized")
        logger.info(f"  - Emergency brake: {self.config.emergency_brake_enabled}")
        logger.info(f"  - Max speed: {self.config.max_speed_kmh} km/h")
        logger.info(f"  - Max steering: {self.config.max_steering_angle} rad")

    def check_speed_limit(
        self,
        vehicle_state: Dict[str, Any],
        throttle: float,
        brake: float
    ) -> Tuple[float, float]:
        """
        Check and enforce speed limit.

        Args:
            vehicle_state: Current vehicle state with 'velocity' (m/s)
            throttle: Proposed throttle value
            brake: Proposed brake value

        Returns:
            (throttle, brake) with safety overrides applied
        """
        velocity_ms = vehicle_state.get('velocity', 0.0)
        speed_kmh = velocity_ms * 3.6

        # Check maximum speed
        if speed_kmh > self.config.max_speed_kmh:
            if self.config.emergency_brake_enabled:
                logger.warning(
                    f"⚠️  SPEED LIMIT EXCEEDED: {speed_kmh:.1f} km/h > {self.config.max_speed_kmh:.1f} km/h"
                )
                self.override_count += 1
                self.last_safety_override = 'speed_limit'
                return 0.0, 1.0  # Full brake, no throttle

        # Check minimum speed (prevent negative)
        if speed_kmh < self.config.min_speed_kmh:
            # This is handled by control logic, but we log it
            if speed_kmh < -0.1:  # Negative speed is invalid
                logger.warning(f"⚠️  Invalid negative speed: {speed_kmh:.1f} km/h")
                return throttle, min(brake, 0.5)  # Reduce brake if negative

        return throttle, brake

    def check_throttle_brake_conflict(
        self,
        throttle: float,
        brake: float
    ) -> Tuple[float, float]:
        """
        Check for throttle/brake conflict.

        If both throttle and brake are applied simultaneously, brake wins
        for safety and throttle is zeroed.

        Args:
            throttle: Proposed throttle value
            brake: Proposed brake value

        Returns:
            (throttle, brake) with conflict resolved
        """
        if throttle > 0.1 and brake > 0.1:
            logger.warning(
                f"⚠️  THROTTLE/BRAKE CONFLICT: throttle={throttle:.3f}, "
                f"brake={brake:.3f} - brake wins, throttle zeroed"
            )
            self.override_count += 1
            self.last_safety_override = 'throttle_brake_conflict'
            return 0.0, brake

        return throttle, brake

    def check_steering_limit(
        self,
        steering: float
    ) -> float:
        """
        Check and enforce steering angle limits.

        Args:
            steering: Proposed steering angle (-1.0 to 1.0)

        Returns:
            Steering angle with safety limits applied
        """
        # Clamp steering to physical limits
        max_steer = self.config.max_steering_angle

        if abs(steering) > max_steer:
            clamped_steer = np.clip(steering, -max_steer, max_steer)
            logger.warning(
                f"⚠️  STEERING LIMIT EXCEEDED: {steering:.3f} > {max_steer:.3f}, "
                f"clamped to {clamped_steer:.3f}"
            )
            self.override_count += 1
            self.last_safety_override = 'steering_limit'
            return clamped_steer

        return steering

    def check_control_validity(
        self,
        steering: float,
        throttle: float,
        brake: float
    ) -> Tuple[float, float, float]:
        """
        Check control values for validity (NaN, Inf, out of range).

        Args:
            steering: Proposed steering value
            throttle: Proposed throttle value
            brake: Proposed brake value

        Returns:
            (steering, throttle, brake) with invalid values replaced
        """
        # Check for NaN or Inf
        if np.isnan(steering) or np.isinf(steering):
            logger.error(f"❌ Invalid steering value: {steering}, using 0.0")
            steering = 0.0
            self.override_count += 1
            self.last_safety_override = 'invalid_steering'

        if np.isnan(throttle) or np.isinf(throttle):
            logger.error(f"❌ Invalid throttle value: {throttle}, using 0.0")
            throttle = 0.0
            self.override_count += 1
            self.last_safety_override = 'invalid_throttle'

        if np.isnan(brake) or np.isinf(brake):
            logger.error(f"❌ Invalid brake value: {brake}, using 0.0")
            brake = 0.0
            self.override_count += 1
            self.last_safety_override = 'invalid_brake'

        # Clamp to valid ranges
        steering = np.clip(steering, -1.1, 1.1)
        throttle = np.clip(throttle, -0.1, 1.1)
        brake = np.clip(brake, -0.1, 1.1)

        return steering, throttle, brake

    def apply_safety_override(
        self,
        vehicle_state: Dict[str, Any],
        steering: float,
        throttle: float,
        brake: float
    ) -> Tuple[float, float, float]:
        """
        Apply all safety overrides to control commands.

        This is the main entry point for safety checks.
        It applies all safety rules in order and returns
        safe control commands.

        Args:
            vehicle_state: Current vehicle state
            steering: Proposed steering angle
            throttle: Proposed throttle value
            brake: Proposed brake value

        Returns:
            (steering, throttle, brake) with all safety overrides applied
        """
        # Step 1: Check control validity (NaN, Inf, ranges)
        steering, throttle, brake = self.check_control_validity(
            steering, throttle, brake
        )

        # Step 2: Check steering limits
        steering = self.check_steering_limit(steering)

        # Step 3: Check CTE-based intervention (NEW)
        throttle, brake = self.check_cte_intervention(
            vehicle_state, throttle, brake
        )

        # Step 4: Check speed limits
        throttle, brake = self.check_speed_limit(
            vehicle_state, throttle, brake
        )

        # Step 5: Check throttle/brake conflict
        throttle, brake = self.check_throttle_brake_conflict(throttle, brake)

        # Step 6: Jerk limiting (steering change rate)
        if self._prev_steer == 0.0 and steering != 0.0:
            self._prev_steer = steering  # Initialize, no rate limit on first frame
        else:
            # Apply jerk limiting
            delta = steering - self._prev_steer
            if abs(delta) > self.config.max_steer_rate:
                steering = self._prev_steer + math.copysign(
                    self.config.max_steer_rate, delta
                )
                logger.warning(
                    f"⚠️  STEER RATE LIMITED: delta={delta:.3f} > "
                    f"{self.config.max_steer_rate:.3f}, "
                    f"clamped to {steering:.3f}"
                )
                self.override_count += 1
                self.last_safety_override = 'steer_rate_limit'
            self._prev_steer = steering

        return steering, throttle, brake

    def check_cte_intervention(
        self,
        vehicle_state: Dict[str, Any],
        throttle: float,
        brake: float
    ) -> Tuple[float, float]:
        """
        Check CTE-based safety intervention.

        When CTE exceeds limit, reduce speed to allow more time for correction
        while maintaining steering authority for path correction.

        Args:
            vehicle_state: Current vehicle state (must include 'cte')
            throttle: Proposed throttle
            brake: Proposed brake

        Returns:
            (throttle, brake) with CTE intervention applied
        """
        # Get CTE from vehicle state
        current_cte = vehicle_state.get('cte', 0.0)

        # CTE limit configuration (from config)
        cte_limit = SAFETY_CTE_LIMIT_M
        max_speed_reduction = SAFETY_CTE_MAX_SPEED_REDUCTION

        if abs(current_cte) > cte_limit:
            # CTE exceeds limit - reduce speed to allow correction
            cte_excess = abs(current_cte) - cte_limit

            # Calculate speed reduction based on CTE excess
            speed_reduction_factor = min(max_speed_reduction, cte_excess * 0.25)

            # Apply speed reduction
            if throttle > 0:
                throttle *= (1.0 - speed_reduction_factor)

            # Add proportional braking for severe CTE
            if cte_excess > 1.0:  # More than 1m over limit
                proportional_brake = min(0.3, cte_excess * 0.1)
                brake = max(brake, proportional_brake)

            # Log safety intervention
            self.override_count += 1
            self.last_safety_override = f"CTE intervention: CTE={current_cte:.2f}m, reduction={speed_reduction_factor:.2f}"

            logger.warning(f"CTE safety intervention: CTE={current_cte:.2f}m > {cte_limit}m, speed reduction={speed_reduction_factor:.2f}")

        return throttle, brake

    def get_safety_status(self) -> Dict[str, Any]:
        """
        Get current safety system status.

        Returns:
            Dictionary with safety status information
        """
        return {
            'override_count': self.override_count,
            'last_override': self.last_safety_override,
            'config': {
                'emergency_brake_enabled': self.config.emergency_brake_enabled,
                'max_speed_kmh': self.config.max_speed_kmh,
                'max_steering_angle': self.config.max_steering_angle
            }
        }

    def reset_counters(self) -> None:
        """Reset safety override counters."""
        self.override_count = 0
        self.last_safety_override = None

