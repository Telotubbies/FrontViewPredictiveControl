#!/usr/bin/env python3
"""
Pure Pursuit Controller for Lane Following.

Simple and stable geometric controller that follows a lookahead point.
Best for straight roads and gentle curves.
"""

import math
from typing import Tuple


class PurePursuitController:
    """
    Pure Pursuit lateral controller.

    Computes steering angle to follow a path by looking ahead
    to a target point and steering towards it.
    """

    def __init__(
        self,
        wheelbase: float = 2.875,      # Tesla Model 3 wheelbase
        min_lookahead: float = 4.0,    # Minimum lookahead distance (m) - reduced for faster response
        max_lookahead: float = 20.0,   # Maximum lookahead distance (m) - reduced for tighter control
        lookahead_gain: float = 0.6,   # Lookahead = gain * speed - reduced for faster steering
        max_steer: float = 0.7,        # Max steering angle (rad)
    ):
        self.L = wheelbase
        self.min_ld = min_lookahead
        self.max_ld = max_lookahead
        self.ld_gain = lookahead_gain
        self.max_steer = max_steer

        # Smoothing
        self._prev_steer = 0.0
        self._alpha = 0.5  # EMA smoothing factor - increased for faster response

        # Debug info (updated during compute_steering)
        self.last_lookahead = 0.0
        self.last_target_y = 0.0
        self.last_raw_steer = 0.0
        self.last_smoothed_steer = 0.0

    def compute_lookahead(self, speed_ms: float, curvature: float = 0.0) -> float:
        """Compute speed-dependent lookahead distance.

        Applies curvature-adaptive reduction: when |curvature| > 0.02 the
        maximum lookahead is reduced by 30% so sharper curves use a shorter
        lookahead for better tracking.
        """
        max_ld = self.max_ld
        if abs(curvature) > 0.02:
            max_ld = self.max_ld * 0.7
        ld = self.ld_gain * speed_ms
        return max(self.min_ld, min(max_ld, ld))

    def compute_steering(
        self,
        cte: float,
        heading_err: float,
        speed_ms: float,
        curvature: float = 0.0,
    ) -> float:
        """
        Compute steering angle using Pure Pursuit geometry.

        Args:
            cte: Cross-track error (m), positive = right of center
            heading_err: Heading error (rad), positive = pointing right
            speed_ms: Current speed (m/s)
            curvature: Road curvature (1/m)

        Returns:
            Steering angle in radians
        """
        ld = self.compute_lookahead(speed_ms, curvature)

        # Speed-dependent steering smoothing alpha:
        # low speed (< 3 m/s) -> responsive (0.3), high speed (> 15 m/s) -> smooth (0.8)
        if speed_ms < 3.0:
            alpha = 0.3
        elif speed_ms > 15.0:
            alpha = 0.8
        else:
            alpha = self._alpha

        # Target point in vehicle frame
        # x = lookahead distance ahead
        # y = lateral offset = cte + heading_err * ld (first-order approximation)
        target_y = cte + heading_err * ld

        # Add curvature feed-forward
        # For a curved road, the target point shifts by curvature * ld^2 / 2
        target_y += 0.5 * curvature * ld * ld

        # Pure pursuit formula: delta = atan(2 * L * sin(alpha) / ld)
        # where alpha = atan(target_y / ld)
        # Simplified: delta = atan(2 * L * target_y / ld^2)
        if ld > 0.1:
            steer = math.atan(2.0 * self.L * target_y / (ld * ld))
        else:
            steer = 0.0

        # Clamp steering
        steer = max(-self.max_steer, min(self.max_steer, steer))

        # Store raw (pre-smoothing) steer for debugging
        raw_steer = steer

        # Smooth steering
        steer = alpha * steer + (1 - alpha) * self._prev_steer
        self._prev_steer = steer

        # Store debug info
        self.last_lookahead = ld
        self.last_target_y = target_y
        self.last_raw_steer = raw_steer
        self.last_smoothed_steer = steer

        return steer

    def get_debug_info(self) -> dict:
        """Return debug info from the last compute_steering call."""
        return {
            "last_lookahead": self.last_lookahead,
            "last_target_y": self.last_target_y,
            "last_raw_steer": self.last_raw_steer,
            "last_smoothed_steer": self.last_smoothed_steer,
        }

    def steer_to_carla(self, steer_rad: float) -> float:
        """Convert steering angle to CARLA control [-1, 1]."""
        # CARLA steering: -1 = full left, +1 = full right
        # Assuming max_steer corresponds to ±1
        return max(-1.0, min(1.0, steer_rad / self.max_steer))

    def reset(self):
        """Reset controller state."""
        self._prev_steer = 0.0


class HybridController:
    """
    Hybrid controller: Pure Pursuit for straight roads, MPC for curves.

    Switches based on road curvature:
    - Low curvature (< threshold): Pure Pursuit (stable, smooth)
    - High curvature (>= threshold): MPC (better tracking in curves)
    """

    # Curvature threshold for switching
    CURV_THRESHOLD_LOW = 0.015   # Below this: Pure Pursuit only
    CURV_THRESHOLD_HIGH = 0.025  # Above this: MPC only

    def __init__(self, mpc_controller, wheelbase: float = 2.875):
        self.mpc = mpc_controller
        self.pp = PurePursuitController(wheelbase=wheelbase)
        self._prev_steer = 0.0
        self._mode = "PP"  # "PP" or "MPC"

    def compute_steering(
        self,
        cte: float,
        heading_err: float,
        speed_ms: float,
        curvature: float,
        confidence: float = 1.0,
        **mpc_kwargs,
    ) -> Tuple[float, str]:
        """
        Compute steering using hybrid approach.

        Returns:
            (steer_rad, mode): Steering angle and controller mode used
        """
        abs_curv = abs(curvature)

        # Determine blend weight
        if abs_curv < self.CURV_THRESHOLD_LOW:
            # Straight road: Pure Pursuit only
            pp_weight = 1.0
            self._mode = "PP"
        elif abs_curv > self.CURV_THRESHOLD_HIGH:
            # Sharp curve: MPC only
            pp_weight = 0.0
            self._mode = "MPC"
        else:
            # Transition zone: blend
            t = (abs_curv - self.CURV_THRESHOLD_LOW) / (self.CURV_THRESHOLD_HIGH - self.CURV_THRESHOLD_LOW)
            pp_weight = 1.0 - t
            self._mode = "HYBRID"

        # Pure Pursuit steering
        pp_steer = self.pp.compute_steering(cte, heading_err, speed_ms, curvature)

        if pp_weight >= 1.0:
            # Pure Pursuit only
            return pp_steer, self._mode

        # MPC steering (if needed)
        try:
            mpc_steer_rad, _, _, _ = self.mpc.solve(
                x0=0, y0=cte, psi0=heading_err, v0=speed_ms,
                v_ref=mpc_kwargs.get('v_ref', speed_ms),
                cte=cte, heading_err=heading_err,
                curvature=curvature, confidence=confidence,
            )
            mpc_steer = self.mpc.steer_to_carla(mpc_steer_rad)
        except Exception:
            # MPC failed, use Pure Pursuit
            return pp_steer, "PP"

        if pp_weight <= 0.0:
            # MPC only
            return mpc_steer, self._mode

        # Blend
        blended = pp_weight * pp_steer + (1 - pp_weight) * mpc_steer

        # Smooth transition
        alpha = 0.3
        blended = alpha * blended + (1 - alpha) * self._prev_steer
        self._prev_steer = blended

        return blended, self._mode

    def reset(self):
        """Reset controller state."""
        self.pp.reset()
        self._prev_steer = 0.0
        self._mode = "PP"
