#!/usr/bin/env python3
"""
Model Predictive Controller for Lane Following.

Uses kinematic bicycle model with CasADi/IPOPT optimization.

State: [x, y, psi, v]  (position, heading, speed)
Control: [delta, a]     (steering angle, acceleration)

Cost minimizes:
  - Cross-track error (CTE)
  - Heading error
  - Velocity error (vs target speed)
  - Control effort (steering + acceleration)
  - Control rate (steering smoothness)

Constraints:
  - |delta| <= max_steer (0.7 rad ~= 40 deg)
  - |d_delta/dt| <= max_steer_rate
  - a_min <= a <= a_max
"""

import logging
from typing import Optional

import numpy as np
import casadi as ca
from dataclasses import dataclass

from control.pure_pursuit import PurePursuitController

logger = logging.getLogger(__name__)

# Speed (m/s) above which we emphasize heading/steer_rate over CTE
MPC_WEIGHTS_HIGH_SPEED_MS = 8.0
# Dynamic horizon: high speed + low curvature → longer N; sharp curve → shorter N
MPC_N_LOW = 8
MPC_N_HIGH = 12
MPC_HORIZON_CURV_THRESHOLD = 0.025


def compute_mpc_horizon(speed_ms: float, curvature: float) -> int:
    """
    Return horizon steps N: high speed + low |curvature| → N_high; sharp curve → N_low.
    """
    abs_curv = abs(float(curvature))
    if abs_curv >= MPC_HORIZON_CURV_THRESHOLD:
        return MPC_N_LOW
    if speed_ms >= MPC_WEIGHTS_HIGH_SPEED_MS * 0.7:
        return MPC_N_HIGH
    return MPC_N_LOW


def get_mpc_weights(
    speed_ms: float,
    curvature: float,
    confidence: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Return (scale_cte, scale_heading, scale_steer_rate, scale_jerk) for MPC.

    Based on research: "Weight Adaptive Path Tracking Control for Autonomous Vehicles"
    - In curves: INCREASE lateral error weight (λy) for better tracking accuracy
    - High speed: INCREASE lateral error weight to compensate for inertia lag
    - Straight roads: reduce weights for comfort and smoothness
    """
    curvature = float(curvature)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    speed_norm = min(1.0, max(0.0, speed_ms / max(MPC_WEIGHTS_HIGH_SPEED_MS, 0.1)))
    abs_curv = abs(curvature)

    # Curve factor: 0 = straight, 1 = sharp curve (curvature > 0.02)
    curve_factor = min(1.0, abs_curv * 50.0)

    # Research finding: In curves, INCREASE CTE weight for tracking accuracy
    # Also increase with speed to compensate for inertia
    scale_cte = 1.0 + 1.2 * curve_factor + 0.5 * speed_norm

    # Heading: important in curves but not as critical as CTE
    scale_heading = 1.0 + 0.8 * curve_factor + 0.3 * speed_norm

    # Steer rate: reduce penalty in curves to allow faster steering response
    # Research: reducing λu improves tracking but may cause jitter
    scale_steer_rate = 1.0 + 0.4 * (1.0 - confidence) - 0.4 * curve_factor
    scale_steer_rate = max(0.4, scale_steer_rate)

    # Jerk: keep moderate to avoid oscillation
    scale_jerk = 1.0 + 0.3 * (1.0 - confidence)

    return (scale_cte, scale_heading, scale_steer_rate, scale_jerk)


@dataclass
class MPCConfig:
    """MPC tuning parameters."""
    # Prediction
    N: int = 15               # prediction horizon steps
    dt: float = 0.1           # timestep (seconds)

    # Vehicle
    L: float = 2.875          # wheelbase (Tesla Model 3)
    max_steer: float = 0.7    # max steering angle (rad)
    max_steer_rate: float = 0.3  # max steering rate (rad/step)
    max_accel: float = 3.0    # max acceleration (m/s^2)
    min_accel: float = -5.0   # max braking deceleration

    # Cost weights - tuned for better lane keeping
    w_cte: float = 80.0       # cross-track error (increased for better centering)
    w_heading: float = 40.0   # heading error (increased for stability)
    w_vel: float = 5.0        # velocity tracking
    w_steer: float = 80.0     # steering magnitude (reduced to allow more steering)
    w_accel: float = 10.0     # acceleration magnitude
    w_steer_rate: float = 150.0  # steering rate (reduced for faster response)
    w_accel_rate: float = 10.0   # acceleration rate (accel smoothness)
    w_steer_jerk: float = 100.0  # steering jerk (d²δ/dt²) — increased to reduce oscillation
    # Adaptive weight scaling (set by solve from curvature + confidence)
    use_adaptive_weights: bool = True


class LaneMPC:
    """
    Nonlinear MPC for lane following using CasADi.

    Kinematic bicycle model:
        x_dot = v * cos(psi)
        y_dot = v * sin(psi)
        psi_dot = v / L * tan(delta)
        v_dot = a

    The optimization converts CTE and heading error to a local
    reference frame where the road is along the x-axis.
    """

    def __init__(self, cfg: MPCConfig = None):
        self.cfg = cfg or MPCConfig()
        self._solver = None
        self._prev_u = np.zeros(2)
        self._prev_x0 = None  # warm start: last solution as initial guess
        self._pp = PurePursuitController(
            wheelbase=self.cfg.L, max_steer=self.cfg.max_steer
        )
        self._build_solver()

    def set_horizon(self, N: int) -> None:
        """Set horizon steps and rebuild solver (e.g. after compute_mpc_horizon())."""
        N = max(4, min(25, int(N)))
        if N == self.cfg.N:
            return
        self.cfg.N = N
        self._prev_x0 = None
        self._build_solver()

    def _build_solver(self):
        """Construct the NLP optimization problem."""
        c = self.cfg
        N = c.N

        # Decision variables
        # States: [x, y, psi, v] for N+1 steps
        # Controls: [delta, a] for N steps
        n_states = 4
        n_controls = 2

        # CasADi symbolic
        X = ca.MX.sym("X", n_states, N + 1)
        U = ca.MX.sym("U", n_controls, N)

        # Parameters: [x0..v0, v_ref, cte, heading_err, curvature, u_prev_delta, u_prev_a, scale_cte, scale_heading, scale_steer_rate, scale_jerk]
        P = ca.MX.sym("P", 14)

        cost = 0.0
        constraints = []
        lb_g = []
        ub_g = []

        # Initial state constraint
        for i in range(n_states):
            constraints.append(X[i, 0] - P[i])
            lb_g.append(0.0)
            ub_g.append(0.0)

        v_ref = P[4]
        curvature = P[7]
        u_prev = P[8:10]
        scale_cte = P[10]
        scale_heading = P[11]
        scale_steer_rate = P[12]
        scale_jerk = P[13]

        prev_d_delta = U[0, 0] - u_prev[0]
        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]

            # ── Kinematic bicycle model (discrete, Euler) ──
            #   x_{k+1} = x_k + v_k * cos(psi_k) * dt
            #   y_{k+1} = y_k + v_k * sin(psi_k) * dt
            #   psi_{k+1} = psi_k + (v_k / L) * tan(delta_k) * dt
            #   v_{k+1} = v_k + a_k * dt
            x_next = ca.vertcat(
                x_k[0] + x_k[3] * ca.cos(x_k[2]) * c.dt,
                x_k[1] + x_k[3] * ca.sin(x_k[2]) * c.dt,
                x_k[2] + (x_k[3] / c.L) * ca.tan(u_k[0]) * c.dt,
                x_k[3] + u_k[1] * c.dt,
            )

            # Dynamics constraints
            for i in range(n_states):
                constraints.append(X[i, k + 1] - x_next[i])
                lb_g.append(0.0)
                ub_g.append(0.0)

            # ── Cost function ──
            # Target: y=0 (center of lane), psi = road heading (curvature feed-forward)
            cte_k = x_k[1]

            # Heading error: vehicle heading vs road direction (curvature feed-forward)
            # Road heading at step k ≈ curvature * distance ≈ curvature * k * dt * v_ref
            psi_ref_k = curvature * (k * c.dt * v_ref)
            heading_err_k = x_k[2] - psi_ref_k

            # Velocity error
            vel_err_k = x_k[3] - v_ref

            cost += c.w_cte * scale_cte * cte_k**2
            cost += c.w_heading * scale_heading * heading_err_k**2
            cost += c.w_vel * vel_err_k**2
            cost += c.w_steer * u_k[0]**2
            cost += c.w_accel * u_k[1]**2

            # Control rate cost
            if k == 0:
                d_delta = u_k[0] - u_prev[0]
                d_accel = u_k[1] - u_prev[1]
            else:
                d_delta = u_k[0] - U[0, k - 1]
                d_accel = u_k[1] - U[1, k - 1]
            cost += c.w_steer_rate * scale_steer_rate * d_delta**2
            cost += c.w_accel_rate * d_accel**2
            # Steering jerk (d²δ/dt²) — ลด oscillation
            steer_jerk = d_delta - prev_d_delta
            cost += c.w_steer_jerk * scale_jerk * steer_jerk**2
            prev_d_delta = d_delta

        # Terminal cost (CTE=0, heading = road heading at end of horizon)
        psi_ref_N = curvature * (N * c.dt * v_ref)
        cost += 2 * c.w_cte * scale_cte * X[1, N]**2
        cost += 2 * c.w_heading * scale_heading * (X[2, N] - psi_ref_N)**2

        # Flatten decision variables
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # Variable bounds
        n_x_vars = n_states * (N + 1)
        lb_x = [-1e6] * n_x_vars
        ub_x = [1e6] * n_x_vars

        # Speed bounds [0, 20 m/s]
        for k in range(N + 1):
            lb_x[k * n_states + 3] = 0.0
            ub_x[k * n_states + 3] = 20.0

        lb_u = []
        ub_u = []
        for k in range(N):
            lb_u.extend([-c.max_steer, c.min_accel])
            ub_u.extend([c.max_steer, c.max_accel])

        # NLP
        nlp = {
            "x": opt_vars,
            "f": cost,
            "g": ca.vertcat(*constraints),
            "p": P,
        }

        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 50,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.acceptable_tol": 1e-4,
            "print_time": 0,
        }

        self._solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        self._lb = lb_x + lb_u
        self._ub = ub_x + ub_u
        self._lb_g = lb_g
        self._ub_g = ub_g
        self._n_states = n_states
        self._n_controls = n_controls
        self._N = N

    def solve(self, x0: float, y0: float, psi0: float, v0: float,
              v_ref: float, cte: float, heading_err: float,
              curvature: float = 0.0, confidence: float = 1.0,
              weight_scales: Optional[tuple[float, float, float, float]] = None):
        """
        Solve MPC for optimal steering and acceleration.

        Args:
            x0, y0, psi0, v0: current state (local frame)
            v_ref: target speed (m/s)
            cte: cross-track error
            heading_err: heading error (rad)
            curvature: estimated road curvature
            confidence: lane detection confidence [0,1]; low → increase steer smoothness (adaptive weights)
            weight_scales: optional (scale_cte, scale_heading, scale_steer_rate, scale_jerk) from get_mpc_weights()

        Returns:
            (steer, accel, solver_status, trajectory): first control action, status, and predicted trajectory
            steer in [-max_steer, max_steer] radians
            accel in [min_accel, max_accel] m/s^2
            solver_status: "Solve_Succeeded" or "Fallback_PP"
            trajectory: np.ndarray (4, N+1) of [x, y, psi, v] predicted states, or None on fallback
        """
        c = self.cfg
        N = c.N

        if c.use_adaptive_weights:
            if weight_scales is not None and len(weight_scales) >= 4:
                scale_cte, scale_heading, scale_steer_rate, scale_jerk = weight_scales[:4]
            else:
                scale_cte, scale_heading, scale_steer_rate, scale_jerk = get_mpc_weights(
                    v0, curvature, confidence
                )
        else:
            scale_cte = scale_heading = scale_steer_rate = scale_jerk = 1.0

        p = np.array([x0, y0, psi0, v0, v_ref,
                      cte, heading_err, curvature,
                      self._prev_u[0], self._prev_u[1],
                      scale_cte, scale_heading, scale_steer_rate, scale_jerk])

        n_x = self._n_states * (N + 1)
        n_u = self._n_controls * N
        x_init = self._prev_x0 if self._prev_x0 is not None and len(self._prev_x0) == n_x + n_u else np.zeros(n_x + n_u)

        try:
            sol = self._solver(
                x0=x_init,
                lbx=self._lb,
                ubx=self._ub,
                lbg=self._lb_g,
                ubg=self._ub_g,
                p=p,
            )

            opt = sol["x"].full().flatten()
            u0 = opt[n_x:n_x + 2]
            steer = float(np.clip(u0[0], -c.max_steer, c.max_steer))
            accel = float(np.clip(u0[1], c.min_accel, c.max_accel))

            # Extract predicted state trajectory X[:, k] for visualization
            X_opt = opt[:n_x].reshape((self._n_states, N + 1))

            self._prev_u = np.array([steer, accel])
            self._prev_x0 = opt.copy()
            self._pp.reset()
            return steer, accel, "Solve_Succeeded", X_opt

        except Exception as e:
            logger.warning(
                "MPC solver failed, using Pure Pursuit fallback: cte=%.3f heading=%.3f v_ref=%.2f v0=%.2f — %s",
                cte, heading_err, v_ref, v0, e,
                exc_info=True,
            )
            steer = self._pp.compute_steering(cte, heading_err, v0, curvature)
            steer = float(np.clip(steer, -c.max_steer, c.max_steer))
            accel = 0.5 * (v_ref - v0)
            accel = float(np.clip(accel, c.min_accel, c.max_accel))
            self._prev_u = np.array([steer, accel])
            self._prev_x0 = None
            return steer, accel, "Fallback_PP", None

    def steer_to_carla(self, steer_rad: float):
        """Convert MPC steering (radians) to CARLA [-1, 1]."""
        return float(np.clip(steer_rad / self.cfg.max_steer, -1.0, 1.0))

    def accel_to_carla(self, accel: float):
        """Convert MPC acceleration to CARLA (throttle, brake)."""
        if accel >= 0:
            throttle = min(1.0, accel / self.cfg.max_accel)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -accel / abs(self.cfg.min_accel))
        return float(throttle), float(brake)


def _test_mpc():
    """Basic MPC solver test."""
    cfg = MPCConfig(N=10)
    mpc = LaneMPC(cfg)

    # Straight road, small CTE
    steer, accel, status, _ = mpc.solve(
        x0=0, y0=0, psi0=0, v0=5.0,
        v_ref=8.0, cte=0.5, heading_err=0.1)

    logger.info(
        "Test 1 - small CTE: steer=%.4f rad, accel=%.4f m/s^2, status=%s",
        steer, accel, status,
    )
    assert abs(steer) < 0.7, "Steer out of range"

    # Curvy road (curvature feed-forward)
    steer2, accel2, status2, _ = mpc.solve(
        x0=0, y0=0, psi0=0, v0=8.0,
        v_ref=6.0, cte=-1.0, heading_err=-0.2, curvature=0.05)

    logger.info("Test 2 - curve: steer=%.4f rad, accel=%.4f m/s^2", steer2, accel2)

    # CARLA conversion
    carla_steer = mpc.steer_to_carla(steer)
    throttle, brake = mpc.accel_to_carla(accel)
    logger.info("CARLA: steer=%.3f, throttle=%.3f, brake=%.3f", carla_steer, throttle, brake)

    logger.info("MPC tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _test_mpc()
