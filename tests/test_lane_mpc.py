"""Unit tests for control.lane_mpc.LaneMPC.solve (no CARLA, no GPU)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.lane_mpc import LaneMPC, MPCConfig


def test_mpc_solve_small_cte():
    """Small CTE and heading: steer and accel in range, status Solve_Succeeded or Fallback."""
    cfg = MPCConfig(N=5, dt=0.1)
    mpc = LaneMPC(cfg)
    steer, accel, status = mpc.solve(
        x0=0, y0=0, psi0=0, v0=5.0,
        v_ref=8.0, cte=0.5, heading_err=0.1,
    )
    assert -mpc.cfg.max_steer <= steer <= mpc.cfg.max_steer
    assert mpc.cfg.min_accel <= accel <= mpc.cfg.max_accel
    assert status in ("Solve_Succeeded", "Fallback")


def test_mpc_solve_curve():
    """Curved road (curvature): solution still bounded."""
    cfg = MPCConfig(N=5, dt=0.1)
    mpc = LaneMPC(cfg)
    steer, accel, status = mpc.solve(
        x0=0, y0=0, psi0=0, v0=8.0,
        v_ref=6.0, cte=-1.0, heading_err=-0.2, curvature=0.05,
    )
    assert -mpc.cfg.max_steer <= steer <= mpc.cfg.max_steer
    assert mpc.cfg.min_accel <= accel <= mpc.cfg.max_accel
    assert status in ("Solve_Succeeded", "Fallback")


def test_mpc_steer_to_carla_bounds():
    """steer_to_carla clips to [-1, 1]."""
    cfg = MPCConfig(max_steer=0.7)
    mpc = LaneMPC(cfg)
    assert -1.0 <= mpc.steer_to_carla(0.0) <= 1.0
    assert mpc.steer_to_carla(10.0) == 1.0
    assert mpc.steer_to_carla(-10.0) == -1.0


def test_mpc_accel_to_carla():
    """accel_to_carla returns (throttle, brake) in [0,1]."""
    cfg = MPCConfig(max_accel=3.0, min_accel=-5.0)
    mpc = LaneMPC(cfg)
    th, br = mpc.accel_to_carla(1.0)
    assert 0 <= th <= 1.0
    assert br == 0.0
    th, br = mpc.accel_to_carla(-2.0)
    assert th == 0.0
    assert 0 <= br <= 1.0
