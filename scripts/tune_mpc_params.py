#!/usr/bin/env python3
"""
Tune MPC parameters for maximum stability (minimum jitter + tracking error).

Modes:
  --synthetic   Generate a short scenario (straight + curve) and run MPC in a loop
                with different param sets. No CARLA. Fast.
  --replay-csv  Read meta.csv from a recorded run; replay states through MPC
                with different params. One recording, many param sets.

Stability score (lower = better):
  score = mean(cte^2) + w_steer_std * std(steer) + w_fail * solver_fail_rate
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.lane_mpc import LaneMPC, MPCConfig


def generate_synthetic_scenario(n_steps: int = 200, seed: int = 42) -> list[tuple]:
    """Generate (cte_m, heading_rad, curvature, speed_ms) for straight + gentle curve."""
    rng = np.random.default_rng(seed)
    out = []
    v_ref = 6.0  # m/s
    for i in range(n_steps):
        t = i * 0.1
        # Slight curve in the middle
        curv = 0.02 * np.sin(t * 0.3) + 0.005
        # Small CTE and heading drift
        cte = 0.2 * np.sin(t * 0.2) + 0.1 * rng.standard_normal()
        head = 0.02 * np.sin(t * 0.25) + 0.01 * rng.standard_normal()
        speed = v_ref + 0.5 * rng.standard_normal()
        speed = max(2.0, min(10.0, speed))
        out.append((float(cte), float(head), float(curv), float(speed)))
    return out


def load_replay_csv(path: Path) -> list[tuple]:
    """Load (cte_m, heading_rad, curvature, speed_ms) from meta.csv."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cte = float(row.get("cte_m", 0))
                head = float(row.get("heading_rad", 0))
                curv = float(row.get("curvature", 0))
                speed_kmh = float(row.get("speed_kmh", 25))
                speed_ms = speed_kmh / 3.6
                rows.append((cte, head, curv, speed_ms))
            except (KeyError, ValueError):
                continue
    return rows


def run_mpc_sequence(
    scenario: list[tuple],
    cfg: MPCConfig,
    v_ref: float = 6.0,
) -> tuple[list[float], list[float], list[str]]:
    """Run MPC over scenario; return steers, accels, statuses."""
    mpc = LaneMPC(cfg)
    steers, accels, statuses = [], [], []
    for cte, head, curv, speed in scenario:
        st, acc, status = mpc.solve(
            x0=0.0, y0=cte, psi0=head, v0=speed,
            v_ref=v_ref, cte=cte, heading_err=head, curvature=curv, confidence=0.9,
        )
        steers.append(st)
        accels.append(acc)
        statuses.append(status)
    return steers, accels, statuses


def stability_score(
    scenario: list[tuple],
    steers: list[float],
    statuses: list[str],
    w_cte: float = 1.0,
    w_steer_std: float = 10.0,
    w_fail: float = 50.0,
) -> float:
    """Lower is better. Combines tracking error, steer smoothness, solver reliability."""
    cte_vals = [s[0] for s in scenario]
    mean_cte_sq = np.mean([c * c for c in cte_vals])
    steer_std = float(np.std(steers)) if len(steers) > 1 else 0.0
    fail_count = sum(1 for s in statuses if s != "Solve_Succeeded")
    fail_rate = fail_count / max(1, len(statuses))
    return w_cte * mean_cte_sq + w_steer_std * steer_std + w_fail * fail_rate


def grid_search(
    scenario: list[tuple],
    base_cfg: MPCConfig,
    grid: dict,
) -> tuple[dict, float]:
    """Try all combinations in grid; return best params and best score."""
    keys = list(grid.keys())
    values = list(grid.values())

    def expand(i: int, combo: dict) -> None:
        if i == len(keys):
            cfg = MPCConfig(
                N=combo.get("N", base_cfg.N),
                dt=combo.get("dt", base_cfg.dt),
                w_cte=combo.get("w_cte", base_cfg.w_cte),
                w_heading=combo.get("w_heading", base_cfg.w_heading),
                w_steer_rate=combo.get("w_steer_rate", base_cfg.w_steer_rate),
                w_steer_jerk=combo.get("w_steer_jerk", base_cfg.w_steer_jerk),
                max_steer_rate=combo.get("max_steer_rate", base_cfg.max_steer_rate),
                w_vel=base_cfg.w_vel,
                w_steer=base_cfg.w_steer,
                w_accel=base_cfg.w_accel,
                w_accel_rate=base_cfg.w_accel_rate,
                use_adaptive_weights=base_cfg.use_adaptive_weights,
                L=base_cfg.L,
                max_steer=base_cfg.max_steer,
                max_accel=base_cfg.max_accel,
                min_accel=base_cfg.min_accel,
            )
            steers, _, statuses = run_mpc_sequence(scenario, cfg, v_ref=6.0)
            score = stability_score(scenario, steers, statuses)
            results.append((combo.copy(), score))
            return
        for v in values[i]:
            combo[keys[i]] = v
            expand(i + 1, combo)

    results = []
    expand(0, {})
    best = min(results, key=lambda x: x[1])
    return best[0], best[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune MPC for stability (grid search)")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic scenario (no CARLA)")
    ap.add_argument("--replay-csv", type=str, default=None, help="Path to meta.csv from a run")
    ap.add_argument("--steps", type=int, default=200, help="Synthetic scenario length")
    ap.add_argument("--write-config", action="store_true", help="Write best mpc section to config.yaml")
    ap.add_argument("--quick", action="store_true", help="Smaller grid (fewer combinations)")
    args = ap.parse_args()

    if args.replay_csv:
        scenario = load_replay_csv(Path(args.replay_csv))
        if not scenario:
            print("No rows in CSV or wrong format. Need: cte_m, heading_rad, curvature, speed_kmh")
            sys.exit(1)
        print("Replay from %s: %d steps" % (args.replay_csv, len(scenario)))
    else:
        scenario = generate_synthetic_scenario(n_steps=args.steps)
        print("Synthetic scenario: %d steps (straight + curve + noise)" % len(scenario))

    try:
        from config import (
            MPC_N, MPC_DT, MPC_L, MPC_MAX_STEER, MPC_MAX_STEER_RATE,
            MPC_MAX_ACCEL, MPC_MIN_ACCEL, MPC_W_CTE, MPC_W_HEADING,
            MPC_W_VEL, MPC_W_STEER, MPC_W_ACCEL, MPC_W_STEER_RATE,
            MPC_W_ACCEL_RATE, MPC_W_STEER_JERK, MPC_USE_ADAPTIVE_WEIGHTS,
        )
        base_cfg = MPCConfig(
            N=MPC_N, dt=MPC_DT, L=MPC_L, max_steer=MPC_MAX_STEER,
            max_steer_rate=MPC_MAX_STEER_RATE, max_accel=MPC_MAX_ACCEL, min_accel=MPC_MIN_ACCEL,
            w_cte=MPC_W_CTE, w_heading=MPC_W_HEADING, w_vel=MPC_W_VEL,
            w_steer=MPC_W_STEER, w_accel=MPC_W_ACCEL,
            w_steer_rate=MPC_W_STEER_RATE, w_accel_rate=MPC_W_ACCEL_RATE,
            w_steer_jerk=MPC_W_STEER_JERK, use_adaptive_weights=MPC_USE_ADAPTIVE_WEIGHTS,
        )
    except ImportError:
        base_cfg = MPCConfig(N=10, dt=0.1, w_cte=200.0, w_heading=85.0, w_steer_rate=500.0)

    if args.quick:
        grid = {
            "w_cte": [100.0, 200.0, 400.0],
            "w_heading": [50.0, 85.0, 120.0],
            "w_steer_rate": [300.0, 500.0, 800.0],
        }
    else:
        grid = {
            "w_cte": [100.0, 200.0, 350.0, 500.0],
            "w_heading": [50.0, 85.0, 120.0, 160.0],
            "w_steer_rate": [300.0, 500.0, 800.0, 1200.0],
        }

    n_runs = int(np.prod([len(v) for v in grid.values()]))
    print("Grid search over w_cte, w_heading, w_steer_rate (%d runs)..." % n_runs)
    best_params, best_score = grid_search(scenario, base_cfg, grid)
    print("Best stability score: %.4f" % best_score)
    print("Best params: w_cte=%.1f w_heading=%.1f w_steer_rate=%.1f" % (
        best_params.get("w_cte"), best_params.get("w_heading"), best_params.get("w_steer_rate")))

    if args.write_config:
        config_path = PROJECT_ROOT / "config.yaml"
        if not config_path.exists():
            print("config.yaml not found, skip write")
            return
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Find mpc section and update values
        in_mpc = False
        new_lines = []
        for line in lines:
            if line.strip() == "mpc:":
                in_mpc = True
                new_lines.append(line)
                continue
            if in_mpc and line.startswith("  MPC_") and ":" in line:
                key = line.split(":")[0].strip()
                if key == "MPC_W_CTE":
                    new_lines.append("  MPC_W_CTE: %.1f\n" % best_params.get("w_cte", 200.0))
                elif key == "MPC_W_HEADING":
                    new_lines.append("  MPC_W_HEADING: %.1f\n" % best_params.get("w_heading", 85.0))
                elif key == "MPC_W_STEER_RATE":
                    new_lines.append("  MPC_W_STEER_RATE: %.1f\n" % best_params.get("w_steer_rate", 500.0))
                else:
                    new_lines.append(line)
                continue
            if in_mpc and line.strip() and not line.startswith(" "):
                in_mpc = False
            new_lines.append(line)
        with open(config_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print("Updated config.yaml mpc section with best params.")


if __name__ == "__main__":
    main()
