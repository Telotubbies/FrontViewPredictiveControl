#!/usr/bin/env python3
"""
run_metrics.py — วิเคราะห์ metrics จาก CSV ที่ MetricsCollector บันทึก

ใช้:
  python scripts/run_metrics.py --csv metrics_output/run_20250712_001234.csv
  python scripts/run_metrics.py --dir metrics_output/  # วิเคราะห์ไฟล์ล่าสุด

Output: พิมพ์สรุปทุก metric + สร้าง plot (ถ้า matplotlib มี)
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_csv(csv_path: str) -> List[Dict[str, str]]:
    """Load CSV into list of dicts"""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze(frames: List[Dict[str, str]]) -> Dict[str, object]:
    """Analyze frame data and return summary"""
    if not frames:
        return {"error": "no frames"}

    n = len(frames)

    def col(name: str, cast=float, default=0.0) -> List[float]:
        return [cast(r.get(name, default)) for r in frames]

    def col_bool(name: str) -> List[bool]:
        return [str(r.get(name, "")).lower() in ("true", "1", "yes") for r in frames]

    cte = col("cte_m")
    heading = col("heading_err_rad")
    conf = col("lane_conf")
    speed = col("speed_ms")
    steer = col("steering")
    throttle = col("throttle")
    brake = col("brake")
    fps = col("fps")
    solve_time = col("mpc_solve_time_ms")
    loop_time = col("loop_time_ms")

    geom_valid = col_bool("geometry_valid")
    p1 = col_bool("lane_phase_p1")
    p2 = col_bool("lane_phase_p2")
    p3 = col_bool("lane_phase_p3")
    p4 = col_bool("lane_phase_p4")
    p5 = col_bool("lane_phase_p5")
    mpc_fallback = col_bool("mpc_fallback")
    aeb = col_bool("aeb_active")
    acc = col_bool("acc_active")
    ldw_warn = col_bool("ldw_warning_active")
    safety = col_bool("safety_active")
    stuck = col_bool("stuck_recovery_active")

    def stats(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
        }

    def rate(vals: List[bool]) -> float:
        return round(sum(vals) / max(1, n), 4)

    summary: Dict[str, object] = {
        "total_frames": n,
        "duration_s": round(float(col("timestamp")[-1]) - float(col("timestamp")[0]), 2) if n > 1 else 0.0,

        "perception": {
            "cte_m": stats(cte),
            "cte_rmse_m": round(float(np.sqrt(np.mean(np.square(cte)))), 4),
            "cte_max_abs_m": round(float(np.max(np.abs(cte))), 4),
            "heading_err_rad": stats(heading),
            "heading_err_max_abs_rad": round(float(np.max(np.abs(heading))), 6),
            "lane_conf": stats(conf),
            "geometry_valid_rate": rate(geom_valid),
            "phase_p1_ok_rate": rate(p1),
            "phase_p2_ok_rate": rate(p2),
            "phase_p3_ok_rate": rate(p3),
            "phase_p4_ok_rate": rate(p4),
            "phase_p5_ok_rate": rate(p5),
        },

        "control": {
            "mpc_solve_time_ms": stats(solve_time),
            "mpc_fallback_count": sum(mpc_fallback),
            "mpc_fallback_rate": rate(mpc_fallback),
            "steering": stats(steer),
            "throttle": stats(throttle),
            "brake": stats(brake),
            "speed_ms": stats(speed),
        },

        "performance": {
            "fps": stats(fps),
            "loop_time_ms": stats(loop_time),
        },

        "safety": {
            "safety_active_count": sum(safety),
            "safety_active_rate": rate(safety),
            "stuck_recovery_count": sum(stuck),
            "aeb_trigger_count": sum(aeb),
            "acc_active_count": sum(acc),
            "ldw_warning_count": sum(ldw_warn),
        },
    }

    return summary


def print_summary(summary: Dict[str, object]) -> None:
    """Pretty print summary"""
    print("\n" + "=" * 70)
    print("RUN METRICS SUMMARY")
    print("=" * 70)

    print(f"\nFrames: {summary['total_frames']}")
    print(f"Duration: {summary['duration_s']:.1f}s")

    perc = summary["perception"]
    print("\n── Perception ──")
    print(f"  CTE:        mean={perc['cte_m']['mean']:.4f}m  std={perc['cte_m']['std']:.4f}m  "
          f"max_abs={perc['cte_max_abs_m']:.4f}m  RMSE={perc['cte_rmse_m']:.4f}m")
    print(f"  Heading:    mean={perc['heading_err_rad']['mean']:.6f}rad  "
          f"max_abs={perc['heading_err_max_abs_rad']:.6f}rad")
    print(f"  Confidence: mean={perc['lane_conf']['mean']:.4f}  min={perc['lane_conf']['min']:.4f}")
    print(f"  Geometry valid: {perc['geometry_valid_rate']*100:.1f}%")
    print(f"  Phase OK: P1={perc['phase_p1_ok_rate']*100:.1f}% P2={perc['phase_p2_ok_rate']*100:.1f}% "
          f"P3={perc['phase_p3_ok_rate']*100:.1f}% P4={perc['phase_p4_ok_rate']*100:.1f}% "
          f"P5={perc['phase_p5_ok_rate']*100:.1f}%")

    ctrl = summary["control"]
    print("\n── Control / MPC ──")
    print(f"  Solve time: mean={ctrl['mpc_solve_time_ms']['mean']:.2f}ms  "
          f"p95={ctrl['mpc_solve_time_ms']['p95']:.2f}ms  max={ctrl['mpc_solve_time_ms']['max']:.2f}ms")
    print(f"  Fallback:   {ctrl['mpc_fallback_count']} ({ctrl['mpc_fallback_rate']*100:.1f}%)")
    print(f"  Steering:   mean={ctrl['steering']['mean']:.4f}  std={ctrl['steering']['std']:.4f}  "
          f"max_abs={ctrl['steering']['max']:.4f}")
    print(f"  Speed:      mean={ctrl['speed_ms']['mean']:.2f}m/s  max={ctrl['speed_ms']['max']:.2f}m/s")

    perf = summary["performance"]
    print("\n── Performance ──")
    print(f"  FPS:        mean={perf['fps']['mean']:.1f}  min={perf['fps']['min']:.1f}  std={perf['fps']['std']:.1f}")
    print(f"  Loop time:  mean={perf['loop_time_ms']['mean']:.2f}ms  "
          f"p95={perf['loop_time_ms']['p95']:.2f}ms  max={perf['loop_time_ms']['max']:.2f}ms")

    sfty = summary["safety"]
    print("\n── Safety ──")
    print(f"  Safety active:    {sfty['safety_active_count']} ({sfty['safety_active_rate']*100:.1f}%)")
    print(f"  Stuck recovery:   {sfty['stuck_recovery_count']}")
    print(f"  AEB triggers:     {sfty['aeb_trigger_count']}")
    print(f"  ACC active:       {sfty['acc_active_count']}")
    print(f"  LDW warnings:     {sfty['ldw_warning_count']}")

    print("\n" + "=" * 70)


def find_latest_csv(directory: str) -> Optional[str]:
    """Find latest CSV in directory"""
    d = Path(directory)
    if not d.exists():
        return None
    csvs = sorted(d.glob("run_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(csvs[0]) if csvs else None


def main():
    parser = argparse.ArgumentParser(description="Analyze run metrics CSV")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    parser.add_argument("--dir", type=str, default="metrics_output", help="Directory to find latest CSV")
    parser.add_argument("--output", type=str, default=None, help="Save JSON summary to this path")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        csv_path = find_latest_csv(args.dir)
        if csv_path is None:
            print(f"No CSV found in {args.dir}/")
            return 1

    print(f"Analyzing: {csv_path}")
    frames = load_csv(csv_path)
    summary = analyze(frames)
    print_summary(summary)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
