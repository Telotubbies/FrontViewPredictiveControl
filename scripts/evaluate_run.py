#!/usr/bin/env python3
"""
Evaluate LKA run from a log dir (logs/run_YYYYMMDD_HHMMSS).
Reads meta.csv and optionally summary.json; computes C1–C5 and prints tuning suggestions.
Usage:
  python scripts/evaluate_run.py logs/run_20250301_123456
  python scripts/evaluate_run.py logs/run_latest   # if symlink
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


# Pass thresholds (from lane_keeping_testloop)
C1_MEAN_CTE_M = 0.30
C1_MAX_CTE_M = 0.80
C3_MIN_DURATION_S = 60.0  # consider "complete" if ran at least this long
C4_SOLVER_FAIL_PCT_MAX = 5.0
C5_STEER_ROLLING_WINDOW = 30  # frames (~3s at 10 Hz)
C5_STEER_STD_MAX = 0.08


def evaluate_run(log_dir: str | Path) -> dict:
    """
    Evaluate one run. Returns dict with C1..C5 bools and details.
    """
    log_path = Path(log_dir)
    meta_file = log_path / "meta.csv"
    if not meta_file.exists():
        return {
            "error": f"meta.csv not found in {log_path}",
            "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
            "details": {},
        }

    try:
        with open(meta_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        return {
            "error": str(e),
            "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
            "details": {},
        }

    if len(rows) < 2:
        return {
            "error": "Too few rows in meta.csv",
            "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
            "details": {"frame_count": len(rows)},
        }

    cols = list(rows[0].keys())
    cte_vals = np.array([float(r["cte_m"]) for r in rows])
    steer_vals = np.array([float(r["steer"]) for r in rows])
    abs_cte = np.abs(cte_vals)
    mean_cte = float(np.mean(abs_cte))
    max_cte = float(np.max(abs_cte))
    details = {}
    details["mean_cte_m"] = round(mean_cte, 4)
    details["max_cte_m"] = round(max_cte, 4)

    C1 = (mean_cte < C1_MEAN_CTE_M) and (max_cte < C1_MAX_CTE_M)

    # C2: no collision/lane_invasion — we don't log events yet; assume pass if no explicit file
    events_file = log_path / "events.json"
    if events_file.exists():
        try:
            events = json.loads(events_file.read_text(encoding="utf-8"))
            collisions = len([e for e in events if e.get("type") == "collision"])
            lane_inv = len([e for e in events if e.get("type") == "lane_invasion" and e.get("duration", 0) > 0.5])
            C2 = (collisions == 0) and (lane_inv == 0)
            details["collisions"] = collisions
            details["lane_invasions"] = lane_inv
        except Exception:
            C2 = True
            details["C2_note"] = "events.json invalid, assumed pass"
    else:
        C2 = True
        details["C2_note"] = "no events.json (assumed pass)"

    # C3: duration / complete run
    frame_count = len(rows)
    fps_approx = 10.0
    duration_s = frame_count / fps_approx
    summary_file = log_path / "summary.json"
    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            duration_s = float(summary.get("duration_sec", duration_s))
        except Exception:
            pass
    details["duration_sec"] = round(duration_s, 1)
    details["frame_count"] = frame_count
    C3 = duration_s >= C3_MIN_DURATION_S

    # C4: solver fail rate
    if "solver_status" in cols:
        fail_count = sum(1 for r in rows if r.get("solver_status") != "Solve_Succeeded")
        fail_rate = 100.0 * fail_count / len(rows)
        details["solver_fail_count"] = fail_count
        details["solver_fail_pct"] = round(fail_rate, 2)
        C4 = fail_rate < C4_SOLVER_FAIL_PCT_MAX
    else:
        C4 = True
        details["C4_note"] = "no solver_status column (assumed pass)"

    # C5: steer oscillation (rolling std of delta steer)
    delta_steer = np.abs(np.diff(steer_vals))
    n = len(delta_steer)
    max_rolling_std = 0.0
    for i in range(n):
        window = delta_steer[max(0, i - C5_STEER_ROLLING_WINDOW + 1) : i + 1]
        if len(window) > 0:
            std_val = float(np.std(window))
            if std_val > max_rolling_std:
                max_rolling_std = std_val
    details["steer_rolling_std_max"] = round(max_rolling_std, 4)
    C5 = max_rolling_std < C5_STEER_STD_MAX

    return {
        "log_dir": str(log_path.resolve()),
        "C1": C1,
        "C2": C2,
        "C3": C3,
        "C4": C4,
        "C5": C5,
        "details": details,
        "pass_all": (C1 and C2 and C3 and C4 and C5),
    }


def tuning_suggestions(result: dict) -> list[str]:
    """Return list of tuning suggestions based on failed criteria."""
    suggestions = []
    if not result.get("C1"):
        d = result.get("details", {})
        mean_cte = d.get("mean_cte_m", 0)
        max_cte = d.get("max_cte_m", 0)
        if mean_cte >= C1_MEAN_CTE_M:
            suggestions.append("C1 (CTE): mean CTE สูง → config.yaml → mpc: MPC_W_CTE (ลอง 300 หรือ 400)")
        if max_cte >= C1_MAX_CTE_M:
            suggestions.append("C1 (CTE): max CTE สูง → config.yaml → mpc: MPC_W_HEADING, MPC_W_CTE; ตรวจ perception ในโค้ง")
    if not result.get("C4"):
        suggestions.append("C4 (Solver): IPOPT fail บ่อย → config.yaml → mpc: MPC_N ลดเป็น 8")
    if not result.get("C5"):
        suggestions.append("C5 (Oscillation): steer ส่าย → config.yaml → mpc: MPC_W_STEER_RATE, MPC_W_STEER_JERK; reference_mpc: STEER_MAX_DELTA_PER_FRAME ลดเป็น 0.08")
    return suggestions


def main():
    ap = argparse.ArgumentParser(description="Evaluate LKA run and print tuning suggestions")
    ap.add_argument("log_dir", nargs="?", default=None, help="e.g. logs/run_20250301_123456")
    ap.add_argument("--json", action="store_true", help="Print only eval JSON to stdout")
    ap.add_argument("--write", action="store_true", help="Write eval_<timestamp>.json into log_dir")
    args = ap.parse_args()

    if not args.log_dir:
        # Find latest run
        logs = Path("logs")
        if not logs.exists():
            print("No logs/ dir. Run with: python run_unet_mpc.py --record --town Town04", file=sys.stderr)
            sys.exit(2)
        runs = sorted(logs.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            print("No logs/run_* found. Run with --record first.", file=sys.stderr)
            sys.exit(2)
        log_dir = str(runs[0])
        print(f"Using latest: {log_dir}", file=sys.stderr)
    else:
        log_dir = args.log_dir

    result = evaluate_run(log_dir)
    if "error" in result:
        print(result["error"], file=sys.stderr)
        sys.exit(1)

    if args.write:
        out_file = Path(log_dir) / "eval_result.json"
        out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_file}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return

    # Human-readable report
    d = result["details"]
    print("━" * 50)
    print(f"EVAL: {result['log_dir']}")
    print("━" * 50)
    print(f"C1 CTE         : {'✅' if result['C1'] else '❌'}  mean={d.get('mean_cte_m', 0):.3f}m  max={d.get('max_cte_m', 0):.3f}m  (threshold: mean<{C1_MEAN_CTE_M}, max<{C1_MAX_CTE_M})")
    print(f"C2 No crash    : {'✅' if result['C2'] else '❌'}  {d.get('C2_note', '')}")
    print(f"C3 Complete    : {'✅' if result['C3'] else '❌'}  duration={d.get('duration_sec', 0):.1f}s  frames={d.get('frame_count', 0)}")
    print(f"C4 Solver OK   : {'✅' if result['C4'] else '❌'}  fail_rate={d.get('solver_fail_pct', 0):.1f}%  (threshold <{C4_SOLVER_FAIL_PCT_MAX}%)")
    print(f"C5 No oscillate: {'✅' if result['C5'] else '❌'}  steer_std_max={d.get('steer_rolling_std_max', 0):.4f}  (threshold <{C5_STEER_STD_MAX})")
    print("━" * 50)
    print(f"VERDICT: {'PASS' if result['pass_all'] else 'FAIL'}")
    suggestions = tuning_suggestions(result)
    if suggestions:
        print("\nปรับค่าแนะนำ:")
        for s in suggestions:
            print(f"  • {s}")
    print()


if __name__ == "__main__":
    main()
