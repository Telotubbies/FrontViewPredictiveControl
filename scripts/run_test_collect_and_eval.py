#!/usr/bin/env python3
"""
รันทดสอบ LKA เก็บผล แล้วประเมินผลและแนะนำการปรับค่า

ต้องเปิด CARLA server (port 2000) ก่อนรัน

Usage:
  cd carla_mpc_classical
  PYTHONPATH=. python scripts/run_test_collect_and_eval.py --duration 60
  PYTHONPATH=. python scripts/run_test_collect_and_eval.py --duration 90 --town Town04 --no-eval  # เก็บอย่างเดียว ไม่รัน eval
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Run LKA test with recording then evaluate")
    ap.add_argument("--duration", type=float, default=60.0, help="Test duration seconds")
    ap.add_argument("--town", default="Town04", help="CARLA map")
    ap.add_argument("--speed", type=float, default=25.0, help="Target speed km/h")
    ap.add_argument("--no-eval", action="store_true", help="Only run test, do not run evaluator")
    ap.add_argument("--eval-only", nargs="?", metavar="LOG_DIR", const="latest",
                    help="Only evaluate existing log dir. Omit to use latest (logs/run_eval_* or logs/run_*)")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    run_script = project_root / "run_unet_mpc.py"
    eval_script = project_root / "scripts" / "evaluate_run.py"

    if args.eval_only is not None:
        log_dir = Path(args.eval_only) if args.eval_only != "latest" else None
        if log_dir is None or args.eval_only == "latest":
            logs_dir = project_root / "logs"
            if not logs_dir.exists():
                print("No logs/ dir.", file=sys.stderr)
                sys.exit(2)
            available = sorted(
                [d for d in logs_dir.iterdir() if d.is_dir() and (d / "meta.csv").exists()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not available:
                print("No log dir with meta.csv in logs/", file=sys.stderr)
                sys.exit(2)
            log_dir = available[0]
            print(f"Using latest: {log_dir.name}", file=sys.stderr)
        elif not log_dir.is_absolute():
            log_dir = project_root / log_dir
        if not (log_dir / "meta.csv").exists():
            print(f"meta.csv not found in {log_dir}", file=sys.stderr)
            logs_dir = project_root / "logs"
            if logs_dir.exists():
                available = sorted(
                    [d for d in logs_dir.iterdir() if d.is_dir() and (d / "meta.csv").exists()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if available:
                    print(f"  ใช้ log ล่าสุด: {available[0].name}", file=sys.stderr)
                    print(f"  ตัวอย่าง: --eval-only logs/{available[0].name}", file=sys.stderr)
            sys.exit(2)
        ret = subprocess.run(
            [sys.executable, str(eval_script), str(log_dir), "--write"],
            cwd=str(project_root),
        )
        sys.exit(ret.returncode)

    record_dir = project_root / "logs" / ("run_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    record_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording to {record_dir} for {args.duration}s ...", file=sys.stderr)
    print("(เปิด CARLA server ก่อน รอรถวิ่งครบเวลาหรือกด Ctrl+C เพื่อหยุด)", file=sys.stderr)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    ret = subprocess.run(
        [
            sys.executable,
            str(run_script),
            "--record-dir", str(record_dir),
            "--duration", str(args.duration),
            "--town", args.town,
            "--speed", str(args.speed),
        ],
        cwd=str(project_root),
        env=env,
    )
    if ret.returncode != 0 and ret.returncode != 130:  # 130 = Ctrl+C
        print("run_unet_mpc exited with code", ret.returncode, file=sys.stderr)
        sys.exit(ret.returncode)

    if args.no_eval:
        print(f"Log dir: {record_dir}", file=sys.stderr)
        return

    if not (record_dir / "meta.csv").exists():
        print("meta.csv not found (run too short?). Skipping eval.", file=sys.stderr)
        sys.exit(0)

    print("\n--- Evaluate ---", file=sys.stderr)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    ret = subprocess.run(
        [sys.executable, str(eval_script), str(record_dir), "--write"],
        cwd=str(project_root),
        env=env,
    )
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
