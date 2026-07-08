#!/usr/bin/env python3
"""System check — ทดสอบทุก module หลักของระบบ"""
import sys
import os

sys.path.insert(0, "src")
sys.path.insert(0, ".")

print("=" * 60)
print("SYSTEM CHECK — FrontViewPredictiveControl")
print("=" * 60)
print(f"Python: {sys.version.split()[0]}")
print()

# 1. Device detection
print("[1/8] Device Detection")
from utils.device_utils import get_device, get_device_info, is_rocm_available, is_cuda_available
device = get_device()
info = get_device_info()
print(f"  Device: {device}")
print(f"  Type: {info['device_type']} | Name: {info['device_name']}")
print(f"  ROCm={is_rocm_available()} CUDA={is_cuda_available()}")
print("  OK")
print()

# 2. Config
print("[2/8] Config")
from config import TARGET_SPEED_KMH, CONTROL_HZ, MPC_N, USE_TRAJECTORY_PIPELINE
print(f"  target={TARGET_SPEED_KMH}km/h hz={CONTROL_HZ} mpc_N={MPC_N}")
print(f"  trajectory_pipeline={USE_TRAJECTORY_PIPELINE}")
print("  OK")
print()

# 3. Pipeline import
print("[3/8] Pipeline Import")
from pipeline import LKAPipeline
print("  LKAPipeline: OK")
print()

# 4. MPC
print("[4/8] MPC Solver")
from control.lane_mpc import LaneMPC, MPCConfig
cfg = MPCConfig()
mpc = LaneMPC(cfg)
steer, accel, status = mpc.solve(x0=0, y0=0.5, psi0=0.1, v0=5.0, v_ref=8.0, cte=0.5, heading_err=0.1)
print(f"  steer={steer:.4f}rad accel={accel:.2f}m/s2 status={status}")
steer2, accel2, status2 = mpc.solve(x0=0, y0=-1.0, psi0=-0.2, v0=8.0, v_ref=6.0, cte=-1.0, heading_err=-0.2, curvature=0.05)
print(f"  curve: steer={steer2:.4f}rad accel={accel2:.2f}m/s2 status={status2}")
print("  OK")
print()

# 5. MetricsCollector
print("[5/8] MetricsCollector")
from telemetry.metrics_collector import MetricsCollector, FrameMetrics
mc = MetricsCollector(output_dir="metrics_output")
for i in range(10):
    mc.record(FrameMetrics(
        frame_idx=i, speed_ms=5.0 + i * 0.3, cte_m=0.1 * i, lane_conf=0.85,
        geometry_valid=(i % 3 != 0), mpc_solve_time_ms=8.0 + i * 0.5,
        mpc_solver_status="Solve_Succeeded" if i < 8 else "Fallback_PP",
        mpc_fallback=(i >= 8), fps=20.0, loop_time_ms=50.0,
        aeb_active=(i == 9), safety_active=(i < 2),
    ))
summary = mc.get_summary()
print(f"  Frames: {summary['total_frames']}")
print(f"  Fallback: {summary['fallback_count']} ({summary['fallback_rate']*100:.0f}%)")
print(f"  CTE mean: {summary['perception']['cte_mean_m']}m")
print(f"  FPS mean: {summary['performance']['fps_mean']}")
print(f"  AEB triggers: {summary['safety']['aeb_trigger_count']}")
paths = mc.save(prefix="test_run")
print(f"  Saved: CSV={paths['csv']} JSON={paths['json']}")
print("  OK")
print()

# 6. ADAS
print("[6/8] ADAS Suite")
from adas.adas_manager import ADASManager
adas = ADASManager()
print("  ADASManager: OK")
print()

# 7. Safety
print("[7/8] Safety Systems")
from safety.override import SafetyOverride
from safety.stuck_recovery import StuckRecovery
from safety.emergency_braking_adaptive_cruise_control import AEBACC
so = SafetyOverride({"max_speed_kmh": 30.0, "max_steering_angle": 0.45})
sr = StuckRecovery()
aeb = AEBACC()
print("  SafetyOverride + StuckRecovery + AEBACC: OK")
print()

# 8. Telemetry
print("[8/8] Telemetry")
from telemetry.influxdb_exporter import TelemetryExporter
te = TelemetryExporter(enabled=False)
print(f"  InfluxDB exporter: enabled={te.enabled}")
print("  OK")
print()

# Cleanup test files
import glob
for f in glob.glob("metrics_output/test_run_*.csv"):
    os.remove(f)
    print(f"  Cleaned: {f}")
for f in glob.glob("metrics_output/test_run_*_summary.json"):
    os.remove(f)
    print(f"  Cleaned: {f}")

print()
print("=" * 60)
print("ALL SYSTEM CHECKS PASSED")
print("=" * 60)
