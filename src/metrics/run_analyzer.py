"""
RunAnalyzer — วิเคราะห์ข้อมูลหลังรัน สร้าง report + plots

อ่าน frame_log.csv จาก run directory แล้ว:
1. คำนวณ KPIs (CTE RMSE, in-lane %, AEB events, speed tracking, etc.)
2. สร้าง plots (matplotlib): CTE, speed, steering, events timeline
3. เขียน report.md (มนุษย์อ่านได้) + report.json (machine-readable)
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


class RunAnalyzer:
    """
    Post-run analysis — อ่าน CSV แล้วสร้าง KPIs + plots + report.

    Usage:
        analyzer = RunAnalyzer("runs/2026-07-08_15-30-21_Town10HD_Opt")
        analyzer.analyze()
        # → สร้าง report.md, report.json, plots/*.png
    """

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.csv_path = self.run_dir / "frame_log.csv"
        self.events_path = self.run_dir / "events.log"
        self.meta_path = self.run_dir / "run_meta.json"
        self.plots_dir = self.run_dir / "plots"

        self._frames: List[Dict[str, Any]] = []
        self._events: List[str] = []
        self._meta: Dict[str, Any] = {}

    def analyze(self) -> Dict[str, Any]:
        """
        วิเคราะห์ข้อมูลทั้งหมด สร้าง report + plots.

        Returns:
            KPI dict
        """
        self._load_data()
        if not self._frames:
            logger.warning(f"No frames in {self.csv_path}")
            return {}

        kpis = self._compute_kpis()
        self._generate_plots(kpis)
        self._write_report(kpis)
        self._write_json(kpis)

        logger.info(f"Analysis complete → {self.run_dir / 'report.md'}")
        return kpis

    def _load_data(self) -> None:
        """โหลด CSV + events + metadata"""
        # Load frames
        if self.csv_path.exists():
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    for key in row:
                        if key in ("frame_idx",):
                            row[key] = int(row[key])
                        elif key in ("geometry_valid", "reverse", "aeb_active",
                                     "acc_active", "ldw_warning_active", "tja_active",
                                     "stop_and_go_stopped", "safety_override_active",
                                     "stuck_recovery_active", "adas_override_active"):
                            row[key] = row[key].lower() in ("true", "1", "yes")
                        else:
                            try:
                                row[key] = float(row[key])
                            except (ValueError, TypeError):
                                pass
                    self._frames.append(row)

        # Load events
        if self.events_path.exists():
            with open(self.events_path, "r", encoding="utf-8") as f:
                self._events = f.readlines()

        # Load metadata
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)

        logger.info(f"Loaded {len(self._frames)} frames, {len(self._events)} events")

    def _compute_kpis(self) -> Dict[str, Any]:
        """คำนวณ KPIs จาก frame data"""
        frames = self._frames
        n = len(frames)

        # Extract arrays (ensure float conversion — CSV may return strings)
        def _to_float(val, default=0.0):
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        cte = np.array([_to_float(f.get("cte_m", 0.0)) for f in frames])
        speed_ms = np.array([_to_float(f.get("speed_ms", 0.0)) for f in frames])
        speed_kmh = speed_ms * 3.6
        steer = np.array([_to_float(f.get("steer", 0.0)) for f in frames])
        heading_deg = np.array([_to_float(f.get("heading_err_deg", 0.0)) for f in frames])
        lane_conf = np.array([_to_float(f.get("lane_conf", 0.0)) for f in frames])
        mpc_time = np.array([_to_float(f.get("mpc_solve_time_ms", 0.0)) for f in frames])
        loop_time = np.array([_to_float(f.get("loop_time_ms", 0.0)) for f in frames])
        timestamps = np.array([_to_float(f.get("timestamp", 0.0)) for f in frames])

        # Speed tracking
        target_kmh = self._meta.get("target_speed_kmh", 0.0)
        speed_error = speed_kmh - target_kmh if target_kmh > 0 else speed_kmh

        # In-lane percentage
        in_lane_mask = np.abs(cte) <= 1.75  # lane half-width
        in_lane_pct = float(np.mean(in_lane_mask) * 100)

        # Departure detection (consecutive departed frames)
        departed_mask = ~in_lane_mask
        departure_events = 0
        in_departure = False
        departure_durations = []
        dep_start = 0
        for i, dep in enumerate(departed_mask):
            if dep and not in_departure:
                in_departure = True
                departure_events += 1
                dep_start = timestamps[i]
            elif not dep and in_departure:
                in_departure = False
                departure_durations.append(timestamps[i] - dep_start)

        # Event counts
        aeb_count = sum(1 for f in frames if f.get("aeb_active", False))
        stuck_count = sum(1 for f in frames if f.get("stuck_recovery_active", False))
        safety_count = sum(1 for f in frames if f.get("safety_override_active", False))
        fallback_count = sum(1 for f in frames if "Fallback" in str(f.get("mpc_solver_status", "")))
        geometry_valid_count = sum(1 for f in frames if f.get("geometry_valid", False))

        # Duration
        duration = timestamps[-1] - timestamps[0] if n > 1 else 0

        # FPS
        fps = n / duration if duration > 0 else 0

        kpis = {
            "run_info": {
                "map": self._meta.get("map_name", "unknown"),
                "vehicle": self._meta.get("vehicle_type", "unknown"),
                "target_speed_kmh": target_kmh,
                "duration_s": round(duration, 2),
                "total_frames": n,
                "fps_mean": round(fps, 2),
            },
            "lane_compliance": {
                "cte_mean_m": round(float(np.mean(cte)), 4),
                "cte_std_m": round(float(np.std(cte)), 4),
                "cte_rmse_m": round(float(np.sqrt(np.mean(cte ** 2))), 4),
                "cte_max_abs_m": round(float(np.max(np.abs(cte))), 4),
                "heading_err_mean_deg": round(float(np.mean(heading_deg)), 4),
                "heading_err_std_deg": round(float(np.std(heading_deg)), 4),
                "heading_err_max_abs_deg": round(float(np.max(np.abs(heading_deg))), 4),
                "in_lane_pct": round(in_lane_pct, 2),
                "departure_events": departure_events,
                "departure_total_s": round(sum(departure_durations), 2),
                "departure_max_s": round(max(departure_durations) if departure_durations else 0, 2),
            },
            "speed_tracking": {
                "speed_mean_kmh": round(float(np.mean(speed_kmh)), 2),
                "speed_std_kmh": round(float(np.std(speed_kmh)), 2),
                "speed_max_kmh": round(float(np.max(speed_kmh)), 2),
                "speed_min_kmh": round(float(np.min(speed_kmh)), 2),
                "speed_error_mean_kmh": round(float(np.mean(speed_error)), 2) if target_kmh > 0 else 0,
                "speed_error_rmse_kmh": round(float(np.sqrt(np.mean(speed_error ** 2))), 2) if target_kmh > 0 else 0,
            },
            "control": {
                "steer_mean": round(float(np.mean(steer)), 4),
                "steer_std": round(float(np.std(steer)), 4),
                "steer_max_abs": round(float(np.max(np.abs(steer))), 4),
                "mpc_solve_time_mean_ms": round(float(np.mean(mpc_time)), 3),
                "mpc_solve_time_p95_ms": round(float(np.percentile(mpc_time, 95)), 3),
                "mpc_solve_time_max_ms": round(float(np.max(mpc_time)), 3),
                "fallback_count": fallback_count,
                "fallback_rate_pct": round(fallback_count / max(1, n) * 100, 2),
            },
            "perception": {
                "lane_conf_mean": round(float(np.mean(lane_conf)), 4),
                "lane_conf_min": round(float(np.min(lane_conf)), 4),
                "geometry_valid_rate": round(geometry_valid_count / max(1, n) * 100, 2),
            },
            "safety": {
                "aeb_events": aeb_count,
                "stuck_events": stuck_count,
                "safety_override_events": safety_count,
            },
            "performance": {
                "fps_mean": round(fps, 2),
                "loop_time_mean_ms": round(float(np.mean(loop_time)), 2),
                "loop_time_p95_ms": round(float(np.percentile(loop_time, 95)), 2),
                "loop_time_max_ms": round(float(np.max(loop_time)), 2),
            },
            "events_log_count": len(self._events),
        }

        return kpis

    def _generate_plots(self, kpis: Dict[str, Any]) -> None:
        """สร้าง plots ด้วย matplotlib"""
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
            return

        self.plots_dir.mkdir(exist_ok=True)
        frames = self._frames
        timestamps = np.array([f.get("timestamp", 0.0) for f in frames])

        # 1. CTE over time
        fig, ax = plt.subplots(figsize=(12, 4))
        cte = np.array([f.get("cte_m", 0.0) for f in frames])
        ax.plot(timestamps, cte, linewidth=0.8, color="blue", label="CTE")
        ax.axhline(y=1.75, color="red", linestyle="--", alpha=0.5, label="Lane boundary (+1.75m)")
        ax.axhline(y=-1.75, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("CTE (m)")
        ax.set_title(f"Cross-Track Error (RMSE={kpis['lane_compliance']['cte_rmse_m']:.2f}m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "cte_over_time.png", dpi=100)
        plt.close(fig)

        # 2. Speed over time
        fig, ax = plt.subplots(figsize=(12, 4))
        speed = np.array([f.get("speed_kmh", 0.0) for f in frames])
        ax.plot(timestamps, speed, linewidth=0.8, color="green", label="Actual speed")
        target = self._meta.get("target_speed_kmh", 0)
        if target > 0:
            ax.axhline(y=target, color="orange", linestyle="--", alpha=0.5, label=f"Target ({target} km/h)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (km/h)")
        ax.set_title("Speed Tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "speed_over_time.png", dpi=100)
        plt.close(fig)

        # 3. Steering over time
        fig, ax = plt.subplots(figsize=(12, 4))
        steer = np.array([f.get("steer", 0.0) for f in frames])
        ax.plot(timestamps, steer, linewidth=0.8, color="purple", label="Steering")
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Steering [-1, 1]")
        ax.set_title("Steering Commands")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "steering_over_time.png", dpi=100)
        plt.close(fig)

        # 4. Events timeline
        fig, ax = plt.subplots(figsize=(12, 4))
        aeb = np.array([1 if f.get("aeb_active", False) else 0 for f in frames])
        stuck = np.array([2 if f.get("stuck_recovery_active", False) else 0 for f in frames])
        safety = np.array([3 if f.get("safety_override_active", False) else 0 for f in frames])
        fallback = np.array([4 if "Fallback" in str(f.get("mpc_solver_status", "")) else 0 for f in frames])
        ax.fill_between(timestamps, 0, aeb, alpha=0.5, color="red", label="AEB")
        ax.fill_between(timestamps, 1, 1 + stuck, alpha=0.5, color="orange", label="Stuck")
        ax.fill_between(timestamps, 2, 2 + safety / 3, alpha=0.5, color="yellow", label="Safety")
        ax.fill_between(timestamps, 3, 3 + fallback / 4, alpha=0.5, color="blue", label="Fallback")
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(["AEB", "Stuck", "Safety", "Fallback"])
        ax.set_xlabel("Time (s)")
        ax.set_title("Safety & Control Events Timeline")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "events_timeline.png", dpi=100)
        plt.close(fig)

        # 5. MPC solve time histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        mpc_times = np.array([f.get("mpc_solve_time_ms", 0.0) for f in frames])
        ax.hist(mpc_times[mpc_times > 0], bins=50, color="cyan", edgecolor="black", alpha=0.7)
        ax.axvline(x=np.mean(mpc_times), color="red", linestyle="--", label=f"Mean={np.mean(mpc_times):.1f}ms")
        ax.set_xlabel("Solve Time (ms)")
        ax.set_ylabel("Count")
        ax.set_title("MPC Solve Time Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "mpc_solve_time_hist.png", dpi=100)
        plt.close(fig)

        logger.info(f"Generated 5 plots in {self.plots_dir}")

    def _write_report(self, kpis: Dict[str, Any]) -> None:
        """เขียน report.md (มนุษย์อ่านได้)"""
        report_path = self.run_dir / "report.md"
        lc = kpis["lane_compliance"]
        st = kpis["speed_tracking"]
        ct = kpis["control"]
        pc = kpis["perception"]
        sf = kpis["safety"]
        pf = kpis["performance"]
        ri = kpis["run_info"]

        # Count events by type
        event_types: Dict[str, int] = {}
        for line in self._events:
            parts = line.strip().split()
            if len(parts) >= 4:
                etype = parts[3].rstrip(":")
                event_types[etype] = event_types.get(etype, 0) + 1

        report = f"""# Run Analysis Report

## Run Info
- **Map**: {ri['map']}
- **Vehicle**: {ri['vehicle']}
- **Target Speed**: {ri['target_speed_kmh']} km/h
- **Duration**: {ri['duration_s']}s ({ri['total_frames']} frames)
- **Mean FPS**: {ri['fps_mean']}

## Lane Compliance
| Metric | Value |
|--------|-------|
| CTE Mean | {lc['cte_mean_m']} m |
| CTE Std | {lc['cte_std_m']} m |
| CTE RMSE | {lc['cte_rmse_m']} m |
| CTE Max | {lc['cte_max_abs_m']} m |
| Heading Error Mean | {lc['heading_err_mean_deg']}° |
| Heading Error Max | {lc['heading_err_max_abs_deg']}° |
| In-Lane % | {lc['in_lane_pct']}% |
| Departure Events | {lc['departure_events']} |
| Departure Total | {lc['departure_total_s']}s |
| Departure Max | {lc['departure_max_s']}s |

## Speed Tracking
| Metric | Value |
|--------|-------|
| Speed Mean | {st['speed_mean_kmh']} km/h |
| Speed Std | {st['speed_std_kmh']} km/h |
| Speed Max | {st['speed_max_kmh']} km/h |
| Speed Min | {st['speed_min_kmh']} km/h |
| Speed Error Mean | {st['speed_error_mean_kmh']} km/h |
| Speed Error RMSE | {st['speed_error_rmse_kmh']} km/h |

## Control & MPC
| Metric | Value |
|--------|-------|
| Steer Mean | {ct['steer_mean']} |
| Steer Std | {ct['steer_std']} |
| Steer Max | {ct['steer_max_abs']} |
| MPC Solve Mean | {ct['mpc_solve_time_mean_ms']} ms |
| MPC Solve P95 | {ct['mpc_solve_time_p95_ms']} ms |
| MPC Solve Max | {ct['mpc_solve_time_max_ms']} ms |
| Fallback Count | {ct['fallback_count']} ({ct['fallback_rate_pct']}%) |

## Perception
| Metric | Value |
|--------|-------|
| Lane Conf Mean | {pc['lane_conf_mean']} |
| Lane Conf Min | {pc['lane_conf_min']} |
| Geometry Valid | {pc['geometry_valid_rate']}% |

## Safety Events
| Event | Count |
|-------|-------|
| AEB | {sf['aeb_events']} |
| Stuck Recovery | {sf['stuck_events']} |
| Safety Override | {sf['safety_override_events']} |

## Performance
| Metric | Value |
|--------|-------|
| FPS Mean | {pf['fps_mean']} |
| Loop Time Mean | {pf['loop_time_mean_ms']} ms |
| Loop Time P95 | {pf['loop_time_p95_ms']} ms |
| Loop Time Max | {pf['loop_time_max_ms']} ms |

## Event Log Summary
"""
        if event_types:
            for etype, count in sorted(event_types.items()):
                report += f"- **{etype}**: {count} events\n"
        else:
            report += "- No events logged\n"

        report += f"""
## Plots
- `plots/cte_over_time.png` — CTE over time with lane boundaries
- `plots/speed_over_time.png` — Speed tracking vs target
- `plots/steering_over_time.png` — Steering commands
- `plots/events_timeline.png` — Safety events timeline
- `plots/mpc_solve_time_hist.png` — MPC solve time distribution

## Overall Assessment
"""
        # Auto-assessment
        if lc['in_lane_pct'] > 90:
            report += "- ✅ **Lane keeping**: Excellent (>90% in-lane)\n"
        elif lc['in_lane_pct'] > 75:
            report += "- ⚠️ **Lane keeping**: Good (>75% in-lane)\n"
        else:
            report += "- ❌ **Lane keeping**: Poor (<75% in-lane)\n"

        if lc['cte_rmse_m'] < 0.5:
            report += "- ✅ **CTE**: Excellent (RMSE < 0.5m)\n"
        elif lc['cte_rmse_m'] < 1.0:
            report += "- ⚠️ **CTE**: Acceptable (RMSE < 1.0m)\n"
        else:
            report += "- ❌ **CTE**: Poor (RMSE > 1.0m)\n"

        if ct['fallback_rate_pct'] < 5:
            report += "- ✅ **MPC**: Stable (<5% fallback)\n"
        elif ct['fallback_rate_pct'] < 20:
            report += "- ⚠️ **MPC**: Some fallbacks (<20%)\n"
        else:
            report += "- ❌ **MPC**: Unstable (>20% fallback)\n"

        if sf['aeb_events'] == 0:
            report += "- ✅ **Safety**: No AEB events\n"
        else:
            report += f"- ⚠️ **Safety**: {sf['aeb_events']} AEB events\n"

        if pf['fps_mean'] > 15:
            report += "- ✅ **Performance**: Good (>15 FPS)\n"
        elif pf['fps_mean'] > 10:
            report += "- ⚠️ **Performance**: Acceptable (>10 FPS)\n"
        else:
            report += "- ❌ **Performance**: Poor (<10 FPS)\n"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Report written to {report_path}")

    def _write_json(self, kpis: Dict[str, Any]) -> None:
        """เขียน KPIs เป็น JSON"""
        report_path = self.run_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(kpis, f, indent=2, default=str)
        logger.info(f"JSON report written to {report_path}")
