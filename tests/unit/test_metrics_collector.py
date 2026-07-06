"""
Tests for MetricsCollector — ระบบวัดผลทุกข้อมูลและ output
"""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from telemetry.metrics_collector import MetricsCollector, FrameMetrics


class TestFrameMetrics(unittest.TestCase):
    """Test FrameMetrics dataclass"""

    def test_default_values(self):
        fm = FrameMetrics()
        self.assertEqual(fm.frame_idx, 0)
        self.assertEqual(fm.speed_ms, 0.0)
        self.assertEqual(fm.mpc_solver_status, "unknown")
        self.assertFalse(fm.mpc_fallback)

    def test_custom_values(self):
        fm = FrameMetrics(
            frame_idx=42,
            speed_ms=8.5,
            cte_m=0.3,
            mpc_solver_status="Solve_Succeeded",
            mpc_solve_time_ms=12.5,
        )
        self.assertEqual(fm.frame_idx, 42)
        self.assertEqual(fm.cte_m, 0.3)
        self.assertFalse(fm.mpc_fallback)


class TestMetricsCollector(unittest.TestCase):
    """Test MetricsCollector class"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.collector = MetricsCollector(output_dir=self.tmpdir)

    def test_empty_summary(self):
        summary = self.collector.get_summary()
        self.assertEqual(summary["total_frames"], 0)
        self.assertEqual(summary["duration_s"], 0.0)

    def test_record_single_frame(self):
        fm = FrameMetrics(
            frame_idx=1,
            speed_ms=5.0,
            cte_m=0.2,
            heading_err_rad=0.01,
            lane_conf=0.85,
            geometry_valid=True,
            mpc_solve_time_ms=10.0,
            mpc_solver_status="Solve_Succeeded",
            fps=20.0,
            loop_time_ms=50.0,
        )
        self.collector.record(fm)

        summary = self.collector.get_summary()
        self.assertEqual(summary["total_frames"], 1)
        self.assertEqual(summary["fallback_count"], 0)
        self.assertAlmostEqual(summary["perception"]["cte_mean_m"], 0.2, places=4)
        self.assertAlmostEqual(summary["perception"]["lane_conf_mean"], 0.85, places=4)
        self.assertAlmostEqual(summary["control"]["mpc_solve_time_mean_ms"], 10.0, places=3)

    def test_record_fallback(self):
        fm = FrameMetrics(mpc_solver_status="Fallback_PP", mpc_fallback=True)
        self.collector.record(fm)
        summary = self.collector.get_summary()
        self.assertEqual(summary["fallback_count"], 1)
        self.assertEqual(summary["fallback_rate"], 1.0)

    def test_record_multiple_frames(self):
        for i in range(10):
            fm = FrameMetrics(
                frame_idx=i,
                speed_ms=5.0 + i * 0.5,
                cte_m=0.1 * i,
                lane_conf=0.8,
                geometry_valid=(i % 2 == 0),
                mpc_solve_time_ms=10.0 + i,
                mpc_solver_status="Solve_Succeeded" if i < 8 else "Fallback_PP",
                mpc_fallback=(i >= 8),
                fps=20.0,
                loop_time_ms=50.0,
            )
            self.collector.record(fm)

        summary = self.collector.get_summary()
        self.assertEqual(summary["total_frames"], 10)
        self.assertEqual(summary["fallback_count"], 2)
        self.assertAlmostEqual(summary["fallback_rate"], 0.2, places=4)
        self.assertAlmostEqual(summary["perception"]["geometry_valid_rate"], 0.5, places=4)
        self.assertAlmostEqual(summary["perception"]["cte_mean_m"], 0.45, places=4)

    def test_record_from_frame_state(self):
        """Test record_from_frame_state with a mock frame_state object"""
        class MockFrameState:
            cte_m = 0.15
            heading_rad = 0.02
            curvature = 0.005
            lane_conf = 0.9
            geometry_valid = True
            phase_p1_ok = True
            phase_p2_ok = True
            phase_p3_ok = False
            phase_p4_ok = False
            phase_p5_ok = False
            solver_status = "Solve_Succeeded"
            mpc_solve_time_ms = 8.5
            ldw_state = "in_lane"
            aeb_active = False
            aeb_ttc = -1.0
            acc_active = False
            acc_distance_m = -1.0

        self.collector.record_from_frame_state(
            frame_idx=0,
            speed_ms=6.0,
            steer=0.1,
            throttle=0.3,
            brake=0.0,
            frame_state=MockFrameState(),
            mpc_solve_time_ms=8.5,
            mpc_solver_status="Solve_Succeeded",
            loop_time_ms=45.0,
        )

        summary = self.collector.get_summary()
        self.assertEqual(summary["total_frames"], 1)
        self.assertAlmostEqual(summary["perception"]["cte_mean_m"], 0.15, places=4)
        self.assertAlmostEqual(summary["perception"]["lane_conf_mean"], 0.9, places=4)
        self.assertAlmostEqual(summary["control"]["mpc_solve_time_mean_ms"], 8.5, places=3)

    def test_save_csv(self):
        for i in range(5):
            fm = FrameMetrics(
                frame_idx=i,
                speed_ms=5.0,
                cte_m=0.1 * i,
                mpc_solver_status="Solve_Succeeded",
            )
            self.collector.record(fm)

        paths = self.collector.save(prefix="test")
        self.assertTrue(os.path.exists(paths["csv"]))
        self.assertTrue(os.path.exists(paths["json"]))

        # Verify CSV
        with open(paths["csv"], "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 5)
        self.assertEqual(int(rows[0]["frame_idx"]), 0)
        self.assertEqual(int(rows[4]["frame_idx"]), 4)

    def test_save_json_summary(self):
        for i in range(3):
            fm = FrameMetrics(
                frame_idx=i,
                speed_ms=5.0,
                cte_m=0.2,
                lane_conf=0.85,
                mpc_solver_status="Solve_Succeeded",
                fps=20.0,
                loop_time_ms=50.0,
            )
            self.collector.record(fm)

        paths = self.collector.save(prefix="test")
        with open(paths["json"], "r") as f:
            summary = json.load(f)
        self.assertEqual(summary["total_frames"], 3)
        self.assertIn("perception", summary)
        self.assertIn("control", summary)
        self.assertIn("performance", summary)
        self.assertIn("safety", summary)

    def test_reset(self):
        for i in range(5):
            self.collector.record(FrameMetrics(frame_idx=i))
        self.assertEqual(self.collector._total_frames, 5)

        self.collector.reset()
        self.assertEqual(self.collector._total_frames, 0)
        self.assertEqual(len(self.collector.frames), 0)

    def test_safety_metrics(self):
        for i in range(10):
            fm = FrameMetrics(
                frame_idx=i,
                safety_active=(i < 3),
                stuck_recovery_active=(i == 5),
                aeb_active=(i == 7),
            )
            self.collector.record(fm)

        summary = self.collector.get_summary()
        self.assertEqual(summary["safety"]["safety_active_count"], 3)
        self.assertEqual(summary["safety"]["stuck_recovery_count"], 1)
        self.assertEqual(summary["safety"]["aeb_trigger_count"], 1)

    def test_phase_rates(self):
        for i in range(10):
            fm = FrameMetrics(
                frame_idx=i,
                lane_phase_p1=True,
                lane_phase_p2=(i < 8),
                lane_phase_p3=(i < 6),
                lane_phase_p4=(i < 4),
                lane_phase_p5=(i < 2),
            )
            self.collector.record(fm)

        summary = self.collector.get_summary()
        self.assertAlmostEqual(summary["perception"]["phase_p1_ok_rate"], 1.0, places=4)
        self.assertAlmostEqual(summary["perception"]["phase_p2_ok_rate"], 0.8, places=4)
        self.assertAlmostEqual(summary["perception"]["phase_p3_ok_rate"], 0.6, places=4)
        self.assertAlmostEqual(summary["perception"]["phase_p4_ok_rate"], 0.4, places=4)
        self.assertAlmostEqual(summary["perception"]["phase_p5_ok_rate"], 0.2, places=4)


class TestRunMetricsScript(unittest.TestCase):
    """Test the run_metrics.py analysis script"""

    def test_analyze_empty(self):
        from scripts.run_metrics import analyze
        result = analyze([])
        self.assertEqual(result["error"], "no frames")

    def test_analyze_basic(self):
        from scripts.run_metrics import analyze
        frames = [
            {"cte_m": "0.1", "heading_err_rad": "0.01", "lane_conf": "0.9",
             "speed_ms": "5.0", "steering": "0.1", "throttle": "0.3", "brake": "0.0",
             "fps": "20.0", "mpc_solve_time_ms": "10.0", "loop_time_ms": "50.0",
             "geometry_valid": "true", "mpc_fallback": "false",
             "aeb_active": "false", "acc_active": "false", "ldw_warning_active": "false",
             "safety_active": "false", "stuck_recovery_active": "false",
             "lane_phase_p1": "true", "lane_phase_p2": "true",
             "lane_phase_p3": "true", "lane_phase_p4": "true", "lane_phase_p5": "true",
             "timestamp": "0.0"},
            {"cte_m": "0.2", "heading_err_rad": "0.02", "lane_conf": "0.8",
             "speed_ms": "6.0", "steering": "0.2", "throttle": "0.4", "brake": "0.0",
             "fps": "18.0", "mpc_solve_time_ms": "12.0", "loop_time_ms": "55.0",
             "geometry_valid": "false", "mpc_fallback": "true",
             "aeb_active": "true", "acc_active": "false", "ldw_warning_active": "true",
             "safety_active": "true", "stuck_recovery_active": "false",
             "lane_phase_p1": "true", "lane_phase_p2": "false",
             "lane_phase_p3": "false", "lane_phase_p4": "false", "lane_phase_p5": "false",
             "timestamp": "0.05"},
        ]
        result = analyze(frames)
        self.assertEqual(result["total_frames"], 2)
        self.assertAlmostEqual(result["perception"]["cte_m"]["mean"], 0.15, places=4)
        self.assertEqual(result["control"]["mpc_fallback_count"], 1)
        self.assertEqual(result["safety"]["aeb_trigger_count"], 1)


if __name__ == "__main__":
    unittest.main()
