"""
Obstacle Avoidance Test Report Generator

สร้าง test report สำหรับระบบหลบสิ่งกีดขวาง (obstacle avoidance)
ตามมาตรฐานสากล:
  - Euro NCAP AEB C2C v4.3.1 (2024)
  - Euro NCAP ESS Assessment v1.0
  - NHTSA FMVSS No. 127 AEB
  - ISO 21448 SOTIF
  - SAE J2944 surrogate safety metrics

รองรับทั้ง:
  - Simulation (CARLA) — ทดสอบใน sim ก่อน
  - Real-world — ทดสอบในสนามจริง

เก็บ metrics ที่ต้องเช็คตอนหลบ:
  1. TTC (Time-to-Collision) — เวลาก่อนชน
  2. Minimum lateral clearance — ระยะหลบด้านข้างน้อยสุด
  3. Speed reduction — ลดความเร็งได้กี่ %
  4. Lateral acceleration — ความเร่งด้านข้าง (comfort)
  5. Steering response time — เวลาตอบสนองเลี้ยว
  6. Lane departure — ออกจาก lane ไหม
  7. Collision avoided — หลบสำเร็จไหม
  8. Stability — รถเสถียรหลังหลบไหม (yaw rate, oscillation)
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TestEnvironment(Enum):
    """สภาพแวดล้อมการทดสอบ"""
    SIMULATION = "simulation"      # CARLA
    REAL_WORLD = "real_world"      # สนามจริง
    HIL = "hil"                    # Hardware-in-the-loop


class TestScenario(Enum):
    """
    สถานการณ์ทดสอบตาม Euro NCAP / NHTSA
    """
    # Car-to-Car
    CCRS = "CCRs"          # Car-to-Car Rear stationary
    CCRM = "CCRm"          # Car-to-Car Rear moving
    CCRB = "CCRb"          # Car-to-Car Rear braking
    # VRU
    CPNA = "CPNA"          # Child Pedestrian Nearside Adult
    CPNC = "CPNC"          # Child Pedestrian Nearside Child
    CBLA = "CBLA"          # Cyclist Bicyclist Lateral Adult
    # Steering avoidance
    ESS = "ESS"            # Emergency Steering Support
    CUT_IN = "CutIn"       # Vehicle cuts in
    CUT_OUT = "CutOut"     # Vehicle cuts out, reveals stopped car
    STATIONARY = "Stationary"  # Stationary obstacle ahead


class TestSpeed(Enum):
    """ความเร็วทดสอบตามมาตรฐาน"""
    LOW = 20       # km/h — ทางเมือง
    MEDIUM = 40    # km/h — Euro NCAP AEB VRU
    HIGH = 60      # km/h — ทางหลวงชานเมือง
    HIGHWAY = 90   # km/h — ทางหลวง


@dataclass
class AvoidanceMetrics:
    """
    Metrics ที่ต้องเช็คตอนหลบสิ่งกีดขวาง

    อ้างอิง:
      - Euro NCAP AEB C2C v4.3.1
      - NHTSA FMVSS 127
      - SAE J2944 surrogate safety metrics
      - ISO 21448 SOTIF validation
    """
    # ── 1. Collision outcome ──────────────────────────────────────────────
    collision_avoided: bool = False       # หลบสำเร็จไหม
    collision_occurred: bool = False      # ชนไหม
    impact_speed_kmh: float = 0.0         # ความเร็วตอนชน (0 = หลบสำเร็จ)

    # ── 2. TTC (Time-to-Collision) ────────────────────────────────────────
    ttc_initial: float = float('inf')     # TTC ตอนเริ่มเห็น obstacle
    ttc_min: float = float('inf')         # TTC น้อยสุดระหว่างหลบ
    ttc_at_steer: float = float('inf')    # TTC ตอนเริ่มเลี้ยวหลบ
    ttc_at_brake: float = float('inf')    # TTC ตอนเริ่มเบรก

    # ── 3. Lateral clearance ──────────────────────────────────────────────
    min_lateral_clearance_m: float = float('inf')  # ระยะหลบด้านข้างน้อยสุด
    max_lateral_offset_m: float = 0.0              # ระยะเลื้อยด้านข้างสูงสุด
    lateral_offset_at_pass_m: float = 0.0          # ระยะเลื้อยตอนผ่าน obstacle

    # ── 4. Speed reduction ────────────────────────────────────────────────
    speed_initial_kmh: float = 0.0        # ความเร็วตอนเริ่ม
    speed_at_obstacle_kmh: float = 0.0    # ความเร็วตอนผ่าน obstacle
    speed_min_kmh: float = 0.0            # ความเร็วน้อยสุดระหว่างหลบ
    speed_reduction_pct: float = 0.0      # % ลดความเร็ว

    # ── 5. Lateral acceleration (comfort) ─────────────────────────────────
    max_lateral_accel_mss: float = 0.0    # ความเร่งด้านข้างสูงสุด
    mean_lateral_accel_mss: float = 0.0   # ความเร่งด้านข้างเฉลี่ย
    # Euro NCAP comfort threshold: < 3.0 m/s² (comfortable)
    #                                  < 5.0 m/s² (acceptable)
    #                                  > 5.0 m/s² (aggressive)

    # ── 6. Steering response ──────────────────────────────────────────────
    steer_response_time_s: float = 0.0    # เวลาจากเห็น obstacle → เริ่มเลี้ยว
    steer_max_rad: float = 0.0            # องศาเลี้ยวสูงสุด
    steer_rate_max_rads: float = 0.0      # อัตราเลี้ยวสูงสุด (rad/s)
    # Euro NCAP ESS: steering rate 150°/s = 2.62 rad/s

    # ── 7. Lane departure ─────────────────────────────────────────────────
    lane_departure: bool = False          # ออกจาก lane ไหม
    lane_departure_duration_s: float = 0.0  # ออกจาก lane นานแค่ไหน
    # Euro NCAP ESS: ห้ามออกจาก adjacent lane เกิน -0.3m นานเกิน 2s

    # ── 8. Stability (post-avoidance) ─────────────────────────────────────
    yaw_rate_max_degs: float = 0.0        # yaw rate สูงสุด
    yaw_rate_std_degs: float = 0.0        # yaw rate std (oscillation)
    heading_overshoot_deg: float = 0.0    # heading overshoot หลังหลบ
    oscillation_count: int = 0            # จำนวนครั้งที่ oscillation
    recovery_time_s: float = 0.0          # เวลากลับสู่เส้นทางเดิม

    # ── 9. Braking ────────────────────────────────────────────────────────
    brake_applied: bool = False
    brake_max: float = 0.0                # brake สูงสุด (0-1)
    brake_response_time_s: float = 0.0    # เวลาจากเห็น obstacle → เบรก

    # ── 10. Overall score ─────────────────────────────────────────────────
    overall_pass: bool = False
    score: float = 0.0                    # 0-100
    grade: str = "F"                      # A/B/C/D/F


@dataclass
class TestRun:
    """ผลการทดสอบ 1 ครั้ง"""
    run_id: int
    scenario: str
    speed_kmh: float
    environment: str  # simulation / real_world
    metrics: AvoidanceMetrics
    timestamp: str = ""
    notes: str = ""


class AvoidanceTestReport:
    """
    สร้าง test report สำหรับ obstacle avoidance

    ใช้:
        report = AvoidanceTestReport(environment="simulation")
        report.start_run(scenario="CCRs", speed_kmh=20)
        # ... run test, collect data ...
        report.record_ttc(ttc_value)
        report.record_lateral_clearance(clearance_m)
        # ...
        run = report.end_run(collision_avoided=True)
        report.save("/tmp/avoidance_report.json")
        report.print_summary()
    """

    # Euro NCAP / NHTSA thresholds
    TTC_WARNING = 2.5       # s — ควรเริ่มเตือน
    TTC_CRITICAL = 1.0      # s — ควรเบรก/หลบ
    TTC_MIN_SAFE = 0.5      # s — อันตรายถ้าน้อยกว่านี้

    LATERAL_CLEARANCE_MIN = 0.3    # m — ระยะหลบข้างน้อยสุดที่ปลอดภัย
    LATERAL_ACCEL_COMFORT = 3.0    # m/s² — สบาย
    LATERAL_ACCEL_MAX = 5.0        # m/s² — ยอมรับได้
    STEER_RATE_MAX = 2.62          # rad/s — Euro NCAP ESS (150°/s)
    LANE_DEPARTURE_MAX_S = 2.0     # s — ห้ามออกจาก lane เกิน 2s

    def __init__(self, environment: str = "simulation") -> None:
        self.environment = environment
        self.runs: List[TestRun] = []
        self._current_run: Optional[TestRun] = None
        self._current_metrics: Optional[AvoidanceMetrics] = None
        self._start_time: float = 0.0
        self._obstacle_detected: bool = False
        self._steer_applied: bool = False
        self._brake_applied: bool = False
        self._obstacle_detected_time: float = 0.0
        self._lateral_accel_history: List[float] = []
        self._yaw_rate_history: List[float] = []
        self._lane_departure_start: Optional[float] = None
        self._lane_departure_total: float = 0.0

    def start_run(self, scenario: str, speed_kmh: float) -> None:
        """เริ่มการทดสอบ 1 ครั้ง"""
        self._current_metrics = AvoidanceMetrics(
            speed_initial_kmh=speed_kmh,
            ttc_initial=float('inf'),
        )
        self._start_time = time.time()
        self._obstacle_detected = False
        self._steer_applied = False
        self._brake_applied = False
        self._obstacle_detected_time = 0.0
        self._lateral_accel_history = []
        self._yaw_rate_history = []
        self._lane_departure_start = None
        self._lane_departure_total = 0.0
        logger.info(f"Test run started: {scenario} @ {speed_kmh} km/h ({self.environment})")

    def record_obstacle_detected(self, ttc: float) -> None:
        """บันทึกว่าตรวจจับ obstacle แล้ว"""
        if not self._obstacle_detected:
            self._obstacle_detected = True
            self._obstacle_detected_time = time.time()
            if self._current_metrics:
                self._current_metrics.ttc_initial = ttc

    def record_ttc(self, ttc: float) -> None:
        """บันทึก TTC ปัจจุบัน"""
        if self._current_metrics:
            self._current_metrics.ttc_min = min(self._current_metrics.ttc_min, ttc)

    def record_steer(self, steer_rad: float, ttc: float = float('inf')) -> None:
        """บันทึกค่าเลี้ยว"""
        if self._current_metrics:
            self._current_metrics.steer_max_rad = max(
                self._current_metrics.steer_max_rad, abs(steer_rad)
            )
            if not self._steer_applied and abs(steer_rad) > 0.05:
                self._steer_applied = True
                if self._obstacle_detected:
                    self._current_metrics.steer_response_time_s = (
                        time.time() - self._obstacle_detected_time
                    )
                self._current_metrics.ttc_at_steer = ttc

    def record_brake(self, brake: float, ttc: float = float('inf')) -> None:
        """บันทึกค่าเบรก"""
        if self._current_metrics:
            self._current_metrics.brake_max = max(self._current_metrics.brake_max, brake)
            if not self._brake_applied and brake > 0.1:
                self._brake_applied = True
                if self._obstacle_detected:
                    self._current_metrics.brake_response_time_s = (
                        time.time() - self._obstacle_detected_time
                    )
                self._current_metrics.ttc_at_brake = ttc
                self._current_metrics.brake_applied = True

    def record_lateral_clearance(self, clearance_m: float) -> None:
        """บันทึกระยะหลบด้านข้างน้อยสุด"""
        if self._current_metrics:
            self._current_metrics.min_lateral_clearance_m = min(
                self._current_metrics.min_lateral_clearance_m, clearance_m
            )

    def record_lateral_offset(self, offset_m: float) -> None:
        """บันทึกระยะเลื้อยด้านข้าง"""
        if self._current_metrics:
            self._current_metrics.max_lateral_offset_m = max(
                self._current_metrics.max_lateral_offset_m, abs(offset_m)
            )
            self._current_metrics.lateral_offset_at_pass_m = offset_m

    def record_speed(self, speed_kmh: float) -> None:
        """บันทึกความเร็วปัจจุบัน"""
        if self._current_metrics:
            self._current_metrics.speed_min_kmh = min(
                self._current_metrics.speed_min_kmh, speed_kmh
            ) if self._current_metrics.speed_min_kmh > 0 else speed_kmh
            self._current_metrics.speed_at_obstacle_kmh = speed_kmh

    def record_lateral_accel(self, accel_mss: float) -> None:
        """บันทึกความเร่งด้านข้าง"""
        if self._current_metrics:
            self._current_metrics.max_lateral_accel_mss = max(
                self._current_metrics.max_lateral_accel_mss, abs(accel_mss)
            )
            self._lateral_accel_history.append(abs(accel_mss))

    def record_yaw_rate(self, yaw_rate_degs: float) -> None:
        """บันทึก yaw rate"""
        if self._current_metrics:
            self._current_metrics.yaw_rate_max_degs = max(
                self._current_metrics.yaw_rate_max_degs, abs(yaw_rate_degs)
            )
            self._yaw_rate_history.append(yaw_rate_degs)

    def record_lane_departure(self, departed: bool) -> None:
        """บันทึกว่าออกจาก lane ไหม"""
        if self._current_metrics:
            if departed and self._lane_departure_start is None:
                self._lane_departure_start = time.time()
                self._current_metrics.lane_departure = True
            elif not departed and self._lane_departure_start is not None:
                self._lane_departure_total += time.time() - self._lane_departure_start
                self._lane_departure_start = None

    def end_run(self, collision_avoided: bool, impact_speed_kmh: float = 0.0,
                notes: str = "") -> TestRun:
        """จบการทดสอบ 1 ครั้ง คำนวณ metrics สรุป"""
        if not self._current_metrics:
            raise RuntimeError("No active test run")

        m = self._current_metrics
        m.collision_avoided = collision_avoided
        m.collision_occurred = not collision_avoided
        m.impact_speed_kmh = impact_speed_kmh

        # Speed reduction
        if m.speed_initial_kmh > 0:
            m.speed_reduction_pct = (
                (m.speed_initial_kmh - m.speed_min_kmh) / m.speed_initial_kmh * 100
            )

        # Lateral accel mean
        if self._lateral_accel_history:
            m.mean_lateral_accel_mss = float(np.mean(self._lateral_accel_history))

        # Yaw rate std (oscillation)
        if self._yaw_rate_history:
            m.yaw_rate_std_degs = float(np.std(self._yaw_rate_history))
            # Count oscillations (sign changes)
            signs = np.sign(np.array(self._yaw_rate_history))
            sign_changes = np.sum(np.diff(signs) != 0)
            m.oscillation_count = int(sign_changes)

        # Lane departure duration
        if self._lane_departure_start is not None:
            m.lane_departure_duration_s = (
                self._lane_departure_total + time.time() - self._lane_departure_start
            )
        else:
            m.lane_departure_duration_s = self._lane_departure_total

        # Compute score
        m.score = self._compute_score(m)
        m.overall_pass = m.score >= 60
        m.grade = self._score_to_grade(m.score)

        run = TestRun(
            run_id=len(self.runs) + 1,
            scenario="unknown",  # set by caller
            speed_kmh=m.speed_initial_kmh,
            environment=self.environment,
            metrics=m,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            notes=notes,
        )
        self.runs.append(run)
        self._current_run = run
        self._current_metrics = None

        logger.info(
            f"Test run ended: {'AVOIDED' if collision_avoided else 'COLLISION'} "
            f"score={m.score:.1f} grade={m.grade}"
        )
        return run

    def _compute_score(self, m: AvoidanceMetrics) -> float:
        """
        คำนวณ score 0-100 ตามเกณฑ์ Euro NCAP / NHTSA

        น้ำหนัก:
          - Collision avoided: 40 points (must-pass)
          - TTC min > 0.5s: 15 points
          - Lateral clearance > 0.3m: 15 points
          - Lateral accel < 5.0 m/s²: 10 points (comfort)
          - No lane departure > 2s: 10 points
          - Stability (yaw std < 2°/s): 10 points
        """
        score = 0.0

        # 1. Collision avoided (40 pts)
        if m.collision_avoided:
            score += 40

        # 2. TTC min (15 pts)
        if m.ttc_min > self.TTC_MIN_SAFE:
            score += 15
        elif m.ttc_min > 0.2:
            score += 7.5

        # 3. Lateral clearance (15 pts)
        if m.min_lateral_clearance_m > self.LATERAL_CLEARANCE_MIN:
            score += 15
        elif m.min_lateral_clearance_m > 0.1:
            score += 7.5

        # 4. Lateral accel (10 pts)
        if m.max_lateral_accel_mss < self.LATERAL_ACCEL_COMFORT:
            score += 10
        elif m.max_lateral_accel_mss < self.LATERAL_ACCEL_MAX:
            score += 5

        # 5. Lane departure (10 pts)
        if m.lane_departure_duration_s < self.LANE_DEPARTURE_MAX_S:
            score += 10
        elif m.lane_departure_duration_s < 4.0:
            score += 5

        # 6. Stability (10 pts)
        if m.yaw_rate_std_degs < 2.0:
            score += 10
        elif m.yaw_rate_std_degs < 5.0:
            score += 5

        return score

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def save(self, filepath: str) -> None:
        """บันทึก report เป็น JSON"""
        data = {
            "environment": self.environment,
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "standards": [
                "Euro NCAP AEB C2C v4.3.1 (2024)",
                "Euro NCAP ESS Assessment v1.0",
                "NHTSA FMVSS No. 127",
                "ISO 21448 SOTIF",
                "SAE J2944",
            ],
            "thresholds": {
                "ttc_warning_s": self.TTC_WARNING,
                "ttc_critical_s": self.TTC_CRITICAL,
                "ttc_min_safe_s": self.TTC_MIN_SAFE,
                "lateral_clearance_min_m": self.LATERAL_CLEARANCE_MIN,
                "lateral_accel_comfort_mss": self.LATERAL_ACCEL_COMFORT,
                "lateral_accel_max_mss": self.LATERAL_ACCEL_MAX,
                "steer_rate_max_rads": self.STEER_RATE_MAX,
                "lane_departure_max_s": self.LANE_DEPARTURE_MAX_S,
            },
            "runs": [asdict(r) for r in self.runs],
            "summary": self._summary(),
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Report saved to {filepath}")

    def _summary(self) -> dict:
        if not self.runs:
            return {}
        n = len(self.runs)
        avoided = sum(1 for r in self.runs if r.metrics.collision_avoided)
        scores = [r.metrics.score for r in self.runs]
        return {
            "total_runs": n,
            "collisions_avoided": avoided,
            "collisions_occurred": n - avoided,
            "avoidance_rate_pct": avoided / n * 100,
            "mean_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "mean_ttc_min": float(np.mean([r.metrics.ttc_min for r in self.runs])),
            "mean_lateral_clearance": float(
                np.mean([r.metrics.min_lateral_clearance_m for r in self.runs])
            ),
        }

    def print_summary(self) -> None:
        """พิมพ์สรุปผลการทดสอบ"""
        s = self._summary()
        if not s:
            print("No test runs recorded.")
            return

        print("\n" + "=" * 70)
        print(f"  OBSTACLE AVOIDANCE TEST REPORT — {self.environment.upper()}")
        print("=" * 70)
        print(f"  Standards: Euro NCAP AEB v4.3.1 / NHTSA FMVSS 127 / ISO 21448")
        print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)
        print(f"  Total runs:           {s['total_runs']}")
        print(f"  Collisions avoided:   {s['collisions_avoided']}/{s['total_runs']} "
              f"({s['avoidance_rate_pct']:.1f}%)")
        print(f"  Collisions occurred:  {s['collisions_occurred']}")
        print(f"  Mean score:           {s['mean_score']:.1f}/100")
        print(f"  Score range:          {s['min_score']:.1f} - {s['max_score']:.1f}")
        print(f"  Mean min TTC:         {s['mean_ttc_min']:.2f}s")
        print(f"  Mean lateral clear:   {s['mean_lateral_clearance']:.2f}m")
        print("-" * 70)
        print("  Per-run details:")
        print(f"  {'ID':>3} {'Scenario':<12} {'Speed':>6} {'Avoid':>6} "
              f"{'Score':>6} {'Grade':>5} {'TTCmin':>7} {'LatClr':>7}")
        for r in self.runs:
            m = r.metrics
            print(f"  {r.run_id:>3} {r.scenario:<12} {r.speed_kmh:>5.0f}k "
                  f"{'YES' if m.collision_avoided else 'NO':>6} "
                  f"{m.score:>5.1f} {m.grade:>5} "
                  f"{m.ttc_min:>6.2f}s {m.min_lateral_clearance_m:>6.2f}m")
        print("=" * 70)

    def print_checklist(self) -> None:
        """พิมพ์ checklist สิ่งที่ต้องเช็คตอนหลบ"""
        print("\n" + "=" * 70)
        print("  CHECKLIST: สิ่งที่ต้องเช็คตอนหลบสิ่งกีดขวาง")
        print("=" * 70)
        checks = [
            ("1. การหลบสำเร็จ", "Collision avoided", "ห้ามชนเด็ดขาด"),
            ("2. TTC น้อยสุด", f"TTC min > {self.TTC_MIN_SAFE}s",
             f"ต้อง > {self.TTC_MIN_SAFE}s (ปลอดภัย) / > 0.2s (ตกลง)"),
            ("3. ระยะหลบข้าง", f"Clearance > {self.LATERAL_CLEARANCE_MIN}m",
             f"ต้อง > {self.LATERAL_CLEARANCE_MIN}m จาก obstacle"),
            ("4. ความเร่งข้าง", f"Lat accel < {self.LATERAL_ACCEL_MAX} m/s²",
             f"< {self.LATERAL_ACCEL_COMFORT} สบาย / < {self.LATERAL_ACCEL_MAX} ยอมรับได้"),
            ("5. ออกจาก lane", f"Departure < {self.LANE_DEPARTURE_MAX_S}s",
             f"ห้ามออกจาก lane เกิน {self.LANE_DEPARTURE_MAX_S}s (Euro NCAP ESS)"),
            ("6. เสถียรภาพ", "Yaw std < 2°/s",
             "ห้าม oscillation หลังหลบ ต้องกลับสู่เส้นทางเดิม"),
            ("7. เวลาตอบสนอง", "Steer response < 1.0s",
             "เวลาจากเห็น obstacle → เริ่มเลี้ยว ต้อง < 1s"),
            ("8. อัตราเลี้ยว", f"Steer rate < {self.STEER_RATE_MAX} rad/s",
             f"ห้ามเลี้ยวเร็วเกิน {self.STEER_RATE_MAX} rad/s (150°/s Euro NCAP)"),
            ("9. ลดความเร็ว", "Speed reduction > 20%",
             "ควรลดความเร็วอย่างน้อย 20% ตอนเข้าใกล้ obstacle"),
            ("10. เบรก", "Brake response < 1.0s",
             "ถ้าเบรก ต้องตอบสนองภายใน 1s"),
        ]
        for name, criteria, detail in checks:
            print(f"  [ ] {name:<20} {criteria:<25} — {detail}")
        print("=" * 70)
        print("  มาตรฐานอ้างอิง:")
        print("    - Euro NCAP AEB C2C v4.3.1 (2024)")
        print("    - Euro NCAP ESS Assessment v1.0")
        print("    - NHTSA FMVSS No. 127 AEB")
        print("    - ISO 21448 SOTIF")
        print("    - SAE J2944 surrogate safety metrics")
        print("=" * 70)
