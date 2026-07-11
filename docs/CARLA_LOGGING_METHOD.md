# CARLA Logging Methodology — ระบบบันทึกข้อมูลและวิเคราะห์พฤติกรรม

> **เอกสารอธิบายวิธีที่ระบบ CARLA MPC Lane Keeping บันทึกข้อมูล ตรวจจับความผิดปกติ และสร้างรายงาน AI-readable**
>
> This document describes the complete telemetry pipeline: what data is logged each frame, how lane detection failures are detected, how root-cause analysis works, what behavior patterns are tracked, and how the AI-readable report is structured.

---

## สารบัญ / Table of Contents

1. [ภาพรวมระบบ Telemetry](#1-ภาพรวมระบบ-telemetry)
2. [ข้อมูลที่บันทึกจาก CARLA ทุก frame](#2-ข้อมูลที่บันทึกจาก-carla-ทุก-frame)
3. [การตรวจจับ Lane Detection ล้มเหลว](#3-การตรวจจับ-lane-detection-ล้มเหลว)
4. [Root-Cause Analysis (LaneLossDetector: P1–P5)](#4-root-cause-analysis-lanelossdetector-p1p5)
5. [Behavior Patterns ที่ติดตาม](#5-behavior-patterns-ที่ติดตาม)
6.[โครงสร้างรายงาน AI-Readable (6 Sections)](#6-โครงสร้างรายงาน-ai-readable-6-sections)
7. [วิธีอ่านและตีความรายงานสำหรับ AI/LLM](#7-วิธีอ่านและตีความรายงานสำหรับ-aillm)
8. [ไฟล์ Output ทั้งหมด](#8-ไฟล์-output-ทั้งหมด)
9. [Workflow: รัน CARLA → ตรวจรายงาน → ส่งให้ LLM](#9-workflow-รัน-carla--ตรวจรายงาน--ส่งให้-llm)

---

## 1. ภาพรวมระบบ Telemetry

ระบบแบ่งการบันทึกเป็น 4 ชั้น (layers) ที่ทำงานร่วมกันใน main loop:

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Loop (src/main.py)                   │
│                                                              │
│  pipeline.step() → frame_state                               │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ RunLogger   │  │ BehaviorLogger   │  │ LaneLossDetect │  │
│  │ (low-level  │  │ (behavior events │  │ (root-cause    │  │
│  │  per-frame  │  │  + pattern detect)│  │  analysis)     │  │
│  │  CSV)       │  │                  │  │                │  │
│  └─────────────┘  └──────────────────┘  └────────────────┘  │
│       │                  │                       │          │
│       │          ┌──────────────────┐            │          │
│       │          │ ContextLogger    │            │          │
│       │          │ (black box: pre/ │            │          │
│       │          │  post-event ctx) │            │          │
│       │          └──────────────────┘            │          │
│       ▼                  ▼                       ▼          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           AiReportGenerator (cleanup)                │   │
│  │  BehaviorSummary + Events + Incidents + RunMeta      │   │
│  │     → ai_behavior_report.txt  (natural language)     │   │
│  │     → ai_behavior_report.json (structured JSON)      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### โมดูลที่เกี่ยวข้อง

| โมดูล | หน้าที่ | ไฟล์ |
|-------|--------|-----|
| `RunLogger` | บันทึก per-frame CSV + events.log | `src/metrics/run_logger.py` |
| `BehaviorLogger` | ตรวจจับ behavior patterns + สร้าง events | `src/telemetry/behavior_logger.py` |
| `BehaviorContextLogger` | Black box recorder: pre/post-event context | `src/telemetry/behavior_context_logger.py` |
| `LaneLossDetector` | Root-cause analysis ของ lane loss (P1–P5) | `src/telemetry/lane_loss_detector.py` |
| `AiReportGenerator` | สร้างรายงาน AI-readable 6 sections | `src/telemetry/ai_report_generator.py` |
| `TelemetryExporter` | ส่งข้อมูลไป InfluxDB (real-time) | `src/telemetry/influxdb_exporter.py` |
| `MetricsCollector` | รวบรวม metrics สำหรับ summary | `src/telemetry/metrics_collector.py` |

### จุดที่ telemetry ถูกเรียกใน main loop

ใน `src/main.py` บรรทัด ~479 (หลัง frame_state ถูก finalize ด้วย final_steer/throttle/brake + vehicle pose):

```python
# ── Behavior + Lane Loss telemetry ──────────────────────
if frame_state and self.behavior_logger:
    try:
        ctx = {"frame_idx": self.frame_count, "speed_ms": current_speed_ms,
               "loop_time_ms": loop_time_ms}
        self.behavior_logger.update(frame_state, ctx)
        if self.context_logger:
            self.context_logger.update(frame_state, ctx)
        if self.lane_loss_detector:
            incident = self.lane_loss_detector.update(frame_state, ctx)
            if incident:
                logger.warning("Lane loss detected: %s", incident.root_cause)
    except Exception as e:
        logger.debug(f"Behavior telemetry update skipped: {e}")
```

> **สำคัญ:** telemetry ทั้งหมดถูกห่อด้วย `try/except` เพื่อไม่ให้ telemetry crash การขับขี่

---

## 2. ข้อมูลที่บันทึกจาก CARLA ทุก frame

### 2.1 FrameState — ข้อมูลหลักที่ไหลผ่านทั้งระบบ

`FrameState` (ใน `src/state.py`) เป็น dataclass ที่รวบรวมข้อมูลทั้งหมดของ 1 frame:

| กลุ่ม | ฟิลด์ | คำอธิบาย |
|-------|-------|---------|
| **Vehicle State** | `speed_kmh`, `steer`, `throttle`, `brake` | ความเร็วและ control จาก MPC |
| **Lane State** | `cte_m`, `heading_rad`, `curvature` | Cross-track error, heading error, ความโค้ง |
| **Lane Detection** | `lane_conf`, `geometry_valid` | Confidence ของ lane mask, geometry valid หรือไม่ |
| **Phase Report** | `phase_p1_ok`..`phase_p5_ok`, `phase_p2_case`, `phase_p3_case` | ผลการตรวจสอบ 5 phases ของ lane detection |
| **Final Control** | `final_steer`, `final_throttle`, `final_brake` | ค่าจริงที่ส่งให้ CARLA (หลัง arbitration) |
| **Vehicle Pose** | `vehicle_x`, `vehicle_y`, `vehicle_z`, `vehicle_yaw` | พิกัดโลกของรถ |
| **MPC** | `mpc_solve_time_ms`, `solver_status` | เวลา solve, สถานะ solver (Solve_Succeeded / Fallback) |
| **ADAS** | `aeb_active`, `acc_active`, `ldw_state`, `lka_pro_assist`, `tja_state`, `tsr_*`, `bsw_*` | สถานะ ADAS ทั้งชุด |
| **Safety** | `safety_override_active`, `stuck_recovery_active`, `adas_override_active` | override ที่ทำงาน |

### 2.2 Context dict (ส่งเพิ่มจาก main loop)

```python
ctx = {
    "frame_idx": self.frame_count,     # เลข frame
    "speed_ms": current_speed_ms,       # ความเร็ว m/s (จาก CARLA)
    "loop_time_ms": loop_time_ms,       # เวลา 1 iteration ของ loop
}
```

### 2.3 สิ่งที่ CARLA ส่งให้ทุก frame

- **RGB camera image** — ภาพจากกล้องหน้ารถ (ส่งให้ perception pipeline)
- **Vehicle transform** — ตำแหน่ง (x, y, z) + rotation (pitch, yaw, roll)
- **Vehicle velocity** — ความเร็ว m/s
- **Vehicle control** — steering, throttle, brake (ส่งกลับไปควบคุมรถ)
- **Waypoint data** — จาก CARLA map API (ใช้สำหรับ fusion และ CTE คำนวณ)

---

## 3. การตรวจจับ Lane Detection ล้มเหลว

### 3.1 BehaviorLogger — ตรวจจับ confidence drop

`BehaviorLogger` ตรวจจับการล้มเหลวของ lane detection ด้วย thresholds:

| เหตุการณ์ | เงื่อนไข | Threshold |
|-----------|---------|-----------|
| `LANE_CONF_DROP` | confidence ตกต่ำกว่า threshold | `CONF_DROP_THRESHOLD = 0.3` |
| `LANE_LOST` | confidence ต่ำติดต่อกัน | ต่ำกว่า 0.3 ต่อเนื่อง |
| `LANE_RECOVERED` | confidence ขึ้นเหนือ threshold | `CONF_RECOVER_THRESHOLD = 0.5` |
| `GEOMETRY_INVALID` | `geometry_valid = False` | ตรวจจับจาก `frame_state.geometry_valid` |
| `PHASE_FAILURE` | phase P1–P5 ใดล้มเหลว | ตรวจจาก `phase_p1_ok`..`phase_p5_ok` |

### 3.2 LaneLossDetector — กำหนดว่า frame ไหน "lane lost"

`LaneLossDetector._is_lane_lost()` ใช้เกณฑ์ต่อไปนี้:

```python
def _is_lane_lost(self, conf, geom_valid, p1_ok, p2_ok, p3_ok, p4_ok, p5_ok):
    # confidence ต่ำกว่า threshold → lost
    if conf < self.conf_loss_threshold:    # default 0.3
        return True
    # geometry invalid → lost
    if not geom_valid:
        return True
    # P1 และ P4 ล้มเหลวทั้งคู่ → ไม่มี lane state ใช้ได้
    if not p1_ok and not p4_ok:
        return True
    return False
```

### 3.3 การตรวจจับแบบ state machine

Lane loss ทำงานเป็น state machine:

```
NORMAL → (lane lost) → LOST → (lane recovered) → NORMAL
                          │
                          ├── บันทึก: root_cause, conf_curve, CTE, speed, mode
                          ├── ติดตาม: cte_max_during_loss, drift_direction
                          └── ปิด incident เมื่อ recover หรือ finalize()
```

---

## 4. Root-Cause Analysis (LaneLossDetector: P1–P5)

### 4.1 5 Phases ของ Lane Detection Pipeline

| Phase | ชื่อ | ฟิลด์ใน FrameState | ความหมาย |
|-------|-----|-------------------|---------|
| **P1** | Lane Mask + Confidence | `phase_p1_ok` | สร้าง lane mask ได้และ confidence เพียงพอ |
| **P2** | Left/Right Boundary | `phase_p2_ok`, `phase_p2_case` | ตรวจจับขอบเลนซ้าย/ขวา (both / mirror_left / mirror_right / completion) |
| **P3** | Centerline | `phase_p3_ok`, `phase_p3_case` | สร้าง centerline ได้ (from_both / completion) |
| **P4** | Lane State | `phase_p4_ok` | คำนวณ CTE, heading, curvature ได้ |
| **P5** | Lane Phase/Mode | `phase_p5_ok` | mode ปกติ (ไม่ fallback) |

### 4.2 การจำแนก Root Cause

`_classify_root_cause()` ตรวจ phase แรกที่ล้มเหลว (ลำดับ P1 → P2 → P3 → P4 → P5):

| Root Cause | Phase ที่ล้มเหลว | เงื่อนไข |
|------------|-----------------|---------|
| `LOW_CONFIDENCE` | P1 | confidence ต่ำกว่า threshold |
| `BOUNDARY_LOST` | P2 | ไม่ได้ left/right boundary (มี case: both / mirror_left / mirror_right / completion) |
| `CENTERLINE_FAILED` | P3 | สร้าง centerline ไม่ได้ (มี case: from_both / completion) |
| `GEOMETRY_INVALID` | P4 | geometry ไม่ valid (CTE/heading/curvature คำนวณไม่ได้) |
| `MODE_FALLBACK` | P5 | mode เปลี่ยนเป็น fallback |
| `UNKNOWN` | — | ระบุไม่ได้ |

### 4.3 ข้อมูลที่บันทึกในแต่ละ LaneLossIncident

```
LaneLossIncident:
  ├── Timing: timestamp_start, timestamp_end, frame_idx_start, frame_idx_end, duration
  ├── Root Cause: root_cause, failed_phase, phase_case, root_cause_detail
  ├── Confidence: conf_at_loss, conf_at_recovery, conf_curve (30 frames ก่อน loss)
  ├── Lane State @ loss: cte_at_loss, heading_at_loss, curvature_at_loss
  ├── During loss: cte_max_during_loss, cte_mean_during_loss, drift_direction (left/right/none)
  ├── Vehicle context: vehicle_speed_at_loss, vehicle_pose_at_loss (x,y,z,yaw)
  ├── Mode tracking: mode_before, mode_during, mode_after
  ├── Recovery: recovery_method (conf_restored / phase_restored / mode_change / not_recovered)
  └── Severity: LOW / MEDIUM / HIGH / CRITICAL
```

### 4.4 Recovery Method

| Recovery Method | ความหมาย |
|----------------|---------|
| `CONF_RESTORED` | confidence กลับขึ้นเหนือ recover threshold |
| `PHASE_RESTORED` | phase P1–P5 กลับ ok ก่อน conf ขึ้น |
| `MODE_CHANGE` | mode เปลี่ยน (เช่น fallback → normal) |
| `NOT_RECOVERED` | ยังไม่ recover ตอนจบ run (finalize ปิดให้) |

### 4.5 Severity Classification

Severity คำนวณจาก duration + CTE สูงสุดระหว่าง loss:

| Severity | เงื่อนไข |
|----------|---------|
| `LOW` | สั้น + CTE น้อย |
| `MEDIUM` | ปานกลาง |
| `HIGH` | นานหรือ CTE สูง |
| `CRITICAL` | นานมาก + CTE สูงมาก (เสี่ยง off-track) |

---

## 5. Behavior Patterns ที่ติดตาม

`BehaviorLogger` ตรวจจับ behavior patterns ทุก frame โดยใช้ rolling windows:

### 5.1 Patterns และ Thresholds

| Pattern | BehaviorType | Threshold | วิธีตรวจจับ |
|---------|-------------|-----------|-----------|
| **Confidence Drop** | `LANE_CONF_DROP` | conf < 0.3 | เปรียบเทียบกับ `CONF_DROP_THRESHOLD` |
| **Lane Lost** | `LANE_LOST` | conf < 0.3 ต่อเนื่อง | state machine: ตก → lost → recover |
| **Lane Recovered** | `LANE_RECOVERED` | conf > 0.5 | confidence กลับขึ้นเหนือ `CONF_RECOVER_THRESHOLD` |
| **Phase Failure** | `PHASE_FAILURE` | phase_pX_ok = False | ตรวจจาก phase flags |
| **Geometry Invalid** | `GEOMETRY_INVALID` | geometry_valid = False | ตรวจจาก `frame_state.geometry_valid` |
| **Drifting Left** | `DRIFTING_LEFT` | \|CTE\| > 0.8m | `DRIFT_CTE_THRESHOLD`, CTE บวก = left |
| **Drifting Right** | `DRIFTING_RIGHT` | \|CTE\| > 0.8m | CTE ลบ = right |
| **Off-Track** | `OFF_TRACK` | \|CTE\| > 1.75m | `OFF_TRACK_CTE_THRESHOLD` |
| **Oscillation** | `OSCILLATION` | std(steer) > 0.15 ใน 30 frames | `OSCILLATION_WINDOW`, `OSCILLATION_STEER_THRESHOLD` |
| **Sudden Brake** | `SUDDEN_BRAKE` | brake เพิ่ม > 0.5 ใน 1 frame | `SUDDEN_BRAKE_THRESHOLD` |
| **Excessive Steering** | `EXCESSIVE_STEERING` | \|steer\| > 0.35 | `EXCESSIVE_STEER_THRESHOLD` |
| **MPC Unstable** | `MPC_UNSTABLE` | fallback ติดต่อกัน ≥ 5 frames | `MPC_UNSTABLE_CONSECUTIVE` |
| **Mode Change** | `PERCEPTION_MODE_CHANGE` | mode เปลี่ยน | เปรียบเทียบกับ prev_mode |

### 5.2 BehaviorSummary — สรุปทั้ง run

เมื่อเรียก `get_summary()` จะได้ `BehaviorSummary` ที่รวม:

- **Lane detection health**: `lane_conf_mean`, `lane_conf_min`, `lane_conf_drop_count`, `lane_lost_count`, `lane_lost_total_frames`, `lane_lost_max_duration_frames`, `geometry_invalid_count`, `phase_failure_counts`
- **Behavior pattern counts**: `drift_left_count`, `drift_right_count`, `off_track_count`, `off_track_total_frames`, `oscillation_count`, `sudden_brake_count`, `excessive_steering_count`, `mpc_unstable_count`, `mode_change_count`
- **Assessment**: `overall_lane_keeping_quality` (excellent / good / fair / poor / critical), `dominant_failure_mode`, `ai_readable_summary`

### 5.3 BehaviorContextLogger — Black Box Recorder

`BehaviorContextLogger` ทำหน้าที่เหมือน flight data recorder:

- รักษา **rolling window** ของ FrameState ย้อนหลัง 60 frames (default ~3–6 วินาที)
- เมื่อมี behavior event เกิดขึ้น จะ capture:
  - **30 frames ก่อน event** (pre-event context)
  - **frame ณ จุดเกิด event**
  - **15 frames หลัง event** (post-event context)
- สร้าง `BehaviorContextSnapshot` ที่มี:
  - peak/extreme metrics ระหว่าง event (peak_cte, min_conf, max_steer, max_brake)
  - vehicle trajectory ระหว่าง event (x, y coordinates)
  - recovery info (recovered, recovery_frame_idx)
  - AI-readable narrative ของแต่ละ event

---

## 6. โครงสร้างรายงาน AI-Readable (6 Sections)

`AiReportGenerator.generate_report()` สร้างรายงานแบ่งเป็น 6 sections:

### Section 1 — Run Overview

ภาพรวมของ run: map, vehicle, duration, total frames, target speed, overall lane keeping quality.

```
## SECTION 1 — RUN OVERVIEW

Run completed on **Town04** with **vehicle_lincoln_mkz_2017**.
Duration: 120.5s (1800 frames). Target speed: 30 kmh.
Overall lane keeping quality: **good**.
```

### Section 2 — Lane Detection Health

สถิติ lane confidence และความถี่ที่ lane หาย:

```
## SECTION 2 — LANE DETECTION HEALTH

Lane confidence: mean=0.872, min=0.123.
Lane was lost **3** times for total 45 frames (2.5% of run).
Max single loss duration: 20 frames.
Geometry was invalid 5 times.
Dominant failure mode: **lane_conf_drop**.
```

### Section 3 — Lane Loss Incidents (Detailed)

รายละเอียดแต่ละครั้งที่เสียเลน: frame, timestamp, root cause, confidence before/after, speed, curve type, duration, CTE peak, CTE direction, recovery method.

```
## SECTION 3 — LANE LOSS INCIDENTS

- Loss #1 at frame 234 (t=15.6s): low_confidence.
  Confidence dropped 0.85 → 0.12. Speed: 28.3 kmh on a gentle_curve.
  Lasted 12 frames, CTE peaked at 0.45m (center). Recovered via conf_restored.
```

### Section 4 — Behavior Patterns

สรุปพฤติกรรม: drift, off-track, oscillation, sudden brake, excessive steering, MPC unstable, mode change.

```
## SECTION 4 — BEHAVIOR PATTERNS

Drifting: left=2, right=1. Off-track: 0 times.
Oscillation: 1 episode. Sudden brake: 0.
Excessive steering: 3 frames. MPC unstable: 0.
Mode changes: 2.
```

### Section 5 — Failure Analysis

วิเคราะห์ failure แบบสรุป: dominant failure mode, conditions, pattern description, และคำแนะนำการปรับปรุง.

```
## SECTION 5 — FAILURE ANALYSIS

Dominant failure mode: lane_conf_drop.
Conditions: failures occurred primarily on gentle curves at moderate speed.
Pattern: confidence drops were brief (avg 15 frames) and recovered automatically.
Recommendation: tune the confidence recovery threshold and improve mask smoothing
to reduce false lane-loss on gentle curves.
```

### Section 6 — Critical Events Timeline

Timeline ของ events ที่มี severity WARNING ขึ้นไป เรียงตามเวลา:

```
## SECTION 6 — CRITICAL EVENTS TIMELINE

[frame 234, t=15.6s] WARNING: lane_conf_drop — confidence dropped to 0.12
[frame 246, t=16.4s] INFO: lane_recovered — confidence restored to 0.82
[frame 567, t=37.8s] WARNING: drifting_left — CTE=0.92m
[frame 890, t=59.3s] ERROR: lane_lost — lane detection lost for 20 frames
```

---

## 7. วิธีอ่านและตีความรายงานสำหรับ AI/LLM

### 7.1 รายงาน Text (.txt)

รายงาน `.txt` เป็น markdown-style natural language ที่ LLM อ่านแล้วเข้าใจได้ทันที. สามารถ paste ลง prompt ได้โดยตรง:

```python
from telemetry.ai_report_generator import AiReportGenerator

gen = AiReportGenerator()
report_text = open("runs/latest/ai_behavior_report.txt").read()
prompt = gen.generate_prompt_for_llm(
    report_text,
    "Why did the car lose the lane at frame 234? What can be improved?"
)
# ส่ง prompt ให้ LLM
```

### 7.2 รายงาน JSON (.json)

รายงาน `.json` เป็น structured format สำหรับ AI ที่รับ JSON input:

```json
{
  "generated_at": "2025-01-15T14:30:00",
  "run_overview": {
    "map": "Town04",
    "vehicle": "vehicle_lincoln_mkz_2017",
    "duration_s": 120.5,
    "total_frames": 1800,
    "target_speed_kmh": 30.0,
    "overall_lane_keeping_quality": "good"
  },
  "lane_detection_health": { ... },
  "lane_loss_incidents": [ ... ],
  "behavior_patterns": { ... },
  "failure_analysis": {
    "dominant_failure_mode": "lane_conf_drop",
    "conditions": "...",
    "pattern": "...",
    "recommendation": "..."
  },
  "critical_events": [ ... ],
  "summary_card": "Run on Town04 ..."
}
```

### 7.3 วิธีตีความสำหรับ AI

| ฟิลด์ | ตีความ |
|-------|--------|
| `overall_lane_keeping_quality` | excellent = ไม่มีปัญหา, critical = มีปัญหารุนแรง |
| `dominant_failure_mode` | สาเหตุหลักที่ทำให้ lane keeping มีปัญหา |
| `lane_lost_count` | จำนวนครั้งที่ lane หาย — 0 = ดีมาก, >5 = น่ากังวล |
| `lane_lost_total_frames` | รวม frames ที่หาย — เปอร์เซ็นต์ต่ำ = ดี |
| `off_track_count` | จำนวนครั้งที่รถออกจากเลน — ควรเป็น 0 |
| `oscillation_count` | ความถี่ที่ steering สั่น — สูง = MPC ไม่ stable |
| `mpc_unstable_count` | จำนวนครั้งที่ MPC fallback ติดต่อกัน — สูง = solver มีปัญหา |
| `root_cause` (ใน incidents) | P1=confidence, P2=boundary, P3=centerline, P4=geometry, P5=mode |
| `recovery_method` | auto = กลับมาเอง, not_recovered = ไม่กลับมา (ร้ายแรง) |
| `severity` (ใน incidents) | LOW = ไม่เป็นไร, CRITICAL = เสี่ยง off-track |

### 7.4 คำถามที่ AI ควรถามจากรายงาน

1. **"ทำไมรถถึงออกจากเลน?"** → ดู Section 3 (incidents) + Section 5 (failure analysis)
2. **"MPC มีปัญหาไหม?"** → ดู `mpc_unstable_count`, `solver_status` ใน events
3. **"ควรปรับอะไร?"** → ดู Section 5 `recommendation`
4. **"เกิดอะไรขึ้นตอน frame 234?"** → ดู Section 6 (timeline) + context snapshots
5. **"รถขับดีไหม?"** → ดู Section 1 `overall_lane_keeping_quality` + Section 2

---

## 8. ไฟล์ Output ทั้งหมด

ทุก run จะสร้างไฟล์ใน `runs/<timestamp>_<map>/`:

### 8.1 ไฟล์หลัก

| ไฟล์ | ผู้สร้าง | เนื้อหา |
|------|---------|--------|
| `frame_log.csv` | RunLogger | ข้อมูลทุก frame: frame_idx, speed, steer, throttle, brake, cte_m, heading_rad, lane_conf, mode, solver_status, mpc_solve_time_ms, ADAS status, ... |
| `events.log` | RunLogger | Event log: collision, lane_invasion, stuck recovery, ADAS events |
| `run_meta.json` | RunLogger | Run metadata: map, vehicle, target_speed, timestamp |
| `report.md` | RunAnalyzer | รายงาน KPI หลังจบ run (CTE RMSE, in-lane %, fallback %, FPS) |

### 8.2 ไฟล์ Behavior Telemetry

| ไฟล์ | ผู้สร้าง | เนื้อหา |
|------|---------|--------|
| `ai_behavior_report.txt` | AiReportGenerator | รายงาน AI-readable 6 sections (natural language, markdown-style) |
| `ai_behavior_report.json` | AiReportGenerator | รายงาน structured JSON (สำหรับ AI ที่รับ JSON input) |
| `snapshots/` | BehaviorLogger | Snapshot metadata ของ critical events (ไม่เก็บรูป, เก็บเฉพาะข้อมูล) |

### 8.3 ไฟล์ Metrics

| ไฟล์ | ผู้สร้าง | เนื้อหา |
|------|---------|--------|
| `metrics_output/run_*.csv` | MetricsCollector | Per-frame metrics สำหรับ summary |
| `metrics_output/run_*.json` | MetricsCollector | Summary statistics |
| `plots/` | RunAnalyzer | กราฟ CTE, speed, steering ตามเวลา |

### 8.4 ตัวอย่างโครงสร้างไดเรกทอรี

```
runs/
└── 2025-01-15_14-30-00_Town04/
    ├── frame_log.csv              ← ข้อมูลทุก frame
    ├── events.log                 ← event log
    ├── run_meta.json              ← run metadata
    ├── report.md                  ← KPI report (RunAnalyzer)
    ├── ai_behavior_report.txt     ← AI-readable report (6 sections)
    ├── ai_behavior_report.json    ← structured JSON report
    └── snapshots/                 ← behavior event snapshots
        ├── snapshot_001.json
        └── snapshot_002.json
```

---

## 9. Workflow: รัน CARLA → ตรวจรายงาน → ส่งให้ LLM

### Step 1: รัน CARLA

```bash
# เปิด CARLA simulator ก่อน
./CarlaUE4.sh -quality-level=Low -RenderOffScreen

# รันระบบ
cd FrontViewPredictiveControl/src
python3 main.py --model ../model/lane_unet_final.pth --town Town04 --speed 30
```

### Step 2: หาไฟล์ผลลัพธ์

```bash
# ดู runs ล่าสุด
ls -lt runs/ | head -5

# ไดเรกทอรีของ run ล่าสุด
LATEST_RUN=$(ls -t runs/ | head -1)
echo "Run dir: runs/$LATEST_RUN"
```

### Step 3: ตรวจรายงานเบื้องต้น

```bash
# อ่าน AI behavior report (text)
cat "runs/$LATEST_RUN/ai_behavior_report.txt"

# ตรวจ JSON report (สำหรับ AI)
python3 -m json.tool "runs/$LATEST_RUN/ai_behavior_report.json" | head -50

# ดู frame log สำหรับ frame ที่มีปัญหา
head -1 "runs/$LATEST_RUN/frame_log.csv"  # header
grep "Fallback" "runs/$LATEST_RUN/frame_log.csv" | head -10  # frames ที่ solver fail
```

### Step 4: ส่งรายงานให้ LLM วิเคราะห์

```python
#!/usr/bin/env python3
"""ส่ง AI behavior report ให้ LLM วิเคราะห์"""
import sys
sys.path.insert(0, "src")
from telemetry.ai_report_generator import AiReportGenerator

gen = AiReportGenerator()

# อ่านรายงาน
report_text = open("runs/latest/ai_behavior_report.txt").read()

# สร้าง prompt
question = """
Analyze this CARLA MPC lane-keeping run:
1. What were the main causes of lane loss?
2. Which phase (P1-P5) failed most often?
3. What specific improvements would you recommend?
4. Were there any dangerous behavior patterns?
"""
prompt = gen.generate_prompt_for_llm(report_text, question)

# ส่งให้ LLM (เช่น OpenAI API)
print(prompt)
# หรือ: response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
```

### Step 5: เปรียบเทียบรันหลายครั้ง

```bash
# เปรียบเทียบ quality ระหว่าง runs
for dir in runs/*/; do
    if [ -f "$dir/ai_behavior_report.json" ]; then
        echo "=== $dir ==="
        python3 -c "
import json
d = json.load(open('${dir}ai_behavior_report.json'))
ov = d['run_overview']
h = d['lane_detection_health']
print(f\"  Map: {ov['map']}, Quality: {ov['overall_lane_keeping_quality']}\")
print(f\"  Lane lost: {h['lane_lost_count']}x, Conf mean: {h['lane_conf_mean']:.3f}\")
print(f\"  Dominant failure: {h['dominant_failure_mode']}\")
"
    fi
done
```

---

## ภาคผนวก: Configuration และ Thresholds

### BehaviorLogger Thresholds (สามารถ override ได้ใน `src/telemetry/behavior_logger.py`)

| Parameter | Default | ความหมาย |
|-----------|---------|---------|
| `CONF_DROP_THRESHOLD` | 0.3 | confidence ต่ำกว่านี้ = drop |
| `CONF_RECOVER_THRESHOLD` | 0.5 | confidence ขึ้นเหนือนี้ = recovered |
| `DRIFT_CTE_THRESHOLD` | 0.8m | \|CTE\| เกินนี้ = drifting |
| `OFF_TRACK_CTE_THRESHOLD` | 1.75m | \|CTE\| เกินนี้ = off-track |
| `OSCILLATION_WINDOW` | 30 frames | window สำหรับตรวจ oscillation |
| `OSCILLATION_STEER_THRESHOLD` | 0.15 | std(steer) เกินนี้ = oscillation |
| `SUDDEN_BRAKE_THRESHOLD` | 0.5 | brake เพิ่มเกินนี้ใน 1 frame |
| `EXCESSIVE_STEER_THRESHOLD` | 0.35 | \|steer\| เกินนี้ |
| `MPC_UNSTABLE_CONSECUTIVE` | 5 | fallback ติดต่อกันเกินนี้ = unstable |

### LaneLossDetector Thresholds

| Parameter | Default | ความหมาย |
|-----------|---------|---------|
| `conf_loss_threshold` | 0.3 | confidence ต่ำกว่านี้ = lane lost |
| `conf_recover_threshold` | 0.5 | confidence เหนือนี้ = recovered |
| `conf_curve_len` | 30 | จำนวน frames ก่อน loss ที่เก็บใน conf_curve |
| `severe_cte_threshold` | 1.5m | CTE เกินนี้ระหว่าง loss = severity สูง |
| `severe_duration_frames` | 30 | loss นานเกินนี้ = severity สูง |

### BehaviorContextLogger Parameters

| Parameter | Default | ความหมาย |
|-----------|---------|---------|
| `rolling_window` | 60 frames | buffer ย้อนหลัง (~3–6 วินาที) |
| `pre_event_frames` | 30 | frames ก่อน event ที่ capture |
| `post_event_frames` | 15 | frames หลัง event ที่ capture |
| `estimated_fps` | 15.0 | ประมาณ fps สำหรับคำนวณ duration |

---

> **หมายเหตุ:** ระบบ telemetry ทั้งหมดถูกห่อด้วย `try/except` ใน main loop เพื่อให้แน่ใจว่า **telemetry failure จะไม่ crash การขับขี่**. หาก telemetry มีปัญหา ระบบจะ log debug message และทำงานต่อไปได้ปกติ
