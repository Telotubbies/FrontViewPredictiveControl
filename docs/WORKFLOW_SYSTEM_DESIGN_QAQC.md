# Workflow & System Design — หลักการ QA/QC (Senior AI & Autonomous Driving)

เอกสารนี้อธิบาย **workflow ระบบ LKA** และ **system design** ของ `carla_mpc_classical` โดยใช้ **QA/QC เป็นหลัก** — ทุก stage มีเกณฑ์ตรวจสอบ ชัดเจน วัดได้ และผ่าน gate ก่อนส่งต่อ

---

## 1. หลักการออกแบบ (Design Principles)

| หลักการ | ความหมายในโปรเจกต์ |
|--------|----------------------|
| **QA/QC เป็นหลัก** | ทุก stage มี input/output ชัดเจน มี validation gate และ metric ที่วัดได้ |
| **Single source of truth** | State (cte_m, heading_rad, curvature) มาจาก stage เดียว (fusion) ใช้ทั้ง reference path และ MPC |
| **Graceful degradation** | Perception ล้ม → ใช้ waypoint; MPC ล้ม → ใช้ fallback P-control; ไม่ crash |
| **Latency awareness** | Perception thread vs control thread แยก; 10 Hz control, ไม่ block บน inference |
| **Observability** | ทุกจุดสำคัญมี log / metric / mode (WP_PRIMARY, WP+UNET_CENTER, LANE+MPC) |

---

## 2. Workflow แบบ Stage + QA Gate

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PRE-FLIGHT QA (ก่อนรัน)                                                     │
│  ✓ Model file มีอยู่  ✓ CARLA port 2000  ✓ Imports (perception, control, safety) │
│  → ไม่ผ่าน = ไม่รัน  (run_adas_full.preflight)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: PERCEPTION (Camera → Lane Mask)                                    │
│  Input:  RGB (H×W×3)                                                        │
│  Output: binary mask (H×W), confidence ∈ [0,1]                               │
│  QC:     mask quality (pixel count, strip continuity); confidence ≥ threshold │
│  Gate:   invalid → geometry_valid=False; low conf → waypoint_only / WP_PRIMARY │
│  Code:   perception/lane_detector.py, lane_trajectory._lane_detection        │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: GEOMETRY (Mask → Lane State)                                       │
│  Input:  mask, BEV warp params                                               │
│  Output: (cte_m, heading_rad, curvature), left/right coeffs, center coeffs   │
│  QC:     MIN_POINTS_POLY, outlier rejection (RANSAC-style in _fit_poly_robust) │
│  Gate:   poly fit fail → mirror from other side / Kalman predict; mask_quality invalid → geometry_valid=False │
│  Code:   perception/lane_trajectory.py (BEV, sliding_window_search, polynomial_fit, Kalman) │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: FUSION (WP + Lane → Single State)                                  │
│  Input:  wp_state (cte_wp, head_wp, curv_wp), lane state, lane_conf         │
│  Output: (cte_m, head_s, curv_s), mode (WP_PRIMARY | WP+UNET_CENTER | LANE+MPC) │
│  QC:     FUSION_CONF_LOW/HIGH; clip CTE/heading/curvature อยู่ในขอบเขต      │
│  Gate:   conf < LOW → WP only; conf ≥ LOW → blend UNet สำหรับ CTE (จัดกลาง)  │
│  Code:   alg/fusion.apply_fusion, config (WP_CTE_WEIGHT, UNET_CENTER_WEIGHT) │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: STATE SMOOTHING (Optional QC for stability)                        │
│  Input:  fused (cte_m, head_s, curv_s)                                      │
│  Output: smoothed state (EMA) ก่อนส่งเข้า MPC                                │
│  QC:     STATE_EMA_ALPHA; ลด jitter ก่อน control                             │
│  Code:   alg/step.py (post-fusion state EMA)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: REFERENCE PATH                                                     │
│  Input:  (cte_m, head_s, curv_s), lookahead_m                               │
│  Output: reference_path [(s_m, lat_m), ...]                                  │
│  QC:     REF_PATH_LOOKAHEAD_M, LOOKAHEAD_MIN_M, LOOKAHEAD_TIME_S             │
│  Code:   alg/reference.get_reference_path, dynamic_lookahead_m               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: CONTROL (MPC)                                                      │
│  Input:  state, reference, v_ref, speed_ms                                   │
│  Output: steer_rad, accel (แล้วแปลงเป็น steer, throttle, brake)             │
│  QC:     solver status (Solve_Succeeded); NaN/Inf check; adaptive weight by confidence │
│  Gate:   solver fail → fallback (ถ้ามี); output clamp ใน SafetyOverride     │
│  Code:   control/lane_mpc.py, alg/step.py                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 7: SAFETY OVERRIDE (QC บังคับ)                                        │
│  Input:  steer, throttle, brake, velocity                                   │
│  Output: clamped steer/throttle/brake; emergency brake ถ้าจำเป็น            │
│  QC:     max_steering_angle, max_speed_kmh; NaN/Inf → replace with safe default │
│  Gate:   ไม่มีข้อยกเว้น — safety override ทำงานเสมอ                         │
│  Code:   safety/safety_override.py                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                              [CARLA Vehicle Actuation]
```

---

## 3. QA/QC Checklist ต่อ Stage

### 3.1 Pre-flight (ก่อนรัน)

| รายการ | เกณฑ์ | ที่ตรวจ |
|--------|--------|--------|
| Model file | มีอยู่และอ่านได้ | `run_adas_full.check_model` |
| CARLA server | พอร์ต 2000 ตอบ | `run_adas_full.check_carla` |
| Imports | perception, control, safety โหลดได้ | `run_adas_full.check_imports` |

### 3.2 Perception

| รายการ | เกณฑ์ | ที่ตรวจ |
|--------|--------|--------|
| Mask quality | total pixels, strip continuity | `lane_trajectory.check_mask_quality` |
| Confidence | ใช้เทียบ FUSION_CONF_LOW/HIGH | `lane_conf` จาก pipeline |
| Morphology | mask close → full line (ไม่ dashed) | `_lane_detection` (MORPH_CLOSE) |

### 3.3 Geometry

| รายการ | เกณฑ์ | ที่ตรวจ |
|--------|--------|--------|
| Poly fit | MIN_POINTS_POLY, outlier rejection | `_fit_poly_robust`, `polynomial_fit` |
| Fallback | left/right mirror ถ้าฝั่งใด fit ไม่ได้ | `_mirror_coeffs` ใน `polynomial_fit` |
| Kalman | predict เมื่อไม่มี measurement | `KalmanPoly.predict` |
| Curvature | rate limit + EMA | `CurvatureSmoother` |

### 3.4 Fusion

| รายการ | เกณฑ์ | ที่ตรวจ |
|--------|--------|--------|
| Mode | WP_PRIMARY / WP+UNET_CENTER / LANE+MPC | `alg/fusion.apply_fusion` |
| Clipping | CTE, heading, curvature อยู่ในขอบเขต | ใน `apply_fusion` |
| State EMA | ลด jitter ก่อน MPC | `STATE_EMA_ALPHA` ใน `alg/step` |

### 3.5 Control & Safety

| รายการ | เกณฑ์ | ที่ตรวจ |
|--------|--------|--------|
| Solver | IPOPT status (Solve_Succeeded) | `lane_mpc.solve` |
| Output | steer/throttle/brake ไม่มี NaN/Inf | `safety_override` |
| Clamp | steer ≤ max_steering_angle, speed ≤ max_speed_kmh | `safety_override` |

---

## 4. โครงสร้างโมดูล (Mapping กับ Code)

| Layer | โฟลเดอร์/ไฟล์หลัก | หน้าที่ (QA view) |
|-------|---------------------|-------------------|
| Entry | `adas/run.py`, `run_adas_full.py` | Pre-flight QA, เรียก main loop |
| Algorithm | `alg/step.py`, `alg/fusion.py`, `alg/reference.py` | Fusion, state EMA, reference path, เรียก perception + control |
| Perception | `perception/lane_trajectory.py`, `lane_detector.py`, `road_perception.py` | Mask, BEV, poly fit, Kalman, quality check |
| Control | `control/lane_mpc.py` | MPC solve, adaptive weight, solver status |
| Safety | `safety/safety_override.py` | Clamp, NaN check, emergency brake |
| Config | `config.py` | ค่าคงที่ทุก stage (threshold, weight, limit) |

รายละเอียดโฟลเดอร์เต็มดู [STRUCTURE.md](STRUCTURE.md)

---

## 5. Runtime QC และ Observability

- **Mode string:** ทุก frame มี `mode` (WP_PRIMARY, WP+UNET_CENTER, LANE+MPC) → รู้ว่า state มาจาก waypoint หรือ lane หรือผสม
- **Log สรุป:** `f=N spd=... st=... cte=... mode=... fps=...` ทุก 200 frames
- **SafetyOverride log:** เมื่อ steer เกิน limit จะมี WARNING
- **Dashboard:** Pygame/Tk fallback แสดง overlay, BEV, path, telemetry

การเพิ่ม QC ในอนาคต (แนะนำ): บันทึก `mode`, `cte_m`, `solver_status` ลง CSV ต่อ frame แล้ววิเคราะห์ offline (เช่น mean CTE, solver fail rate) ตาม [HOW_TO_COLLECT_AND_VIEW_RESULTS.md](HOW_TO_COLLECT_AND_VIEW_RESULTS.md)

---

## 6. สรุปหลักการ QA/QC

1. **Gate ก่อนส่งต่อ:** แต่ละ stage มีเกณฑ์ผ่าน (mask quality, poly fit, solver success) ถ้าไม่ผ่านใช้ fallback หรือ flag (geometry_valid, waypoint_only)
2. **Metric วัดได้:** confidence, cte_m, mode, solver_status, FPS
3. **Safety ไม่ bypass:** SafetyOverride ทำงานทุก frame; AI/MPC output ต้องผ่าน clamp เสมอ
4. **Config ที่เดียว:** `config.py` เป็น single source สำหรับ threshold และ limit ใช้ทั้ง pipeline และ QC

---

*เอกสารนี้จัดระเบียบ workflow และ system design ตามหลัก Senior AI & Autonomous Driving Engineering โดยใช้ QA/QC เป็นหลัก*
