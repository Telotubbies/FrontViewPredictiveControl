# Apollo MPC + GUI Design + Line Detection — อ้างอิงและปรับใช้

อ้างอิงจาก [Apollo Auto (GitHub)](https://github.com/ApolloAuto/apollo) และโค้ดใน repo นี้ (carla_mpc_classical, carla_apollo_bridge) สำหรับ **MPC**, **design หน้า GUI ตอนรัน**, และ **line/lane detection**.

---

## 1. MPC — จาก Apollo มาใช้กับโปรเจกต์

### 1.1 โครงสร้าง MPC ของ Apollo

ใน Apollo ([modules/control](https://github.com/ApolloAuto/apollo)):
- **Controller หลัก:** `MPC_CONTROLLER` (หรือ Lat + Lon แยก)
- **State space:** lateral (lateral error, heading error, …) + longitudinal
- **Solver:** QP (OSQP) ใช้ `matrix_q`, `matrix_r`, gain scheduler ตามความเร็ว
- **Config:** `control_conf.pb.txt` → `mpc_controller_conf` (ts, cf/cr, mass, matrix_q/r, feedforward, lookahead)

ใน **carla_apollo_bridge** มี **refined controller** (แก้บั๊กของ Apollo 8.0):
- `carla_bridge/refined_controller/mpc_controller.cc` — C++ ใช้กับ Apollo ใน Docker
- `carla_bridge/refined_controller/control_conf.pb.txt` — ค่า matrix_q, matrix_r, lat_err_gain_scheduler, heading_err_gain_scheduler, feedforward, lookahead

แนวคิดที่นำมาใช้ได้:
- **Gain scheduler ตามความเร็ว:** ลด weight lateral/heading เมื่อความเร็วสูง (ใน control_conf มี `lat_err_gain_scheduler`, `heading_err_gain_scheduler`)
- **Feedforward จาก curvature:** `enable_mpc_feedforward_compensation` + feedforward term จาก curvature
- **Lookahead สำหรับ planning point:** `query_relative_time`, lookahead_station — เลือกจุดบน trajectory ที่ใช้คำนวณ error

### 1.2 โปรเจกต์เรา: `control/lane_mpc.py` (LaneMPC)

| หัวข้อ | Apollo (refined C++) | carla_mpc_classical (Python) |
|--------|----------------------|------------------------------|
| โมเดล | Lateral dynamics (cf, cr, mass, lf, lr) + longitudinal | Kinematic bicycle (x, y, ψ, v) |
| Solver | QP / OSQP | CasADi + IPOPT (NLP) |
| State | lateral_error, heading_error, … | CTE, heading_err, v ผ่าน parameter P |
| Cost | matrix_q, matrix_r | w_cte, w_heading, w_vel, w_steer, w_steer_rate, w_accel |
| ปรับใช้ได้ | Gain scheduler ตาม speed, feedforward curvature | มีแล้ว: scale_cte/scale_heading ตาม confidence; curvature ส่งเข้า P |

**สิ่งที่นำจาก Apollo มาใส่ใน LaneMPC ได้ (ถ้าต้องการ):**
1. **Speed-based weight scaling**  
   - อ่านจาก `control_conf`: `lat_err_gain_scheduler`, `heading_err_gain_scheduler` (speed → ratio).  
   - ใน Python: สร้าง `get_lat_gain(speed_kmh)`, `get_heading_gain(speed_kmh)` แล้วคูณกับ `w_cte`, `w_heading` ก่อน solve.
2. **Feedforward steering จาก curvature**  
   - Apollo: `feedforwardterm_gain_scheduler` + curvature → steer_feedforward.  
   - ใน `lane_mpc.py`: เพิ่ม `u_ff = f(curvature, v_ref)` เป็น initial guess หรือบวกเข้า cost reference (แล้วแต่ formulation).
3. **Lookahead reference point**  
   - Apollo: `query_relative_time`, lookahead_station.  
   - เรา: มี `REF_PATH_LOOKAHEAD_M`, `LOOKAHEAD_TIME_S` ใน config — สามารถเทียบกับ Apollo ว่าใช้เวลา/ระยะเทียบกับ trajectory อย่างไร แล้วเลือกจุดอ้างอิงหนึ่งจุดสำหรับ CTE/heading.

ไฟล์ที่เกี่ยวข้องใน repo:
- `carla_mpc_classical/control/lane_mpc.py` — ตำแหน่งหลักของ MPC
- `carla_mpc_classical/config.py` — REF_PATH_LOOKAHEAD_M, weights
- `carla_apollo_bridge-master/carla_bridge/refined_controller/control_conf.pb.txt` — อ้างอิงค่า Q/R และ gain scheduler
- `carla_apollo_bridge-master/carla_bridge/refined_controller/mpc_controller.cc` — logic feedforward, matrix, filters

---

## 2. Design หน้า GUI ตอนรัน

### 2.1 ตัวเลือก GUI ในโปรเจกต์

| ตัวเลือก | ไฟล์ | เทคโนโลยี | ใช้เมื่อ |
|----------|------|-----------|----------|
| **PyQt ADAS Dashboard** | `gui/adas_dashboard.py`, `gui/main_window.py` | PyQt5/PySide6 | รัน pipeline จริง + แสดง camera, charts, control panel |
| **Apollo-style (matplotlib)** | `gui/apollo_dashboard.py` | Matplotlib | HUD แบบ Apollo DreamView (pipeline status, camera, BEV, MPC state, control, log) |
| **In-run (Pygame)** | `run_unet_mpc.py` class `Dashboard` | Pygame | แสดงในลูปรันจริง (FPS, สัญลักษณ์ถนน, overlay) |

### 2.2 Layout แบบ Apollo (อ้างอิง DreamView)

จาก `gui/apollo_dashboard.py` — layout ที่มีอยู่แล้ว:

```
┌──────────────────────────┬───────────────────────┬──────────────────────┐
│  [A] PIPELINE STATUS     │  [B] CAMERA / LANE    │  [C] BEV MAP          │
│  (workflow steps)        │  (UNet overlay)       │  (bird-eye view)     │
├──────────────────────────┼───────────────────────┼──────────────────────┤
│  [D] MPC STATE           │  [E] CONTROL SIGNALS  │  [F] SYSTEM LOG      │
│  (speed, CTE, heading)   │  (steer, thr, brake)   │  (live text)         │
└──────────────────────────┴───────────────────────┴──────────────────────┘
```

- **A:** SENSOR → PREPROCESS → UNET → IPM → CENTERLINE FIT → MPC SOLVE → CONTROL APPLY  
- **B:** ภาพจาก camera + lane overlay (และ mask px ถ้ามี)  
- **C:** BEV / reference path  
- **D:** MPC state (CTE, heading, curvature, converged/FAIL)  
- **E:** Steer (°), throttle (%), brake (%)  
- **F:** Log ข้อความแบบ real-time  

การเชื่อมกับ pipeline จริง: สร้าง `SimState` (หรือเทียบเท่า) จาก `run_unet_mpc` แต่ละ frame — ส่ง rgb, cte_m, heading_rad, curvature, steer, throttle, brake, lane_conf, reference_path, mode → อัปเดต `apollo_dashboard.render_frame(state)` ใน thread แยก (เช่น 10 Hz) เพื่อไม่ block control loop.

### 2.3 สรุปการออกแบบ GUI

- **ต้องการ UI เต็ม (ปุ่ม Start/Stop, Town, sliders):** ใช้ **PyQt** `gui/adas_dashboard.py` — โครงสร้างอยู่ที่ `gui/main_window.py`, `gui/panels/`, `gui/theme.py`.
- **ต้องการ HUD แบบ Apollo (status + camera + MPC + log):** ใช้ **Apollo-style** `gui/apollo_dashboard.py` — ปรับให้รับ state จาก `run_unet_mpc` แทน mock data (ดู comment ใน apollo_dashboard.py เรื่อง INTEGRATION).
- **ต้องการเบา ๆ ตอนรันในลูป:** ใช้ **Pygame** ใน `run_unet_mpc.py` (Dashboard ปัจจุบัน) หรือสลับให้เปิดหน้าต่าง Apollo matplotlib แยก process/thread ก็ได้.

---

## 3. Line / Lane Detection

### 3.1 โฟลว์ในโปรเจกต์ (Perception → MPC)

```
Camera RGB
    → [P1] Lane mask (UNet หรือ classical edge)
    → [P2] BEV warp + morphology + mask_quality
    → [P3] Left/right boundaries + centerline (sliding window / ego tracker)
    → [P4] Kalman smooth center coeffs, geometry_valid
    → [P5] Trajectory + (cte_m, heading_rad, curvature, v_ref)
    → Fusion (waypoint + lane)
    → MPC (LaneMPC.solve)
```

ไฟล์หลัก:
- **Mask / UNet:** `perception/lane_detector.py`, pipeline ผ่าน `pipeline.py` / `perception/lane_trajectory.py`
- **BEV + boundaries + center:** `perception/lane_trajectory.py` — `_phase2_bev_and_quality`, `_phase3_boundaries_and_center`, `_phase4_smooth_center`, `_phase5_trajectory_and_state`
- **Classical (ไม่ใช้ UNet):** `perception/classical_lane.py` — edge → perspective → sliding window → poly → cte/heading/curvature
- **Fusion:** `alg/fusion.py` — WP primary, UNet ช่วย center + lane check
- **ส่งเข้า MPC:** `alg/step.py` → state (cte_m, head_s, curv_s) → `control/lane_mpc.py`

### 3.2 การ “จับเส้น” ให้เนียน (เต็มเส้นจากเส้นปะ)

- **Morphology:** ใน `lane_trajectory.py` ใช้ `cv2.morphologyEx(..., MORPH_CLOSE)` ทั้งที่ mask และที่ BEV เพื่อเติมช่องว่างระหว่างเส้นปะ
- **Polynomial fit:** ใช้จุดที่หนาแน่นขึ้น (`_densify_xy`) และ fallback `_mirror_coeffs` เมื่อขอบหนึ่งข้าง fit ไม่ได้
- **Temporal smoothing:** Kalman + EMA ใน `lane_trajectory` และ `alg/step.py` (STATE_EMA_ALPHA, LANE_EMA_ALPHA) ลด jitter ก่อนส่งเข้า MPC

ถ้าต้องการเทียบกับ “line detection” แบบ Apollo: Apollo ใช้ HD map + localization เป็นหลัก; เราใช้ camera-based lane (UNet/classical) + waypoint fusion — โครงสร้างที่อธิบายข้างต้นคือส่วน “detection line” ของเราและเป็น input หลักของ MPC.

---

## 4. สรุปการนำไปใช้

| เป้าหมาย | แนะนำ |
|----------|--------|
| **เอา logic MPC จาก Apollo มาใช้** | ดู refined `mpc_controller.cc` และ `control_conf.pb.txt` → นำ gain scheduler ตาม speed, feedforward curvature, lookahead มาใส่ใน `lane_mpc.py` (หรือ config.py) |
| **Design หน้า GUI ตอนรัน** | ใช้ layout Apollo ใน `apollo_dashboard.py`; ถ้าต้องปุ่ม/control เต็มใช้ `adas_dashboard.py` (PyQt); ถ้าเบาใช้ Pygame ใน `run_unet_mpc.py` |
| **Line / lane detection** | ใช้โฟลว์ P1–P5 ใน `perception/lane_trajectory.py` + fusion ใน `alg/fusion.py`; ปรับ morphology และ poly fit ใน `lane_trajectory.py` ถ้าต้องการเส้นเต็มจากเส้นปะ |

ลิงก์อ้างอิง:
- [Apollo GitHub](https://github.com/ApolloAuto/apollo) — โครงสร้าง control, docs
- [Apollo Control (อ่านเพิ่ม)](https://github.com/ApolloAuto/apollo/blob/master/modules/control/control_component/README_cn.md) (ถ้ามีใน branch ที่ใช้)
- Refined controller ใน repo นี้: `carla_apollo_bridge-master/carla_bridge/refined_controller/`
