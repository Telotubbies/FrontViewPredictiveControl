# Back to Basic — การจับเลนและตีเส้นแบ่ง Phase (ทำทีละขั้น)

เอกสารนี้อธิบาย **การจับเลน (lane detection)** และ **การตีเส้นแบ่ง phase** แบบกลับไปพื้นฐาน ทำทีละ phase ชัดเจน วัดได้ และมี gate ก่อนไปขั้นถัดไป

---

## 1. สรุป Phase ทั้งหมด (ภาพรวม)

| Phase | ชื่อ | Input | Output | เป้าหมาย |
|-------|------|--------|--------|----------|
| **P0** | Pre-flight | — | go / no-go | ระบบพร้อมรัน |
| **P1** | Lane mask (pixel) | RGB | binary mask, confidence | ได้พิกเซลว่า “ตรงไหนคือเลน” |
| **P2** | Left/Right boundary | mask (BEV) | เส้นซ้าย, เส้นขวา (poly หรือ point set) | ตีเส้นแบ่งขอบเลนซ้าย–ขวา |
| **P3** | Centerline | left + right | centerline (poly), width | ได้เส้นกลางเลน + ความกว้าง |
| **P4** | Lane state (สำหรับ control) | centerline + ego pose | cte_m, heading_rad, curvature | ตี “phase” ของเลนในมุมควบคุม |
| **P5** | Lane phase / mode | state + confidence + WP | mode (WP_PRIMARY, WP+UNET_CENTER, …) | ตีว่า “ตอนนี้ใช้เลนจากอะไร” |

---

## 2. ทำทีละ Phase (ค่อยๆ ทำใหม่)

### Phase 1 — การจับเลนระดับพิกเซล (Lane Mask)

**วัตถุประสงค์:** จากภาพ RGB ได้ binary mask ว่า pixel ไหนเป็น “เลน/ถนน” และความเชื่อมั่น (confidence)

- **Input:** RGB (H×W×3)
- **Output:**
  - `mask` (H×W) binary: 1 = lane/road, 0 = อื่น
  - `confidence` ∈ [0, 1]
- **ขั้นตอนย่อย (ทำตามลำดับ):**
  1. โหลดภาพ → resize ถ้าต้องการ
  2. ตัดพื้นที่ที่ไม่ใช่ถนน (เช่น sky) ออก
  3. UNet (หรือ classical edge) → mask
  4. Morphological close (เติมช่องว่างเส้นปะ) → mask สุดท้าย
  5. คำนวณ confidence จากจำนวน pixel ใน ROI
- **QC:** จำนวน pixel ขั้นต่ำ, strip continuity (ดู `check_mask_quality`)
- **Code ปัจจุบัน:** `perception/lane_trajectory.py` → `_lane_detection`, `perception/lane_detector.py`

**ตรวจว่า P1 ผ่าน:** mask มีพื้นที่เลนต่อเนื่องพอ, confidence สูงเมื่อเห็นเลนชัด  

**รัน P1 อย่างเดียว (ลุย P1):**
```bash
cd carla_mpc_classical
python scripts/run_phase1_only.py --image path/to/frame.png
python scripts/run_phase1_only.py --image frame.png --out-dir out_p1   # บันทึก mask + overlay
```

---

### Phase 2 — ตีเส้นแบ่งขอบเลน (Left / Right Boundary)

**วัตถุประสงค์:** จาก mask ในมุม bird’s eye ได้ “เส้นซ้าย” และ “เส้นขวา” ของเลน (เป็น polynomial หรือชุดจุด)

- **Input:** mask ใน perspective → warp เป็น BEV → `bev_bin`
- **Output:**
  - `left_boundary`: poly coeffs หรือจุด (x_vehicle, y_vehicle)
  - `right_boundary`: เหมือนกัน
- **ขั้นตอนย่อย:**
  1. Warp mask → BEV (trapezoid → rectangle)
  2. (Optional) Morphology บน BEV เพื่อเติมช่องว่าง
  3. Histogram / sliding window แยกซ้าย–ขวา (หรือ ego-lane tracker)
  4. Fit polynomial ต่อจุดซ้าย และจุดขวา → left_coeffs, right_coeffs
  5. ถ้าฝั่งใด fit ไม่ได้ ใช้ mirror จากอีกฝั่ง (full line)
- **QC:** MIN_POINTS_POLY, outlier rejection, mirror fallback
- **Code ปัจจุบัน:** `perception/lane_trajectory.py` → `sliding_window_search`, `polynomial_fit`, `ego_lane_tracker`

**ตรวจว่า P2 ผ่าน:** ได้ left_coeffs และ right_coeffs ที่วาดแล้วตรงกับขอบเลนใน BEV

---

### Phase 3 — เส้นกลางเลน (Centerline) และความกว้าง

**วัตถุประสงค์:** จากขอบซ้าย–ขวา ได้ “เส้นกลางเลน” และ (ถ้าต้องการ) ความกว้างเลน

- **Input:** left_coeffs, right_coeffs
- **Output:**
  - `center_coeffs`: พหุนามเส้นกลาง (มักใช้ (left + right) / 2)
  - `lane_width_m`: ความกว้างเลน (optional)
- **ขั้นตอนย่อย:**
  1. center_coeffs = (left_coeffs + right_coeffs) / 2
  2. (Optional) คำนวณ width จาก left/right ที่ x=0 หรือเฉลี่ยตามระยะ
- **QC:** center อยู่ระหว่าง left กับ right; width อยู่ในช่วงสมเหตุสมผล
- **Code ปัจจุบัน:** `polynomial_fit` คืน center; `lane_trajectory` ใช้ center สำหรับ trajectory

**ตรวจว่า P3 ผ่าน:** centerline อยู่กลางระหว่างเส้นซ้าย–ขวา ใน BEV / planning view

---

### Phase 4 — Lane State (ตี “phase” สำหรับ control)

**วัตถุประสงค์:** จาก centerline + ตำแหน่ง/heading ของ ego ได้ตัวเลขที่ control ใช้: CTE, heading error, curvature

- **Input:** center_coeffs, ego pose (หรือใช้ waypoint แทนเมื่อไม่มีเลน)
- **Output:**
  - `cte_m`: cross-track error (เมตร)
  - `heading_rad`: มุม heading error (rad)
  - `curvature`: ความโค้งถนน
- **ขั้นตอนย่อย:**
  1. สร้าง reference path จาก center (หรือจาก waypoint)
  2. หา CTE, heading, curvature ที่จุด ego (หรือจุดใกล้ที่สุดบน path)
  3. (Optional) Kalman / EMA เพื่อลด jitter
- **QC:** clip ค่าให้อยู่ในขอบเขต (CTE, heading, curvature)
- **Code ปัจจุบัน:** `lane_trajectory` → `mpc_reference_output`, `algorithms/reference.py`, `algorithms/step.py` (state EMA)

**ตรวจว่า P4 ผ่าน:** ค่า cte_m, heading_rad, curvature ตรงกับที่เห็นในภาพ (รถอยู่ซ้ายของ center = cte < 0 ฯลฯ)

---

### Phase 5 — Lane Phase / Mode (ใช้เลนจากอะไร)

**วัตถุประสงค์:** ตี “phase” ว่า **ตอนนี้ระบบถือว่าเลนมาจากอะไร** — waypoint ล้วน, เลนจาก UNet ล้วน, หรือผสม

- **Input:** wp_state (มี/ไม่มี), lane state, confidence
- **Output:**
  - `mode`: WP_PRIMARY | WP+UNET_CENTER | LANE+MPC
  - state ที่ผสมแล้ว (ถ้า fusion): cte_m, head_s, curv_s
- **ขั้นตอนย่อย:**
  1. ถ้าไม่มี waypoint → ใช้เลนจาก perception ล้วน (LANE+MPC)
  2. ถ้ามี waypoint + confidence ต่ำ → WP_PRIMARY (ใช้ WP ล้วน)
  3. ถ้ามี waypoint + confidence สูง → WP+UNET_CENTER (ใช้ WP เป็นหลัก + UNet ช่วยจัดกลาง)
  4. Clip และส่ง state สุดท้ายเข้า reference path + MPC
- **QC:** FUSION_CONF_LOW/HIGH; ไม่มี state ที่เป็น NaN/Inf
- **Code ปัจจุบัน:** `algorithms/fusion.apply_fusion`, `config.py` (FUSION_CONF_*, WP_CTE_WEIGHT, UNET_CENTER_WEIGHT)

**ตรวจว่า P5 ผ่าน:** mode เปลี่ยนตามสถานะ (เห็นเลนชัด → WP+UNET_CENTER; ไม่เห็น → WP_PRIMARY)

---

## 3. ลำดับการทำใหม่ (ค่อยๆ ทำ)

แนะนำทำตามลำดับนี้ แล้วทดสอบแต่ละ phase ก่อนไปขั้นถัดไป:

1. **P1 อยู่ที่เดิมหรือปรับเล็กน้อย**  
   - ตรวจว่า mask + confidence ออกมาถูก (ดูจาก Raw mask ใน UI)  
   - ถ้าจะ “ทำใหม่” อาจเพิ่ม script แยกที่รับ RGB → โหลด UNet → ส่งออกเฉพาะ mask + confidence แล้ว plot/บันทึก

2. **P2 แยกให้ชัด**  
   - รับ bev_bin → ส่งออกเฉพาะ left_coeffs, right_coeffs (และ optional left_px, right_px)  
   - วาดเฉพาะเส้นซ้าย–ขวาบน BEV ไม่ต้องต่อถึง P3/P4  
   - ตรวจว่าเส้นตรงกับขอบเลนจริง

3. **P3 แยกจาก P2**  
   - รับ left_coeffs, right_coeffs → ส่งออก center_coeffs (และ width ถ้ามี)  
   - วาด centerline บน BEV/planning view ตรวจว่าอยู่กลาง

4. **P4 รับ center + ego**  
   - รับ center_coeffs (หรือ waypoint) + ego → ส่งออก cte_m, heading_rad, curvature  
   - เทียบกับค่าจาก waypoint หรือจากการวัดบนภาพ (sanity check)

5. **P5 รับ state + confidence + WP**  
   - รับ wp_state, lane state, confidence → ส่งออก mode และ fused state  
   - ตรวจว่า mode ตรงกับที่คาด (ลองบังกล้อง = WP_PRIMARY, เปิดกล้องเห็นเลน = WP+UNET_CENTER)

จากนั้นค่อยรวมกลับเข้า pipeline เดิม (เช่น `LaneTrajectoryPipeline.process` และ `algorithms/step`) โดยให้แต่ละบล็อกเรียกผลจาก phase ก่อนหน้าเท่านั้น

---

## 4. สรุปนิยาม “Phase” ในเอกสารนี้

- **Phase 1 (Lane mask):** ระดับพิกเซล — ตรงไหนคือเลน
- **Phase 2 (Boundary):** ตีเส้นแบ่ง — เส้นซ้าย เลน, เส้นขวาเลน
- **Phase 3 (Centerline):** เส้นกลางเลน + ความกว้าง
- **Phase 4 (Lane state):** ตี “phase” เป็นตัวเลขสำหรับ control — CTE, heading, curvature
- **Phase 5 (Lane phase/mode):** ตี “phase” ว่าใช้แหล่งข้อมูลไหน — WP only / UNet only / Fusion

ทำทีละ phase แล้วทดสอบให้ผ่านก่อน จะได้ pipeline การจับเลนและตีเส้นแบ่ง phase ที่ชัดและดูแลง่าย

---

## 5. โค้ดหลัง refactor (การจับเส้นใหม่หมด)

ใน `perception/lane_trajectory.py` — **LaneTrajectoryPipeline** ถูก refactor เป็น phase ชัดเจน:

| Phase | Method | หน้าที่ |
|-------|--------|--------|
| P1 | `_phase1_lane_mask(rgb)` | RGB → mask, confidence |
| P2 | `_phase2_bev_and_quality(mask)` | mask → BEV warp, fill gaps, mask_quality |
| P3 | `_phase3_boundaries_and_center(bev_bin)` | BEV → left/right/center coeffs (ego tracker หรือ sliding window) |
| P4 | `_phase4_smooth_center(center_raw, mask_quality)` | Kalman → center_coeffs, geometry_valid |
| P5 | `_phase5_trajectory_and_state(center_coeffs)` | center → x_ref, y_ref, cte, heading, curvature, v_ref |
| P6 | `_phase6_visualization(...)` | BEV window vis + lane overlay |

`process(rgb)` เรียก P1 → P2 → … → P6 ตามลำดับ แล้วคืน `TrajectoryOutput` — ไม่เปลี่ยน behavior เฉพาะจัดระเบียบใหม่

---

## 6. Phase case labels และเงื่อนไขผ่าน/ไม่ผ่าน (P2 / P3)

Pipeline ส่งค่า **phase_p2_case** และ **phase_p3_case** ออกมาใน `TrajectoryOutput` และ `FrameState` เพื่อให้เทสและ dashboard รายงานได้ว่าแต่ละ frame อยู่ใน case ไหน และ phase ผ่านหรือไม่

### P2 case (ได้ left/right อย่างไร)

| Case | ความหมาย | ผ่าน P2 เมื่อ |
|------|----------|----------------|
| `both` | จับได้ทั้งซ้ายและขวาจาก sliding window | ✅ ผ่าน (geometry_valid ขึ้นกับ P4) |
| `mirror_left` | จับได้แค่ซ้าย → สร้างขวาจาก mirror | ✅ ผ่าน |
| `mirror_right` | จับได้แค่ขวา → สร้างซ้ายจาก mirror | ✅ ผ่าน |
| `completion` | ใช้ ego tracker / completion จากเฟรมก่อน | ✅ ผ่าน |
| `none` | ไม่ได้ทั้งซ้ายและขวา | ❌ ไม่ผ่าน |

**เงื่อนไขผ่าน P2 (ใน step):** `phase_p2_ok = (phase_p2_case in ("both", "mirror_left", "mirror_right", "completion"))`

### P3 case (ได้ center อย่างไร)

| Case | ความหมาย | ผ่าน P3 เมื่อ |
|------|----------|----------------|
| `from_both` | center = (left + right) / 2 จาก detection | ✅ ผ่าน |
| `completion` | center จาก ego tracker (ใช้เฟรมก่อน) | ✅ ผ่าน |
| `none` | ไม่มี center (fit ไม่ได้หรือไม่มี left/right) | ❌ ไม่ผ่าน |

**เงื่อนไขผ่าน P3 (ใน step):** `phase_p3_ok = (phase_p3_case in ("from_both", "completion"))`

### P1 / P4 / P5 (สรุป)

- **P1:** ผ่านเมื่อ `lane_conf > 0.01` (มี mask พอ + confidence สูงพอ)
- **P4:** ผ่านเมื่อ cte_m, heading_rad, curvature เป็น finite และอยู่ในช่วงที่กำหนด
- **P5:** ผ่านเมื่อ `mode` ไม่ว่าง (fusion ตั้ง mode ได้)

สคริปต์ `scripts/run_test_phase1_phase5.py` รองรับการเก็บ `phase_p2_case` / `phase_p3_case` ต่อ frame และรายงานสรุปแยกตาม case (จำนวนครั้งที่เกิดแต่ละ case และอัตราผ่านใน case นั้น)
