# Lane-Level Features — ระดับเดียวกับระบบที่ไม่มีข้ามเลน

## ✅ สถานะ: ทำครบทั้ง 5 อย่าง + แก้ perception

| # | รายการ | ไฟล์/ตำแหน่ง | สถานะ |
|---|--------|----------------|--------|
| 1 | Lane candidate clustering | `perception/lane_clustering.py` | ✅ |
| 2 | Ego-lane selection + temporal tracking | `perception/ego_lane_tracker.py` | ✅ |
| 3 | Lane completion (poly + temporal) | Kalman predict + ego prev in `lane_trajectory.py` | ✅ |
| 4 | Adaptive weight MPC | `control/lane_mpc.py` (P[10–13], confidence) | ✅ |
| 5 | Steering rate cost + jerk penalty | `lane_mpc.py` (w_steer_rate, w_steer_jerk) | ✅ |
| — | Curvature smoothing (ลด spike) | `CurvatureSmoother` in `lane_trajectory.py` | ✅ |
| — | ส่ง confidence เข้า MPC | `pipeline.py` → `mpc.solve(..., confidence=...)` | ✅ |
| — | เปิด ego tracker ใน trajectory | `LaneTrajectoryPipeline(use_ego_lane_tracker=True)` default | ✅ |

---

## ปัญหา: ทำไมข้ามเลนบ่อย

จากประสบการณ์ **ไม่ใช่เพราะ MPC แย่** แต่เพราะ **perception layer ยังไม่ complete**:

- ไม่มี **lane ID tracking** → เลือก left/right ผิดคู่
- ไม่มี **temporal consistency** → frame ต่อ frame กระตุก
- ไม่มี **ego-relative filtering** → ใช้เลนอื่นแทนเลน ego
- **curvature spike จาก mask noise** → MPC ได้ curvature กระโดด

---

## 5 อย่างที่ต้องมี (และที่ทำแล้ว)

### 1️⃣ Lane candidate clustering

- **ไฟล์:** `perception/lane_clustering.py`
- **ทำอะไร:** จาก BEV binary mask หา **หลาย lane candidates** (histogram peaks หลายจุด → trace แต่ละ peak → fit poly ต่อ cluster) แทนที่จะถือแค่ left/right จาก histogram เดียว
- **Ego-relative:** กรองเฉพาะ candidates ที่ `|lateral_at_ego_m| <= EGO_LATERAL_FILTER_M`

### 2️⃣ Ego-lane selection + temporal tracking

- **ไฟล์:** `perception/ego_lane_tracker.py`
- **ทำอะไร:** จาก list of candidates เลือกคู่ **left+right ที่เป็น ego lane** (center ที่ x=0 ใกล้ 0, width อยู่ในช่วงสมเหตุสมผล) แล้ว **match กับ frame ก่อน** (lateral distance) เพื่อ temporal consistency
- **Lane completion:** ถ้าวันนี้ไม่มี detection ใช้ **previous frame** (used_completion=True)

### 3️⃣ Lane completion (poly + temporal)

- **ใน:** `perception/lane_trajectory.py`
- **Kalman:** เมื่อไม่มี center_coeffs ใช้ `Kalman.predict()` (เดิมอยู่แล้ว)
- **Ego tracker:** เมื่อไม่มี candidates ใช้ `EgoLaneResult` จาก frame ก่อน (left/right/center_coeffs)

### 4️⃣ Adaptive weight MPC

- **ไฟล์:** `control/lane_mpc.py`
- **ทำอะไร:** ใช้ `confidence` และ `curvature` ปรับ scale ของ cost:
  - โค้งมาก → เพิ่ม `scale_heading`
  - confidence ต่ำ → เพิ่ม `scale_steer_rate` และ `scale_jerk` (ให้ steer นุ่มลง)
- **เปิด/ปิด:** `MPCConfig.use_adaptive_weights = True`

### 5️⃣ Steering rate constraint + jerk penalty

- **ไฟล์:** `control/lane_mpc.py`
- **Constraint:** `max_steer_rate` ใช้เป็น bound อยู่แล้ว (lb_u/ub_u)
- **Cost:** `w_steer_rate` (dδ/dt) + **w_steer_jerk** (d²δ/dt²) เพื่อลด oscillation

---

## Curvature smoothing (ลด spike จาก mask)

- **ใน:** `perception/lane_trajectory.py` — `CurvatureSmoother`
- **EMA + rate limit:** ก่อนส่ง curvature ให้ MPC ผ่าน `CurvatureSmoother.update()` เพื่อไม่ให้ mask noise กระโดด

---

## การเชื่อมใน pipeline

- **Pipeline:** `pipeline.py` ส่ง `confidence=lane_conf` เข้า `mpc.solve(..., confidence=...)`
- **Trajectory:** `LaneTrajectoryPipeline(use_ego_lane_tracker=True)` (default) ใช้ cluster → ego tracker ก่อน sliding window fallback
- **Reset:** เมื่อเปลี่ยน map หรือเริ่มใหม่ เรียก `pipeline.reset()` จะ reset Kalman, ego_tracker, curvature_smoother

---

## สรุปไฟล์ที่เพิ่ม/แก้

| ไฟล์ | การเปลี่ยนแปลง |
|------|------------------|
| `perception/lane_clustering.py` | **ใหม่** — cluster lane candidates + ego lateral filter |
| `perception/ego_lane_tracker.py` | **ใหม่** — ego-lane selection + temporal tracking + completion |
| `perception/lane_trajectory.py` | CurvatureSmoother, optional ego_tracker path, curvature smoothing |
| `control/lane_mpc.py` | P 14 params, adaptive weights, w_steer_jerk |
| `pipeline.py` | ส่ง confidence เข้า mpc.solve() |
