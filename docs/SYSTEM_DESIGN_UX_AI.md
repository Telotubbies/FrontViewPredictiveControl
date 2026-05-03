# ออกแบบระบบ LKA — UX/UI + AI Engineering

## 1. Pipeline (AI Engineering) — ตั้งแต่จับเส้น

```
[Camera RGB]
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  PERCEPTION (จับเส้น)                                             │
│  • UNet → binary lane mask (ทุก frame, ตั้งแต่เริ่ม)              │
│  • Optional: geometric validation, trapezoid ROI                 │
└─────────────────────────────────────────────────────────────────┘
     │ mask (H×W)
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  GEOMETRY (จาก mask → state)                                     │
│  • BEV warp → sliding window → polynomial fit → center line      │
│  • Output: (cte_m, heading_rad, curvature) + confidence           │
│  • Kalman smoothing on coefficients (temporal)                   │
└─────────────────────────────────────────────────────────────────┘
     │ lane state
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  FUSION (optional)                                               │
│  • Waypoint (map) + lane state → single (cte, heading, curvature)│
│  • Weight: e.g. lane 70%, waypoint 30%                          │
└─────────────────────────────────────────────────────────────────┘
     │ fused state
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  REFERENCE PATH                                                  │
│  • state → path (s, lateral_m) in vehicle frame                  │
│  • lateral(s) = cte + heading*s + 0.5*curv*s²                    │
└─────────────────────────────────────────────────────────────────┘
     │ reference_path
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  CONTROL (MPC)                                                   │
│  • state + path + v_ref → steer, throttle, brake                 │
│  • Safety override (speed, steer, NaN)                           │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
[CARLA Vehicle]
```

**หลักการ AI Eng:**
- **Single source of truth:** state = (cte_m, heading_rad, curvature) มาจาก stage เดียว (geometry/fusion) แล้วใช้ทั้ง reference path และ MPC
- **จับเส้น = UNet เท่านั้น** สำหรับ overlay และ input ให้ geometry; classical edge เป็น optional fallback
- **Interface ชัด:** Perception → mask; Geometry → state; Path → list of (s, lat); Control → (steer, throttle, brake)

---

## 2. UX/UI Design — หลักการ

### 2.1 Information hierarchy (ลำดับการมอง)

1. **Primary:** รถอยู่เลนไหม / ทางที่ระบบจะไป (planned path) → ต้องเห็นใน 1–2 วินาที
2. **Secondary:** raw perception (UNet overlay), map view, speed/steer
3. **Tertiary:** mode, confidence, curvature, graphs สำหรับ debug

### 2.2 Layout แนะนำ (Left → Right = Perception → Planning → Control)

| Zone        | เนื้อหา                    | เป้าหมาย UX                          |
|------------|----------------------------|--------------------------------------|
| **ซ้าย 1** | กล้อง + UNet overlay       | “เห็นว่าระบบจับเลนตรงไหน” (เขียว)   |
| **กลาง**   | Bird’s Eye / 3D vector     | “ทางที่ไป + ทิศรถ” แบบแผนที่         |
| **ขวา**    | Telemetry + graphs         | “สถานะควบคุม + ความเชื่อมั่น”        |

### 2.3 Color & state (สอดคล้องทั้งจอ)

- **เขียว:** เลน / valid / อยู่ในเลน
- **ส้ม/เหลือง:** planned path / คำเตือน
- **แดง:** invalid / error / ออกจากเลน
- **เทา/ขาว:** เส้นขอบ, กริด, ข้อความปกติ

### 2.4 Label ให้คนและ AI เข้าใจตรงกัน

- ป้ายแต่ละ panel บอก **หน้าที่** ไม่ใช่แค่ชื่อเทคนิค (เช่น “Perception — UNet Lane” ไม่ใช่แค่ “Camera”)
- สถานะสำคัญมีคำเดียวชัดเจน: IN LANE / LANE INVALID / WAYPOINT ONLY / FUSION

---

## 3. Implementation checklist

- [ ] **Pipeline:** ทุก step รับ-ส่ง type ชัด (mask, state, reference_path)
- [ ] **Dashboard:** รับ state เดียว (e.g. FrameState) แทน parameter หลายตัว
- [ ] **Labels:** ป้าย panel = Perception | Planning | Control
- [ ] **Colors:** ใช้ constant (GREEN_VALID, ORANGE_PATH, RED_INVALID) ทั้ง run
- [ ] **Docs:** pipeline diagram + UX หลักการอยู่ใน repo (ไฟล์นี้)
