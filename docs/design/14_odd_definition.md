# ODD (Operational Design Domain) Definition — ข้อกำหนดขอบเขตการทำงาน

> เอกสารหมายเลข **14** ในชุด System Design ของ `carla_mpc_classical` (CARLA + ADAS + MPC + Perception pipeline)
> อ้างอิง: [WORKFLOW_SYSTEM_DESIGN_QAQC.md](../WORKFLOW_SYSTEM_DESIGN_QAQC.md), [SYSTEM_DESIGN_UX_AI.md](../SYSTEM_DESIGN_UX_AI.md), [STRUCTURE.md](../STRUCTURE.md)
> หลักการ: **No reinforcement learning** — ทุก behavior กำหนดด้วย rule/state-machine + classical perception + MPC

---

## 0. ภาพรวม (Overview)

ODD คือ **เซตของเงื่อนไขที่ระบบ autonomous ถูกออกแบบมาให้ทำงานได้อย่างปลอดภัย** หากสภาพแวดล้อมออกนอกเซตนี้ (OOD = Out-of-Domain) ระบบต้อง **ตรวจจับได้ ลดระดับการทำงาน (graceful degradation) และส่งควบคุมคืน driver อย่างปลอดภัย**

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ODD LIFECYCLE                                  │
│                                                                      │
│  [IN_ODD] ──monitor──▶ [IN_ODD]  (ปกติ: MPC + perception ทำงาน)      │
│      │                                                               │
│      │ ตรวจพบ condition เปลี่ยน                                       │
│      ▼                                                               │
│  [DEGRADING] ──▶ ลดความเร็ว/เพิ่ม following distance                 │
│      │                                                               │
│      │ รุนแรงขึ้น / คงอยู่นาน                                          │
│      ▼                                                               │
│  [HANDOFF_REQUEST] ──10s buffer──▶ [DRIVER_TAKEOVER]                 │
│      │ driver ไม่ตอบ                                                  │
│      ▼                                                               │
│  [SAFE_STOP] ──▶ จอดข้างทาง / emergency brake                        │
└──────────────────────────────────────────────────────────────────────┘
```

หลักการสำคัญ (สอดคล้องกับ QA/QC หลักของโปรเจกต์):
- **ODD คือ gate สูงสุด** — perception/MPC ทำงานได้ดีเท่าไหร่ก็ตาม ถ้าออกนอก ODD ต้องไม่อ้างว่า "ปลอดภัย"
- **Graceful degradation ก่อน safe stop** — ลด speed, เพิ่ม gap, ลด confidence weight ของ perception ก่อนตัดสินใจหยุด
- **Observability** — ทุก ODD dimension มี confidence + log (เหมือน mode log ของ fusion: WP_PRIMARY / LANE+MPC)
- **No RL** — behavior map ทั้งหมด rule-based อ่าน/ตรวจสอบได้ (deterministic)

---

## 1. ODD Definition (นิยามขอบเขตการทำงาน)

> **ODD ของระบบนี้** = เซตของ (road type × speed × weather × lighting × traffic × infrastructure × geography) ที่ perception pipeline (UFLDv2/UNet + Kalman) และ MPC controller ถูก validate แล้วว่าทำงานได้โดยคง CTE และ heading error อยู่ในเกณฑ์ C1–C5

ระบบถูกออกแบบให้ทำงานปลอดภัยเมื่อ **ทุก dimension อยู่ในช่วงที่กำหนดพร้อมกัน** (AND logic) หาก dimension ใดออกนอกช่วง ระบบจะเข้าสู่ degradation/handoff flow ในส่วนที่ 4

**Scope ของโปรเจกต์นี้ (prototype):**
- เป้าหมายหลัก: **Lane Keeping Assist (LKA) + Adaptive Cruise Control (ACC) + Traffic Jam Assist (TJA)**
- ไม่รวม: full door-to-door autonomous, unprotected left turn ในทางแยกซับซ้อน, off-road
- ระดับ automation: **SAE Level 2–3** (driver ต้องพร้อมรับควบคุมภายใน 10 วินาที)

---

## 2. ODD Dimensions (มิติของขอบเขต)

### 2.1 ตาราง ODD Dimension

| Dimension | ค่าที่อยู่ใน ODD (IN) | ค่านอก ODD (OUT) | หน่วยตรวจวัด (sensor) |
|-----------|----------------------|-------------------|------------------------|
| **Road type** | highway, urban, rural, parking, intersection | construction zone, off-road, roundabout (complex) | map (HD map) + perception (road class) |
| **Speed range** | 0–30, 30–60, 60–90, 90–120 km/h | > 120 km/h | vehicle speed (CAN/IMU) |
| **Weather** | clear, rain (light–moderate), fog (light), night | heavy rain, dense fog (<50 m vis), snow (accumulating), storm | camera visibility + radar SNR |
| **Lighting** | day, dusk, night (with streetlight), tunnel | pitch black (no streetlight), glare sunrise/sunset | camera exposure (EV) |
| **Traffic** | free flow, congested, stopped, mixed | aggressive cut-in (repeated), emergency vehicle approaching | perception (vehicle count/density) |
| **Infrastructure** | lane markings good/poor, traffic lights, signs | no markings + no map, broken/missing signs | lane marking quality score + map |
| **Geography** | flat, hilly | mountainous (sharp switchback), steep grade (>10%) | map elevation + IMU pitch |

> หมายเหตุ: ค่าในวงเล็บ เช่น "light–moderate" คือเกณฑ์เชิงปริมาณที่ define ในส่วน 3 (ODD monitoring)

### 2.2 รายละเอียดแต่ละ dimension

#### Road type (ประเภทถนน)
- **highway**: ทางด่วน ไม่มีไฟแดง ไม่มี pedestrian บนผิวจราจร ความเร็วสูง
- **urban**: ถนนในเมือง มีไฟแดง มี pedestrian crossing ความเร็วปานกลาง
- **rural**: ถนนนอกเมือง รถน้อย อาจไม่มีกึ่งกลางชัด
- **parking**: พื้นที่จอดรถ ความเร็วต่ำมาก ต้องการ precise positioning
- **intersection**: ทางแยก (signalized หรือ stop-sign) ต้องจัดการ right-of-way

#### Speed range (ช่วงความเร็ว)
| ช่วง (km/h) | ODD behavior หลัก | เหตุผล |
|-------------|-------------------|--------|
| 0–30 | TJA / parking โหมด | รถติด/จอด ต้องการ precise stop & go |
| 30–60 | urban ACC + LKA | ในเมือง ต้องระวัง pedestrian |
| 60–90 | rural / urban arterial | ระยะตามรถขยาย |
| 90–120 | highway ACC + LKA | ความเร็วสูง lookahead ขยาย (REF_PATH_LOOKAHEAD_M) |

> ความเร็ว > 120 km/h = OUT เพราะ MPC ไม่ได้ tune สำหรับ dynamics ที่ความเร็วสูง (slip term และ latency budget ไม่พอ)

#### Weather (สภาพอากาศ)
- **clear**: visibility > 200 m (baseline)
- **rain (light–moderate)**: visibility 100–200 m, รถยังตรวจจับ lane ได้ (อาจลด confidence)
- **fog (light)**: visibility 100–200 m
- **night**: ใช้กล้อง + หรือไฟถนน (lighting แยก dimension)
- **OUT**: heavy rain (vis < 100 m), dense fog (vis < 50 m), snow ที่ปกคลุมเส้น lane หรือสะสมบนผิวจราจร

#### Lighting (แสงสว่าง)
- **day**: EV ปกติ, ความคมชัดสูง
- **dusk**: EV ต่ำกลาง อาจมี glare
- **night (with streetlight)**: EV ต่ำ แต่มีแสงเสริม
- **tunnel**: มืดกะทันหัน → ต้อง transition ภายใน 2 วินาที
- **OUT**: pitch black (ไม่มี streetlight และไม่มี headlight ที่เพียงพอ), glare sunrise/sunset ที่ทำให้ lane detection พลาด

#### Traffic (ความหนาแน่นจราจร)
- **free flow**: รถน้อย ระยะห่าง > 5 s headway
- **congested**: รถหนาแน่น ความเร็วต่ำ ระยะ 2–5 s
- **stopped**: รถหยุดนิ่ง TJA ทำงาน
- **mixed**: สลับ free/congested
- **OUT**: aggressive cut-in ซ้ำ ๆ (รถตัดหน้าในระยะ < 1 s) หรือ emergency vehicle ใกล้

#### Infrastructure (โครงสร้างพื้นฐาน)
- **lane markings good**: เส้นชัด ต่อเนื่อง UFLDv2 confidence > 0.8
- **lane markings poor**: เส้นซีด/ขาดบางช่วง confidence 0.4–0.8 → ใช้ fusion กับ waypoint
- **lane markings none**: confidence < 0.4 → ใช้ WP_PRIMARY (เหมือน mode ปัจจุบันของ fusion)
- **traffic lights / signs**: ต้องมี perception ตรวจจับได้ (urban ODD)
- **OUT**: ไม่มีเส้น + ไม่มี HD map → ไม่มี reference path ที่เชื่อถือได้

#### Geography (ภูมิประเทศ)
- **flat**: grade < 3% MPC ทำงานปกติ
- **hilly**: grade 3–8% ต้องปรับ weight ความเร็ว/brake
- **OUT**: mountainous (switchback แคบ) หรือ grade > 10% (MPC ไม่ได้ tune สำหรับ longitudinal ที่ steep)

---

## 3. ODD Monitoring (Runtime Detection)

ODD monitoring ทำงานแบบ real-time ทุก frame (10 Hz เท่ากับ control loop) โดยใช้ข้อมูลจาก perception + map + vehicle state แล้วออกมาเป็น `ODDState` (ส่วน 7)

### 3.1 Weather detection (camera + radar visibility)

```
Input:  camera RGB, radar return SNR
Process:
  1. คำนวณ image contrast / sharpness (Laplacian variance) → ต่ำ = มีฝน/ฝ้า
  2. คำนวณ sky region brightness (top 20% ของภาพ)
  3. ตรวจจับ rain streak ด้วย edge density ผิดปกติ
  4. radar: นับ return ที่อยู่นอก expected range (clutter = ฝน/หิมะ)
Output: weather_class ∈ {clear, rain, fog, snow, night}, visibility_m, confidence
```

เกณฑ์เชิงปริมาณ:
- `visibility_m` ≈ จาก contrast decay ตามระยะ (calibrate กับ CARLA weather presets)
- `visibility_m < 50` → weather = OUT (dense fog)
- contrast < 0.15 + radar clutter > threshold → rain (heavy) → OUT

### 3.2 Road type detection (map + perception)

```
Input:  HD map (road class), GPS, perception (detect traffic light / intersection)
Process:
  1. อ่าน road class จาก map ที่ตำแหน่ง ego (primary)
  2. cross-check ด้วย perception: เห็น traffic light → urban/intersection
  3. ถ้า map บอก highway แต่ perception เห็น traffic light → flag mismatch (degrade confidence)
Output: road_type ∈ {highway, urban, rural, parking, intersection}, confidence
```

> ใน CARLA ใช้ `world.get_map().get_waypoint(location)` เพื่ออ่าน road info ได้โดยตรง

### 3.3 Lane marking quality score

ใช้ผลจาก perception pipeline ที่มีอยู่แล้ว (UFLDv2/UNet + Kalman):

```
Input:  lane confidence (จาก LaneDetector / LaneTrajectoryPipeline), mask pixel count
Process:
  1. lane_confidence ∈ [0,1] (มีอยู่แล้วใน fusion stage)
  2. mask continuity score = จำนวน strip ที่มี lane pixel / total strip
  3. quality_score = 0.6 * lane_confidence + 0.4 * continuity
Output: lane_quality ∈ {good (>0.8), poor (0.4–0.8), none (<0.4)}, quality_score
```

สอดคล้องกับ fusion mode ปัจจุบัน:
- `good` → `LANE+MPC`
- `poor` → `WP+UNET_CENTER` (blend)
- `none` → `WP_PRIMARY`

### 3.4 Lighting estimation (camera exposure)

```
Input:  camera RGB
Process:
  1. คำนวณ mean brightness (V channel ใน HSV) ของทั้งภาพ
  2. คำนวณ dynamic range (P95 - P5 ของ brightness)
  3. ตรวจจับ glare: มี region สว่างมาก (>240) ที่ top ของภาพ
Output: lighting ∈ {day, dusk, night, tunnel}, exposure_ev, glare_flag
```

เกณฑ์:
- mean brightness > 100 → day
- 50–100 → dusk
- < 50 → night (ตรวจ streetlight ด้วย hot spot)
- transition รวดเร็ว (delta > 60 ภายใน 2 s) → tunnel entry/exit

### 3.5 Traffic density estimation

```
Input:  perception bounding boxes (vehicles), radar tracks
Process:
  1. นับ vehicle ใน lane เดียวกัน + lane ข้าง ภายในระยะ 80 m
  2. headway_s = distance_to_lead / ego_speed
  3. density = count / 80 m
Output: traffic ∈ {free_flow, congested, stopped, mixed}, headway_s, density
```

เกณฑ์:
- headway > 5 s → free flow
- 2–5 s → congested
- ego speed < 1 m/s และ lead < 5 m → stopped
- สลับ → mixed

---

## 4. ODD Exit Handling (การจัดการเมื่อออกนอก ODD)

### 4.1 State Machine

```
                        ┌─────────────────────────────────────────────┐
                        │                                             │
                        ▼                                             │
   ┌──────────┐   all IN   ┌──────────┐   any OUT (soft)  ┌───────────────┐
   │ INIT     │───────────▶│ IN_ODD   │─────────────────▶│ DEGRADING     │
   └──────────┘            └──────────┘                  └───────────────┘
                                ▲                              │
                                │ all IN (≥3 s)                │ severe OR
                                │                              │ degraded ≥10 s
                                └──────────────┐               ▼
                                               │        ┌───────────────┐
                                               │        │ HANDOFF_REQ   │
                                               │        │ (10s buffer)  │
                                               │        └───────────────┘
                                               │              │
                                               │              │ driver take
                                               │              ▼
                                               │        ┌───────────────┐
                                               │        │ DRIVER_CTRL   │
                                               │        └───────────────┘
                                               │              │
                                               │              │ no take (10s)
                                               │              ▼
                                               │        ┌───────────────┐
                                               │        │ SAFE_STOP     │
                                               │        └───────────────┘
                                               │              │
                                               │              │ stopped
                                               │              ▼
                                               └────────  ┌───────────┐
                                                          │ STOPPED   │
                                                          └───────────┘
```

### 4.2 นิยาม transition

| From → To | เงื่อนไข | การกระทำ (action) |
|-----------|----------|--------------------|
| IN_ODD → DEGRADING | ≥1 dimension OUT แบบ "soft" (เช่น lane poor, rain light) | ลด v_ref 20%, เพิ่ม following distance ×1.5, ลด perception confidence weight |
| DEGRADING → IN_ODD | ทุก dimension IN ติดต่อกัน ≥ 3 s | คืนค่า v_ref/distance ปกติ |
| DEGRADING → HANDOFF_REQ | dimension OUT แบบ "severe" (visibility < 50 m, lane none + no map, speed > 120) หรือ อยู่ใน DEGRADING ≥ 10 s ไม่กลับ | เปิด handoff request, แจ้ง driver (visual + audio) |
| HANDOFF_REQ → DRIVER_CTRL | driver แตะ steering หรือกด pedal | ปลด autonomous, โหมด manual |
| HANDOFF_REQ → SAFE_STOP | ครบ 10 s driver ไม่ตอบ | emergency brake + hazard light, ชะลอไปขอบทาง |
| SAFE_STOP → STOPPED | ความเร็ว < 0.5 m/s | ปลด throttle, คง brake, รอ driver |

### 4.3 Severity classification

| Severity | ตัวอย่าง | การตอบสนอง |
|----------|----------|------------|
| **Soft** (degrade) | lane poor, rain light, dusk, congested | ลด speed + เพิ่ม gap + ลด conf weight |
| **Severe** (handoff) | visibility < 50 m, lane none + no map, speed > 120, snow accumulating | handoff request ทันที |
| **Critical** (safe stop now) | perception ล้มทั้งหมด (mask + waypoint invalid), MPC solver fail ติดต่อกัน, hardware fault | safe stop ทันที ไม่รอ 10 s |

> Critical = ข้าม HANDOFF_REQ ไป SAFE_STOP โดยตรง (เพราะระบบไม่สามารถควบคุมได้ปลอดภัยแล้ว)

### 4.4 Graceful degradation details

เมื่อเข้าสู่ DEGRADING:
1. **v_ref** × 0.8 (ลด 20%) — ปรับใน MPC reference speed
2. **following distance** × 1.5 (ACC gap setting)
3. **REF_PATH_LOOKAHEAD_M** × 0.7 (ลด lookahead เพราะ visibility จำกัด)
4. **perception confidence weight** ลด (เน้น waypoint มากขึ้น → เลื่อนไปทาง WP_PRIMARY)
5. **steer jerk limit** ลด (ขับนุ่มขึ้น)
6. แจ้ง driver ด้วย UI: สีเหลือง + ข้อความ "ODD: DEGRADED — <reason>"

### 4.5 Driver handoff request (10s buffer)

- แสดง **สีแดง** + ข้อความ "TAKE CONTROL — <reason>" + เสียงเตือน
- นับเวลาถอยหลัง 10 → 0 (แสดงบน dashboard)
- ตรวจ driver input: steering torque > threshold หรือ pedal > threshold
- หาก driver ตอบภายใน 10 s → โหมด manual (DRIVER_CTRL)
- หากครบ 10 s → SAFE_STOP

---

## 5. ODD-Specific Behavior (พฤติกรรมเฉพาะ ODD)

แต่ละ ODD subset กำหนด behavior ของ ADAS suite (ACC / LKA / TJA / AEB) แบบ rule-based:

### 5.1 Highway ODD
```
road_type = highway, speed 60–120 km/h
  • ACC: ทำงาน (maintain set speed + following)
  • LKA: ทำงาน (lane keeping ด้วย MPC)
  • Lane change: เฉพาะเมื่อ speed ≥ 60 km/h และ target lane ว่าง (headway > 4 s)
  • ไม่ตอบสนอง traffic light (ไม่มีบน highway)
  • Pedestrian: ไม่คาดหวัง (flag เป็น anomaly → handoff)
  • REF_PATH_LOOKAHEAD_M: ค่าสูง (highway = มองไกล)
```

### 5.2 Urban ODD
```
road_type = urban, speed 0–60 km/h
  • TJA: ทำงาน (stop & go ในรถติด)
  • ACC: ทำงาน แต่ v_ref จำกัดตาม limit
  • LKA: ทำงาน แต่ยอม lane poor (ใช้ fusion กับ waypoint)
  • Traffic light: perception ตรวจจับ → หยุดที่แดง, ชะลอที่เหลือง
  • Pedestrian priority: ตรวจจับ pedestrian ใน crossing → AEB + yield
  • Lane change: จำกัด (เฉพาะที่จำเป็น และ speed < 40 km/h)
  • REF_PATH_LOOKAHEAD_M: ค่าต่ำ (urban = มองใกล้)
```

### 5.3 Parking ODD
```
road_type = parking, speed 0–10 km/h
  • ปิด ACC (ไม่ใช้ high-speed following)
  • LKA: ปิด (ไม่มี lane ชัด ใน parking)
  • Precise positioning: ใช้ waypoint + perception ระยะใกล้ (object รอบรถ)
  • AEB: ไวขึ้น (threshold ต่ำ เพราะความเร็วต่ำ)
  • ใช้ MPC ในโหมด "low-speed waypoint tracking"
  • ตรวจจับ slot/obstacle รอบรถ
```

### 5.4 Intersection ODD
```
road_type = intersection
  • หยุดที่ stop line (ถ้ามี sign/light)
  • จัดการ right-of-way (rule-based: รถทางขวามีสิทธิ์, รถใน roundabout มีสิทธิ์)
  • ไม่ตัดสินใจ overtake ใน intersection
  • เฝ้าระวัง cross-traffic
  • หากไม่มี traffic light และไม่มี stop sign → handoff (ซับซ้อนเกิน ODD)
```

### 5.5 Rural ODD
```
road_type = rural, speed 30–90 km/h
  • ACC + LKA ทำงาน
  • ยอม lane marking poor (ใช้ waypoint เป็นหลัก)
  • เฝ้าระวัง oncoming traffic (ไม่มีกึ่งกลาง)
  • ไม่ overtake (เกิน ODD)
  • REF_PATH_LOOKAHEAD_M: ปานกลาง
```

### 5.6 Behavior matrix สรุป

| Feature | Highway | Urban | Rural | Parking | Intersection |
|---------|---------|-------|-------|---------|--------------|
| ACC | ✓ | ✓ (limited) | ✓ | ✗ | ✗ |
| LKA | ✓ | ✓ (fusion) | ✓ (WP) | ✗ | ✗ |
| TJA | ✗ | ✓ | ✗ | ✗ | ✗ |
| AEB | ✓ | ✓ (sensitive) | ✓ | ✓ (very sensitive) | ✓ |
| Lane change | ✓ (≥60) | limited | ✗ | ✗ | ✗ |
| Traffic light resp | ✗ | ✓ | ✗ | ✗ | ✓ |
| Pedestrian yield | ✗ | ✓ | ✓ (caution) | ✓ | ✓ |
| Lookahead | long | short | medium | very short | short |

---

## 6. CARLA Integration — Test Scenarios

CARLA 0.9.16 รองรับ weather presets + multiple towns ทำให้ทดสอบ ODD dimension ได้ครบ

### 6.1 Scenario list (ตาม ODD dimension)

| ID | Scenario | Town | Weather preset | จุดตรวจ (pass criteria) |
|----|----------|------|----------------|--------------------------|
| S-ODD-01 | Highway clear day | Town04 (loop) | ClearNoon | LKA CTE < 0.3 m ที่ 90 km/h |
| S-ODD-02 | Highway rain | Town04 | HardRainNoon | DEGRADING ภายใน 5 s, speed ลด 20% |
| S-ODD-03 | Highway fog dense | Town04 | FogSunset (dense) | HANDOFF_REQ ภายใน 3 s (vis < 50 m) |
| S-ODD-04 | Highway night | Town04 | ClearNight | LKA ทำงาน CTE < 0.5 m |
| S-ODD-05 | Urban clear day | Town05 | ClearNoon | TJA + traffic light stop ถูกต้อง |
| S-ODD-06 | Urban congested | Town05 | ClearNoon + heavy traffic | TJA stop&go ไม่กระตุก, headway ปลอดภัย |
| S-ODD-07 | Urban pedestrian | Town05 | ClearNoon + walker | AEB หยุดให้ pedestrian ที่ crossing |
| S-ODD-08 | Urban night | Town05 | ClearNight | traffic light detection ยังถูก |
| S-ODD-09 | Rural clear | Town03 | ClearNoon | LKA บน road ไม่มีกึ่งกลาง ใช้ WP |
| S-ODD-10 | Rural dusk glare | Town03 | SunsetIllumination | DEGRADING (glare), ลด speed |
| S-ODD-11 | Parking low speed | Town04 parking lot | ClearNoon | จอดใน slot ความเร็ว < 5 km/h |
| S-ODD-12 | Intersection signalized | Town05 | ClearNoon | หยุดที่แดง, ออกที่เขียว |
| S-ODD-13 | Intersection stop-sign | Town03 | ClearNoon | หยุด 2 s แล้ว yield |
| S-ODD-14 | Tunnel transition | Town05 (tunnel) | ClearNoon → tunnel | lighting transition ภายใน 2 s, ไม่ handoff |
| S-ODD-15 | Lane marking none | Town04 (off-lane area) | ClearNoon | WP_PRIMARY, DEGRADING |
| S-ODD-16 | Speed > 120 | Town04 | ClearNoon | HANDOFF_REQ (out of speed ODD) |
| S-ODD-17 | Mixed weather night rain | Town04 | WetNight | DEGRADING, LKA ยังทำงาน CTE < 0.6 m |
| S-ODD-18 | Driver handoff timeout | Town04 | FogSunset dense | 10 s ไม่ตอบ → SAFE_STOP |
| S-ODD-19 | Driver handoff accept | Town04 | FogSunset dense | driver แตะ steering → DRIVER_CTRL |
| S-ODD-20 | MPC solver fail | Town04 | ClearNoon (inject fail) | Critical → SAFE_STOP ทันที |

### 6.2 CARLA weather presets mapping

| CARLA preset | ODD weather | visibility estimate |
|--------------|-------------|---------------------|
| ClearNoon / ClearSunset | clear | > 200 m |
| CloudyNoon / CloudySunset | clear | > 200 m |
| WetNoon / WetSunset | rain (light) | 150–200 m |
| HardRainNoon / HardRainSunset | rain (heavy) | < 100 m → OUT |
| SoftRainNoon / SoftRainSunset | rain (moderate) | 100–150 m |
| ClearNight | night | 150 m (with headlight) |
| WetNight / WetCloudyNight | night + rain | 100–150 m |
| HardRainNight | night + heavy rain | < 100 m → OUT |
| FogSunset / FogNight | fog | 50–150 m (dense → OUT) |
| MidRainSunset | rain (moderate) | 100–150 m |

### 6.3 วิธีรัน scenario (ตัวอย่าง)

```bash
# จาก root CARLA_0.9.16
./run_carla_visible.sh

# รัน ODD scenario (สคริปต์สมมติ: scripts/run_odd_scenario.py)
python scripts/run_odd_scenario.py --scenario S-ODD-02 --town Town04 \
    --weather HardRainNoon --speed 90 --frames 500

# ผล: บันทึก ODDState log + CTE + mode transitions
# ตรวจสอบ: ระบบเข้า DEGRADING ภายใน 5 s และ speed ลด 20%
```

---

## 7. Data Structure — ODDState

Python dataclass สำหรับเก็บสถานะ ODD ทุก dimension + confidence รวม

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums สำหรับ ODD dimension ──────────────────────────────────────────
class RoadType(Enum):
    HIGHWAY = "highway"
    URBAN = "urban"
    RURAL = "rural"
    PARKING = "parking"
    INTERSECTION = "intersection"
    UNKNOWN = "unknown"          # ยังตรวจไม่ได้ / นอก ODD


class WeatherClass(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"
    SNOW = "snow"
    NIGHT = "night"              # night แยกจาก lighting เพราะ weather อาจเป็น clear-night


class LightingClass(Enum):
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"
    TUNNEL = "tunnel"


class TrafficClass(Enum):
    FREE_FLOW = "free_flow"
    CONGESTED = "congested"
    STOPPED = "stopped"
    MIXED = "mixed"


class LaneQuality(Enum):
    GOOD = "good"        # quality_score > 0.8
    POOR = "poor"        # 0.4–0.8
    NONE = "none"        # < 0.4


class GeographyClass(Enum):
    FLAT = "flat"
    HILLY = "hilly"
    MOUNTAINOUS = "mountainous"  # OUT


class ODDStatus(Enum):
    """สถานะของ ODD lifecycle (state machine ในส่วน 4)"""
    IN_ODD = "in_odd"
    DEGRADING = "degrading"
    HANDOFF_REQUEST = "handoff_request"
    DRIVER_CONTROL = "driver_control"
    SAFE_STOP = "safe_stop"
    STOPPED = "stopped"


class Severity(Enum):
    """ระดับความรุนแรงของ ODD violation"""
    NONE = "none"
    SOFT = "soft"          # degrade
    SEVERE = "severe"      # handoff
    CRITICAL = "critical"  # safe stop now


# ── ODDState ─────────────────────────────────────────────────────────────
@dataclass
class ODDDimensionStatus:
    """สถานะราย dimension: ค่า + confidence + flag ว่า IN/OUT"""
    value: object                  # enum value ของ dimension นั้น
    confidence: float = 1.0        # 0.0–1.0
    in_odd: bool = True            # True = อยู่ใน ODD, False = ออกนอก ODD
    severity: Severity = Severity.NONE
    reason: str = ""               # คำอธิบาย กรณี out (สำหรับ log/UI)


@dataclass
class ODDState:
    """สถานะ ODD ทั้งหมด ใช้ส่งระหว่าง ODD monitor → planner → dashboard"""
    # ── แต่ละ dimension ──
    road_type: ODDDimensionStatus = field(
        default_factory=lambda: ODDDimensionStatus(RoadType.UNKNOWN))
    speed: ODDDimensionStatus = field(
        default_factory=lambda: ODDDimensionStatus(0.0))          # value = km/h (float)
    weather: ODDDimensionStatus = field(
        default_factory=lambda: ODDDimensionStatus(WeatherClass.CLEAR))
    lighting: ODDDimensionStatus = field(
        default_factory=lambda: ODDDimensionStatus(LightingClass.DAY))
    traffic: ODDDimensionStatus = field(
        default_factory=lambda: ODDDimensionStatus(TrafficClass.FREE_FLOW))
    lane_quality: ODDDimensionStatus = field(
        default_factory=lambda: ODDDimensionStatus(LaneQuality.GOOD))
    geography: ODDDimensionStatus = field(
        default_factory=lambda: ODDDimensionStatus(GeographyClass.FLAT))

    # ── ค่าวัดดิบ (raw measurements) ──
    visibility_m: float = 200.0          # จาก weather detection
    headway_s: float = 10.0              # จาก traffic estimation
    traffic_density: float = 0.0         # vehicles / 80 m
    exposure_ev: float = 0.0             # จาก lighting estimation
    lane_quality_score: float = 1.0      # 0.0–1.0
    glare_flag: bool = False
    road_grade_pct: float = 0.0          # % slope

    # ── สรุประดับสูง ──
    status: ODDStatus = ODDStatus.IN_ODD
    overall_severity: Severity = Severity.NONE
    overall_confidence: float = 1.0      # min ของ confidence ทุก dimension
    in_odd: bool = True                  # True เมื่อทุก dimension in_odd

    # ── timing สำหรับ state machine ──
    time_in_current_status: float = 0.0  # วินาทีที่อยู่ใน status ปัจจุบัน
    handoff_countdown_s: float = 10.0    # นับถอยหลัง handoff (10 → 0)

    # ── เหตุผลหลัก (สำหรับ UI/log) ──
    primary_reason: str = ""             # เช่น "visibility < 50m (dense fog)"

    # ── timestamp ──
    timestamp: float = 0.0               # sim time (CARLA world snapshot)

    def update_summary(self) -> None:
        """คำนวณ overall_severity, overall_confidence, in_odd จากทุก dimension
        เรียกหลัง update แต่ละ dimension"""
        dims = [self.road_type, self.speed, self.weather, self.lighting,
                self.traffic, self.lane_quality, self.geography]
        self.in_odd = all(d.in_odd for d in dims)
        self.overall_confidence = min(d.confidence for d in dims)
        # severity สูงสุดเป็น overall
        sev_order = {Severity.NONE: 0, Severity.SOFT: 1,
                     Severity.SEVERE: 2, Severity.CRITICAL: 3}
        self.overall_severity = max(dims, key=lambda d: sev_order[d.severity]).severity
        # หา reason ของ dimension ที่ severity สูงสุด
        worst = max(dims, key=lambda d: sev_order[d.severity])
        self.primary_reason = worst.reason if not worst.in_odd else ""

    def is_critical(self) -> bool:
        """Critical = ต้อง safe stop ทันที"""
        return self.overall_severity == Severity.CRITICAL

    def to_log_dict(self) -> dict:
        """แปลงเป็น dict สำหรับ logging (observability)"""
        return {
            "t": round(self.timestamp, 3),
            "status": self.status.value,
            "in_odd": self.in_odd,
            "severity": self.overall_severity.value,
            "conf": round(self.overall_confidence, 3),
            "road": self.road_type.value.value if hasattr(self.road_type.value, "value") else str(self.road_type.value),
            "speed_kmh": round(float(self.speed.value), 1),
            "weather": self.weather.value.value,
            "lighting": self.lighting.value.value,
            "traffic": self.traffic.value.value,
            "lane_q": self.lane_quality.value.value,
            "geo": self.geography.value.value,
            "vis_m": round(self.visibility_m, 1),
            "headway_s": round(self.headway_s, 2),
            "reason": self.primary_reason,
        }
```

### 7.1 ตัวอย่างการใช้งานใน pipeline

```python
# ใน ODD monitor (รัน 10 Hz คู่กับ control loop)
odd_state = ODDState()
odd_state.road_type = ODDDimensionStatus(
    value=RoadType.HIGHWAY, confidence=0.95, in_odd=True)
odd_state.weather = ODDDimensionStatus(
    value=WeatherClass.FOG, confidence=0.7, in_odd=False,
    severity=Severity.SEVERE, reason="visibility 45m < 50m threshold")
odd_state.update_summary()
# → odd_state.in_odd = False
# → odd_state.overall_severity = SEVERE
# → state machine: IN_ODD → HANDOFF_REQUEST

# ส่งไป planner เพื่อปรับ behavior และ dashboard เพื่อแสดง
planner.apply_odd(odd_state)
dashboard.update_odd_panel(odd_state)
logger.log(odd_state.to_log_dict())
```

---

## 8. การเชื่อมต่อกับ pipeline ที่มี (Integration)

ODD module เสียบเข้ากับ pipeline ที่มีอยู่ (ตาม WORKFLOW_SYSTEM_DESIGN_QAQC.md) โดยเป็น **stage ใหม่ก่อน STAGE 6 (CONTROL)**:

```
STAGE 5 (REFERENCE PATH)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  STAGE 5.5: ODD MONITOR (ใหม่)                   │
│  Input:  perception confidence, camera, radar,   │
│          map, vehicle speed                      │
│  Output: ODDState (status, severity, conf)       │
│  QC:     ทุก dimension มี confidence + in_odd     │
│  Gate:   status != IN_ODD → ปรับ MPC + behavior   │
│  Code:   odd/odd_monitor.py (ใหม่)                │
└─────────────────────────────────────────────────┘
        │ ODDState
        ▼
STAGE 6 (CONTROL — MPC)  ← รับ ODDState ปรับ v_ref / lookahead / weight
        │
        ▼
STAGE 7 (SAFETY OVERRIDE)  ← รับ ODDStatus: SAFE_STOP → emergency brake
```

การปรับ MPC ตาม ODD:
- `DEGRADING` → v_ref × 0.8, lookahead × 0.7, confidence weight ลด
- `HANDOFF_REQUEST` → คง MPC แต่เตรียม safe stop, เปิด handoff UI
- `SAFE_STOP` → สั่ง emergency brake (ส่งไป SafetyOverride เหมือนเดิม)
- ODD subset (highway/urban/...) → เลือก behavior profile ของ ACC/LKA/TJA

---

## 9. สรุป (Summary)

| หัวข้อ | จุดสำคัญ |
|--------|----------|
| **ODD definition** | เซตของ 7 dimension ที่ระบบ validate แล้ว ใช้ AND logic |
| **Dimensions** | road type, speed, weather, lighting, traffic, infrastructure, geography |
| **Monitoring** | 5 detector รัน 10 Hz ออกมาเป็น ODDState + confidence |
| **Exit handling** | state machine: IN_ODD → DEGRADING → HANDOFF_REQ → SAFE_STOP |
| **Severity** | soft (degrade) / severe (handoff) / critical (safe stop now) |
| **Handoff** | 10 s buffer พร้อม countdown; ไม่ตอบ → safe stop |
| **ODD behavior** | 5 profile: highway / urban / parking / intersection / rural |
| **CARLA** | 20 scenario ครอบคลุมทุก dimension + weather preset mapping |
| **Data structure** | `ODDState` dataclass + `ODDDimensionStatus` + enums |
| **Integration** | stage 5.5 ใหม่ ก่อน MPC, ปรับ v_ref/lookahead/weight |

หลักการสำคัญที่ยึดตลอดเอกสาร:
1. **No RL** — ทุก behavior และ transition เป็น rule/deterministic ตรวจสอบได้
2. **Graceful degradation ก่อน safe stop** — ลด speed/gap ก่อน ไม่ตัดสินใจหยุดกะทันหัน
3. **Observability** — ODDState log ทุก frame เหมือน mode log ของ fusion
4. **Reuse pipeline ที่มี** — ODD ใช้ lane confidence จาก perception ที่มีแล้ว ไม่ต้องสร้างใหม่ทั้งหมด
5. **Driver-in-the-loop** — SAE L2–3, handoff 10 s เป็นข้อกำหนดหลัก

---

> ไฟล์ที่เกี่ยวข้อง (สร้างในขั้นถัดไป):
> - `carla_mpc_classical/odd/odd_monitor.py` — ODD monitoring implementation
> - `carla_mpc_classical/odd/odd_state.py` — ODDState dataclass (ส่วน 7)
> - `carla_mpc_classical/odd/behavior_profiles.py` — ODD-specific behavior (ส่วน 5)
> - `scripts/run_odd_scenario.py` — CARLA scenario runner (ส่วน 6)
> - `tests/test_odd_state.py` — unit test สำหรับ ODD state machine + dataclass
