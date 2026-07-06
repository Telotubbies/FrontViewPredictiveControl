# Safety & Redundancy — ความปลอดภัยและระบบสำรอง

> เอกสารหมายเลข **07** ในชุด System Design ของ `carla_mpc_classical` (CARLA + ADAS + MPC + Perception pipeline)
> อ้างอิง: [14_odd_definition.md](14_odd_definition.md), [05_path_planning.md](05_path_planning.md), [02_vision_perception.md](02_vision_perception.md)
> หลักการ: **No reinforcement learning** — ทุก safety behavior กำหนดด้วย rule/state-machine + classical perception + deterministic fallback
> มาตรฐานอ้างอิง: **ISO 26262:2018** (Functional Safety), **ISO 21448:2022** (SOTIF), **UNECE R79** (steering), **UNECE R13** (braking)

---

## 0. ภาพรวม (Overview)

เอกสารนี้อธิบายสถาปัตยกรรมความปลอดภัยของระบบ autonomous prototype ซึ่งออกแบบตามแนวทาง **ISO 26262** โดยมีเป้าหมายหลักคือ:

1. **ไม่มี single-point-of-failure** ที่นำไปสู่การควบคุมคำสั่งที่อันตราย (hazardous control command)
2. **Safety layer แยกอิสระ** จาก control/planning layer — safety สามารถ override ได้เสมอ และไม่ถูก bypass โดย MPC output
3. **Graceful degradation** — ลดระดับการทำงานทีละขั้น ไม่หยุดทำงานกะทันหันโดยไม่มี fallback
4. **Driver-in-the-loop** — ระบบเตือนและส่งควบคุมคืน driver ก่อนเข้าสู่ safe stop เสมอ (ยกเว้น AEB ที่ time-critical)

```
┌────────────────────────────────────────────────────────────────────────┐
│                    SAFETY ARCHITECTURE (LAYERED)                       │
│                                                                        │
│  Layer 4: Driver Monitoring (DMS)  ← camera-based attention tracking   │
│      │                                                                 │
│  Layer 3: Safety Override (override.py) ← independent, never bypassed  │
│      │   ├── speed limit enforcement                                   │
│      │   ├── steering limit enforcement                                │
│      │   ├── CTE limit enforcement                                     │
│      │   └── emergency brake command                                   │
│      │                                                                 │
│  Layer 2: ADAS Functions (AEB, LDW/LKA, ACC, Stuck Recovery)           │
│      │   ├── AEB: TTC-based emergency brake (aeb_acc.py)               │
│      │   ├── LDW/LKA Pro: lane departure warning + corrective steer    │
│      │   ├── ACC: time-gap following distance                          │
│      │   └── Stuck Recovery: brake → reverse → forward                 │
│      │                                                                 │
│  Layer 1: Redundant Hardware (dual compute, dual brake, dual power)    │
│      ├── Compute A (primary MPC) + Compute B (safety monitor)         │
│      ├── Brake Primary + Brake Secondary (independent actuators)       │
│      └── Power A + Power B (isolated rails, OR-diode redundancy)       │
└────────────────────────────────────────────────────────────────────────┘
```

> **หมายเหตุสำคัญ**: โปรโตไทป์นี้ทำงานใน CARLA simulation เป็นหลัก แต่สถาปัตยกรรมนี้ออกแบบให้
> สามารถพอร์ตไปยัง vehicle platform จริงได้ โดย ASIL D requirements จะถูก mark ไว้ชัดเจน

---

## 1. ISO 26262 Compliance & ASIL Classification

### 1.1 Hazard Analysis & Risk Assessment (HARA)

การจัด ASIL ใช้กรอบของ ISO 26262 Part 3 โดยพิจารณา 3 ปัจจัย: **Severity (S)**, **Exposure (E)**, **Controllability (C)**

| Hazard Scenario | S | E | C | ASIL | Safety Goal |
|-----------------|---|---|---|------|-------------|
| Unintended steering > 0.45 rad at 30 km/h | S3 | E4 | C3 | **ASIL D** | จำกัด steering angle ไม่เกิน safety limit เสมอ |
| Unintended acceleration (throttle stuck) | S3 | E4 | C2 | ASIL C | ตัด throttle + apply brake เมื่อ speed > max |
| Loss of braking (brake actuator fail) | S3 | E3 | C3 | **ASIL D** | Secondary brake path ต้องทำงานภายใน 200 ms |
| AEB false-negative (ไม่เบรกเมื่อควร) | S3 | E4 | C3 | **ASIL D** | TTC < 1.0 s → full brake โดยอิสระ |
| AEB false-positive (เบรกผิดจังหวะ) | S2 | E3 | C2 | ASIL B | จำกัด decel ของ AEB ไม่เกิน 5 m/s² ยกเว้น critical |
| Lane departure ไม่ถูกตรวจจับ | S2 | E4 | C2 | ASIL B | LDW + LKA Pro corrective steer |
| MPC compute lockup / hang | S3 | E3 | C3 | **ASIL D** | Watchdog → safe stop ภายใน timeout |
| Sensor degradation (camera ตาบอด) | S2 | E3 | C2 | ASIL B | ODD exit → degraded mode → handoff |
| Stuck vehicle (off-road) | S1 | E3 | C3 | ASIL A | Stuck recovery ≤ 3 attempts แล้ว safe stop |

> **ASIL D** = highest integrity → ต้องมี hardware redundancy + diagnostic coverage ≥ 99%
> **ASIL A/B** = ต้องมี basic diagnostic + warning

### 1.2 Safety Goals (SG)

| ID | Safety Goal | ASIL | Safe State |
|----|-------------|------|------------|
| SG-01 | ห้าม steering command เกิน ±0.45 rad โดยไม่มี safety check | D | Clamp steering + warn driver |
| SG-02 | ห้าม speed เกิน 30 km/h ใน ODD | D | Cut throttle + apply brake |
| SG-03 | ห้าม collision ด้วยความเร็ว > 0 เมื่อ TTC < 1.0 s | D | Full emergency brake |
| SG-04 | ห้าม lane departure โดยไม่มี warning/assist | B | LDW + LKA Pro corrective |
| SG-05 | ห้าม MPC output ไปถึง actuator โดยไม่ผ่าน safety layer | D | Watchdog → safe stop |
| SG-06 | ห้ามติดค้างใน stuck state เกิน 3 recovery attempts | A | Safe stop + driver handoff |
| SG-07 | ห้ามทำงานต่อเมื่อ driver ไม่ responsive เกิน 15 s | B | Escalating warning → safe stop |

---

## 2. Safety Case Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TOP-LEVEL SAFETY CLAIM                               │
│                                                                         │
│  "ระบบไม่สร้าง hazardous control command ภายใต้ ODD ที่กำหนด"             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
   ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
   │  G1: Hazard   │   │  G2: Fault    │   │  G3: Driver       │
   │  prevented    │   │  detected &   │   │  always has       │
   │  by limits    │   │  mitigated    │   │  override path    │
   └───────┬───────┘   └───────┬───────┘   └─────────┬─────────┘
           │                   │                     │
   ┌───────┴───────┐   ┌───────┴───────┐    ┌────────┴────────┐
   │ S1: override  │   │ S4: watchdog  │    │ S7: mechanical  │
   │ .py clamps    │   │ timeout 200ms │    │ brake pedal +   │
   │ steer/speed   │   │ → safe stop   │    │ steering wheel  │
   │               │   │               │    │ override        │
   │ S2: AEB TTC   │   │ S5: dual      │    │                 │
   │ < 1.0 → brake │   │ compute vote  │    │ S8: DMS detects │
   │               │   │ disagreement  │    │ unresponsive →  │
   │ S3: LDW/LKA   │   │ → degrade     │    │ handoff request │
   │ keeps in lane │   │               │    │                 │
   └───────────────┘   │ S6: dual brake│    └─────────────────┘
                       │ failover 200ms│
                       └───────────────┘
```

**Argument structure (GSN-style):**
- **G1** (Hazard prevented) ← supported by **S1, S2, S3** (deterministic safety functions)
- **G2** (Fault mitigated) ← supported by **S4, S5, S6** (redundancy + watchdog)
- **G3** (Driver override) ← supported by **S7, S8** (mechanical + monitoring)

---

## 3. Redundancy Architecture

### 3.1 Dual Compute (ASIL D requirement)

```
┌─────────────────────────────┐    ┌─────────────────────────────┐
│      COMPUTE A (Primary)    │    │   COMPUTE B (Safety Monitor)│
│                             │    │                             │
│  • Perception (YOLO + lane) │    │  • Independent TTC calc     │
│  • MPC path planning        │    │  • Speed/steer limit check  │
│  • ACC target speed         │    │  • Watchdog heartbeat       │
│  • Output: (steer, thr, br) │    │  • Output: veto / override  │
│                             │    │                             │
│  ASIL QM (no proof needed)  │    │  ASIL D (independent)       │
└──────────┬──────────────────┘    └──────────┬──────────────────┘
           │                                  │
           └────────────┬─────────────────────┘
                        ▼
              ┌──────────────────┐
              │   VOTE / GATE    │
              │                  │
              │  if A == B: pass │
              │  if A != B:      │
              │    → use B (safe)│
              │    → log mismatch│
              │    → degrade     │
              └────────┬─────────┘
                       ▼
                  [Actuator Bus]
```

- Compute A ทำงานหลัก (MPC, perception)
- Compute B ทำ safety monitor เท่านั้น — คำนวณ TTC และ limit check อิสระ ไม่ใช้ output ของ A
- หาก Compute A ไม่ส่ง heartbeat ภายใน **200 ms** → Compute B เข้าควบคุม safe stop
- หาก output ของ A และ B ขัดแย้งเกิน threshold → ใช้ค่า conservative ของ B + log incident

### 3.2 Dual Brake (ASIL D requirement)

```
                    Brake Command (from safety layer)
                           │
                    ┌──────┴──────┐
                    ▼             ▼
            ┌──────────────┐  ┌──────────────┐
            │ BRAKE PRIMARY│  │ BRAKE SECOND │
            │ (service     │  │ (independent │
            │  brake, EHB) │  │  e-park +    │
            │              │  │  friction)   │
            └──────┬───────┘  └──────┬───────┘
                   │                 │
                   └────────┬────────┘
                            ▼
                    [Wheel brake pressure]
```

| Failure Mode | Detection | Mitigation | Latency Budget |
|--------------|-----------|------------|----------------|
| Primary brake actuator stuck | Pressure sensor ≠ command | Switch to secondary | ≤ 200 ms |
| Primary brake pressure loss | Pressure < threshold | Secondary brake + warning | ≤ 200 ms |
| Both brakes fail | N/A (double fault) | Mechanical parking brake + driver | Manual |

### 3.3 Dual Power

```
  Power Rail A ──┬──▶ Compute A + Brake Primary + Front sensors
                 │
                 ├── [OR-diode] ──▶ Safety-critical bus (always powered)
                 │
  Power Rail B ──┴──▶ Compute B + Brake Secondary + Rear sensors
```

- แต่ละ rail มี battery แยก + DC-DC converter แยก
- Safety-critical bus (watchdog, brake secondary, DMS) ได้เลือกจาก rail ใดก็ได้ผ่าน OR-diode
- หาก Rail A ตก → Compute A ตาย → watchdog trigger → Compute B รับช่อง safe stop

---

## 4. Fail-Safe Modes (State Machine)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FAIL-SAFE STATE MACHINE                            │
│                                                                      │
│  [NORMAL]                                                            │
│    │ fault detected (minor)                                          │
│    ▼                                                                 │
│  [DEGRADED]  ← ลด speed 50%, เพิ่ม following distance, warn driver   │
│    │ fault worsens / ODD exit                                        │
│    ▼                                                                 │
│  [LIMP]  ← จำกัด 15 km/h, steering assist only, handoff request     │
│    │ driver not responsive / critical fault                          │
│    ▼                                                                 │
│  [SAFE_STOP]  ← AEB-style decel -3 m/s² จนหยุด, hazard lights on    │
│    │ shoulder available                                              │
│    ▼                                                                 │
│  [PULL_OVER]  ← ขยับไปขอบทางด้วยความเร็วต่ำ, park brake engage       │
│    │                                                                │
│    ▼                                                                 │
│  [SHUTDOWN]  ← log EDR, park brake, power off non-critical          │
└──────────────────────────────────────────────────────────────────────┘
```

| Mode | Speed Limit | Steering | Brake | ADAS | Driver Action |
|------|-------------|----------|-------|------|---------------|
| NORMAL | 30 km/h | MPC full | Normal | All on | Hands-on optional |
| DEGRADED | 15 km/h | MPC + LKA | Normal | AEB only | Hands-on required |
| LIMP | 10 km/h | Assist only | Ready | AEB only | Takeover requested |
| SAFE_STOP | → 0 | Hold straight | -3 m/s² | AEB active | Take control |
| PULL_OVER | 5 km/h | Lane to shoulder | Park | Off | Take control |

**Transition triggers:**
- NORMAL → DEGRADED: sensor confidence < 0.7, OR compute mismatch, OR ODD exit
- DEGRADED → LIMP: fault persists > 10 s, OR driver no response > 5 s
- LIMP → SAFE_STOP: driver no response > 15 s, OR critical fault (brake/compute)
- SAFE_STOP → PULL_OVER: shoulder detected within 50 m, speed < 5 km/h

---

## 5. Watchdog System

```
┌──────────────────────────────────────────────────────────────────────┐
│                       WATCHDOG ARCHITECTURE                          │
│                                                                      │
│  Compute A ──heartbeat (every 50ms)──▶ [Hardware Watchdog]           │
│                                            │                        │
│  Compute B ──heartbeat (every 50ms)──▶─────┤                        │
│                                            │                        │
│                            if either missing > 200ms:                │
│                                            ▼                        │
│                                  [FAULT TRIGGER]                     │
│                                            │                        │
│                            ┌───────────────┼───────────────┐        │
│                            ▼               ▼               ▼        │
│                       Cut throttle    Safe stop      Log to EDR     │
│                       Apply brake     state machine   (black box)   │
└──────────────────────────────────────────────────────────────────────┘
```

| Watchdog Parameter | Value | Rationale |
|--------------------|-------|-----------|
| Heartbeat period | 50 ms | Control loop runs at 20 Hz |
| Timeout (single miss) | 200 ms | 4 missed frames = fault |
| Timeout (hard) | 500 ms | Must enter safe stop |
| Watchdog type | External hardware timer | Independent of both computes |
| Reset | Requires power cycle | Prevents auto-recovery masking faults |

> ใน CARLA simulation ใช้ software watchdog (thread-based) แทน hardware
> แต่ interface เหมือนกันเพื่อ port ได้

---

## 6. Existing Safety Modules (Implementation)

### 6.1 SafetyOverride (`src/safety/override.py`)

โมดูลนี้เป็น **safety layer อิสระ** ที่ไม่ถูก bypass โดย MPC output:

```python
# จาก override.py (lines 1-6)
"""
Independent Safety Override System.
This module provides safety checks that must NEVER be bypassed by AI output.
Safety logic is completely independent from control module.
"""
```

| Check | Parameter | Default | Action on Violation |
|-------|-----------|---------|---------------------|
| Speed limit | `max_speed_kmh` | 30.0 km/h | Cut throttle + apply brake |
| Steering limit | `max_steering_angle` | 0.45 rad (SAFETY_MAX_STEER_RAD) | Clamp to limit |
| CTE limit | `SAFETY_CTE_LIMIT_M` | 2.0 m | Reduce speed by 50% |
| Emergency brake | `emergency_deceleration` | -5.0 m/s² | Full brake command |
| Collision timeout | `collision_timeout` | 5.0 s | Safe stop if persistent |

**Key design principle (lines 34-41):** `SafetyOverride` รับ proposed command จาก control module แล้วตรวจสอบ หากผิด limit จะ override ค่าก่อนส่งไป actuator — ไม่มี code path ที่ MPC output ไปถึง actuator โดยไม่ผ่าน layer นี้

### 6.2 AEB (`src/safety/aeb_acc.py`)

Automatic Emergency Braking ทำงานแบบ TTC-based:

```python
# จาก aeb_acc.py (lines 22-26)
AEB_TTC_THRESHOLD_S = 2.5       # TTC < นี่ → AEB active
AEB_TTC_CRITICAL_S = 1.0        # TTC < นี่ → full brake
AEB_BRAKE_PROPORTIONAL = 0.6    # brake gain for proportional AEB
AEB_FULL_BRAKE = 1.0            # full brake
AEB_LATERAL_THRESHOLD_M = 1.8   # half lane width (~3.5m) + margin
```

| TTC Range | AEB Action | Brake Level |
|-----------|------------|-------------|
| TTC > 2.5 s | Inactive | 0.0 |
| 1.0 s < TTC ≤ 2.5 s | Proportional brake | 0.0 – 0.6 (gain × margin) |
| TTC ≤ 1.0 s | Full brake | 1.0 (AEB_FULL_BRAKE) |

- พิจารณาเฉพาะ obstacle ในเลนเดียวกัน (`AEB_LATERAL_THRESHOLD_M = 1.8 m`)
- AEB ส่ง `brake_override` ซึ่ง override.py ต้องยอมรับเสมอ (priority สูงสุด)
- ACC ทำงาน parallel: ปรับ `target_speed_ms` ตามรถข้างหน้า (`ACC_TIME_GAP_S = 1.8 s`)

### 6.3 LDW / LKA Pro (`src/safety/ldw.py`)

Lane Departure Warning + Lane Keeping Assist:

```
State machine (จาก ldw.py lines 8-11):
    IN_LANE  ──(|cte| > 0.3m)──▶  DEPARTING
    DEPARTING ──(|cte| > 0.4m)──▶  DEPARTED
    DEPARTED/DEPARTING ──(|cte| < 0.2m)──▶  IN_LANE
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `departure_threshold_m` | 0.3 m | เริ่มพิจารณาออกนอกเลน |
| `departure_hysteresis_m` | 0.1 m | กัน state flicker |
| `warning_duration_s` | 2.0 s | warning ค้างหลัง DEPARTED |
| `lka_pro_assist_gain` | 0.5 | corrective steer gain |
| `LDW_MIN_SPEED_MS` | 5.0 m/s | ไม่ทำงานตอนจอด |

- LKA Pro เพิ่ม corrective steering แบบ **additive** กับ MPC output (ไม่ replace)
- ไม่ทำงานเมื่อความเร็ว < 5 m/s (parking scenario)

### 6.4 Stuck Recovery (`src/safety/stuck_recovery.py`)

```
Recovery sequence (จาก stuck_recovery.py):
    stuck detected (280 frames, speed < 0.18 m/s, throttle > 0.35)
        → BRAKE (12 frames)
        → REVERSE (40 frames)
        → FORWARD
    จำกัด MAX_RECOVERY_ATTEMPTS = 3 ครั้ง
        → เกิน 3 ครั้ง: safe stop + driver handoff
```

| Parameter | Value |
|-----------|-------|
| `STUCK_THRESHOLD` | 280 frames |
| `STUCK_CONFIRM_FRAMES` | 5 |
| `STUCK_SPEED_MS` | 0.18 m/s |
| `STUCK_THROTTLE_MIN` | 0.35 |
| `BRAKE_FRAMES` | 12 |
| `REVERSE_FRAMES` | 40 |
| `MAX_RECOVERY_ATTEMPTS` | 3 |
| `COOLDOWN_AFTER_RECOVERY` | 100 frames |

---

## 7. Driver Monitoring System (DMS)

### 7.1 Driver State Detection

```
┌──────────────────────────────────────────────────────────┐
│                  DMS STATE MACHINE                       │
│                                                          │
│  [ATTENTIVE] ──gaze away > 3s──▶ [DISTRACTED]            │
│      ▲                            │                      │
│      │ gaze return < 1s           │ > 8s                 │
│      └────────────────            ▼                      │
│  [ATTENTIVE]              [UNRESPONSIVE]                 │
│                               │ > 15s                    │
│                               ▼                          │
│                         [SAFE_STOP trigger]              │
└──────────────────────────────────────────────────────────┘
```

| State | Detection Method | Warning Level | System Action |
|-------|------------------|---------------|---------------|
| ATTENTIVE | Gaze on road, hands on wheel | None | Normal operation |
| DISTRACTED | Gaze away > 3 s | Visual + audio | Increase following distance |
| UNRESPONSIVE | No gaze/hand response > 8 s | Escalating audio | Handoff request → safe stop at 15 s |
| DRIVER_ABSENT | No face detected | Critical | Immediate safe stop |

### 7.2 DMS Sensors

| Sensor | Purpose | Redundancy |
|--------|---------|------------|
| Cabin IR camera | Gaze tracking, head pose | Primary |
| Steering wheel torque sensor | Hands-on detection | Secondary |
| Seat occupancy sensor | Driver present | Tertiary |

> ใน CARLA simulation DMS ใช้ flag จำลอง (driver_attentive = True/False)
> บน vehicle จริงใช้ IR camera + torque sensor

---

## 8. Collision Avoidance Strategy

```
┌────────────────────────────────────────────────────────────────────┐
│              COLLISION AVOIDANCE — LAYERED DEFENSE                  │
│                                                                    │
│  Layer 1: Planning (MPC)                                           │
│    • Path planning หลีกเลี่ยง obstacle ตั้งแต่ตอนวาง trajectory     │
│    • Horizon = 5 s lookahead                                       │
│                                                                    │
│  Layer 2: ACC (proactive)                                          │
│    • ปรับ target speed ตามรถข้างหน้า (time gap 1.8 s)              │
│    • Decel limit 3.0 m/s² (นุ่มกว่า AEB)                           │
│                                                                    │
│  Layer 3: AEB (reactive)                                           │
│    • TTC < 2.5 s → proportional brake                              │
│    • TTC < 1.0 s → full brake (-5 m/s²)                           │
│    • ทำงานแม้ MPC ไม่เห็น obstacle (independent detection)         │
│                                                                    │
│  Layer 4: Safety Override (last resort)                            │
│    • หาก AEB ไม่ทำงาน → override.py ตรวจ collision_timeout        │
│    • Force emergency_deceleration = -5.0 m/s²                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 9. Operational Design Domain (ODD) Safety Constraints

> รายละเอียดเต็มดู [14_odd_definition.md](14_odd_definition.md)

| ODD Parameter | Value | OOD Action |
|---------------|-------|------------|
| Speed range | 0 – 30 km/h | Clamp + warning |
| Weather | Clear / light rain | Heavy rain → degrade → handoff |
| Road type | Marked lanes, flat | Unmarked → safe stop |
| Lighting | Day / street-lit night | Dark → degrade |
| Lane width | ≥ 3.0 m | Narrow → limp mode |
| Obstacle density | Low-medium | Dense → reduce speed |

**OOD detection → safety response:**
1. Perception confidence < 0.7 → DEGRADED
2. Lane marking not detected > 10 frames → DEGRADED → LIMP
3. Weather sensor (if equipped) reports heavy rain → DEGRADED
4. Any OOD persistent > 30 s → SAFE_STOP

---

## 10. FMEA (Failure Mode and Effects Analysis)

### 10.1 FMEA — Compute System

| Item | Function | Failure Mode | Cause | Effect | Severity | Detection | Mitigation | RPN |
|------|----------|--------------|-------|--------|----------|-----------|------------|-----|
| Compute A | MPC planning | Output freeze | SW hang, HW fault | No control update | High | Watchdog heartbeat miss | Compute B takes over safe stop | Low |
| Compute A | MPC planning | Wrong output | Perception error | Hazardous steer/speed | High | Compute B vote disagreement | Use B conservative + degrade | Med |
| Compute B | Safety monitor | Output freeze | SW hang | No safety check | Critical | Independent watchdog | Hardware watchdog → safe stop | Low |
| Vote gate | Command selection | Stuck on A | Logic fault | Unsafe command passes | Critical | B-side heartbeat | Force safe stop if B silent | Low |
| CAN bus | Communication | Loss | Wiring, ECU fail | No command to actuator | High | Timeout 100 ms | Safe stop (brake default-on) | Low |

### 10.2 FMEA — Brake System

| Item | Function | Failure Mode | Cause | Effect | Severity | Detection | Mitigation | RPN |
|------|----------|--------------|-------|--------|----------|-----------|------------|-----|
| Primary brake | Service brake | No pressure | Actuator stuck | No decel | Critical | Pressure sensor | Secondary brake within 200 ms | Low |
| Primary brake | Service brake | Reduced pressure | Fluid leak | Longer stopping | High | Pressure < command | Secondary + warning | Med |
| Secondary brake | Backup brake | No engage | Actuator stuck | No backup | Critical | Self-test at startup | Mechanical parking brake + driver | Low |
| Brake command | Signal path | Lost | CAN failure | No brake | Critical | Timeout | Brake default-on (fail-safe) | Low |

### 10.3 FMEA — Perception & ADAS

| Item | Function | Failure Mode | Cause | Effect | Severity | Detection | Mitigation | RPN |
|------|----------|--------------|-------|--------|----------|-----------|------------|-----|
| Camera | Lane detection | No lane | Glare, weather | LDW/LKA off | Med | Confidence < 0.5 | Degrade + handoff | Med |
| Camera | Obstacle detection | False negative | Occlusion, small object | AEB miss | High | TTC from radar (if equipped) | Override collision_timeout safe stop | Med |
| AEB | Emergency brake | False positive | Ghost obstacle | Unnecessary brake | Low | Multi-frame confirm (3 frames) | Limit decel to 3 m/s² unless critical | Low |
| LDW | Lane departure | No warning | CTE calc error | Unnoticed departure | Med | Cross-check with camera lane | LKA Pro corrective steer | Low |
| Stuck recovery | Unstick vehicle | Loop forever | Logic fault | Repeated reverse | Low | MAX_RECOVERY_ATTEMPTS = 3 | Safe stop after 3 attempts | Low |

### 10.4 FMEA — Power System

| Item | Function | Failure Mode | Cause | Effect | Severity | Detection | Mitigation | RPN |
|------|----------|--------------|-------|--------|----------|-----------|------------|-----|
| Power Rail A | Primary supply | Voltage drop | Battery drain, alternator | Compute A dies | High | Voltage monitor | Rail B via OR-diode | Low |
| Power Rail B | Secondary supply | Voltage drop | Battery drain | Compute B dies | Critical | Voltage monitor | Watchdog → safe stop on Rail A | Low |
| Both rails | All power | Total loss | Double fault | Full shutdown | Critical | N/A | Mechanical brake (spring-applied) | Low |

---

## 11. Incident Response & EDR (Event Data Recorder)

### 11.1 EDR Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        EDR (BLACK BOX)                                │
│                                                                      │
│  Trigger: any of                                                     │
│    • AEB activation (TTC < 2.5 s)                                    │
│    • Safety override activation (speed/steer/CTE clamp)              │
│    • Watchdog fault trigger                                          │
│    • Crash detected (acceleration spike > 5 g)                       │
│    • Manual emergency stop                                           │
│                                                                      │
│  Pre-event buffer: 30 s @ 20 Hz = 600 frames                         │
│  Post-event buffer: 15 s @ 20 Hz = 300 frames                        │
│                                                                      │
│  Recorded data:                                                      │
│    • Vehicle state (speed, steer, throttle, brake, accel, yaw)       │
│    • Perception output (lane confidence, obstacle list, TTC)         │
│    • MPC command (proposed steer, throttle, brake)                   │
│    • Safety override action (what was clamped, why)                  │
│    • ADAS state (AEB/ACC/LDW active flags)                           │
│    • DMS state (driver attentive/distracted)                         │
│    • Compute health (heartbeat timestamps, vote results)             │
│    • System mode (NORMAL/DEGRADED/LIMP/SAFE_STOP/PULL_OVER)          │
└──────────────────────────────────────────────────────────────────────┘
```

### 11.2 EDR Data Schema

| Field | Type | Unit | Source |
|-------|------|------|--------|
| timestamp | float | s | system clock |
| speed | float | m/s | vehicle state |
| steering_cmd | float | rad | MPC output |
| steering_actual | float | rad | override output |
| throttle_cmd | float | [0,1] | MPC output |
| throttle_actual | float | [0,1] | override output |
| brake_cmd | float | [0,1] | MPC output |
| brake_actual | float | [0,1] | override output |
| cte | float | m | path tracking |
| ttc | float | s | AEB calc |
| aeb_active | bool | - | aeb_acc.py |
| ldw_state | str | - | ldw.py |
| override_reason | str | - | override.py |
| compute_a_heartbeat | bool | - | watchdog |
| compute_b_heartbeat | bool | - | watchdog |
| system_mode | str | - | fail-safe FSM |
| dms_state | str | - | DMS |
| perception_confidence | float | [0,1] | perception |

### 11.3 Incident Response Workflow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. DETECT    │───▶│ 2. RECORD    │───▶│ 3. ANALYZE   │───▶│ 4. REPORT    │
│              │    │              │    │              │    │              │
│ Trigger event│    │ Freeze EDR   │    │ Post-process │    │ Export log   │
│ → start EDR  │    │ buffer       │    │ classify     │    │ to file      │
│              │    │              │    │ severity     │    │ + notify     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

1. **Detect**: EDR trigger condition met (see above)
2. **Record**: Freeze 30 s pre-event + 15 s post-event buffer to persistent storage
3. **Analyze**: Offline tool classifies incident (AEB save, override clamp, crash, etc.)
4. **Report**: Generate incident report with timeline, root cause hypothesis, recommended fix

---

## 12. Safety Validation & Testing

### 12.1 Fault Injection Testing

| Test ID | Fault Injected | Expected Response | Pass Criteria |
|---------|----------------|-------------------|---------------|
| FI-01 | Compute A freeze | Compute B safe stop | Stop within 500 ms |
| FI-02 | Compute A bad steer | Override clamps | Steer ≤ 0.45 rad |
| FI-03 | Primary brake fail | Secondary engages | Decel within 200 ms |
| FI-04 | Camera blackout | OOD → degrade | Degrade within 10 s |
| FI-05 | AEB false trigger | Limited decel | Decel ≤ 3 m/s² (non-critical) |
| FI-06 | Driver unresponsive | Safe stop | Stop within 20 s |
| FI-07 | Power Rail A loss | Rail B continues | No interruption |
| FI-08 | MPC output > speed limit | Override cuts throttle | Speed ≤ 30 km/h |

### 12.2 Simulation Scenarios (CARLA)

| Scenario | Town | Weather | Safety Function Tested |
|----------|------|---------|------------------------|
| Pedestrian cut-in | Town04 | Clear | AEB full brake |
| Lead vehicle hard brake | Town01 | Clear | ACC + AEB |
| Lane drift (distraction) | Town02 | Clear | LDW + LKA Pro |
| Stuck at curb | Town03 | Clear | Stuck recovery |
| Sensor blackout (fog) | Town05 | Fog | OOD → safe stop |
| Compute hang injection | Any | Clear | Watchdog safe stop |
| Speed limit exceed | Town04 | Clear | Override speed clamp |

---

## 13. Summary & Traceability

| Safety Goal | Module(s) | ASIL | Test |
|-------------|-----------|------|------|
| SG-01 (steer limit) | override.py | D | FI-02 |
| SG-02 (speed limit) | override.py | D | FI-08 |
| SG-03 (collision) | aeb_acc.py, override.py | D | FI-05, AEB scenario |
| SG-04 (lane departure) | ldw.py | B | LDW scenario |
| SG-05 (MPC bypass prevention) | override.py + watchdog | D | FI-01 |
| SG-06 (stuck) | stuck_recovery.py | A | Stuck scenario |
| SG-07 (driver monitoring) | DMS | B | FI-06 |

> **Design principle สำคัญ**: ระบบนี้ไม่ใช้ reinforcement learning ใด ๆ
> ทุก safety decision มาจาก deterministic rule, state machine, หรือ classical
> perception — สามารถ verify และ trace ได้ 100% ตามข้อกำหนด ISO 26262

---

*สิ้นสุดเอกสาร 07 — Safety & Redundancy*
