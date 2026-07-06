# Coding Standards — FrontViewPredictiveControl

> ไฟล์นี้เป็นมาตรฐานอ้างอิงสำหรับทุกคนที่เขียน/แก้โค้ดใน repo
> ใช้คู่กับ `agent/*.md` (subagent บทบาท) และ `WORKLOG.md` (บันทึกการเปลี่ยนแปลง)

---

## 1. โครงสร้างโปรเจกต์

```
FrontViewPredictiveControl/
├── main.py                  # entry point wrapper (เรียก src/main.py)
├── pyproject.toml           # build + pytest + ruff config
├── config/
│   ├── __init__.py          # config loader (flatten YAML → module constants)
│   └── default.yaml         # source of truth สำหรับทุก parameter
├── src/                     # โค้ด production ทั้งหมด
│   ├── main.py              # CARLAMPCSystem + main()
│   ├── pipeline.py          # LKAPipeline: Perception → Fusion → Reference → MPC → Safety
│   ├── carla_input_output.py          # CARLA waypoint/geometry utilities
│   ├── state.py             # FrameState dataclass (shared state)
│   ├── algorithms/                 # algorithm layer
│   │   ├── lane_keep_assist_step.py      # LKAStep — main step orchestrator
│   │   ├── reference.py     # reference path generation
│   │   └── fusion.py        # WP + UNet fusion
│   ├── control/             # control logic
│   │   ├── lane_mpc.py      # MPC solver (CasADi/IPOPT)
│   │   └── pure_pursuit.py  # Pure Pursuit fallback
│   ├── perception/          # lane detection & AI
│   │   ├── lane_trajectory.py
│   │   ├── lane_detector.py
│   │   ├── birds_eye_view_lane_pipeline.py
│   │   ├── spline_lane_fitting.py
│   │   ├── kalman_lane_tracker.py
│   │   ├── ego_lane_mask.py
│   │   ├── ego_lane_tracker.py
│   │   ├── road_perception.py
│   │   ├── classical/       # classical detector (no GPU)
│   │   └── unet/            # UNet-specific
│   ├── safety/              # safety systems
│   │   ├── override.py      # SafetyOverride
│   │   └── stuck_recovery.py
│   ├── core/                # core infrastructure
│   │   ├── carla_interface.py
│   │   └── mpc_runner.py
│   ├── managers/            # system managers
│   │   ├── control_manager.py
│   │   ├── perception_manager.py
│   │   ├── carla_manager.py
│   │   └── display_manager.py
│   ├── gui/                 # visualization
│   │   ├── dashboard.py
│   │   ├── adas_dashboard.py
│   │   ├── main_window.py
│   │   ├── panels/
│   │   └── ...
│   ├── temporal/            # temporal processing
│   │   └── lane_lstm.py
│   ├── bridge/              # bridge layer
│   │   ├── obstacles.py
│   │   └── transforms.py
│   └── utils/               # utilities
│       ├── type_hints.py
│       └── device_utils.py
├── tests/
│   ├── conftest.py          # sys.path setup
│   ├── unit/                # unit tests (no CARLA)
│   └── integration/         # integration tests (CARLA required)
├── dsunet_training/         # DSUNet training pipeline
├── scripts/                 # shell + utility scripts
├── docs/                    # documentation
├── agent/                   # subagent team definitions
├── .github/
│   ├── workflows/ci.yml     # GitHub Actions CI
│   ├── CONTRIBUTING.md
│   └── pull_request_template.md
└── WORKLOG.md               # บันทึกการเปลี่ยนแปลง
```

---

## 2. กฎการตั้งชื่อไฟล์

| กฎ | ตัวอย่างที่ถูก | ตัวอย่างที่ผิด | เหตุผล |
|----|---------------|---------------|--------|
| ใช้ `snake_case.py` | `lane_mpc.py` | `LaneMPC.py`, `laneMpc.py` | PEP 8 |
| ชื่อบอก **หน้าที่** ไม่ใช่ implementation detail | `spline_lane_fitting.py` | `improved_lane_fitting.py` | "improved" subjective, "spline" บอกวิธีการ |
| ห้ามมี codename/project name ในชื่อ | `dashboard.py` | `apollo_dashboard.py` | "apollo" เป็น codename ไม่บอกหน้าที่ |
| ห้ามมี implementation detail ในชื่อ | `birds_eye_view_lane_pipeline.py` | `gpu_bev_transform.py` | "gpu" เป็น implementation detail |
| ห้าม redundant prefix ซ้ำกับชื่อโฟลเดอร์ | `safety/override.py` | `safety/safety_override.py` | "safety_" ซ้ำกับโฟลเดอร์ |
| ห้ามชื่อกว้างเกินไป | `algorithms/lane_keep_assist_step.py` | `algorithms/step.py` | "step" ไม่บอกว่า step อะไร |
| ไฟล์ test ตั้งชื่อตรงกับโมดูล | `test_override.py` สำหรับ `safety/override.py` | `test_safety_override.py` | ลด redundant prefix |

---

## 3. กฎการตั้งชื่อในโค้ด

### Class / Function / Variable

| ประเภท | รูปแบบ | ตัวอย่าง |
|--------|-------|---------|
| Class | `PascalCase` | `LaneMPC`, `SafetyOverride`, `StuckRecovery` |
| Function / Method | `snake_case` | `compute_steering()`, `apply_safety_override()` |
| Variable | `snake_case` | `speed_ms`, `heading_err`, `lane_conf` |
| Constant (module-level) | `UPPER_SNAKE` | `MPC_DT`, `MPC_MAX_STEER`, `CAM_W` |
| Private member | `_prefix` | `self._prev_steer`, `self._phase` |
| Dataclass | `PascalCase` | `FrameState`, `MPCConfig`, `TrajectoryOutput` |

### ตัวแปรคณิตศาสตร์

ใช้ชื่อที่อ่านง่าย พร้อมคอมเมนต์สัญลักษณ์:

```python
delta = 0.0  # steering angle δ (rad)
psi = 0.0    # heading angle ψ (rad)
kappa = 0.0  # curvature κ (1/m)
```

ห้ามใช้สัญลักษณ์ตรงๆ เป็นชื่อตัวแปร: `δ`, `ψ`, `κ`

---

## 4. กฎการ import

```python
# ✅ ถูก — absolute import จาก package ชื่อ
from control.lane_mpc import LaneMPC, MPCConfig
from safety.override import SafetyOverride
from perception.spline_lane_fitting import BSplineLaneFitter

# ✅ ถูก — relative import ภายใน package เดียวกัน
from .reference import get_reference_path
from ..utils.type_hints import ControlCommand

# ❌ ผิด — import จากไฟล์ที่ถูกลบ
from run_unet_mpc import Dashboard
from adas_v2.fusion import apply_fusion

# ❌ ผิด — wildcard import (ยกเว้น config)
from control.lane_mpc import *  # ห้าม
from config import *  # อนุญาต (เพราะ config เป็น flat constants)
```

### Layer boundary — ห้ามข้าม

| ฝั่ง | ห้าม import จาก |
|------|-----------------|
| `control/`, `algorithms/`, `safety/`, `core/` | `gui/`, `pygame`, `PyQt5` |
| `perception/` | `control/` โดยตรง (ต้องผ่าน `managers/` หรือ `state.py`) |
| `safety/` | `control/` (ต้องรับค่าผ่าน parameter) |

---

## 5. กฎ config — `config/default.yaml` เป็น source of truth

```yaml
# config/default.yaml
mpc:
  MPC_DT: 0.1          # ✅ ค่าอยู่ใน YAML
  MPC_MAX_STEER: 0.7
```

```python
# ❌ ผิด — hardcode ใน .py
MPC_DT = 0.1  # ห้าม! ต้องอยู่ใน config/default.yaml

# ✅ ถูก — โหลดจาก config
from config import MPC_DT
```

กฎ:
- ทุก parameter ที่ user ปรับได้ ต้องอยู่ใน `config/default.yaml`
- ห้าม hardcode ค่าใหม่ใน `.py` ไฟล์
- ห้ามมี fallback ค่าใน `config/__init__.py` — ถ้า key ขาดให้ error ทันที
- ถ้าเปลี่ยน parameter ใน YAML ต้องเช็คว่าไม่มีที่อื่น hardcode ค่าเดิมซ้ำ

---

## 6. กฎ docstring

```python
def compute_steering(self, cte: float, heading_err: float, speed_ms: float) -> float:
    """Compute steering angle using Pure Pursuit geometry.

    Args:
        cte: Cross-track error (m), positive = right of center.
        heading_err: Heading error (rad), positive = pointing right.
        speed_ms: Current speed (m/s).

    Returns:
        Steering angle in radians, clamped to [-max_steer, max_steer].
    """
```

กฎ:
- ทุก public function/class ต้องมี docstring
- อธิบาย **หน่วย** ของทุก parameter (m, rad, m/s, km/h, 1/m)
- อธิบาย **sign convention** (positive = right? left?)
- สำหรับสมการคณิตศาสตร์ ใส่สมการใน docstring พร้อมสัญลักษณ์

---

## 7. กฎ type hints

```python
# ✅ ถูก
def solve(self, x0: float, y0: float, psi0: float, v0: float,
          v_ref: float, cte: float, heading_err: float) -> Tuple[float, float, str]:

# ❌ ผิด
def solve(self, x0, y0, psi0, v0, v_ref, cte, heading_err):
```

ใช้ type hints จาก `utils/type_hints.py`:
- `ControlCommand`, `PerceptionResult`, `SafetyStatus`, `VehicleState`
- `CameraFrame`, `DetectionMode`, `ControlMode`

---

## 8. กฎ error handling

```python
# ✅ ถูก — handle จุดที่จริงๆ ต้อง handle ไม่ใช่ try/catch ทุกบรรทัด
try:
    sol = self.solver(...)
    steer, accel = parse_solution(sol)
    return steer, accel, "Solve_Succeeded"
except Exception as e:
    logger.warning("MPC solver failed: %s", e)
    return self._pp_fallback(cte, heading_err, v0, v_ref)

# ❌ ผิด — try/catch ทุกบรรทัด
try:
    x = compute_x()
    try:
        y = compute_y()
        try:
            result = x + y
        except:
            pass
    except:
        pass
except:
    pass
```

กฎ:
- Error ที่คาดเดาได้ (solver fail, file not found) → handle + log + fallback
- Error ที่ไม่คาดเดาได้ → ปล่อยให้ crash (อย่ากลืน error)
- ห้าม `except: pass` เด็ดขาด

---

## 9. กฎ logging

```python
import logging
logger = logging.getLogger(__name__)

# ✅ ถูก
logger.info("MPC solve succeeded: steer=%.3f accel=%.3f", steer, accel)
logger.warning("MPC solver failed, using PP fallback: %s", e)

# ❌ ผิด
print("MPC solve succeeded")  # ห้ามใช้ print ใน production code
print(f"steer={steer}")       # ห้าม f-string ใน logger (ใช้ % formatting)
```

กฎ:
- ใช้ `logging` module ไม่ใช้ `print()` ใน production code
- ใช้ `%` formatting ใน logger ไม่ใช้ f-string (lazy evaluation)
- Level: `DEBUG` = detail, `INFO` = normal operation, `WARNING` = fallback/recover, `ERROR` = ล้มเหลว

---

## 10. กฎ test

```
tests/
├── conftest.py              # sys.path setup (src/ + root)
├── unit/                    # unit test (no CARLA, no GPU, no model file)
│   ├── test_lane_mpc.py
│   ├── test_reference.py
│   ├── test_override.py
│   ├── test_stuck_recovery.py
│   └── test_spline_lane_fitting.py
└── integration/             # integration test (CARLA required)
    ├── test_integration.py
    ├── test_carla_input_output.py
    └── test_carla_manager.py
```

กฎ:
- ชื่อ test file ตรงกับชื่อโมดูล: `test_override.py` สำหรับ `safety/override.py`
- Unit test ต้องรันได้โดยไม่ต้องการ CARLA/GPU/model file
- Integration test ใช้ marker `@pytest.mark.integration`
- ทุก bug fix ต้องมี regression test
- Safety-critical path ต้องมี test ครอบคลุม edge case

---

## 11. กฎ commit (Conventional Commits)

```
<type>(<scope>): <subject>

<body>

<footer>
```

| type | ใช้เมื่อ |
|------|---------|
| `feat` | เพิ่ม feature ใหม่ |
| `fix` | แก้ bug |
| `refactor` | เปลี่ยนโครงสร้างโค้ดโดยไม่เปลี่ยน behavior |
| `test` | เพิ่ม/แก้ test |
| `docs` | เปลี่ยน documentation |
| `chore` | งานบ้าน (dependencies, config) |

| scope | โมดูล |
|-------|-------|
| `control` | `src/control/` |
| `perception` | `src/perception/` |
| `safety` | `src/safety/` |
| `alg` | `src/algorithms/` |
| `core` | `src/core/`, `src/main.py`, `src/pipeline.py` |
| `gui` | `src/gui/` |
| `config` | `config/` |
| `test` | `tests/` |
| `ci` | `.github/` |

กฎ:
- Subject ≤ 72 ตัวอักษร, ไม่มี period ท้ายบรรทัด
- Body อธิบาย **อะไรเปลี่ยน + ทำไม**
- มี `BREAKING CHANGE:` ใน footer ถ้ากระทบ API

---

## 12. กฎ branch

| branch | ใช้สำหรับ |
|--------|-----------|
| `main` | stable, production-ready |
| `project-ADAS` | active development |
| `feature/<name>` | feature ใหม่ จาก `project-ADAS` |
| `fix/<name>` | bugfix จาก `project-ADAS` |
| `refactor/<name>` | refactor จาก `project-ADAS` |

---

## 13. กฎ CI

GitHub Actions (`.github/workflows/ci.yml`) รันทุก push/PR:

1. **Lint** (ruff) — `ruff check . --select=E,F,W --ignore=E501,E402`
2. **Smoke imports** — ตรวจว่า import หลักไม่ error
3. **Unit tests** — `pytest tests/unit/ -v`
4. **Config/pipeline tests** — `continue-on-error: true` (มี known failures)

ห้าม merge PR ถ้า CI ไม่ผ่าน (ยกเว้น `continue-on-error` steps)

---

## 14. กฎ safety-critical code

สำหรับโค้ดใน `safety/` และจุดที่ MPC ส่งต่อไป actuator:

- ทุกค่า steer/throttle/brake ต้องผ่าน constraint check ก่อนส่งออก
- `|δ| ≤ MPC_MAX_STEER` และ rate limit ต้องไม่ถูกละเมิด
- MPC solver fail ต้อง fallback ไป Pure Pursuit ได้ ไม่ throw exception ค้างระบบ
- Safety override ต้องเบรกจริงเมื่อ CTE > threshold
- Stuck recovery ต้องกลับสู่ idle ได้ ไม่ loop ค้าง
- ห้ามมี edge case ที่ปล่อยค่า steer/throttle/brake ที่ไม่ผ่าน constraint ออกไปได้

---

## 15. กฎ real-time constraint

- Control loop: `MPC_DT = 0.1s` (10 Hz)
- Camera input: 20 FPS
- MPC solve time ต้อง < 80ms (เหลือ 20ms สำหรับ perception + safety)
- ห้ามมี blocking call, busy-loop, หรือ heavy allocation ในลูปหลัก
- Pure Pursuit fallback ต้องเร็วกว่า MPC_DT เสมอ

---

## 16. อ้างอิง

| ไฟล์ | หน้าที่ |
|------|--------|
| `agent/README.md` | บทบาท subagent 6 ตัว |
| `agent/orchestrator.md` | orchestrator: วางแผน + มอบหมายงาน |
| `agent/dev-control.md` | dev-control: control/ algorithms/ safety/ core/ |
| `agent/dev-ai.md` | dev-ai: perception/ dsunet_training/ temporal/ |
| `agent/tech-lead.md` | tech-lead: architecture review |
| `agent/qa-reviewer.md` | qa-reviewer: test + safety edge case |
| `agent/git-ops.md` | git-ops: commit/push/PR |
| `WORKLOG.md` | บันทึกการเปลี่ยนแปลงทุกงาน |
| `.github/CONTRIBUTING.md` | วิธี setup + รัน test + branch strategy |
| `.github/pull_request_template.md` | PR template พร้อม safety checklist |
| `pyproject.toml` | build + pytest + ruff config |
| `config/default.yaml` | ทุก parameter (source of truth) |
