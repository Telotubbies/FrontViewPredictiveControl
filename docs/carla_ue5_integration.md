# CARLA UE5 Integration — FrontViewPredictiveControl

> **Date:** 2026-07-06
> **CARLA version:** 0.10.0 (UE5.5)
> **Integration method:** Option A — CARLA Python API (ไม่ใช้ ROS2)

## Overview

FVPC เชื่อมต่อกับ CARLA UE5 ผ่าน **CARLA Python API** โดยตรง (เหมือนเดิม)
เปลี่ยนเฉพาะวิธีติดตั้ง Python package จาก local egg → pip wheel และ map จาก Town04 → Town10

## Prerequisites

### 1. CARLA UE5 Server

- รัน `CarlaUE5.exe` (หรือ `CarlaUE5.sh` บน Linux) ก่อนเริ่ม FVPC
- Default port: `2000` (RPC), `2001` (streaming)

### 2. Python Package

> `carla` ไม่ได้อยู่บน PyPI — ต้องติดตั้ง manual จาก CARLA release wheel

ดาวน์โหลด wheel จาก [CARLA releases](https://github.com/carla-simulator/carla/releases) (UE5 build)
แล้วติดตั้งด้วยคำสั่ง:

```bash
pip install carla-0.10.0-cp311-cp311-win_amd64.whl
```

> ไม่ต้องใช้ local egg หรือ `PythonAPI/` path — wheel ติดตั้ง `.pyd` ลง site-packages โดยตรง

### 3. FVPC Dependencies

```bash
pip install -e .
# หรือ
pip install -r requirements.txt
```

## Configuration

ไฟล์ `config/default.yaml` มี section ใหม่:

```yaml
carla:
  CARLA_MAP: Town10   # UE5 default (Town01-07 ถูกลบใน UE5 0.10.0)
```

Camera config คงเดิม:
- Resolution: 640×480
- FOV: 90°
- Pitch: -8°
- Control rate: 10 Hz (MPC_DT=0.1, MPC_N=10)

## วิธีรัน

### ใช้ entry point ใหม่ (แนะนำ)

```bash
# Default: localhost:2000, Town10
python scripts/run_carla_ue5.py

# ระบุ host/port
python scripts/run_carla_ue5.py --host 192.168.1.10 --port 2000

# ใช้ classical lane detection (ไม่ใช้ UNet)
python scripts/run_carla_ue5.py --classical --no-gui

# ระบุ map อื่น
python scripts/run_carla_ue5.py --town Town10_Opt
```

### ใช้ main.py เดิม

```bash
python src/main.py --town Town10
```

> **หมายเหตุ:** `main.py` มี `--town` arg แต่ไม่ได้ load_world อัตโนมัติ
> แนะนำใช้ `scripts/run_carla_ue5.py` ซึ่งโหลด map ก่อนเริ่ม pipeline

## ไฟล์ที่เปลี่ยน

| ไฟล์ | การเปลี่ยนแปลง |
|------|----------------|
| `src/core/carla_interface.py` | `setup_carla_paths()` ใช้ pip-installed carla แทน local egg |
| `src/managers/carla_manager.py` | ลบ local egg fallback, ใช้ `import carla` โดยตรง |
| `requirements.txt` | comment ออก `carla>=0.10.0` (ไม่มีใน PyPI ต้องติดตั้ง wheel เอง) |
| `pyproject.toml` | ย้าย `carla>=0.10.0` ไป optional-dependencies ในกลุ่ม `[carla]` |
| `config/default.yaml` | เพิ่ม `carla.CARLA_MAP: Town10` |
| `scripts/run_carla_ue5.py` | Entry point ใหม่สำหรับ CARLA UE5 |

## API Compatibility

CARLA Python API เข้ากันได้ 100% ระหว่าง 0.9.15 (UE4) และ 0.10.0 (UE5)
ไม่ต้องแก้ perception, control, safety, bridge, GUI modules

ดูรายละเอียดเพิ่มเติม: [docs/design/11_carla_ue5_migration.md](design/11_carla_ue5_migration.md)
