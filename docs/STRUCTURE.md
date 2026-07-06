# โครงสร้างโปรเจกต์ carla_mpc_classical

**Workflow & System Design (หลัก QA/QC):** [WORKFLOW_SYSTEM_DESIGN_QAQC.md](WORKFLOW_SYSTEM_DESIGN_QAQC.md)  
**Back to Basic — การจับเลนและตีเส้นแบ่ง Phase:** [BACK_TO_BASIC_LANE_AND_PHASE.md](BACK_TO_BASIC_LANE_AND_PHASE.md)  
**Apollo MPC + GUI design + Line detection:** [APOLLO_MPC_GUI_AND_LINE_DETECTION.md](APOLLO_MPC_GUI_AND_LINE_DETECTION.md)  
**Lane ระยะไกล (Display + UNet):** ดูหัวข้อ "Lane ระยะไกล" ใน [../README.md](../README.md) — REF_PATH_DISPLAY_LOOKAHEAD_M, debug mask, retrain options.

## สรุปการใช้งาน (Entry points)

### วิธีรันที่แนะนำ (จาก root โปรเจกต์ CARLA_0.9.16)

1. **เปิด CARLA ให้เห็นหน้าต่างซิม (optional แต่แนะนำ)**  
   จาก root: `./run_carla_visible.sh`  
   รอโหลดแมปเสร็จ แล้วรัน ADAS ในเทอร์มินัลอีกอัน

2. **รัน ADAS + Dashboard (กล้อง + lane overlay)**  
   จาก root: `./run_adas.sh --no-carla`  
   สคริปต์นี้เรียก `carla_mpc_classical/adas/run.py` → `run_adas_full.py` → `run_unet_mpc.main()`  
   แสดง Dashboard (Pygame) ภาพกล้อง + lane overlay  
   ใช้ค่าคงที่จาก **config.yaml** (โหลดผ่าน config.py — แก้ค่าที่ config.yaml ที่เดียว)

3. **รัน ADAS โดยให้สคริปต์ start CARLA เอง (background)**  
   จาก root: `./run_adas.sh`  
   จะ start CARLA ในพื้นหลัง; อาจไม่เห็นหน้าต่างซิม

4. **เลือก Dashboard**  
   ส่ง `--dashboard pygame` (default) หรือ `--dashboard apollo` ผ่าน `run_adas_full.py`

| วิธีจับเลน | คำสั่งรัน | หมายเหตุ |
|------------|-----------|----------|
| **Classical** (edge + sliding window) | `./scripts/run.sh classical` หรือ `python scripts/run_classical_mpc.py --town Town04` | ไม่ใช้ ML, ไม่ต้องมี model |
| **UNet** (โมเดลที่ train มา) | `./run_adas.sh --no-carla` (จาก root) หรือ `./scripts/run.sh unet` | ต้องมี model/lane_unet_final.pth; config จาก config.yaml |
| **เทส P1–P5** (phase pass rate) | `python scripts/run_test_phase1_phase5.py --town Town04 --frames 200` | ต้องเปิด CARLA ก่อน; รายงาน P1..P5 pass % |

---

## โครงสร้างโฟลเดอร์ (จัดระเบียบ)

```
carla_mpc_classical/
├── README.md
├── config.yaml              # ค่าคงที่รวม (แก้ที่นี่); config.py โหลดจากไฟล์นี้
├── config.py                # โหลด config.yaml + PROJECT_ROOT, UXColors
├── state.py
├── carla_input_output.py
├── pipeline.py              # LKAPipeline (perception → MPC → safety)
├── run_unet_mpc.py          # Entry UNet + Dashboard
├── run_adas_full.py
│
├── perception/
│   ├── classical/           # [Classical] จับเลนแบบไม่ใช้ ML
│   │   ├── __init__.py      # → ClassicalLane
│   │   └── detector.py      # edge → warp → sliding window → poly fit
│   ├── unet/                 # [UNet] จับเลนด้วยโมเดล
│   │   └── __init__.py      # → LaneDetector, LaneUNet
│   ├── edge_detection.py    # shared: threshold, blur, mag_thresh (ใช้โดย classical)
│   ├── classical_lane.py   # re-export จาก perception.classical (backward compat)
│   ├── lane_detector.py     # LaneDetector, LaneUNet (unet path)
│   ├── lane_trajectory.py   # LaneTrajectoryPipeline (BEV + Kalman + ego tracker)
│   ├── lane_clustering.py
│   ├── ego_lane_tracker.py
│   ├── geometric_validation.py
│   ├── lane_validator.py
│   ├── road_perception.py   # Legacy UNet row-wise (ใช้ LaneDetector)
│   └── ...
│
├── control/
│   └── lane_mpc.py          # MPC (adaptive weight + steer jerk)
├── temporal/
│   └── lane_lstm.py
├── safety/
│   └── safety_override.py
├── utils/
├── adas/
│
├── scripts/
│   ├── run.sh               # classical | unet
│   ├── run_lka.sh           # alias จุดเข้าเดียวกัน
│   └── run_classical_mpc.py # classical only (ไม่มี UNet)
│
├── docs/
│   ├── WORKFLOW_SYSTEM_DESIGN_QAQC.md   # Workflow + System Design หลัก QA/QC
│   ├── BACK_TO_BASIC_LANE_AND_PHASE.md   # การจับเลนและตีเส้นแบ่ง phase ทีละขั้น
│   ├── STRUCTURE.md                     # ไฟล์นี้
│   ├── LANE_LEVEL_FEATURES.md
│   └── ...
│
├── tests/                               # Unit tests (pytest; ไม่ใช้ CARLA server/GPU)
│   ├── test_reference.py                # alg.reference
│   ├── test_carla_input_output.py     # carla_input_output.waypoints_to_cte_heading
│   ├── test_lane_mpc.py                 # control.lane_mpc
│   └── test_safety_override.py          # safety.safety_override
│
└── model/
    └── lane_unet_final.pth  # ใช้กับ run unet
```

---

## Classical vs UNet

### Classical (`perception/classical`)

- **ที่มา:** `perception/classical/detector.py` (เดิม `classical_lane.py`)
- **วิธี:** Edge (HLS + Sobel) → perspective warp → sliding window → polynomial fit
- **ใช้เมื่อ:** ไม่มี model, ทดสอบ pipeline

### UNet (`perception/unet`)

- **ที่มา:** `perception/lane_detector.py` (LaneUNet + LaneDetector)
- **วิธี:** โมเดล U-Net → mask → LaneTrajectoryPipeline (BEV + clustering + ego tracker + Kalman)
- **ใช้เมื่อ:** มี model/lane_unet_final.pth

---

## วิธีรัน

```bash
# จาก root โปรเจกต์ (CARLA_0.9.16)
# 1) เปิด CARLA ให้เห็นหน้าต่าง (เทอร์มินัลที่ 1)
./run_carla_visible.sh

# 2) รัน ADAS + Dashboard (เทอร์มินัลที่ 2)
./run_adas.sh --no-carla

# Classical (ไม่ใช้ model) — จากโฟลเดอร์ carla_mpc_classical
./scripts/run.sh classical --town Town04
# หรือ
python scripts/run_classical_mpc.py --town Town04

# UNet (ใช้ model)
./scripts/run.sh unet --town Town04
./scripts/run_lka.sh unet --model model/lane_unet_final.pth --town Town04
```

**Config:** ค่าคงที่ทั้งหมด (camera, waypoints, fusion, MPC, safety) อยู่ที่ `config.py` ที่เดียว
