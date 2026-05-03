# เก็บผลและดูจากตาราง (carla_mpc_classical / UFLD+MPC)

ตอนนี้ระบบยังไม่แม่น — เก็บผลแต่ละรันแล้วดูจาก **CSV**, **events** และ **ตารางผล (results_matrix)** ได้ดังนี้

## 1. เก็บผลอัตโนมัติ (รันจาก carla_lstm_mpc_project)

เปิด CARLA ก่อน แล้วรัน:

```bash
cd carla_lstm_mpc_project
./scripts/run_and_collect_results.sh
```

จะได้:
- **CSV**: `logs/run_<timestamp>.csv` — แต่ละ frame มี cte_m, steer_carla, solver_status ฯลฯ
- **Events**: `logs/events_run_<timestamp>.json` — collision, lane_invasion, stuck
- **รายงาน C1–C5** พิมพ์บนจอ และ `logs/eval_run_<timestamp>.json`

ปรับ duration / town / speed ได้ด้วยตัวแปรสภาพแวดล้อม:

```bash
DURATION=60 TOWN=Town04 SPEED=20 ./scripts/run_and_collect_results.sh
```

## 2. เก็บผลด้วยมือ (ระบุ path CSV เอง)

```bash
cd carla_lstm_mpc_project
mkdir -p logs
python3 run_ufld_mpc.py --town Town04 --duration 120 \
  --log_csv logs/run_20250226_120000.csv
```

หลังจบจะได้ `logs/run_20250226_120000.csv` และ `logs/events_run_20250226_120000.json` (ชื่อคู่กัน)

## 3. ดูผลจากตาราง (results_matrix) และ C1–C5

- **ตารางผล**: เปิด `carla_lstm_mpc_project/results_matrix.md` ดู Scenario × C1–C5 ว่า S1–S5 ผ่านหรือยัง
- **ประเมินรันล่าสุด** (แทนที่ `<id>` ด้วย timestamp จากชื่อไฟล์ CSV):

```bash
python3 evaluate_run.py --csv logs/run_<id>.csv --events logs/events_run_<id>.json --report
```

จากรายงานนี้ไปอัปเดต results_matrix.md ว่าแต่ละ scenario เป็น ✅/❌

## 4. ดูจาก CSV โดยตรง

- **CTE (C1)**: คอลัมน์ `cte_m` — mean/max ต้องอยู่เกณฑ์ (mean < 0.30 m, max < 0.80 m)
- **Solver (C4)**: คอลัมน์ `solver_status` — นับว่าไม่ใช่ `Solve_Succeeded` กี่ %
- **Steer (C5)**: คอลัมน์ `steer_carla` — ใช้ rolling std ~3 s ต้อง < 0.08

ตัวอย่างดูด้วย pandas:

```python
import pandas as pd
df = pd.read_csv("logs/run_20250226_120000.csv")
print(df["cte_m"].abs().mean(), df["cte_m"].abs().max())
print((df["solver_status"] != "Solve_Succeeded").mean() * 100)
```
