# 🚀 วิธีรัน ADAS Test

## ✅ การแก้ไขที่ทำแล้ว:

### 1. MPC Weights
- `MPC_W_CTE: 20.0` (ลดจาก 30)
- `MPC_W_HEADING: 50.0` (เพิ่มจาก 40)
- `MPC_W_STEER: 200.0` (เพิ่มจาก 150)
- `MPC_W_STEER_RATE: 400.0` (เพิ่มจาก 300)
- `MPC_W_STEER_JERK: 200.0` (เพิ่มจาก 150)

### 2. Steering Smoothing
- `STEER_SMOOTH_ALPHA: 0.80` (เพิ่มจาก 0.70)
- `STEER_MAX_DELTA_PER_FRAME: 0.03` (ลดจาก 0.05)
- `STEER_MAX_DELTA_HIGH_SPEED: 0.02` (ลดจาก 0.03)

### 3. B-Spline Support
- เพิ่ม `BSplineLaneFitter` พร้อมใช้งาน

---

## 📋 ขั้นตอนการทดสอบ:

### Step 1: เปิด CARLA (Terminal 1)
```bash
cd /home/supawich/Desktop/CARLA_0.9.16
./CarlaUE4.sh -windowed -carla-rpc-port=2000
```

### Step 2: รัน ADAS (Terminal 2)
```bash
cd /home/supawich/Desktop/CARLA_0.9.16/carla_mpc_classical
source venv/bin/activate  # หรือ activate venv ที่มี torch
python adas/run.py
```

---

## 📊 ผลที่คาดหวัง:

### Before (ก่อนแก้):
- ❌ Max Steering: ±32.6° (เกิน limit 25.8°)
- ❌ Steering warnings: 50-100 ครั้ง/นาที
- ❌ Oscillation: สูง
- ❌ Control: รุนแรง

### After (หลังแก้):
- ✅ Max Steering: ±20-25° (ภายใน limit)
- ✅ Steering warnings: 0-10 ครั้ง/นาที
- ✅ Oscillation: ลดลง 50-70%
- ✅ Control: smooth

---

## 🔍 วิธีดู Log:

```bash
# ดู steering angle
grep "steer=" adas.log | tail -20

# นับ steering warnings
grep "STEERING LIMIT" adas.log | wc -l

# ดู mode
grep "mode=" adas.log | tail -20

# ดู phases
grep "phases:" adas.log | tail -10
```

---

## 📈 DSUNet Training Status:

```
Epoch 8/100 (กำลังรัน)
Best IoU: 0.8737 ✅
Best Dice: 0.9313 ✅
Best Loss: 0.0392 ✅

เหลืออีก ~1.5 ชม.
```

---

## ⚠️ Troubleshooting:

### ถ้า CARLA ไม่เชื่อมต่อ:
```bash
# เช็คว่า CARLA รันอยู่ไหม
ps aux | grep CarlaUE4

# เช็ค port
netstat -an | grep 2000
```

### ถ้า torch ไม่มี:
```bash
# ใช้ venv ที่ถูก
source /path/to/correct/venv/bin/activate

# หรือ install torch
pip install torch torchvision
```

---

## 📝 Files ที่เกี่ยวข้อง:

- `config.yaml` - MPC weights และ steering smoothing
- `lane_trajectory.py` - B-spline fitter
- `FIXES_APPLIED.md` - สรุปการแก้ไขทั้งหมด
- `PIPELINE_ISSUES_ANALYSIS.md` - วิเคราะห์ปัญหา

---

**พร้อมทดสอบแล้ว!** เปิด CARLA แล้วรัน ADAS ได้เลยครับ 🚗
