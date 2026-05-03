# ✅ การแก้ไขที่ทำแล้ว (Applied Fixes)

## 📋 สรุปปัญหาและการแก้ไข

### ปัญหาที่พบ:
1. ❌ MPC คำนวณ steering สูงเกินไป (±32.6° > 25.8°)
2. ❌ Polynomial fitting มี oscillation
3. ❌ MPC weights เน้น CTE มากเกินไป
4. ❌ Steering smoothing น้อยเกินไป

---

## ✅ Fix 1: เพิ่ม B-Spline Fitting Support

### การเปลี่ยนแปลง:
```python
# lane_trajectory.py
# เพิ่ม import B-spline fitter
from .improved_lane_fitting import BSplineLaneFitter
BSPLINE_AVAILABLE = True

# เพิ่มใน LaneTrajectoryPipeline.__init__
self._bspline_fitter = BSplineLaneFitter() if BSPLINE_AVAILABLE else None
```

### ประโยชน์:
- ✅ Local control (เปลี่ยนจุดหนึ่งไม่กระทบทั้งเส้น)
- ✅ Smoother curvature (ไม่มี oscillation)
- ✅ เหมาะกับโค้งแหลม
- ✅ Natural representation ของเรขาคณิตถนน

### สถานะ:
- ✅ Code พร้อมใช้งานใน `improved_lane_fitting.py`
- ⏳ ต้อง integrate เข้า polynomial_fit() function
- ⏳ ต้องทดสอบกับ CARLA

---

## ✅ Fix 2: ลด MPC CTE Weight

### การเปลี่ยนแปลง:
```yaml
# config.yaml - MPC weights
MPC_W_CTE: 20.0         # ลดจาก 30 → 20 (-33%)
MPC_W_HEADING: 50.0     # เพิ่มจาก 40 → 50 (+25%)
MPC_W_STEER: 200.0      # เพิ่มจาก 150 → 200 (+33%)
MPC_W_STEER_RATE: 400.0 # เพิ่มจาก 300 → 400 (+33%)
MPC_W_STEER_JERK: 200.0 # เพิ่มจาก 150 → 200 (+33%)
```

### เหตุผล:
- **CTE weight สูง** → MPC พยายามกลับเข้าเลนเร็วเกินไป → steering มาก
- **Heading weight ต่ำ** → ไม่ smooth heading → oscillation
- **Steer/Rate/Jerk weight ต่ำ** → ไม่ penalize การเปลี่ยนแปลง steering

### ผลที่คาดหวัง:
- ✅ Steering angle ลดลง 20-30%
- ✅ Oscillation ลดลง
- ✅ Smoother control
- ⚠️ CTE อาจเพิ่มขึ้นเล็กน้อย (ยอมรับได้)

---

## ✅ Fix 3: เพิ่ม Steering Smoothing

### การเปลี่ยนแปลง:
```yaml
# config.yaml - Steering smoothing
STEER_SMOOTH_ALPHA: 0.80           # เพิ่มจาก 0.70 → 0.80
STEER_MAX_DELTA_PER_FRAME: 0.03    # ลดจาก 0.05 → 0.03
STEER_MAX_DELTA_HIGH_SPEED: 0.02   # ลดจาก 0.03 → 0.02
```

### สูตร:
```python
# EMA smoothing
steer = 0.80 * prev_steer + 0.20 * raw_steer

# Rate limiting
steer = clip(steer, prev_steer - 0.03, prev_steer + 0.03)
```

### ผลที่คาดหวัง:
- ✅ Steering เปลี่ยนแปลงช้าลง
- ✅ ลด jerk และ oscillation
- ✅ Passenger comfort ดีขึ้น
- ⚠️ Response time ช้าลงเล็กน้อย

---

## 📊 สรุปการเปลี่ยนแปลง

### Before (ก่อนแก้):
```
MPC_W_CTE: 30.0
MPC_W_HEADING: 40.0
MPC_W_STEER: 150.0
MPC_W_STEER_RATE: 300.0
MPC_W_STEER_JERK: 150.0
STEER_SMOOTH_ALPHA: 0.70
STEER_MAX_DELTA_PER_FRAME: 0.05
STEER_MAX_DELTA_HIGH_SPEED: 0.03

Result:
- Steering: ±32.6° (เกิน limit 25.8°)
- Oscillation: สูง
- Control: รุนแรง
```

### After (หลังแก้):
```
MPC_W_CTE: 20.0          ⬇️ -33%
MPC_W_HEADING: 50.0      ⬆️ +25%
MPC_W_STEER: 200.0       ⬆️ +33%
MPC_W_STEER_RATE: 400.0  ⬆️ +33%
MPC_W_STEER_JERK: 200.0  ⬆️ +33%
STEER_SMOOTH_ALPHA: 0.80 ⬆️ +14%
STEER_MAX_DELTA_PER_FRAME: 0.03  ⬇️ -40%
STEER_MAX_DELTA_HIGH_SPEED: 0.02 ⬇️ -33%

Expected Result:
- Steering: ±20-25° (ภายใน limit)
- Oscillation: ลดลง 50-70%
- Control: smooth มากขึ้น
```

---

## 🎯 ผลที่คาดหวัง (Expected Improvements)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Max Steering | ±32.6° | ±20-25° | -23 to -30% |
| Steering Oscillation | High | Low | -50 to -70% |
| CTE | 0.3-0.5m | 0.4-0.6m | +20% (ยอมรับได้) |
| Passenger Comfort | ⭐⭐ | ⭐⭐⭐⭐ | +100% |
| Control Smoothness | ⭐⭐ | ⭐⭐⭐⭐⭐ | +150% |

---

## 📝 สิ่งที่ยังต้องทำ (Remaining Tasks)

### Priority 1: Critical
1. ⏳ **Integrate B-spline fitting**
   ```python
   # ใน polynomial_fit() function
   if self._bspline_fitter is not None:
       c_l, c_r, center = self._bspline_fitter.fit(left_xy, right_xy)
   else:
       c_l = _fit_poly_robust(left_xy[:, 0], left_xy[:, 1])
       c_r = _fit_poly_robust(right_xy[:, 0], right_xy[:, 1])
   ```

2. ⏳ **Test กับ CARLA**
   - รัน ADAS ด้วย config ใหม่
   - ดู steering angle log
   - เช็ค oscillation

3. ⏳ **Evaluate DSUNet model** (หลัง training เสร็จ ~1 ชม.)
   - Test IoU, Dice, F1
   - Compare กับ UNet เดิม
   - Integrate ถ้าดีกว่า

### Priority 2: Important
4. ⏳ **Fix BEV Transform**
   - ใช้ calibrated IPM
   - Compensate distortion

5. ⏳ **Tune Sliding Window**
   - ลด histogram weights ระยะไกล
   - ลด Gaussian kernel size

### Priority 3: Nice to have
6. ⏳ **Add Hysteresis ใน Fusion**
   - Smooth mode switching
   - Blend reference paths

---

## 🚀 วิธีทดสอบ (Testing Instructions)

### 1. รัน CARLA:
```bash
cd /home/supawich/Desktop/CARLA_0.9.16
./CarlaUE4.sh -windowed -carla-rpc-port=2000
```

### 2. รัน ADAS:
```bash
cd carla_mpc_classical
source venv/bin/activate
python adas/run.py
```

### 3. ดู Log:
```bash
# ดู steering angle
grep "steer=" adas.log | tail -20

# ดู steering limit warnings
grep "STEERING LIMIT" adas.log | wc -l

# ดู mode
grep "mode=" adas.log | tail -20
```

### 4. เปรียบเทียบ:
```
Before:
- Steering warnings: 50-100 ครั้ง/นาที
- Max steering: ±32.6°
- Oscillation: เห็นชัดเจน

After (Expected):
- Steering warnings: 0-10 ครั้ง/นาที
- Max steering: ±20-25°
- Oscillation: ลดลงมาก
```

---

## 📚 เอกสารเพิ่มเติม

- `PIPELINE_ISSUES_ANALYSIS.md` - วิเคราะห์ปัญหาทั้งระบบ
- `improved_lane_fitting.py` - B-spline implementation
- `LANE_FITTING_IMPROVEMENTS.md` - คำอธิบาย B-spline
- `FIX_SUMMARY.md` - สรุปปัญหาเดิม

---

## ✅ Checklist

- [x] ลด MPC CTE weight (30 → 20)
- [x] เพิ่ม MPC heading weight (40 → 50)
- [x] เพิ่ม MPC steer weights (150/300/150 → 200/400/200)
- [x] เพิ่ม steering smoothing (0.70 → 0.80)
- [x] ลด steering rate limits (0.05/0.03 → 0.03/0.02)
- [x] เพิ่ม B-spline fitter support
- [ ] Integrate B-spline ใน polynomial_fit()
- [ ] Test กับ CARLA
- [ ] Evaluate DSUNet model
- [ ] Fix BEV transform
- [ ] Tune sliding window

---

**สรุป:** แก้ไข 3 ส่วนหลักแล้ว (MPC weights, steering smoothing, B-spline support)  
**ต่อไป:** ทดสอบกับ CARLA และ integrate B-spline เข้า pipeline
