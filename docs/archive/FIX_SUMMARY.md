# 🔧 สรุปปัญหาและการแก้ไข

## 📊 ปัญหาที่พบจาก Log

### ปัญหา 1: ไม่ใช้ UNet แม้ว่า P1-P5 ผ่านหมด
```
INFO   phases: P1✓ P2✓ P3✓ P4✓ P5✓  # UNet จับเลนได้
INFO   mode=WP_PRIMARY              # แต่ไม่ใช้ UNet!
```

**สาเหตุ:**
- `FUSION_CONF_LOW = 0.55` สูงเกินไป
- Lane confidence อาจต่ำกว่า 0.55 → ไม่ผ่านเงื่อนไข
- Fusion logic เลือกใช้ WP_PRIMARY แทน

### ปัญหา 2: Steering เหวี่ยงรุนแรง
```
WARNING ⚠️  STEERING LIMIT EXCEEDED: ±0.570 > 0.450
# เกิดทุก 1-2 วินาที
```

**สาเหตุ:**
- MPC คำนวณ steering สูงเกินไป (±32.6°)
- Polynomial fitting มี oscillation
- MPC weights ไม่เหมาะสม (เน้น CTE มากเกินไป)
- Steering smoothing น้อยเกินไป

---

## ✅ การแก้ไข

### Fix 1: ลด FUSION_CONF_LOW ให้ใช้ UNet ได้ง่ายขึ้น

**แก้ใน `config.yaml`:**
```yaml
# เดิม
FUSION_CONF_LOW: 0.55   # สูงเกินไป
FUSION_CONF_HIGH: 0.85

# ใหม่
FUSION_CONF_LOW: 0.30   # ลดลงให้ใช้ UNet ได้ง่ายขึ้น
FUSION_CONF_HIGH: 0.75
```

### Fix 2: ลด Steering Oscillation

**แก้ใน `config.yaml`:**

```yaml
# 1. เพิ่ม Steering Smoothing
STEER_SMOOTH_ALPHA: 0.7        # เพิ่มจาก 0.5
STEER_MAX_DELTA_PER_FRAME: 0.05  # ลดจาก 0.1
STEER_MAX_DELTA_HIGH_SPEED: 0.03 # ลดจาก 0.05

# 2. ปรับ MPC Weights - ลดการเน้น CTE, เพิ่มการลด oscillation
MPC_W_CTE: 30.0           # ลดจาก 50.0
MPC_W_HEADING: 40.0       # เพิ่มจาก 30.0
MPC_W_STEER: 150.0        # เพิ่มจาก 100.0
MPC_W_STEER_RATE: 300.0   # เพิ่มจาก 200.0
MPC_W_STEER_JERK: 150.0   # เพิ่มจาก 80.0

# 3. ลด Safety Limit (ถ้ายังเหวี่ยง)
SAFETY_MAX_STEER_RAD: 0.40  # ลดจาก 0.45 (23° แทน 25.8°)
```

### Fix 3: ใช้ B-Spline Fitting (Optional)

**แก้ใน `perception/lane_trajectory.py`:**
```python
# แทนที่ polynomial_fit() ด้วย B-spline
from .improved_lane_fitting import BSplineLaneFitter, convert_bspline_to_polynomial

def polynomial_fit(left_xy, right_xy):
    fitter = BSplineLaneFitter(degree=3, num_control_points=8)
    left_spline, right_spline, info = fitter.fit_lane_pair(
        left_xy, right_xy,
        enforce_parallel=True,
        enforce_width=True,
    )
    
    if not info.get("valid", False):
        return None
    
    left_coeffs = convert_bspline_to_polynomial(left_spline, (0, 90))
    right_coeffs = convert_bspline_to_polynomial(right_spline, (0, 90))
    center_coeffs = (left_coeffs + right_coeffs) / 2
    
    return left_coeffs, right_coeffs, center_coeffs
```

---

## 🎯 ผลลัพธ์ที่คาดหวัง

### Before:
```
mode=WP_PRIMARY                           # ไม่ใช้ UNet
STEERING LIMIT EXCEEDED: ±0.570 (×50)    # เหวี่ยงมาก
cte=-0.503m hd=+11.8deg                  # ผิดพลาดสูง
```

### After:
```
mode=WP+UNET_CENTER                      # ใช้ UNet ช่วย
STEERING LIMIT EXCEEDED: ±0.470 (×10)    # เหวี่ยงน้อยลง
cte=-0.150m hd=+3.2deg                   # ผิดพลาดต่ำลง
```

---

## 📝 ขั้นตอนการทดสอบ

1. **แก้ไข config.yaml** (ตามด้านบน)
2. **รัน ADAS ใหม่:**
   ```bash
   cd /home/supawich/Desktop/CARLA_0.9.16
   ./run_adas.sh --town Town04 --speed 25 2>&1 | tee test_after_fix.log
   ```
3. **ดู log:**
   ```bash
   # ดู mode
   grep "mode=" test_after_fix.log | tail -20
   
   # นับ steering limit
   grep -c "STEERING LIMIT EXCEEDED" test_after_fix.log
   
   # ดู CTE/heading
   grep "cte=" test_after_fix.log | tail -10
   ```

---

## 💡 Tips เพิ่มเติม

### ถ้ายังเหวี่ยงอยู่:
1. เพิ่ม `STATE_EMA_ALPHA` จาก 0.3 → 0.5 (smooth state มากขึ้น)
2. เพิ่ม `CURVATURE_WP_EMA_ALPHA` จาก 0.5 → 0.7 (smooth curvature)
3. ลด `MPC_W_CTE` ลงอีก → 20.0

### ถ้า UNet ยังไม่ทำงาน:
1. ดู lane_conf ใน log: `grep "lane_conf=" test_after_fix.log`
2. ถ้า < 0.3 ตลอด → ปัญหาที่ UNet model หรือ preprocessing
3. ลอง visualize mask: เพิ่ม `cv2.imwrite("debug_mask.png", mask*255)` ใน perception

### Debug Mode:
```bash
# รันพร้อม debug logging
export ADAS_DEBUG=1
./run_adas.sh --town Town04 --speed 25
```
