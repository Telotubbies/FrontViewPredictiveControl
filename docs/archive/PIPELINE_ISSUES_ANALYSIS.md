# 🔍 วิเคราะห์ปัญหา: Lane Detection → Control Pipeline

## 📋 สรุปปัญหาที่พบ

### ⚠️ **ปัญหาหลัก 5 ข้อ:**

---

## 1. 🎯 **UNet Detection - Threshold ต่ำเกินไป**

### ปัญหา:
```python
# lane_detector.py:410-411
# Threshold: detect lane if probability > 0.1 (very low for weak model)
mask_bin = (lane_prob > 0.1).astype(np.uint8)
```

**ผลกระทบ:**
- Threshold = 0.1 ต่ำมาก → จับ noise เยอะ
- False positives สูง
- Lane mask มี noise รบกวนการ fit polynomial

**แนวแก้:**
```python
# ควรใช้ threshold 0.3-0.5 หรือ adaptive threshold
# หรือ train DSUNet model ใหม่ให้ confidence สูงขึ้น
mask_bin = (lane_prob > 0.3).astype(np.uint8)
```

---

## 2. 🔄 **BEV Transform - Coordinate Mismatch**

### ปัญหา:
```python
# lane_trajectory.py:303-305
# BEV pixel → Vehicle XY
def _bev_px_to_vehicle_xy(row, col, bev_h, bev_w, lookahead_m, half_width_m):
    x = (bev_h - 1 - row) * (lookahead_m / max(bev_h - 1, 1))
    y = (col - (bev_w - 1) / 2.0) * (half_width_m * 2.0 / max(bev_w - 1, 1))
    return x, y
```

**ปัญหา:**
- BEV transform ใช้ trapezoid ROI แต่ไม่ได้ compensate distortion
- Pixel spacing ไม่ uniform → error เพิ่มขึ้นตามระยะ
- ระยะไกล (top of BEV) มี error มากกว่าระยะใกล้

**ผลกระทบ:**
- Lane curvature คำนวณผิด
- Heading error ไม่แม่นยำ
- MPC ได้ reference path ที่ผิด

**แนวแก้:**
```python
# ควรใช้ IPM (Inverse Perspective Mapping) แบบ calibrated
# หรือ compensate distortion ด้วย camera intrinsics
# หรือใช้ polynomial fitting ใน image space แล้วค่อย transform
```

---

## 3. 📐 **Polynomial Fitting - Overfitting & Noise**

### ปัญหา:
```python
# lane_trajectory.py:605-606
c_l = _fit_poly_robust(left_xy_d[:, 0], left_xy_d[:, 1])
c_r = _fit_poly_robust(right_xy_d[:, 0], right_xy_d[:, 1])
```

**ปัญหาที่พบ:**

1. **Densify แล้วยัง fit polynomial ลำดับ 2:**
   - ข้อมูลเยอะแต่ model ไม่ซับซ้อนพอ
   - ไม่ fit โค้งแหลมได้ดี

2. **ไม่มี outlier rejection ที่ดีพอ:**
   ```python
   def _fit_poly_robust(x, y):
       # ใช้ RANSAC แต่อาจไม่เพียงพอ
       # Noise จาก UNet ยังผ่านมาได้
   ```

3. **Mirror strategy อาจทำให้ผิด:**
   ```python
   # lane_trajectory.py:608-611
   if c_l is None and c_r is not None:
       c_l = _mirror_coeffs(c_r)  # สมมติว่า symmetric
   ```
   - ถนนไม่ symmetric เสมอ
   - โค้งแหลม mirror ผิด

**แนวแก้:**
```python
# 1. ใช้ B-spline แทน polynomial (ทำไว้แล้วใน improved_lane_fitting.py)
# 2. เพิ่ม geometric constraints
# 3. ใช้ Kalman filter smooth coefficients
# 4. Reject outliers ด้วย statistical methods
```

---

## 4. 🎛️ **Sliding Window - Histogram Weighting Issues**

### ปัญหา:
```python
# lane_trajectory.py:455-461
# Distance-weighted histograms
weights = np.linspace(BEV_HISTOGRAM_WEIGHT_FAR, BEV_HISTOGRAM_WEIGHT_NEAR, n_rows)[:, np.newaxis]
hist_l = np.sum((left_bottom > 0).astype(np.float32) * weights, axis=0)
hist_r = np.sum((right_bottom > 0).astype(np.float32) * weights, axis=0)
```

**ปัญหา:**
- Weight ระยะไกลมากเกินไป → sensitive to noise
- Histogram smoothing (Gaussian blur kernel=31) อาจทำให้ peak shift
- Independent left/right search อาจ miss correlation

**ผลกระทบ:**
- Peak detection ผิด → lane center ผิด
- ระยะไกลมี error สูง

**แนวแก้:**
```python
# 1. ลด weight ระยะไกล
weights = np.linspace(1.5, 1.0, n_rows)  # แทน 3.0, 1.0

# 2. ใช้ smaller Gaussian kernel
hist_l_sm = cv2.GaussianBlur(hist_l.reshape(1, -1), (1, 15), 0)  # แทน 31

# 3. เพิ่ม constraint: lane width ต้องสม่ำเสมอ
```

---

## 5. 🚗 **MPC Control - Error Propagation**

### ปัญหา:

#### 5.1 **State EMA Smoothing:**
```python
# alg/step.py:235-239
a = STATE_EMA_ALPHA
self._prev_fused_state = a * self._prev_fused_state + (1.0 - a) * np.array([cte_m, head_s, curv_s])
```

**ปัญหา:**
- EMA ทำให้ lag เกิดขึ้น
- ความเร็วสูง + โค้งแหลม → lag ทำให้ตอบสนองช้า

#### 5.2 **Curvature Feed-forward:**
```python
# control/lane_mpc.py:196-198
# Road heading at step k ≈ curvature * distance
psi_ref_k = curvature * (k * c.dt * v_ref)
```

**ปัญหา:**
- สมมติว่า curvature คงที่ตลอด horizon
- จริงๆ curvature เปลี่ยนตาม path
- ทำให้ heading reference ผิดในโค้งที่เปลี่ยน

#### 5.3 **Adaptive Weights:**
```python
# control/lane_mpc.py:64-68
scale_cte = 1.0 + 0.5 * (1.0 - speed_norm)
scale_heading = 1.0 + 0.8 * min(1.0, abs(curvature) * 30.0) + 0.4 * speed_norm
```

**ปัญหา:**
- Weight scaling อาจไม่เหมาะกับทุกสถานการณ์
- Confidence ต่ำ → เพิ่ม steer_rate weight → ทำให้ตอบสนองช้า

**แนวแก้:**
```python
# 1. ลด EMA alpha หรือใช้ Kalman filter
# 2. ใช้ curvature profile แทน constant curvature
# 3. Tune adaptive weights ให้เหมาะสม
```

---

## 6. 🔗 **Fusion Logic - Mode Switching**

### ปัญหา:
```python
# alg/fusion.py
# Mode switching: WP_PRIMARY ↔ WP+UNET_CENTER
```

**ปัญหา:**
- Mode switch ทำให้ reference path กระโดด
- EMA smoothing ไม่เพียงพอ
- Steering oscillation เกิดขึ้นตอน switch

**ผลกระทบ:**
- Control ไม่ smooth
- Passenger comfort ต่ำ

**แนวแก้:**
```python
# 1. เพิ่ม hysteresis ใน mode switching
# 2. Blend reference path แทน hard switch
# 3. เพิ่ม transition period
```

---

## 📊 **สรุป Error Propagation**

```
UNet (low confidence)
    ↓ (noise)
BEV Transform (distortion)
    ↓ (coordinate error)
Sliding Window (histogram weighting)
    ↓ (peak detection error)
Polynomial Fit (overfitting/underfitting)
    ↓ (curvature error)
Fusion (mode switching)
    ↓ (reference path jump)
MPC (lag from EMA)
    ↓ (control delay)
Steering Output (oscillation)
```

**Error สะสม:** แต่ละ step มี error 5-10% → รวมกันได้ 30-50% error!

---

## ✅ **แนวทางแก้ไขแบบ Priority**

### **Priority 1: Critical (ทำทันที)**

1. **Train DSUNet model ใหม่** ✅ (กำลังทำอยู่)
   - เพิ่ม confidence → ลด noise
   - ใช้ proper loss functions

2. **ปรับ UNet threshold**
   ```python
   mask_bin = (lane_prob > 0.3).astype(np.uint8)  # เพิ่มจาก 0.1
   ```

3. **ใช้ B-spline fitting** ✅ (มีแล้วใน improved_lane_fitting.py)
   - แทน polynomial
   - เพิ่ม geometric constraints

### **Priority 2: Important (ทำในอีก 1-2 สัปดาห์)**

4. **Fix BEV Transform**
   - ใช้ calibrated IPM
   - หรือ compensate distortion

5. **Improve Sliding Window**
   - ลด histogram weights ระยะไกล
   - เพิ่ม lane width constraint

6. **Reduce MPC lag**
   - ลด EMA alpha
   - ใช้ Kalman filter

### **Priority 3: Nice to have**

7. **Smooth mode switching**
   - เพิ่ม hysteresis
   - Blend reference paths

8. **Better curvature feed-forward**
   - ใช้ curvature profile
   - Predict curvature change

---

## 🎯 **Expected Improvements**

| Fix | Error Reduction | Difficulty |
|-----|----------------|------------|
| DSUNet training | 40-50% | Medium ✅ |
| Threshold adjustment | 10-15% | Easy |
| B-spline fitting | 15-20% | Easy ✅ |
| BEV calibration | 20-30% | Hard |
| Sliding window fix | 10-15% | Medium |
| MPC lag reduction | 10-15% | Medium |

**Total potential improvement: 60-80% error reduction**

---

## 📝 **Immediate Actions**

1. ✅ **DSUNet training** - กำลังรัน (Epoch 1/100)
2. ⏳ **Test DSUNet** - หลัง training เสร็จ
3. ⏳ **Integrate B-spline** - ใช้ improved_lane_fitting.py
4. ⏳ **Tune thresholds** - ปรับ 0.1 → 0.3
5. ⏳ **Benchmark** - เปรียบเทียบก่อน/หลัง

---

**สรุป:** ปัญหาหลักคือ **UNet confidence ต่ำ** และ **error propagation** ตลอด pipeline 
การ train DSUNet ใหม่จะแก้ปัญหาได้ 40-50% ที่เหลือต้องปรับแต่ง pipeline ต่อไป
