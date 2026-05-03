# 🚗 การปรับปรุงการจับ Lane ด้วย Research-Based Methods

## 📊 สรุปปัญหาเดิม

### ❌ ปัญหาที่พบในโค้ดเดิม (`lane_trajectory.py`)

1. **Sliding Window แบบ Naive**
   - ใช้ histogram หา peak แบบง่ายๆ
   - Mirror strategy ไม่มีหลักการ (แค่สะท้อนซ้าย-ขวา)
   - ไม่มี temporal consistency
   - ไม่ได้ใช้ geometric constraints

2. **Polynomial Fitting ไม่เพียงพอ**
   ```python
   # โค้ดเดิม: _fit_poly_robust()
   coeffs = np.polyfit(x, y, order)  # แค่ fit ธรรมดา
   # ปัญหา:
   # - Oscillation ในโค้งแหลม
   # - ไม่มี curvature constraints
   # - ไม่มี lane width validation
   ```

3. **ไม่มี Lane Geometry Validation**
   - ไม่เช็คว่าเลนซ้าย-ขวาขนานกันหรือไม่
   - ไม่เช็คความกว้างเลน (ควรเป็น 2.5-4.0m)
   - ไม่มี curvature continuity

---

## ✅ วิธีแก้ไขจาก Research Papers

### 📚 อ้างอิงจาก Papers

1. **3D-SplineNet** (2023)
   - ใช้ B-spline แทน polynomial
   - Local control, smoother curvature
   
2. **LaneATT** (2021)
   - Attention-based feature pooling
   - Learned lane anchors
   
3. **Monocular Lane Detection Survey** (2024)
   - Geometric constraints (parallel, width)
   - Temporal consistency

---

## 🛠️ การปรับปรุงที่ทำ

### 1. **B-Spline Curve Fitting** (`BSplineLaneFitter`)

**ทำไมดีกว่า Polynomial:**

| Feature | Polynomial | B-Spline |
|---------|-----------|----------|
| Oscillation | ❌ มาก (โดยเฉพาะ order สูง) | ✅ น้อย |
| Local Control | ❌ แก้จุดหนึ่งกระทบทั้งเส้น | ✅ แก้จุดหนึ่งกระทบแค่บริเวณนั้น |
| Sharp Curves | ❌ ไม่ดี | ✅ ดี |
| Smoothness | ❌ ต้องระวัง | ✅ Smooth by design |

**ตัวอย่างการใช้:**
```python
from perception.improved_lane_fitting import BSplineLaneFitter

fitter = BSplineLaneFitter(degree=3, num_control_points=8)

# Fit lanes with geometric constraints
left_spline, right_spline, info = fitter.fit_lane_pair(
    left_xy=left_points,      # (N, 2) array
    right_xy=right_points,     # (M, 2) array
    enforce_parallel=True,     # บังคับให้ขนานกัน
    enforce_width=True,        # บังคับความกว้างเลน 2.5-4.0m
)

# ดู quality metrics
print(f"Average lane width: {info['avg_width_m']:.2f}m")
print(f"Width valid: {info['width_valid']}")
print(f"Parallel score: {info['parallel_score']:.2f}")

# Evaluate at specific positions
x_positions = np.linspace(0, 50, 100)  # 0-50m ahead
left_y, right_y = fitter.evaluate_at(x_positions)
```

### 2. **Geometric Constraints**

**Constraints ที่เพิ่มเข้ามา:**

```python
# 1. Parallel Lanes
# เลนซ้าย-ขวาควรมี curvature ใกล้เคียงกัน
parallel_error = sum((left_tangent - right_tangent)^2)

# 2. Lane Width
# ความกว้างเลนควรเป็น 2.5-4.0m (มาตรฐานถนน)
width_error = sum((widths - 3.5m)^2)

# 3. Curvature Limit
# ไม่ควรโค้งแหลมเกินไป (max curvature = 0.08)
```

### 3. **Geometric Validation** (`GeometricLaneValidator`)

**ตรวจสอบความถูกต้องของเลนที่จับได้:**

```python
from perception.improved_lane_fitting import GeometricLaneValidator

validator = GeometricLaneValidator()

valid, info = validator.validate_lane_pair(left_xy, right_xy)

if valid:
    print("✅ Lane detection valid!")
    print(f"   Average width: {info['avg_width_m']:.2f}m")
    print(f"   Width std: {info['width_std_m']:.2f}m")
else:
    print("❌ Lane detection invalid!")
    print(f"   Failed checks: {info['checks']}")
```

**Validation Checks:**
- ✅ Sufficient points (>10 points per lane)
- ✅ Lane width in valid range (2.5-4.0m)
- ✅ Width consistency (std < 0.5m)
- ✅ No crossing lanes

---

## 🔧 วิธีใช้งานกับโค้ดเดิม

### Option 1: แทนที่ Polynomial Fitting ทั้งหมด

```python
# ใน lane_trajectory.py, แก้ function polynomial_fit()

def polynomial_fit(
    left_xy: Optional[np.ndarray],
    right_xy: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """ใช้ B-spline แทน polynomial"""
    from .improved_lane_fitting import BSplineLaneFitter, convert_bspline_to_polynomial
    
    fitter = BSplineLaneFitter(degree=3, num_control_points=8)
    
    left_spline, right_spline, info = fitter.fit_lane_pair(
        left_xy, right_xy,
        enforce_parallel=True,
        enforce_width=True,
    )
    
    if not info.get("valid", False):
        return None
    
    # Convert to polynomial for compatibility
    left_coeffs = convert_bspline_to_polynomial(left_spline, (0, 90))
    right_coeffs = convert_bspline_to_polynomial(right_spline, (0, 90))
    center_coeffs = (left_coeffs + right_coeffs) / 2
    
    return left_coeffs, right_coeffs, center_coeffs
```

### Option 2: เพิ่ม Validation Layer

```python
# ใน lane_trajectory.py, หลัง sliding_window_search()

from .improved_lane_fitting import GeometricLaneValidator

# หลังจาก sliding window
left_xy, right_xy, ... = sliding_window_search(bev_binary)

# Validate ก่อนใช้
validator = GeometricLaneValidator()
valid, info = validator.validate_lane_pair(left_xy, right_xy)

if not valid:
    logger.warning(f"Lane validation failed: {info['checks']}")
    # Fallback to waypoint or previous frame
    return None
```

---

## 📈 ผลลัพธ์ที่คาดหวัง

### Before (Polynomial)
```
❌ Oscillation ในโค้งแหลม
❌ เลนซ้าย-ขวาไม่ขนานกัน
❌ ความกว้างเลนไม่สม่ำเสมอ (1.5m - 5.0m)
❌ Jitter เยอะเมื่อเปลี่ยน frame
```

### After (B-Spline + Constraints)
```
✅ Smooth curves ไม่มี oscillation
✅ เลนซ้าย-ขวาขนานกัน (parallel score > 0.9)
✅ ความกว้างเลนสม่ำเสมอ (3.3-3.7m)
✅ Temporal consistency ดีขึ้น
```

---

## 🎯 Next Steps

### 1. **Integration Test**
```bash
# ทดสอบกับ CARLA
cd /home/supawich/Desktop/CARLA_0.9.16/carla_mpc_classical
python -c "
from perception.improved_lane_fitting import BSplineLaneFitter
import numpy as np

# Test data
left = np.array([[0, -1.5], [10, -1.6], [20, -1.8], [30, -2.0]])
right = np.array([[0, 1.5], [10, 1.6], [20, 1.8], [30, 2.0]])

fitter = BSplineLaneFitter()
l, r, info = fitter.fit_lane_pair(left, right)
print(info)
"
```

### 2. **Performance Tuning**
- ปรับ `num_control_points` (4-12) ตามความโค้งของถนน
- ปรับ constraint weights ใน optimization
- เพิ่ม caching สำหรับ spline evaluation

### 3. **Temporal Smoothing**
- เพิ่ม Kalman filter สำหรับ control points
- ใช้ EMA smoothing บน spline coefficients

---

## 📚 References

1. **3D-SplineNet**: Spline-based 3D Lane Detection (CVPR 2023)
2. **LaneATT**: Attention-based Lane Detection (CVPR 2021)
3. **Monocular Lane Detection Survey** (arXiv 2024)
   - https://arxiv.org/html/2411.16316v2

---

## 💡 Tips

1. **เมื่อไหร่ควรใช้ B-spline:**
   - ถนนโค้งแหลม (curvature > 0.03)
   - ต้องการ smooth curves
   - มีจุดข้อมูลเยอะ (>20 points)

2. **เมื่อไหร่ควรใช้ Polynomial:**
   - ถนนตรง (curvature < 0.01)
   - จุดข้อมูลน้อย (<10 points)
   - ต้องการความเร็ว (polynomial เร็วกว่า)

3. **Hybrid Approach:**
   ```python
   if curvature > 0.03 and len(points) > 20:
       use_bspline()
   else:
       use_polynomial()
   ```
