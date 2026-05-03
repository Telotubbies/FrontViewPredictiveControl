# 🎯 Classical Lane Detector - Usage Guide

## ✅ **สร้างเสร็จแล้ว:**

1. ✅ Clone Peter Moran's repository
2. ✅ Analyzed classical approach
3. ✅ Created `perception/classical_lane_detector.py`
4. ✅ Compatible with existing pipeline

---

## 🔑 **Classical vs UNet:**

### **UNet (Current):**
```python
# Deep learning approach
- Requires: GPU, trained model
- Speed: Medium (GPU inference)
- Accuracy: High
- File size: ~50MB model
```

### **Classical (New Option):**
```python
# Color thresholding approach
- Requires: CPU only, no model
- Speed: Fast (pure OpenCV)
- Accuracy: Good (on highways)
- File size: ~5KB code
```

---

## 🚀 **How to Use:**

### **Option 1: Quick Test (Standalone)**

```python
from perception.classical_lane_detector import create_classical_detector

# Create detector optimized for CARLA
detector = create_classical_detector(carla_optimized=True)

# Use like UNet detector
mask, confidence, _ = detector.detect_lanes(rgb_image)
```

### **Option 2: Replace UNet in Pipeline**

Modify `perception/lane_trajectory.py`:

```python
# At top of file
from perception.classical_lane_detector import create_classical_detector

# In LaneTrajectoryPipeline.__init__():
def __init__(self, detector=None, use_classical=False, ...):
    if use_classical:
        # Use classical detector (no GPU)
        self.detector = create_classical_detector(carla_optimized=True)
    elif detector is None:
        # Use UNet detector (default)
        from perception.lane_detector import LaneDetector
        self.detector = LaneDetector(...)
    else:
        self.detector = detector
```

### **Option 3: Hybrid Approach**

```python
class HybridDetector:
    def __init__(self):
        self.unet = LaneDetector(...)
        self.classical = create_classical_detector(carla_optimized=True)
    
    def detect_lanes(self, image):
        # Try UNet first
        unet_mask, unet_conf, _ = self.unet.detect_lanes(image)
        
        # If UNet confidence low, use classical
        if np.mean(unet_conf) < 0.5:
            return self.classical.detect_lanes(image)
        else:
            return unet_mask, unet_conf, []
```

---

## 🎨 **How It Works:**

### **Multi-Channel Color Thresholding:**

```python
# 1. LAB B-channel (Yellow/White detection)
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
b_channel = lab[:, :, 2]
# Apply CLAHE normalization
# Threshold at 140 (CARLA optimized)

# 2. HSV V-channel (Brightness)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
v_channel = hsv[:, :, 2]
# Threshold at 200

# 3. HLS L-channel (Lightness)
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
l_channel = hls[:, :, 1]
# Threshold at 190

# Combine all channels
scores = lab_binary + hsv_binary + hls_binary
mask = normalize(scores)
```

---

## ⚙️ **Tuning for CARLA:**

### **Default Settings (CARLA Optimized):**
```python
detector = create_classical_detector(carla_optimized=True)
# LAB threshold: 140 (vs 150 for real roads)
# HSV threshold: 200 (vs 220 for real roads)
# HLS threshold: 190 (vs 210 for real roads)
```

### **Custom Settings:**
```python
detector = ClassicalLaneDetector(
    lab_threshold=130,  # Lower = more sensitive
    hsv_threshold=180,
    hls_threshold=170,
    lab_clip_limit=3.0,  # Higher = more contrast
    hsv_clip_limit=6.0,
    hls_clip_limit=3.0,
)
```

---

## 📊 **Expected Performance:**

### **Advantages:**
- ✅ **No GPU required** - runs on CPU
- ✅ **Fast** - pure OpenCV operations
- ✅ **Small** - no model file needed
- ✅ **Robust to lighting** - CLAHE normalization
- ✅ **Good on highways** - designed for lane markings

### **Limitations:**
- ❌ **Sensitive to shadows** - may detect shadow edges
- ❌ **Requires good markings** - won't work on unmarked roads
- ❌ **Fixed thresholds** - may need tuning per scenario
- ❌ **Less flexible** - can't learn complex patterns

---

## 🧪 **Testing:**

### **Step 1: Test Classical Detector Alone**

```python
# test_classical.py
from perception.classical_lane_detector import create_classical_detector
import cv2
import numpy as np

detector = create_classical_detector(carla_optimized=True)

# Load CARLA image
img = cv2.imread("carla_test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect
mask, confidence, _ = detector.detect_lanes(img)

# Visualize
cv2.imshow("Mask", mask)
cv2.imshow("Confidence", (confidence * 255).astype(np.uint8))
cv2.waitKey(0)
```

### **Step 2: Compare with UNet**

```python
# Compare both detectors
unet_detector = LaneDetector(...)
classical_detector = create_classical_detector(carla_optimized=True)

# Run both
unet_mask, _, _ = unet_detector.detect_lanes(img)
classical_mask, _, _ = classical_detector.detect_lanes(img)

# Compare
diff = cv2.absdiff(unet_mask, classical_mask)
print(f"Difference: {np.sum(diff > 0)} pixels")
```

---

## 🔧 **Integration Steps:**

### **Minimal Change (Test Only):**

1. Import classical detector in `run_unet_mpc.py`:
```python
from perception.classical_lane_detector import create_classical_detector

# Replace detector
detector = create_classical_detector(carla_optimized=True)
```

2. Run and observe performance

### **Full Integration:**

1. Add parameter to `LaneTrajectoryPipeline.__init__()`:
```python
def __init__(self, ..., use_classical=False):
    if use_classical:
        self.detector = create_classical_detector(carla_optimized=True)
    else:
        self.detector = LaneDetector(...)  # UNet
```

2. Add config option in `config.yaml`:
```yaml
perception:
  use_classical_detector: false  # Set true to use classical
```

3. Test both modes

---

## 📝 **Files Created:**

1. ✅ `perception/classical_lane_detector.py` - Main detector
2. ✅ `PETER_MORAN_CLASSICAL_ANALYSIS.md` - Analysis
3. ✅ `CLASSICAL_DETECTOR_USAGE.md` - This guide
4. ✅ `highway-lane-tracker/` - Peter Moran's original code

---

## 🎯 **Recommendation:**

### **For Testing:**
Try classical detector first to see if it works in CARLA:
- Faster development (no model needed)
- Easier to debug (visual thresholds)
- Good baseline for comparison

### **For Production:**
Use hybrid approach:
- UNet for complex scenarios
- Classical as fallback when UNet fails
- Best of both worlds

---

## 🚀 **Quick Start:**

```bash
# Test classical detector
cd /home/supawich/Desktop/CARLA_0.9.16/carla_mpc_classical
python -c "
from perception.classical_lane_detector import create_classical_detector
detector = create_classical_detector(carla_optimized=True)
print('✅ Classical detector ready!')
print(f'Settings: LAB=140, HSV=200, HLS=190')
"
```

**Status:** Classical detector ready to use! 🎉
