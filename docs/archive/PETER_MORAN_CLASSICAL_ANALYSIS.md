# 🎯 Peter Moran's Classical Lane Tracker - Analysis

## 📚 **Source:**
- GitHub: https://github.com/peter-moran/highway-lane-tracker
- Cloned to: `highway-lane-tracker/`

---

## 🔑 **Key Components:**

### **1. Classical Lane Detection (No Deep Learning!)**

**Instead of UNet, uses:**
```python
def score_pixels(img):
    """
    Color-based lane detection using:
    1. LAB color space (B channel) - yellow/white detection
    2. HSV color space (V channel) - brightness
    3. HLS color space (L channel) - lightness
    
    Each channel:
    - Apply CLAHE (histogram equalization)
    - Threshold to binary
    - Combine scores
    """
    settings = [
        {'cspace': 'LAB', 'channel': 2, 'threshold': 150},  # B channel
        {'cspace': 'HSV', 'channel': 2, 'threshold': 220},  # Value
        {'cspace': 'HLS', 'channel': 1, 'threshold': 210},  # Lightness
    ]
    
    scores = combine_all_thresholds(img, settings)
    return scores  # Pixel intensity = lane likelihood
```

**Advantages:**
- ✅ No GPU needed
- ✅ Fast (real-time)
- ✅ Works well on highways
- ✅ Robust to lighting changes (CLAHE)

**Disadvantages:**
- ❌ Sensitive to shadows
- ❌ Requires good lane markings
- ❌ May fail in complex scenarios

---

### **2. Sliding Window with Kalman Filters**

```python
# 9 windows per lane (18 total)
# Each window has its own Kalman filter
for level in range(9):
    windows_left.append(Window(level, ...))
    windows_right.append(Window(level, ...))

# Update all windows jointly
joint_sliding_window_update(windows_left, windows_right, pixel_scores)

# Filter valid windows
win_left_valid = filter_window_list(windows_left, 
                                     remove_frozen=False, 
                                     remove_dropped=True)
```

---

### **3. Camera Calibration**

```python
class DashboardCamera:
    def __init__(self, chessboard_imgs, ...):
        # Calibrate using chessboard
        self.camera_matrix, self.distortion_coeffs = calibrate(...)
        
        # Define perspective transform
        self.overhead_transform = cv2.getPerspectiveTransform(src, dst)
        
    def undistort(self, img):
        return cv2.undistort(img, self.camera_matrix, self.distortion_coeffs)
    
    def warp_to_overhead(self, img):
        return cv2.warpPerspective(img, self.overhead_transform, ...)
```

**Issue for CARLA:**
- ❌ Requires chessboard calibration images
- ❌ CARLA camera is already "perfect" (no distortion)
- ✅ Can skip calibration, use only perspective transform

---

## 🔄 **Pipeline Comparison:**

### **Our Current (UNet-based):**
```
RGB Image
  ↓
UNet (Deep Learning)
  ↓
Lane Mask (binary)
  ↓
BEV Transform
  ↓
Sliding Window
  ↓
Polynomial Fit
```

### **Peter Moran's (Classical):**
```
RGB Image
  ↓
Undistort (calibration)
  ↓
BEV Transform
  ↓
Color Thresholding (LAB+HSV+HLS)
  ↓
Pixel Scoring
  ↓
Sliding Window + Kalman
  ↓
Polynomial Fit
```

---

## ✅ **What We Can Adopt:**

### **Option 1: Replace UNet with Classical Detection**

```python
# perception/classical_lane_detector.py

class ClassicalLaneDetector:
    """
    Peter Moran's color-based lane detection
    No deep learning required!
    """
    
    def score_pixels(self, img_overhead):
        """
        Multi-channel color thresholding
        """
        scores = np.zeros(img.shape[0:2], dtype=np.uint8)
        
        # LAB B-channel (yellow/white)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        b_channel = lab[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        b_norm = clahe.apply(b_channel)
        _, b_binary = cv2.threshold(b_norm, 150, 1, cv2.THRESH_BINARY)
        scores += b_binary
        
        # HSV V-channel (brightness)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        v_norm = clahe.apply(v_channel)
        _, v_binary = cv2.threshold(v_norm, 220, 1, cv2.THRESH_BINARY)
        scores += v_binary
        
        # HLS L-channel (lightness)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_norm = clahe.apply(l_channel)
        _, l_binary = cv2.threshold(l_norm, 210, 1, cv2.THRESH_BINARY)
        scores += l_binary
        
        # Normalize to 0-255
        return cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX)
    
    def detect_lanes(self, img):
        """
        Main detection function
        """
        # BEV transform (already have this)
        img_overhead = bev_transform(img, ...)
        
        # Score pixels (classical method)
        pixel_scores = self.score_pixels(img_overhead)
        
        # Convert to binary mask
        mask = (pixel_scores > 127).astype(np.uint8) * 255
        
        return mask, confidence
```

---

### **Option 2: Hybrid Approach**

```python
class HybridLaneDetector:
    """
    Use both UNet AND classical detection
    Combine for robustness
    """
    
    def __init__(self):
        self.unet = UNetDetector()
        self.classical = ClassicalDetector()
    
    def detect_lanes(self, img):
        # Get both detections
        unet_mask = self.unet.detect(img)
        classical_mask = self.classical.detect(img)
        
        # Combine (weighted average or voting)
        combined = cv2.addWeighted(unet_mask, 0.6, classical_mask, 0.4, 0)
        
        # Or use classical as fallback when UNet fails
        if unet_confidence < 0.5:
            return classical_mask
        else:
            return unet_mask
```

---

## 🚀 **Quick Implementation Plan:**

### **Step 1: Extract Classical Detector**

Create `perception/classical/peter_moran_detector.py`:
- Copy `score_pixels()` function
- Adapt for CARLA (skip calibration)
- Use our existing BEV transform

### **Step 2: Integrate into Pipeline**

Modify `LaneTrajectoryPipeline`:
```python
def __init__(self, use_classical=False):
    if use_classical:
        self.detector = ClassicalLaneDetector()
    else:
        self.detector = UNetDetector()  # Current
```

### **Step 3: Test & Compare**

Run both detectors and compare:
- Speed (classical should be faster)
- Accuracy (UNet should be better in complex scenarios)
- Robustness (classical better in good lighting)

---

## 📊 **Expected Performance:**

| Metric | UNet | Classical | Hybrid |
|--------|------|-----------|--------|
| **Speed** | Medium (GPU) | Fast (CPU) | Medium |
| **Accuracy** | High | Medium | High |
| **Lighting** | Good | Excellent | Excellent |
| **Shadows** | Good | Poor | Good |
| **Complex** | Excellent | Poor | Good |
| **GPU Required** | Yes | No | Yes |

---

## ⚠️ **Challenges for CARLA:**

1. **Color Calibration:**
   - Peter Moran's thresholds tuned for real dashcam
   - CARLA has different lighting/colors
   - Need to retune thresholds

2. **No Camera Distortion:**
   - CARLA camera is perfect
   - Can skip undistortion step
   - Simplifies implementation

3. **Lane Markings:**
   - CARLA lanes may differ from real roads
   - Classical method relies on white/yellow markings
   - May need threshold adjustment

---

## 💡 **Recommendation:**

### **For Testing:**
Implement **Option 1** (Classical Only) first:
- Fastest to implement
- No GPU dependency
- Good for testing if classical works in CARLA

### **For Production:**
Use **Option 2** (Hybrid):
- Best of both worlds
- UNet for complex scenarios
- Classical as fallback or validation

---

## 📝 **Next Steps:**

1. ✅ Clone repository (done)
2. ⏳ Extract `score_pixels()` function
3. ⏳ Create `ClassicalLaneDetector` class
4. ⏳ Integrate into pipeline
5. ⏳ Test in CARLA
6. ⏳ Tune thresholds for CARLA lighting
7. ⏳ Compare with UNet performance

---

**Status:** Repository cloned, ready to extract and adapt! 🚀
