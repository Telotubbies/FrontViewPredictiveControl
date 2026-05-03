# Perception Architecture Complete Scan

สรุปการ scan ระบบ perception, marking, และ lane detection ทั้งหมดใน `carla_mpc_classical`

---

## 📁 File Structure Overview

```
perception/
├── __init__.py
├── classical/
│   ├── __init__.py
│   └── detector.py                    # Peter Moran classical detector (color-based)
├── unet/
│   └── __init__.py
├── road_perception.py                 # Legacy UNet wrapper
├── lane_detector.py                   # UNet lane detector
├── classical_lane_detector.py         # Color threshold detector (no GPU)
├── lane_trajectory.py                 # ⭐ MAIN PIPELINE (6 phases)
├── ego_lane_tracker.py                # Ego lane selection + temporal completion
├── kalman_lane_tracker.py             # Kalman filter for lane coefficients
├── temporal_lane_buffer.py            # Frame buffer for temporal smoothing
├── improved_lane_fitting.py           # B-spline lane fitting
├── lane_clustering.py                 # Cluster lane candidates
├── lane_validator.py                  # Geometric validation
├── geometric_validation.py            # Lane geometry checks
├── edge_detection.py                  # Classical edge detection helpers
├── road_symbol.py                     # Road arrow detection
└── optimized_detector.py              # RTX 3080 Ti optimized detector

visualization/
├── peter_moran_viz.py                 # ⭐ NEW: Peter Moran overlay (6 thumbnails + metrics)
└── perception_display.py              # OLD: 4-panel display (not used)
```

---

## 🏗️ Main Architecture: LaneTrajectoryPipeline

**File:** `perception/lane_trajectory.py`

### 6-Phase Pipeline

```
P1: Lane Mask       → RGB → binary mask + confidence
P2: BEV + Quality   → mask → BEV warp, fill gaps, mask_quality
P3: Boundaries      → BEV → left/right/center coeffs (sliding window)
P4: Smooth Center   → center_raw + Kalman → center_coeffs, geometry_valid
P5: Trajectory      → center_coeffs → (x_ref, y_ref, cte, heading, curvature)
P6: Visualization   → BEV window vis + lane overlay + Peter Moran data
```

### Key Components

1. **Detector Options:**
   - **UNet** (default): Deep learning lane segmentation
   - **Classical**: Color thresholding (Peter Moran style, no GPU)

2. **BEV Transform:**
   - Perspective warp to bird's-eye view
   - Parameters: `src_top_ratio`, `src_bot_ratio`, `src_top_margin`, `src_bot_margin`

3. **Sliding Window Search:**
   - Histogram-based lane detection in BEV
   - Raw windows (all detections) + Filtered windows (outlier rejection)
   - Peter Moran's 60% pixel coverage threshold

4. **Ego Lane Tracker:**
   - Selects ego lane from multiple candidates
   - Temporal completion when detection fails
   - Lateral position matching across frames

5. **Kalman Filter:**
   - Smooths lane polynomial coefficients
   - Reduces jitter in lane overlay
   - Separate trackers for left/right/center

6. **Geometric Validation:**
   - Lane width checks (2.0m - 5.5m)
   - Parallelism validation
   - Curvature sanity checks

---

## 🎨 Visualization System

### Current: Peter Moran Overlay Style

**File:** `visualization/peter_moran_viz.py`

**Layout:**
```
┌─────────────────────────────────────────────────────────┐
│ [6 BEV Thumbnails]                    [Raw Cam Thumb]  │
│ Left    Left   Right  Right  Center   Filtered         │
│ Windows Fill   Windows Fill  Detect   Combined         │
│                                                         │
│ Left Curvature: 582.25m                                │
│ Right Curvature: 352.76m                               │
│ Center Alignment: 0.01m Left                           │
│                                                         │
│              ┌─── Full Camera View ───┐                │
│              │  Green lane fill       │                │
│              │  Yellow boundaries     │                │
│              └────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

**Function:** `compose_peter_moran_overlay()`
- Composites 6 diagnostic thumbnails onto lane overlay
- Shows raw/filtered sliding windows
- Displays left/right lane fills
- Shows center detection (yellow left + blue right)
- Overlays metrics: left/right curvature, center alignment

---

## 🔧 Detector Implementations

### 1. UNet Detector
**File:** `perception/lane_detector.py`

- Deep learning segmentation
- Input: 640x480 RGB
- Output: Binary mask
- GPU accelerated (CUDA)
- Model: `model/lane_unet_final.pth`

### 2. Classical Detector (Peter Moran)
**File:** `perception/classical_lane_detector.py`

- Color thresholding (LAB, HSV, HLS channels)
- CLAHE histogram equalization
- No GPU required (CPU only)
- Thresholds optimized for CARLA:
  - LAB: 140
  - HSV: 200
  - HLS: 190

### 3. Legacy Classical
**File:** `perception/classical/detector.py`

- Original Peter Moran implementation
- Edge detection + Hough transform
- Not currently used in main pipeline

---

## 📊 Data Flow

```
RGB Camera (640x480)
    ↓
[P1] Detector (UNet or Classical)
    ↓
Binary Mask + Confidence
    ↓
[P2] BEV Transform + Quality Check
    ↓
BEV Binary Mask
    ↓
[P3] Sliding Window Search
    ↓
Left/Right Lane Points (BEV space)
    ↓
Ego Lane Tracker → Select ego lane
    ↓
Polynomial Fitting (2nd order)
    ↓
[P4] Kalman Filter Smoothing
    ↓
Smooth Lane Coefficients
    ↓
[P5] Trajectory Generation
    ↓
Reference Path (x, y, yaw, v)
CTE, Heading Error, Curvature
    ↓
[P6] Visualization
    ↓
TrajectoryOutput {
  - x_ref, y_ref, yaw_ref, v_ref
  - cte, heading_err, curvature
  - confidence, geometry_valid
  - lane_overlay (RGB with green fill)
  - bev_window_vis (BEV with colored windows)
  - left_mask_vis, right_mask_vis
  - raw_camera
  - left_curvature_m, right_curvature_m
  - Window data for Peter Moran viz
}
    ↓
Dashboard (Pygame)
    ↓
Peter Moran Overlay Composited
```

---

## 🎯 Key Features

### 1. Temporal Completion
- When detection fails, use previous frame's lane
- Prevents sudden jumps in lane overlay
- Tracked via `used_completion` flag

### 2. Outlier Rejection
- Peter Moran's 60% pixel coverage threshold
- Filters noisy sliding windows
- Separate raw/filtered window lists

### 3. Per-Lane Curvature
- Left lane curvature (radius in meters)
- Right lane curvature (radius in meters)
- Computed from polynomial coefficients: R = 1/(2*c₂)

### 4. Geometric Validation
- Lane width: 2.0m - 5.5m
- Parallelism check
- Curvature sanity (not too sharp)
- Sets `geometry_valid` flag

### 5. Multi-Detector Support
- Switch between UNet and Classical via config
- `USE_CLASSICAL_DETECTOR: true/false`
- No code changes needed

---

## 🔍 Current Status

### ✅ Completed
1. ✅ Peter Moran visualization integrated
2. ✅ 6-phase pipeline working
3. ✅ UNet + Classical detector support
4. ✅ Kalman filtering
5. ✅ Ego lane tracking
6. ✅ Temporal completion
7. ✅ Per-lane curvature computation
8. ✅ BEV sliding window search
9. ✅ Geometric validation
10. ✅ Full data flow to Dashboard

### 🎨 Visualization Components
- ✅ 6 diagnostic thumbnails (BEV windows + fills)
- ✅ Raw camera thumbnail
- ✅ Left/Right curvature metrics
- ✅ Center alignment display
- ✅ Green lane fill on main view
- ✅ Yellow lane boundaries

### 📦 Data Structures

**TrajectoryOutput** (perception/lane_trajectory.py:160-192)
```python
@dataclass
class TrajectoryOutput:
    # Reference trajectory
    x_ref: np.ndarray
    y_ref: np.ndarray
    yaw_ref: np.ndarray
    v_ref: np.ndarray
    
    # Lane state
    cte: float
    heading_err: float
    curvature: float
    confidence: float
    
    # Visualization
    mask_vis: np.ndarray
    bev_binary: np.ndarray
    bev_window_vis: np.ndarray
    lane_overlay: np.ndarray
    left_mask_vis: np.ndarray
    right_mask_vis: np.ndarray
    raw_camera: np.ndarray
    
    # Per-lane metrics
    left_curvature_m: float
    right_curvature_m: float
    
    # Flags
    geometry_valid: bool
    tracked: bool
    used_completion: bool
```

**FrameState** (state.py:10-57)
```python
@dataclass
class FrameState:
    # Control state
    rgb: np.ndarray
    speed_kmh: float
    steer: float
    throttle: float
    brake: float
    cte_m: float
    heading_rad: float
    curvature: float
    
    # Perception
    lane_conf: float
    geometry_valid: bool
    lane_overlay: np.ndarray
    mask: np.ndarray
    bev_window_vis: np.ndarray
    
    # Peter Moran viz
    bev_binary: np.ndarray
    raw_windows_left: list
    raw_windows_right: list
    filt_windows_left: list
    filt_windows_right: list
    left_px_bev: np.ndarray
    right_px_bev: np.ndarray
    left_curvature_m: float
    right_curvature_m: float
```

---

## 🚀 Integration Points

### 1. Main Loop
**File:** `run_unet_mpc.py`
- Dashboard.draw() receives Peter Moran data
- Composites overlay onto perception panel

### 2. Algorithm Layer
**File:** `alg/step.py`
- Extracts Peter Moran data from TrajectoryOutput
- Populates FrameState with window lists

### 3. Pipeline Layer
**File:** `pipeline.py`
- Alternative entry point (not used in main)
- Also extracts Peter Moran data

---

## 📝 Configuration

**File:** `config.yaml`

```yaml
# Detector selection
USE_CLASSICAL_DETECTOR: false  # true = color threshold, false = UNet

# BEV transform
BEV_TOP_RATIO: 0.28
BEV_BOT_RATIO: 0.98
BEV_TOP_MARGIN: 0.32
BEV_BOT_MARGIN: 0.10

# Lane width
MIN_LANE_WIDTH_M: 2.0
MAX_LANE_WIDTH_M: 5.5

# Confidence threshold
LANE_CONF_THRESHOLD: 0.55

# Kalman filter
KALMAN_PROCESS_NOISE: 0.01
KALMAN_MEASUREMENT_NOISE: 0.1
```

---

## 🎓 Summary

ระบบ perception ปัจจุบันเป็น **6-phase pipeline** ที่:
1. รองรับทั้ง UNet และ Classical detector
2. ใช้ BEV transform + sliding window search
3. มี Kalman filter สำหรับ smooth lane coefficients
4. มี Ego lane tracker + temporal completion
5. คำนวณ per-lane curvature
6. สร้าง Peter Moran visualization แบบ overlay

**ข้อดี:**
- Modular design (แต่ละ phase แยกชัด)
- Multi-detector support
- Robust temporal handling
- Rich visualization data
- Geometric validation

**ระบบพร้อมใช้งานและทดสอบแล้ว** ✅
