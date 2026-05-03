# 🎨 Peter Moran Visualization - Implementation Guide

## 📋 **Overview:**

สร้าง multi-panel diagnostic view แบบ Peter Moran ที่แสดง:

```
+----------+----------+----------+
| Overhead |   Raw    | Filtered |
|  (BEV)   |  (Red)   | (Green)  |
+----------+----------+----------+
|        Position Panel          |
|     (Lane Overlay + Metrics)   |
+-----------------------------------+
```

---

## ✅ **Files Created:**

### **1. visualization/peter_moran_viz.py**

**Main Class:**
```python
class PeterMoranViz:
    """Multi-panel visualization"""
    
    def create_diagnostic_view(
        overhead_img,           # BEV image
        raw_windows_left,       # Red windows (before filter)
        raw_windows_right,
        filtered_windows_left,  # Green windows (after filter)
        filtered_windows_right,
        position_img,           # Main camera view
        cte_m,                  # Cross-track error
        curvature_m,            # Radius of curvature
    ) -> np.ndarray
```

**Convenience Function:**
```python
create_peter_moran_visualization(
    bev_binary,
    bev_color,
    raw_windows_left,
    raw_windows_right,
    filtered_windows_left,
    filtered_windows_right,
    lane_overlay_img,
    cte_m,
    curvature_m,
)
```

---

## 🔧 **How to Integrate:**

### **Step 1: Import in lane_trajectory.py**

```python
# Add to imports
from visualization.peter_moran_viz import create_peter_moran_visualization
```

### **Step 2: Track Windows in sliding_window_search()**

Currently we only return window rectangles. Need to separate:
- **Raw windows** (all detected, before outlier check)
- **Filtered windows** (after outlier rejection)

**Modify sliding_window_search():**

```python
def sliding_window_search(...):
    # ... existing code ...
    
    # NEW: Track raw vs filtered windows separately
    raw_left_wins = []      # All windows (red)
    raw_right_wins = []
    filtered_left_wins = [] # Only accepted windows (green)
    filtered_right_wins = []
    
    for win_i in range(n_windows):
        # ... existing window setup ...
        
        # Left window
        xl1 = max(0, left_peak - margin_dyn)
        xl2 = min(w, left_peak + margin_dyn)
        
        # Always add to raw (red)
        raw_left_wins.append((xl1, y_lo, xl2, y_hi))
        
        # Check outlier
        search_area_l = (xl2 - xl1) * (y_hi - y_lo)
        is_outlier_l, reason_l = is_window_outlier(len(ys_l), search_area_l)
        
        if len(ys_l) >= min_pix_win and not is_outlier_l:
            # Add to filtered (green)
            filtered_left_wins.append((xl1, y_lo, xl2, y_hi))
            left_pts_px.append(...)
            left_peak = int(np.mean(xs_l + xl1))
        
        # Same for right window...
    
    # Return both raw and filtered
    return (
        left_xy, right_xy,
        left_px, right_px,
        raw_left_wins, raw_right_wins,      # NEW
        filtered_left_wins, filtered_right_wins,  # NEW
        p2_case
    )
```

### **Step 3: Use in LaneTrajectoryPipeline**

```python
class LaneTrajectoryPipeline:
    def process(self, rgb):
        # ... existing phases ...
        
        # Phase 2: BEV warp
        bev_binary = self._phase2_bev_warp(mask)
        
        # Phase 3: Sliding window (now returns raw + filtered)
        (left_xy, right_xy, left_px, right_px,
         raw_left_wins, raw_right_wins,
         filtered_left_wins, filtered_right_wins,
         p2_case) = sliding_window_search(bev_binary, ...)
        
        # ... Phase 4-5: Kalman + trajectory ...
        
        # Phase 6: Visualization
        if self.enable_peter_moran_viz:
            diagnostic_view = create_peter_moran_visualization(
                bev_binary=bev_binary,
                bev_color=None,  # or color BEV if available
                raw_windows_left=raw_left_wins,
                raw_windows_right=raw_right_wins,
                filtered_windows_left=filtered_left_wins,
                filtered_windows_right=filtered_right_wins,
                lane_overlay_img=lane_overlay,
                cte_m=cte,
                curvature_m=1.0/curvature if curvature > 0.001 else 9999,
            )
            
            # Display or save
            cv2.imshow("Peter Moran Diagnostic", diagnostic_view)
```

---

## 🎯 **What Each Panel Shows:**

### **Panel 1: Overhead (BEV)**
- Bird's eye view of the road
- Perspective warped from camera
- Shows lane markings from above

### **Panel 2: Raw Lane Detection (Red)**
- **Red windows** = all detected windows
- Shows where algorithm searched
- Includes outliers and low-quality detections
- No smoothing applied

### **Panel 3: Filtered Lane Detection (Green)**
- **Green windows** = only accepted windows
- After outlier rejection (60% coverage)
- After Kalman filtering
- High-quality detections only

### **Panel 4: Position**
- Main camera view
- Lane overlay (green polygon)
- Metrics:
  - Cross-track error (CTE)
  - Radius of curvature

---

## 📊 **Visual Feedback:**

### **Red Windows (Raw):**
- **Many red, few green** → High noise, good filtering
- **Red jumping around** → Unstable detection
- **Red missing** → No detection in that area

### **Green Windows (Filtered):**
- **Solid green** → Active, updated
- **Faded green** → Frozen (future: state machine)
- **No green** → Dropped or outlier

---

## 🚀 **Quick Implementation:**

### **Option 1: Minimal (Just visualization)**

Add to `run_unet_mpc.py` or `adas/run.py`:

```python
from visualization.peter_moran_viz import create_peter_moran_visualization

# In main loop, after getting trajectory output:
if frame_state.bev_binary is not None:
    diagnostic = create_peter_moran_visualization(
        bev_binary=frame_state.bev_binary,
        bev_color=None,
        raw_windows_left=frame_state.raw_left_wins,
        raw_windows_right=frame_state.raw_right_wins,
        filtered_windows_left=frame_state.filtered_left_wins,
        filtered_windows_right=frame_state.filtered_right_wins,
        lane_overlay_img=frame_state.lane_overlay,
        cte_m=frame_state.cte,
        curvature_m=frame_state.curvature_radius,
    )
    
    cv2.imshow("Diagnostic", diagnostic)
```

### **Option 2: Full Integration**

Modify `LaneTrajectoryPipeline` to:
1. Track raw vs filtered windows
2. Return them in output
3. Create diagnostic view in Phase 6

---

## 📝 **Next Steps:**

1. ✅ **Created:** `visualization/peter_moran_viz.py`
2. ⏳ **TODO:** Modify `sliding_window_search()` to return raw + filtered windows
3. ⏳ **TODO:** Update `LaneTrajectoryPipeline.process()` to use visualization
4. ⏳ **TODO:** Add to main loop display

---

## 🎓 **Benefits:**

### **For Debugging:**
- See exactly where detection fails
- Understand outlier rejection
- Visualize Kalman filtering effect

### **For Tuning:**
- Adjust outlier thresholds
- See impact of Kalman parameters
- Identify problematic areas

### **For Presentation:**
- Professional diagnostic view
- Clear visualization of pipeline
- Easy to understand system behavior

---

## 📚 **Reference:**

- **Peter Moran's Blog:** https://petermoran.org/robust-lane-tracking/
- **GitHub:** https://github.com/peter-moran/highway-lane-tracker
- **Our Implementation:** `PETER_MORAN_IMPLEMENTATION.md`

---

**Status:** Visualization module created, ready to integrate! 🎨
