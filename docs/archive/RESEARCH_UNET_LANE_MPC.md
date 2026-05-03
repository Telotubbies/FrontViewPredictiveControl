# 📚 Research: UNet Lane Detection + MPC Control

## 🎯 วัตถุประสงค์
ค้นคว้าวิธีการใช้ UNet สำหรับ lane detection แล้วนำมาใช้กับ MPC control จาก research papers และ GitHub (โดยเฉพาะ Baidu Apollo)

---

## 📄 Research Papers

### 1. **End-to-End Deep Learning of Lane Detection and Path Prediction** (2021)
**Paper:** https://arxiv.org/abs/2102.04738

**Key Contributions:**
- **DSUNet (Depthwise Separable UNet)**
  - ใช้ depthwise separable convolutions แทน standard convolutions
  - น้ำหนักเบากว่า UNet ถึง **5.16x**
  - เร็วกว่า UNet ถึง **1.61x**
  
- **End-to-End Learning:**
  - Input: RGB image
  - Output: Lane segmentation + Path prediction (curvature, lateral offset)
  - ไม่ต้องแยก lane detection กับ path planning

- **CNN-PP (CNN + Path Prediction):**
  - Integrate path prediction algorithm กับ CNN
  - Output: Predicted curvature และ lateral offset สำหรับ path planning
  - ทดสอบใน simulation และ real car

**ผลลัพธ์:**
- DSUNet-PP ดีกว่า UNet-PP ใน mean average errors
- ทดสอบบนรถจริง: lateral error ต่ำกว่า modified UNet

**การประยุกต์ใช้:**
```python
# Architecture
class DSUNet:
    # Encoder: Depthwise separable convolutions
    # Decoder: Upsampling + skip connections
    # Output: Lane mask + Path parameters (curvature, offset)
    
# Path Prediction
def predict_path(lane_mask):
    # Extract lane center
    # Fit polynomial or spline
    # Compute curvature and lateral offset
    return curvature, lateral_offset

# MPC Integration
def mpc_control(curvature, lateral_offset, speed):
    # Use predicted path as reference
    # MPC optimizes steering and throttle
    return steering, throttle
```

---

### 2. **Development of Testbed for Autonomous Vehicles: MPC + Lane Detection** (2024)
**Paper:** https://arxiv.org/abs/2511.19655

**Key Contributions:**
- **Lane Detection Pipeline:**
  1. Edge detection (Canny)
  2. Sliding window-based straight-line identification
  3. Dynamic ROI extraction
  
- **MPC Controller:**
  - Bicycle vehicle dynamics model
  - Tracks identified lane line
  - Tested in ROS Gazebo simulation

**ผลลัพธ์:**
- RMSE ลดลง **27.65%** เมื่อเทียบกับ baseline
- High robustness and flexibility

**การประยุกต์ใช้:**
```python
# Lane Detection
def detect_lane(image):
    # 1. Edge detection
    edges = cv2.Canny(image, threshold1, threshold2)
    
    # 2. Dynamic ROI
    roi = extract_dynamic_roi(edges)
    
    # 3. Sliding window
    left_lane, right_lane = sliding_window_search(roi)
    
    # 4. Line fitting
    left_fit = np.polyfit(left_lane[:, 0], left_lane[:, 1], 2)
    right_fit = np.polyfit(right_lane[:, 0], right_lane[:, 1], 2)
    
    return left_fit, right_fit

# MPC with Bicycle Model
def mpc_bicycle_model(state, lane_fit, horizon=10):
    # State: [x, y, theta, v]
    # Control: [delta, a] (steering, acceleration)
    
    # Dynamics:
    # x_dot = v * cos(theta)
    # y_dot = v * sin(theta)
    # theta_dot = v / L * tan(delta)
    # v_dot = a
    
    # Cost function:
    # J = sum(w_cte * cte^2 + w_heading * heading_error^2 + 
    #         w_steer * delta^2 + w_steer_rate * delta_dot^2)
    
    return optimal_steering, optimal_accel
```

---

### 3. **Improved Lane Detection with Deep Learning** (2024)
**Paper:** https://ojs.southfloridapublishing.com/ojs/index.php/jdev/article/view/4532

**Key Contributions:**
- **Multi-Sensor Data Fusion:**
  - Camera + LiDAR + Radar
  - Semantic segmentation (UNet) + Edge detection
  - Robust to various lighting conditions

- **Improved UNet Architecture:**
  - Attention mechanisms
  - Multi-scale feature extraction
  - Better performance in complex scenarios

**การประยุกต์ใช้:**
```python
# Multi-Sensor Fusion
def fuse_sensors(camera, lidar, radar):
    # Camera: Semantic segmentation (UNet)
    lane_mask_camera = unet_model(camera)
    
    # LiDAR: 3D lane detection
    lane_points_lidar = detect_lane_3d(lidar)
    
    # Fusion: Weighted combination
    lane_mask_fused = alpha * lane_mask_camera + 
                      beta * project_3d_to_2d(lane_points_lidar)
    
    return lane_mask_fused
```

---

## 🚀 Baidu Apollo Perception

### **Apollo 5.0 Perception Module**
**Source:** https://github.com/ApolloAuto/apollo

**Architecture:**
```
Camera Input
    ↓
Lane Detection (CNN-based)
    ↓
Lane Post-Processing
    ↓
Lane Line Fitting
    ↓
Output: Lane Boundaries + Center Line
```

**Key Features:**

1. **Deep Learning Framework:**
   - Supports **Caffe** and **PaddlePaddle**
   - PaddlePaddle: Baidu's deep learning platform
   - Efficient, flexible, scalable

2. **Lane Detection:**
   - CNN-based semantic segmentation
   - Vanishing point detection
   - Closest In-Path Object (CIPO) detection

3. **Sensor Calibration:**
   - Online sensor calibration service
   - Manual camera calibration
   - Ensures accurate lane detection

4. **Limitations:**
   - ❌ Does NOT support high curvature roads
   - ❌ Does NOT support roads without lane lines
   - ⚠️ Visual detection degrades at night or with strong light

**Apollo Lane Detection Pipeline:**

```python
# Simplified Apollo-style pipeline
class ApolloLaneDetection:
    def __init__(self):
        self.cnn_model = load_paddlepaddle_model()
        self.calibration = CameraCalibration()
    
    def detect(self, image):
        # 1. Preprocessing
        image = self.calibration.undistort(image)
        
        # 2. CNN Inference
        lane_mask = self.cnn_model.predict(image)
        
        # 3. Post-processing
        lane_mask = morphological_operations(lane_mask)
        
        # 4. Lane Line Fitting
        left_lane, right_lane = fit_lane_lines(lane_mask)
        
        # 5. Vanishing Point Detection
        vanishing_point = detect_vanishing_point(left_lane, right_lane)
        
        # 6. Output
        return {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'center_line': (left_lane + right_lane) / 2,
            'vanishing_point': vanishing_point,
            'confidence': compute_confidence(lane_mask)
        }
```

---

## 🔧 Best Practices for UNet + MPC Integration

### **1. Lane Detection → Reference Path**

```python
def lane_to_reference_path(left_lane, right_lane, lookahead=50):
    """Convert lane detection to MPC reference path."""
    
    # 1. Compute center line
    center_x = (left_lane[:, 0] + right_lane[:, 0]) / 2
    center_y = (left_lane[:, 1] + right_lane[:, 1]) / 2
    
    # 2. Fit smooth curve (B-spline or polynomial)
    from scipy.interpolate import splprep, splev
    tck, u = splprep([center_x, center_y], s=0, k=3)
    
    # 3. Sample points along curve
    u_new = np.linspace(0, 1, lookahead)
    x_ref, y_ref = splev(u_new, tck)
    
    # 4. Compute heading and curvature
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    heading = np.arctan2(dy, dx)
    
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    
    return x_ref, y_ref, heading, curvature
```

### **2. MPC Cost Function Design**

```python
def mpc_cost_function(state, control, reference):
    """MPC cost function for lane keeping."""
    
    # State: [x, y, theta, v]
    # Control: [delta, a]
    # Reference: [x_ref, y_ref, theta_ref, v_ref]
    
    # Cross-track error (CTE)
    cte = compute_cte(state, reference)
    
    # Heading error
    heading_error = state[2] - reference[2]
    
    # Speed error
    speed_error = state[3] - reference[3]
    
    # Cost
    cost = (
        w_cte * cte**2 +
        w_heading * heading_error**2 +
        w_speed * speed_error**2 +
        w_steer * control[0]**2 +
        w_accel * control[1]**2 +
        w_steer_rate * (control[0] - prev_control[0])**2 +
        w_jerk * (control[1] - prev_control[1])**2
    )
    
    return cost
```

### **3. Handling UNet Uncertainty**

```python
def robust_lane_detection(unet_output, confidence_threshold=0.3):
    """Handle low confidence UNet predictions."""
    
    # 1. Compute confidence
    confidence = compute_lane_confidence(unet_output)
    
    # 2. If confidence is low, use fallback
    if confidence < confidence_threshold:
        # Option A: Use previous frame (temporal smoothing)
        lane_output = temporal_buffer.get_smoothed()
        
        # Option B: Use waypoint/GPS
        lane_output = get_waypoint_lane()
        
        # Option C: Use lane completion
        lane_output = complete_lane_from_partial(unet_output)
    else:
        lane_output = unet_output
    
    # 3. Kalman filter for smoothing
    lane_output = kalman_filter.update(lane_output)
    
    return lane_output, confidence
```

---

## 📊 Comparison: Different Approaches

| Method | Accuracy | Speed | Robustness | Complexity |
|--------|----------|-------|------------|------------|
| **Classical CV** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| **UNet** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **DSUNet** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Apollo CNN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Multi-Sensor** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎯 Recommendations for Your Project

### **Immediate Improvements:**

1. **Lower UNet Threshold** ✅ (Already done: 0.1)
   - Increases detection coverage
   - May introduce more noise

2. **Add Temporal Smoothing:**
   ```python
   # Kalman filter for lane coefficients
   class LaneKalmanFilter:
       def update(self, measurement):
           # Predict
           self.state = self.A @ self.state
           self.P = self.A @ self.P @ self.A.T + self.Q
           
           # Update
           K = self.P @ self.H.T @ inv(self.H @ self.P @ self.H.T + self.R)
           self.state = self.state + K @ (measurement - self.H @ self.state)
           self.P = (I - K @ self.H) @ self.P
           
           return self.state
   ```

3. **Improve Lane Fitting:**
   - Use B-spline instead of polynomial ✅ (Already implemented)
   - Add geometric constraints (parallel lanes, width)
   - Use RANSAC for outlier rejection

### **Medium-term Improvements:**

4. **Implement DSUNet:**
   - Lighter and faster than UNet
   - Better for real-time applications
   - Can be trained on CARLA data

5. **Add Vanishing Point Detection:**
   ```python
   def detect_vanishing_point(left_lane, right_lane):
       # Fit lines to left and right lanes
       left_line = np.polyfit(left_lane[:, 0], left_lane[:, 1], 1)
       right_line = np.polyfit(right_lane[:, 0], right_lane[:, 1], 1)
       
       # Find intersection (vanishing point)
       vp_x = (right_line[1] - left_line[1]) / (left_line[0] - right_line[0])
       vp_y = left_line[0] * vp_x + left_line[1]
       
       return vp_x, vp_y
   ```

6. **Improve MPC Weights:**
   - Use adaptive weights based on road curvature
   - Increase weights in curves, decrease on straight roads

### **Long-term Improvements:**

7. **Train Better UNet Model:**
   - Collect more CARLA data
   - Use data augmentation
   - Train with proper loss function (focal loss, dice loss)

8. **Multi-Sensor Fusion:**
   - Add LiDAR for 3D lane detection
   - Fuse camera + LiDAR + GPS/IMU

9. **End-to-End Learning:**
   - Train network to output path directly
   - Skip intermediate lane detection step
   - Faster and potentially more accurate

---

## 📚 References

1. **DSUNet Paper:** https://arxiv.org/abs/2102.04738
2. **MPC + Lane Detection:** https://arxiv.org/abs/2511.19655
3. **Apollo Perception:** https://github.com/ApolloAuto/apollo
4. **Improved Lane Detection:** https://ojs.southfloridapublishing.com/ojs/index.php/jdev/article/view/4532
5. **Lane Detection Survey:** https://arxiv.org/html/2411.16316v2

---

## 💡 Key Takeaways

1. **UNet is good but can be improved:**
   - Use DSUNet for better speed/accuracy tradeoff
   - Add attention mechanisms
   - Use proper training data

2. **MPC integration requires:**
   - Smooth reference path from lane detection
   - Proper cost function design
   - Handling of detection uncertainty

3. **Apollo approach:**
   - CNN-based segmentation
   - Robust post-processing
   - Sensor calibration is critical

4. **Best practices:**
   - Temporal smoothing (Kalman filter)
   - Geometric constraints
   - Fallback mechanisms for low confidence

5. **Your current implementation is on the right track:**
   - ✅ UNet detection
   - ✅ Sliding window + polynomial fitting
   - ✅ MPC control
   - ✅ Fusion with waypoints
   - 🔧 Can improve: Better UNet model, B-spline fitting, adaptive weights
