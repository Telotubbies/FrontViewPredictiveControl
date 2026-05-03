# Complete CARLA MPC Workflow Analysis
## จาก Camera Input ถึง Vehicle Control Output

---

## 📊 Full System Workflow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CARLA Camera  │───▶│   UNet Model     │───▶│  Binary Mask    │
│   (640x480)     │    │  (Deep Learning) │    │   (480x640)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Vehicle Ctrl   │◀───│   MPC Solver     │◀───│ Reference Path  │
│  (Steer,Throt)  │    │  (Optimization) │    │   (x,y,yaw)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        ▲
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Safety Override │◀───│   Fusion Logic   │◀───│  Lane Detection │
│   (Limits)      │    │ (Weight+Blend)   │    │ (Coefficients)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        ▲
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CARLA World   │◀───│  Waypoints      │◀───│   BEV Transform │
│   (Position)    │    │  (Route Path)    │    │ (Perspective)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 🔬 Phase-by-Phase Algorithm Analysis

### **Phase 1: Camera Input & UNet Processing**

#### **Input:**
```python
rgb_frame: np.ndarray  # Shape: (480, 640, 3), Range: [0, 255]
# CARLA camera capture at 640x480 RGB
```

#### **UNet Model Processing:**
```python
# Lane Detection Network Architecture
class LaneUNet(nn.Module):
    def __init__(self):
        # Encoder (Downsampling)
        self.enc1 = ConvBlock(3, 64)    # 480x640 → 240x320
        self.enc2 = ConvBlock(64, 128)  # 240x320 → 120x160
        self.enc3 = ConvBlock(128, 256) # 120x160 → 60x80
        self.enc4 = ConvBlock(256, 512) # 60x80 → 30x40
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)  # 30x40 → 30x40
        
        # Decoder (Upsampling)
        self.up4 = UpBlock(1024, 512)   # 30x40 → 60x80
        self.dec4 = ConvBlock(1024, 512) # 60x80 → 60x80
        self.up3 = UpBlock(512, 256)    # 60x80 → 120x160
        self.dec3 = ConvBlock(512, 256) # 120x160 → 120x160
        self.up2 = UpBlock(256, 128)    # 120x160 → 240x320
        self.dec2 = ConvBlock(256, 128) # 240x320 → 240x320
        self.up1 = UpBlock(128, 64)     # 240x320 → 480x640
        self.dec1 = ConvBlock(128, 64)  # 480x640 → 480x640
        
        # Final
        self.final = nn.Conv2d(64, 1, kernel_size=1)  # 480x640 → 480x640

def forward(self, x):
    # Normalize input
    x = (x / 255.0 - 0.5) / 0.5
    
    # Encoder path
    e1 = self.enc1(x)   # 480x640 → 240x320
    e2 = self.enc2(e1)  # 240x320 → 120x160
    e3 = self.enc3(e2)  # 120x160 → 60x80
    e4 = self.enc4(e3)  # 60x80 → 30x40
    
    # Bottleneck
    b = self.bottleneck(e4)  # 30x40 → 30x40
    
    # Decoder path (with skip connections)
    d4 = self.up4(b, e4)    # 30x40 → 60x80
    d4 = self.dec4(d4)
    d3 = self.up3(d4, e3)   # 60x80 → 120x160
    d3 = self.dec3(d3)
    d2 = self.up2(d3, e2)   # 120x160 → 240x320
    d2 = self.dec2(d2)
    d1 = self.up1(d2, e1)   # 240x320 → 480x640
    d1 = self.dec1(d1)
    
    # Final output
    out = torch.sigmoid(self.final(d1))  # 480x640 → 480x640
    return out
```

#### **Output:**
```python
binary_mask: np.ndarray  # Shape: (480, 640), Range: [0, 1]
confidence: float        # Range: [0, 1]
# Lane pixels = 1, background = 0
```

---

### **Phase 2: BEV (Bird's Eye View) Transform**

#### **Perspective Transform Mathematics:**
```python
# Camera to BEV coordinate transformation
def get_bev_transform(cam_w, cam_h, bev_top_ratio, bev_bot_ratio, 
                     bev_top_margin, bev_bot_margin):
    """
    Transform camera view to bird's eye view
    
    Camera coordinates (pixels):
    - Origin: top-left
    - X: rightward, Y: downward
    
    BEV coordinates (meters):
    - Origin: vehicle center
    - X: forward, Y: rightward
    """
    
    # Define source points (camera image)
    # Top trapezoid of camera view
    src_top = int(bev_top_ratio * cam_h)
    src_bot = int(bev_bot_ratio * cam_h)
    src_left = int(bev_top_margin * cam_w)
    src_right = int((1 - bev_top_margin) * cam_w)
    
    src_pts = np.array([
        [src_left, src_top],      # Top-left
        [src_right, src_top],     # Top-right
        [src_right, src_bot],     # Bottom-right
        [src_left, src_bot]       # Bottom-left
    ], dtype=np.float32)
    
    # Define destination points (BEV)
    # BEV dimensions in meters
    lookahead_m = 30.0    # Forward distance
    half_width_m = 8.0   # Lateral distance each side
    
    dst_pts = np.array([
        [-half_width_m, lookahead_m],   # Top-left
        [half_width_m, lookahead_m],    # Top-right
        [half_width_m, 0],              # Bottom-right
        [-half_width_m, 0]              # Bottom-left
    ], dtype=np.float32)
    
    # Convert to pixel coordinates
    bev_w = 450  # BEV width in pixels
    bev_h = 320  # BEV height in pixels
    
    # Scale meters to pixels
    scale_x = bev_w / (2 * half_width_m)    # pixels per meter (lateral)
    scale_y = bev_h / lookahead_m           # pixels per meter (forward)
    
    dst_pts[:, 0] = (dst_pts[:, 0] + half_width_m) * scale_x  # Y → X pixels
    dst_pts[:, 1] = (lookahead_m - dst_pts[:, 1]) * scale_y    # X → Y pixels (inverted)
    
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    
    return M, M_inv, src_pts, dst_pts

# Apply transform
bev_binary = cv2.warpPerspective(
    binary_mask, M, (bev_w, bev_h), 
    flags=cv2.INTER_NEAREST
)
```

#### **Output:**
```python
bev_binary: np.ndarray  # Shape: (320, 450), Range: [0, 255]
# Top = far (30m), Bottom = near (0m)
# Left = left side (-8m), Right = right side (+8m)
```

---

### **Phase 3: Lane Boundary Detection**

#### **Sliding Window Algorithm:**
```python
def sliding_window_search(bev_binary):
    """
    Detect lane boundaries using histogram-based sliding window
    """
    bev_h, bev_w = bev_binary.shape
    
    # 1. Histogram for initial lane positions
    histogram = np.sum(bev_binary[bev_h//2:, :], axis=0)  # Bottom half
    
    # Find peaks (lane centers)
    left_peak = np.argmax(histogram[:bev_w//2])
    right_peak = np.argmax(histogram[bev_w//2:]) + bev_w//2
    
    # 2. Sliding window search
    n_windows = 9
    window_height = bev_h // n_windows
    margin = 50  # Window width/2
    
    left_pixels = []
    right_pixels = []
    
    for window in range(n_windows):
        # Window boundaries
        win_y_low = bev_h - (window + 1) * window_height
        win_y_high = bev_h - window * window_height
        
        # Left window
        win_x_left_low = left_peak - margin
        win_x_left_high = left_peak + margin
        
        # Right window  
        win_x_right_low = right_peak - margin
        win_x_right_high = right_peak + margin
        
        # Find non-zero pixels in windows
        left_window = bev_binary[win_y_low:win_y_high, win_x_left_low:win_x_left_high]
        right_window = bev_binary[win_y_low:win_y_high, win_x_right_low:win_x_right_high]
        
        # Extract pixel coordinates
        left_nonzero = left_window.nonzero()
        right_nonzero = right_window.nonzero()
        
        # Convert to image coordinates
        left_px = [
            [win_y_low + left_nonzero[0][i], win_x_left_low + left_nonzero[1][i]]
            for i in range(len(left_nonzero[0]))
        ]
        right_px = [
            [win_y_low + right_nonzero[0][i], win_x_right_low + right_nonzero[1][i]]
            for i in range(len(right_nonzero[0]))
        ]
        
        left_pixels.extend(left_px)
        right_pixels.extend(right_px)
        
        # Update window centers for next iteration
        if len(left_px) > 0:
            left_peak = int(np.mean([p[1] for p in left_px]))
        if len(right_px) > 0:
            right_peak = int(np.mean([p[1] for p in right_px]))
    
    return np.array(left_pixels), np.array(right_pixels)
```

#### **Polynomial Fitting:**
```python
def fit_lane_polynomial(pixels, bev_h, bev_w, lookahead_m, half_width_m):
    """
    Fit polynomial to lane pixels and convert to vehicle coordinates
    """
    if len(pixels) < 10:  # Minimum points for fitting
        return None
    
    # Convert BEV pixels to vehicle coordinates
    # BEV: (row, col) → Vehicle: (x_forward, y_lateral)
    rows = pixels[:, 0]  # BEV y (inverted)
    cols = pixels[:, 1]  # BEV x
    
    # Convert to meters
    x_m = (bev_h - rows) * lookahead_m / bev_h      # Row → forward distance
    y_m = (cols - bev_w/2) * 2 * half_width_m / bev_w  # Col → lateral distance
    
    # Fit polynomial: y = ax² + bx + c
    # Where x = forward distance, y = lateral offset
    coeffs = np.polyfit(x_m, y_m, deg=2)  # [a, b, c]
    
    return coeffs
```

#### **Output:**
```python
left_coeffs: np.ndarray   # [a, b, c] for left lane: y = ax² + bx + c
right_coeffs: np.ndarray  # [a, b, c] for right lane: y = ax² + bx + c
# x = forward distance (0-30m), y = lateral offset (-8m to +8m)
```

---

### **Phase 4: Reference Path Generation**

#### **Fusion Logic:**
```python
def generate_reference_path(waypoints, left_coeffs, right_coeffs, 
                          wp_weight=0.8, lane_weight=0.2):
    """
    Fuse waypoints and lane detection for reference path
    """
    
    # 1. Waypoint-based reference
    if waypoints is not None and len(waypoints) > 0:
        # Waypoints are in world coordinates
        wp_x = [wp.transform.location.x for wp in waypoints]
        wp_y = [wp.transform.location.y for wp in waypoints]
        wp_yaw = [wp.transform.rotation.yaw for wp in waypoints]
        
        # Convert to vehicle frame
        wp_ref = world_to_vehicle_frame(wp_x, wp_y, wp_yaw)
    
    # 2. Lane-based reference  
    if left_coeffs is not None and right_coeffs is not None:
        # Sample points along lanes
        x_sample = np.linspace(0, 30, 50)  # 0-30m forward
        
        left_y = np.polyval(left_coeffs, x_sample)
        right_y = np.polyval(right_coeffs, x_sample)
        
        # Centerline = average of left and right
        center_y = (left_y + right_y) / 2
        
        # Calculate heading from centerline
        dy_dx = np.gradient(center_y, x_sample)
        lane_yaw = np.arctan2(dy_dx, 1)  # Heading angle
        
        lane_ref = (x_sample, center_y, lane_yaw)
    
    # 3. Fusion
    if wp_ref is not None and lane_ref is not None:
        # Weighted average
        fused_x = wp_weight * wp_ref[0] + lane_weight * lane_ref[0]
        fused_y = wp_weight * wp_ref[1] + lane_weight * lane_ref[1]
        fused_yaw = wp_weight * wp_ref[2] + lane_weight * lane_ref[2]
        
        reference_path = (fused_x, fused_y, fused_yaw)
    elif wp_ref is not None:
        reference_path = wp_ref  # Waypoint-only
    elif lane_ref is not None:
        reference_path = lane_ref  # Lane-only
    else:
        reference_path = None  # No reference
    
    return reference_path
```

#### **Output:**
```python
x_ref: np.ndarray      # Forward distances [m]
y_ref: np.ndarray      # Lateral offsets [m] 
yaw_ref: np.ndarray    # Heading angles [rad]
# Shape: (N,) where N is number of reference points
```

---

### **Phase 5: MPC (Model Predictive Control) Optimization**

#### **MPC Problem Formulation:**
```python
class LaneMPC:
    def __init__(self, N=10, dt=0.1, L=2.7):
        """
        N: Prediction horizon steps
        dt: Time step [s]
        L: Wheelbase [m]
        """
        self.N = N
        self.dt = dt
        self.L = L
        
        # Cost function weights
        self.w_cte = 1.0        # Cross-track error
        self.w_heading = 0.5    # Heading error
        self.w_vel = 0.1        # Velocity error
        self.w_steer = 0.01     # Steering effort
        self.w_accel = 0.01     # Acceleration effort
        self.w_steer_rate = 0.1 # Steering change rate
    
    def solve(self, x_ref, y_ref, yaw_ref, v_ref, current_state):
        """
        Solve MPC optimization problem
        
        State vector: [x, y, yaw, v]
        Control vector: [steer, accel]
        """
        # 1. Kinematic bicycle model
        def vehicle_dynamics(state, control):
            x, y, yaw, v = state
            steer, accel = control
            
            # Bicycle model equations
            x_next = x + v * np.cos(yaw) * self.dt
            y_next = y + v * np.sin(yaw) * self.dt
            yaw_next = yaw + (v / self.L) * np.tan(steer) * self.dt
            v_next = v + accel * self.dt
            
            return np.array([x_next, y_next, yaw_next, v_next])
        
        # 2. Cost function
        def stage_cost(state, control, ref_state):
            # Tracking errors
            cte = state[1] - ref_state[1]  # Cross-track error
            heading_err = state[2] - ref_state[2]
            vel_err = state[3] - ref_state[3]
            
            # Control effort
            steer, accel = control
            
            # Total stage cost
            cost = (
                self.w_cte * cte**2 +
                self.w_heading * heading_err**2 +
                self.w_vel * vel_err**2 +
                self.w_steer * steer**2 +
                self.w_accel * accel**2
            )
            
            return cost
        
        # 3. Constraints
        # Steering angle: [-0.45, 0.45] rad (≈ ±26°)
        # Acceleration: [-3.0, 3.0] m/s²
        # Velocity: [0, 30] m/s
        # Steering rate: [-0.5, 0.5] rad/s
        
        # 4. Optimization (simplified - actual uses OSQP/CVXOPT)
        optimal_controls = self.optimize_mpc(
            current_state, x_ref, y_ref, yaw_ref, v_ref
        )
        
        return optimal_controls[0]  # First control action
```

#### **Output:**
```python
steering_angle: float    # Range: [-0.45, 0.45] rad
acceleration: float     # Range: [-3.0, 3.0] m/s²
```

---

### **Phase 6: Safety Override & Vehicle Control**

#### **Safety Limits:**
```python
class SafetyOverride:
    def __init__(self):
        self.max_steer_rad = 0.45      # ±26° steering
        self.max_accel = 3.0           # 3.0 m/s² acceleration
        self.max_brake = 8.0           # 8.0 m/s² braking
        self.max_speed = 30.0           # 30 m/s max speed
        self.cte_limit = 2.0            # 2m CTE limit
    
    def apply_safety(self, steer, accel, brake, current_cte, current_speed):
        """Apply safety limits to control outputs"""
        
        # 1. Steering limits
        steer_safe = np.clip(steer, -self.max_steer_rad, self.max_steer_rad)
        
        # 2. CTE-based intervention
        if abs(current_cte) > self.cte_limit:
            # Reduce steering if too far from path
            steer_safe *= 0.5
        
        # 3. Speed limits
        if current_speed > self.max_speed:
            brake = max(brake, (current_speed - self.max_speed) * 0.5)
        
        # 4. Acceleration/brake limits
        accel_safe = np.clip(accel, -self.max_brake, self.max_accel)
        brake_safe = np.clip(brake, 0, self.max_brake)
        
        # 5. Conflict resolution
        if accel_safe > 0 and brake_safe > 0:
            # Prefer braking over acceleration
            accel_safe = 0
        
        return steer_safe, accel_safe, brake_safe
```

#### **CARLA Vehicle Control:**
```python
def apply_vehicle_control(steer, accel, brake, vehicle):
    """Apply control to CARLA vehicle"""
    
    # Convert to CARLA control format
    control = carla.VehicleControl()
    
    # Steering: [-1, 1] (our rad to normalized)
    control.steer = np.clip(steer / 0.45, -1.0, 1.0)
    
    # Throttle: [0, 1]
    if accel > 0:
        control.throttle = np.clip(accel / 3.0, 0.0, 1.0)
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = np.clip(brake / 8.0, 0.0, 1.0)
    
    # Apply control
    vehicle.apply_control(control)
```

#### **Final Output:**
```python
carla.VehicleControl(
    steer: float,      # [-1, 1] normalized steering
    throttle: float,   # [0, 1] throttle  
    brake: float       # [0, 1] brake
)
```

---

## 📈 Complete Data Flow Summary

### **Input → Output Transformations:**

```
CARLA Camera (640x480x3 RGB)
        ↓
    UNet Model (Deep Learning)
        ↓
Binary Lane Mask (480x640)
        ↓
BEV Perspective Transform
        ↓
BEV Binary Mask (320x450)
        ↓
Sliding Window Detection
        ↓
Lane Pixel Points
        ↓
Polynomial Fitting
        ↓
Lane Coefficients [a,b,c]
        ↓
Fusion with Waypoints
        ↓
Reference Path [x,y,yaw]
        ↓
MPC Optimization
        ↓
Control Commands [steer,accel]
        ↓
Safety Override
        ↓
CARLA Vehicle Control
```

### **Key Mathematical Functions:**

1. **UNet Forward Pass**: Deep neural network with encoder-decoder architecture
2. **Perspective Transform**: 3x3 homography matrix for camera→BEV conversion
3. **Histogram Peak Detection**: Find lane centers using intensity distribution
4. **Polynomial Fitting**: Least-squares fit to lane boundary points
5. **Coordinate Transform**: World↔Vehicle frame rotations and translations
6. **MPC Optimization**: Constrained optimization using kinematic bicycle model
7. **Safety Clipping**: Bounding control outputs within physical limits

### **Performance Characteristics:**

- **Input Rate**: 20-30 Hz (camera capture)
- **UNet Inference**: ~10-20ms (GPU)
- **BEV Transform**: ~1ms (CPU)
- **Lane Detection**: ~5ms (CPU)
- **MPC Solve**: ~5-10ms (CPU)
- **Total Latency**: ~25-45ms per frame

**🎯 This complete workflow processes raw camera images to vehicle control commands in real-time using deep learning, computer vision, and optimization algorithms!**
