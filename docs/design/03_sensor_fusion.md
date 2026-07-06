# 03 — Multi-Sensor Fusion (Design Document)

> **Project:** FrontViewPredictiveControl — Tesla-like prototype
> **Scope:** Sensor-suite calibration, low/mid/high-level fusion, EKF/UKF state
> estimation, occupancy-grid mapping, degradation handling
> **Status:** Design / partial implementation
> **Owner:** Fusion & State-Estimation team
> **Related code:** `src/alg/fusion.py` (WP + UNet centering blend), `src/perception/`
> (UNet lanes, BEV, Kalman lane tracker), `src/alg/reference.py`, `src/alg/lka_step.py`
> **No reinforcement learning** — all fusion is classical Bayesian filtering +
> deterministic arbitration.

---

## 0. Executive Summary

A real autonomous vehicle cannot rely on a single modality. This document specifies the
**Multi-Sensor Fusion** layer that combines an 8-camera vision stack, a roof-mounted
mechanical LiDAR, front/side radars, ultrasonic parking rings, a 6-DoF IMU, and a
RTK-corrected GNSS into one internally-consistent estimate of (a) the ego pose and
velocity, (b) the surrounding dynamic-object list, and (c) a probabilistic occupancy
grid of the static world.

The design is **Bayesian, geometry-first, and RL-free**. Ego state is estimated with an
Error-State Extended Kalman Filter (EKF) for nominal operation and an Unscented Kalman
Filter (UKF) for high-curvature / high-slip maneuvers where the linearization error of
the EKF becomes unacceptable. Object-level fusion is mid-level (track-centric) with a
low-level radar-vision early-fusion channel for longitudinal distance. The occupancy
grid is a classic Bayesian / Dempster-Shafer grid that ingests LiDAR rays, vision
freespace, and radar returns.

The existing `src/alg/fusion.py` is a **lane-state fusion** module: it blends a
waypoint-derived state `(cte, heading, curvature)` with a UNet-derived lane state using a
confidence-gated, curvature-adaptive weight `w_unet`. This document generalizes that
pattern to the full sensor suite while preserving the same arbitration philosophy:
**primary source + confidence-gated assist + degradation fallback**.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          MULTI-SENSOR FUSION STACK                               │
│                                                                                  │
│  [8 Cam] [LiDAR] [Radar] [USS] [IMU] [GNSS]                                      │
│      │       │       │      │      │      │                                      │
│      ▼       ▼       ▼      ▼      ▼      ▼                                      │
│  ┌──────────────────────────────────────────────────────┐                        │
│  │  Low-Level Fusion (raw / detection level)            │                        │
│  │  radar+vision early fusion, LiDAR clustering,        │                        │
│  │  IMU pre-integration, GNSS pull-in                   │                        │
│  └───────────────────────┬──────────────────────────────┘                        │
│                           ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐                        │
│  │  Mid-Level Fusion (track / feature level)            │                        │
│  │  Joint Probabilistic Data Association (JPDA),        │                        │
│  │  Covariance Intersection across modalities           │                        │
│  └───────────────────────┬──────────────────────────────┘                        │
│                           ▼                                                      │
│  ┌──────────────────┬─────────────────────┐  ┌────────────────────────────┐      │
│  │ EKF / UKF        │ Occupancy Grid Map  │  │ Lane-State Fusion          │      │
│  │ Ego State (x,y,  │ (Bayesian / D-S)    │  │ (existing src/alg/fusion.py)│     │
│  │  v, θ, ω, a, β)  │ 2.5D cells          │  │ WP + UNet centering blend  │      │
│  └────────┬─────────┴──────────┬──────────┘  └─────────────┬──────────────┘      │
│           │                    │                           │                      │
│           └──────────┬─────────┴───────────────────────────┘                      │
│                      ▼                                                            │
│           [Scene Model] → MPC / Behavior Planner / Visualization                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Sensor Suite

### 1.1 Modalities and roles

| Sensor | Count | Role | Rate | Range / Accuracy |
|--------|-------|------|------|------------------|
| Forward narrow (cam) | 1 | Long-range objects, TSR/TLR | 30 Hz | 250 m, ±0.3° |
| Forward main (cam) | 1 | Primary perception, depth | 30 Hz | 150 m |
| Forward wide (cam) | 1 | Close FOV, fisheye, cut-in | 30 Hz | 60 m |
| Forward side (cam) | 2 | Blind-spot, intersection | 30 Hz | 80 m |
| Rear side (cam) | 2 | Lane change, cross-traffic | 30 Hz | 80 m |
| Rear (cam) | 1 | Reverse, rear collision | 30 Hz | 50 m |
| LiDAR (mechanical) | 1 | Geometry, occupancy, ground | 10–20 Hz | 200 m, ±2 cm |
| Long-range radar | 1–3 | Velocity, distance, occlusion | 20 Hz | 250 m, ±0.1 m/s |
| Ultrasonic (USS) | 12 ring | Parking, near-field | 10 Hz | 5 m, ±3 cm |
| IMU (6-DoF) | 1 | Ego dynamics, pre-integration | 200 Hz | — |
| GNSS (RTK) | 1 | Absolute pose pull-in | 10 Hz | ±2 cm (RTK) |

### 1.2 Why a multi-modal suite (not vision-only)

Tesla's pure-vision FSD is economically attractive but trades sensor redundancy for
cost. This prototype keeps a LiDAR + radar + IMU + GNSS suite because:

1. **Geometric ground truth** — LiDAR gives metric depth that vision must *infer*;
   it is the calibration anchor for the camera extrinsics and the occupancy grid.
2. **Velocity directness** — radar Doppler gives radial velocity without optical-flow
   ambiguity; critical for cut-in and emergency-brake distance estimation.
3. **Weather robustness** — radar penetrates fog/rain where cameras degrade; LiDAR
   degrades less than cameras in low light.
4. **Ego-state observability** — IMU + GNSS make the ego pose independent of the
   vision pipeline, so a vision blackout never loses the vehicle.

---

## 2. Calibration

### 2.1 Intrinsic calibration (per camera)

Pinhole + Brown-Conrady distortion:

$$
\mathbf{x}_p = K \, \pi(R \, X_c + t), \qquad
\mathbf{x}_d = \mathbf{x}_p \big(1 + k_1 r^2 + k_2 r^4 + k_3 r^6\big) + \big[2 p_1 x y + p_2 (r^2 + 2 x^2)\big]
$$

with $K = \text{diag}(f_x, f_y, 1)$, principal point $(c_x, c_y)$, and distortion
$(k_1, k_2, k_3, p_1, p_2)$. Estimated with a planar chessboard via Zhang's method;
refined with a LiDAR-camera registration residual against a shared target board.

### 2.2 Extrinsic calibration (sensor-to-vehicle)

Each sensor has a rigid transform $T^{v}_{s} \in SE(3)$ to the rear-axle vehicle frame
$\mathcal{V}$. The calibration graph is:

```
        GNSS antenna
            │  T_v^gnss
            ▼
   ┌────────────────┐  T_v^imu   ┌─────┐
   │  Vehicle frame │◄───────────│ IMU │
   │  (rear axle)   │            └─────┘
   │   𝒱            │  T_v^lidar ┌─────┐
   │                │◄───────────│LiDAR│
   │                │            └─────┘
   │                │  T_v^radar ┌─────┐
   │                │◄───────────│Radar│
   │                │            └─────┘
   │                │  T_v^cam_i ┌─────┐
   │                │◄───────────│ Cam │  × 8
   └────────────────┘            └─────┘
```

LiDAR-camera extrinsics are solved by minimizing the point-to-plane residual between
LiDAR edge points and projected chessboard edges:

$$
\min_{T} \sum_{i} \rho\!\left( \big\| \, \pi\!\big(K (R X_i^{L} + t)\big) - e_i^{img} \, \big\|^2 \right)
$$

where $e_i^{img}$ are sub-pixel image-edge samples and $\rho$ is a Huber kernel.
IMU extrinsics include the lever arm $\ell^{v}_{imu}$ and the axis misalignment
$R^{v}_{imu}$; both are jointly estimated with the EKF bias states during a calibration
drive (figure-8 + braking).

### 2.3 Temporal calibration

All sensors are time-stamped at the hardware level (PTP / PPS). A per-sensor latency
offset $\delta t_s$ is estimated by cross-correlating IMU angular rate with vision-derived
optical-flow rotation, and LiDAR edge sweeps with camera edges. The fusion buffer
re-bases every measurement to the IMU clock:

$$
t_{meas}^{imu} = t_{meas}^{s} - \delta t_s
$$

---

## 3. Fusion Levels

### 3.1 Low-level fusion (raw / detection)

- **Radar + vision early fusion**: radar detections (range, range-rate, azimuth) are
  projected into each camera's image and associated with 2D bounding boxes by IoU + range
  consistency. A radar-confirmed box gets a tighter longitudinal covariance; a
  radar-only return with no vision match is kept as a "ghost track" with high
  longitudinal but low lateral confidence.
- **LiDAR clustering**: Euclidean clustering on ground-removed points yields 3D
  proposals; these are associated to vision boxes by 2D IoU of the projected cluster
  hull. The fused 3D box inherits vision semantic class + LiDAR metric extent.

### 3.2 Mid-level fusion (track / feature)

Each modality runs its own tracker (vision: SORT/ByteTrack on BEV; radar: alpha-beta;
LiDAR: constant-velocity Kalman on cluster centroids). A **Covariance Intersection
(CI)** combiner merges correlated tracks:

$$
P_{fused}^{-1} = \omega P_1^{-1} + (1-\omega) P_2^{-1}, \qquad
\hat{x}_{fused} = P_{fused}\big(\omega P_1^{-1}\hat{x}_1 + (1-\omega) P_2^{-1}\hat{x}_2\big)
$$

with $\omega = \arg\min \text{tr}(P_{fused})$. CI is consistent even when the
cross-covariance between modalities is unknown — the key reason it is preferred over
naive Kalman fusion of dependent estimates.

### 3.3 High-level fusion (scene / state)

High-level fusion produces two artifacts:

1. **Ego state** $\hat{\mathbf{x}}_{ego}$ from the EKF/UKF (Section 4).
2. **Scene model**: a unified object list (class, pose, velocity, covariance, existence
   probability, source flags) + occupancy grid (Section 5) + lane state
   `(cte, heading, curvature)` from `src/alg/fusion.py` (Section 6).

---

## 4. EKF / UKF State Estimation

### 4.1 State vector

We use an error-state (indirect) formulation in the rear-axle frame:

$$
\mathbf{x} = \big[\, \mathbf{p}^n,\; \mathbf{v}^n,\; \mathbf{q}^n_b,\; \mathbf{b}_a,\; \mathbf{b}_g,\; \omega,\; a_x,\; \beta \,\big]^{\top}
$$

where $\mathbf{p}^n$ is position in the navigation frame, $\mathbf{v}^n$ velocity,
$\mathbf{q}^n_b$ the body-to-nav quaternion, $\mathbf{b}_a, \mathbf{b}_g$ IMU biases,
$\omega$ yaw rate, $a_x$ longitudinal acceleration, and $\beta$ the vehicle sideslip
angle. Slip is included because the predictive controller (MPC) needs it for the
bicycle-model feed-forward term.

### 4.2 Kinematic motion model (bicycle + slip)

$$
\dot{x} = v \cos(\theta + \beta), \quad
\dot{y} = v \sin(\theta + \beta), \quad
\dot{\theta} = \frac{v}{L_r}\sin\beta, \quad
\dot{v} = a_x, \quad
\beta = \arctan\!\left(\frac{L_r}{L_r + L_f}\tan\delta_f\right)
$$

with $L_f, L_r$ the front/rear axle distances. The IMU propagates the high-rate
prediction; GNSS, vision pose, and LiDAR pose are measurement updates.

### 4.3 EKF prediction (IMU pre-integration)

Between two keyframes $t_{k}, t_{k+1}$, IMU measurements
$(\tilde{\mathbf{a}}, \tilde{\boldsymbol{\omega}})$ are pre-integrated:

$$
\Delta\tilde{\mathbf{v}}_{k,k+1} = \iint_{t_k}^{t_{k+1}} R^{n}_{b}(\tau)\,\tilde{\mathbf{a}}(\tau)\,d\tau^2, \quad
\Delta\tilde{\mathbf{R}}_{k,k+1} = \prod R(\tilde{\boldsymbol{\omega}}\,dt)
$$

The error-state transition uses the standard 15-DoF IMU Jacobian $F$; the discrete
covariance propagation is:

$$
P_{k+1|k} = \Phi\, P_{k|k}\, \Phi^{\top} + G\, Q_{imu}\, G^{\top}
$$

### 4.4 EKF measurement update

For a measurement $\mathbf{z}$ (GNSS position, vision pose, or LiDAR pose) with
observation model $h(\mathbf{x})$:

$$
\mathbf{K}_k = P_{k|k-1} H_k^{\top}\big(H_k P_{k|k-1} H_k^{\top} + R_k\big)^{-1}
$$
$$
\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k\big(\mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})\big)
$$
$$
P_{k|k} = (I - \mathbf{K}_k H_k)\, P_{k|k-1}
$$

### 4.5 UKF fallback (high curvature / slip)

When $|\kappa| > \kappa_{switch}$ or the innovation normalized residual
$\|\mathbf{r}_k\|_{S_k^{-1}}$ exceeds a $\chi^2$ gate, the estimator switches to a UKF.
The UKF generates $2n+1$ sigma points via the Cholesky factor $S = \text{chol}(P)$:

$$
\mathcal{X}_0 = \hat{\mathbf{x}}, \quad
\mathcal{X}_i = \hat{\mathbf{x}} \pm (\sqrt{(n+\lambda)\,P})_i, \quad
\lambda = \alpha^2(n+\kappa_u) - n
$$

Sigma points are propagated through the *nonlinear* bicycle model and the measurement
model; the posterior mean and covariance are recovered with the standard unscented
weights $W_m, W_c$. The UKF is more expensive but avoids the Jacobian linearization
error that the EKF suffers during tight cornering or low-friction slip.

### 4.6 Sensor-specific update schedule

| Sensor | Update type | Rate | Gate |
|--------|-------------|------|------|
| IMU | Prediction | 200 Hz | always |
| GNSS (RTK) | Position | 10 Hz | $\chi^2$, 95% |
| Vision pose | Pose (x, y, θ) | 30 Hz | $\chi^2$, 99% |
| LiDAR pose | Pose + yaw | 20 Hz | $\chi^2$, 99% |
| Radar ego-v | Longitudinal v | 20 Hz | $\chi^2$, 95% |
| Wheel odometry | v, ω | 100 Hz | $\chi^2$, 95% |

---

## 5. Occupancy Grid

### 5.1 Representation

A 2.5D Bayesian occupancy grid centered on the ego vehicle, rolling with the ego pose.
Each cell $c$ holds $P(\text{occ} \mid \mathbf{z}_{1:t})$ and a height $h_c$ (for
ground/steps). Resolution: $0.2\,\text{m} \times 0.2\,\text{m}$, size $200 \times 200$
($\pm 20$ m).

### 5.2 Inverse sensor models

- **LiDAR**: a Bresenham ray-cast sets free-space along the beam and occupied at the
  endpoint. Log-odds update:
  $$
  \ell(c) \mathrel{+}= \log\frac{P(\text{occ}\mid z)}{1 - P(\text{occ}\mid z)} - \ell_{prior}
  $$
- **Vision freespace**: the BEV freespace mask from `src/perception/` gives
  $P(\text{free}\mid \text{cam})$; applied as a soft free-space prior with a lower
  confidence weight than LiDAR.
- **Radar**: a radar return at range $r$ with high RCS marks the cell occupied; a
  Doppler-only (no range) return with low RCS is ignored to avoid phantom occupancy.

### 5.3 Dempster-Shafer alternative

For cells where vision says "free" but radar says "occupied" (e.g., a low metal bar
invisible to camera), a Dempster-Shafer layer resolves the conflict via mass
combination:

$$
m_{12}(A) = \frac{\sum_{B \cap C = A} m_1(B) m_2(C)}{1 - K}, \quad
K = \sum_{B \cap C = \emptyset} m_1(B) m_2(C)
$$

where $K$ is the conflict mass. High $K$ flags the cell as *conflicted* and routes the
planner around it conservatively.

### 5.4 Temporal decay

Cells unobserved for $\tau > 2\,\text{s}$ decay toward the prior:
$\ell(c) \leftarrow \alpha\, \ell(c) + (1-\alpha)\,\ell_{prior}$, so stale occupancy
from a moving object does not persist as a phantom wall.

---

## 6. Existing `src/alg/fusion.py` Integration

### 6.1 Current behavior

`apply_fusion()` in `src/alg/fusion.py` is a **lane-state fusion** routine. It takes a
waypoint-derived state `wp_state = (cte_wp, head_wp, curv_wp)` and a UNet-derived lane
state `(cte_lane, head_s, curv_s, lane_conf)` and produces a blended
`(cte, heading, curvature, mode)`. The blend weight is:

$$
w_{unet} = W_{unet} \cdot \text{clip}\!\left(\frac{\text{lane\_conf} - C_{low}}{C_{high} - C_{low}},\, 0,\, 1\right)
$$

with an **adaptive curvature reduction**: when $|\kappa| > \kappa_{red}$ the weight is
scaled down, and when `geometry_valid is False` *and* the curve is sharp, $w_{unet}$
is forced to zero (pure waypoint mode). This is exactly the
**primary-source + confidence-gated-assist + degradation-fallback** pattern this
document generalizes.

### 6.2 Generalization to the full suite

The same pattern is applied at each fusion level:

| Layer | Primary | Assist | Degradation |
|-------|---------|-------|-------------|
| Lane state (`fusion.py`) | Waypoint | UNet (lane_conf) | WP-only when geometry invalid |
| Ego pose | IMU + GNSS | Vision pose | IMU-only dead reckoning |
| Object distance | Radar | Vision depth | Radar ghost track |
| Occupancy | LiDAR | Vision freespace + radar | Vision-only grid (lower res) |

### 6.3 Proposed extension to `apply_fusion`

Add optional radar-confirmed longitudinal correction and an EKF-corrected heading, so
the lane-state fusion is consistent with the global ego estimate:

```python
def apply_fusion(
    wp_state, cte_m_lane, head_s, curv_s, lane_conf,
    geometry_valid=None,
    ego_state_ekf=None,      # NEW: (x, y, theta, v) from EKF/UKF
    radar_range=None,        # NEW: lead-vehicle range from radar
) -> Tuple[float, float, float, str]:
    ...
    # If EKF heading is available and lane_conf is low, trust EKF over raw perception
    if ego_state_ekf is not None and lane_conf < FUSION_CONF_LOW:
        head_s = ego_state_ekf[2]   # θ from global estimator
    ...
```

This keeps `fusion.py` as the **lane-level arbitrator** while the EKF/UKF owns the
global pose — a clean separation of concerns.

---

## 7. Tesla Vision-Only vs Waymo LiDAR-Fusion

| Aspect | Tesla (vision-only) | Waymo (LiDAR-fusion) | This prototype |
|--------|---------------------|----------------------|----------------|
| Primary sensor | 8 cameras | LiDAR + cameras + radar | LiDAR + 8 cameras + radar |
| Metric depth | Inferred (stereo / monocular) | Direct (LiDAR) | Direct (LiDAR) + inferred (cam) |
| Velocity | Optical flow / multi-frame | Radar Doppler + LiDAR scene flow | Radar Doppler + LiDAR + OF |
| Ego pose | Vision + IMU + wheels | GNSS + IMU + LiDAR match | GNSS-RTK + IMU + LiDAR + vision |
| Cost | Low | High | Medium (prototype) |
| Weather robustness | Camera-limited | Radar/LiDAR help | Radar/LiDAR help |
| Redundancy | Cross-camera only | Multi-modal | Multi-modal |
| Failure mode | Camera glare → blind | LiDAR blooming in fog | Radar survives both |

The prototype deliberately sits closer to Waymo's philosophy: **redundant modalities
with explicit degradation**, because the research goal is safety analysis, not
production cost minimization. The Tesla comparison is retained as the *vision-only
fallback* mode (Section 8.3).

---

## 8. Degradation Handling

### 8.1 Per-sensor health monitoring

Each sensor publishes a health flag derived from:
- data age $\Delta t_s$ vs nominal rate,
- internal diagnostics (IMU bias magnitude, GNSS HDOP, LiDAR return count, camera
  exposure saturation),
- innovation gate failures in the EKF (consecutive $\chi^2$ rejects).

A sensor is marked **DEGRADED** when $\Delta t_s > 2\,T_s$ or the $\chi^2$ reject rate
exceeds 20% over a 1 s window, and **LOST** when $\Delta t_s > 5\,T_s$.

### 8.2 Fusion reconfiguration

| Lost sensor | Reconfiguration |
|-------------|-----------------|
| GNSS | IMU + LiDAR pose-only (dead reckoning, growing covariance) |
| LiDAR | Vision-only occupancy grid + radar grid (lower resolution) |
| Radar | Vision depth + LiDAR scene flow; longer TTC margins |
| Cameras (some) | Reduce to available subset; widen conservative envelope |
| All cameras | LiDAR + radar only; disengage lane-keep, fallback to safety stop |
| IMU | Emergency: GNSS + wheel odometry at low rate; mandatory pull-over |

### 8.3 Vision-only fallback (Tesla mode)

If LiDAR and radar are both lost, the stack falls back to a **vision-only mode** that
mirrors Tesla's approach: BEV depth from multi-camera stereo + monocular networks,
velocity from multi-frame optical flow, occupancy from the vision freespace mask only.
The controller widens its safety envelope (longer TTC, lower speed cap) because the
metric uncertainty is larger. This mode is explicitly flagged to the behavior planner
so it can choose conservative maneuvers.

### 8.4 Confidence propagation to the controller

Every fused output carries a scalar confidence $\gamma \in [0,1]$ derived from the
trace of its covariance and the health flags. The MPC uses $\gamma$ to scale its
terminal-cost weighting: low confidence → more conservative tracking, larger safety
distance, and earlier braking. This mirrors the `lane_conf` gating already present in
`src/alg/fusion.py`.

---

## 9. Open Issues & Future Work

1. **Online extrinsic drift** — thermal expansion shifts camera-LiDAR extrinsics; an
   online residual minimizer should run in the background during straight driving.
2. **Radar phantom suppression** — learn a static-echo classifier (classical, not RL)
   using RCS + Doppler history to reject guardrail multipath.
3. **UKF switch hysteresis** — the current $\kappa_{switch}$ threshold can chatter;
   add hysteresis bands $[\kappa_{on}, \kappa_{off}]$.
4. **Grid memory** — the rolling grid should persist a coarse low-resolution map for
   re-visited areas (loop closure) without unbounded memory growth.
5. **Dempster-Shafer conflict logging** — high-conflict cells are safety-critical;
   log them for offline analysis and planner audit.

---

## 10. References

- Thrun, Burgard, Fox — *Probabilistic Robotics*, ch. 3–5 (Kalman, grid, CI).
- Forster, Carlone, Dellaert — "IMU Pre-Integration on Manifold" (2017).
- Julier, Uhlmann — "Unscented Filtering and Nonlinear Estimation" (2004).
- Shafer — *A Mathematical Theory of Evidence* (Dempster-Shafer, 1976).
- Existing code: `src/alg/fusion.py`, `src/perception/kalman_lane_tracker.py`,
  `src/alg/reference.py`, `src/alg/lka_step.py`.
