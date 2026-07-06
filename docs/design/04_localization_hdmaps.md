# 04 — Localization & HD Maps (Design Document)

> **Project:** FrontViewPredictiveControl — Tesla-like prototype
> **Scope:** Multi-sensor localization, SLAM, and HD-map representation (NO reinforcement learning)
> **Status:** Design / partial implementation
> **Owner:** Localization & Mapping team
> **Related code:** `src/carla_io.py` (CARLA waypoint + map geometry extraction), `src/perception/`

---

## 0. Executive Summary

This document specifies the **Localization & HD-Map subsystem** that estimates the
vehicle's pose $(x, y, z, \phi, \theta, \psi)$ at high rate (≥100 Hz) and aligns that
pose against a prebuilt **High-Definition (HD) map** so the predictive controller (MPC)
and behavior planner can reason in lane-level coordinates $(s, e_y, e_\psi)$.

The stack is **sensor-fusion-first, classical-estimation-heavy, and RL-free**. It fuses
three complementary sources:

1. **GNSS-RTK + IMU** — absolute, drift-bounded, low-rate global reference.
2. **Visual Odometry / SLAM** — relative, high-rate, texture-rich environments.
3. **LiDAR SLAM** — relative, high-rate, geometrically-precise, weather-robust.

These are bound together by an **error-state Kalman filter (ESKF)** on $SE(3)$ and
matched to the HD map via a **particle filter (global)** followed by **ICP (local)**.
A **SuperPoint + SuperGlue** visual relocalizer provides a learning-augmented fallback
when GNSS is denied (tunnels, urban canyons). The HD map itself is a lane-graph with
sub-decimeter geometry, semantic attributes, and a versioned update protocol.

Design targets: **≤10 cm lateral error at 100 Hz, ≤2° heading error, <50 ms fusion
latency, graceful degradation through 60 s of full GNSS outage.**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       LOCALIZATION & HD-MAP STACK                            │
│                                                                              │
│  [GNSS-RTK]──► ┐                                                             │
│  [IMU 200Hz]──►├──► [ESKF on SE(3)]──► Pose @100Hz                           │
│  [VO/SLAM]──► ─┤        ▲                                                    │
│  [LiDAR SLAM]►─┘        │                                                    │
│                         │                                                    │
│                [Particle Filter]──► global init / recovery                   │
│                         │                                                    │
│                  [ICP vs HD Map]──► map-aligned pose (s,e_y,e_psi)           │
│                         │                                                    │
│                  [SuperPoint+SuperGlue]──► visual relocalization fallback    │
│                         │                                                    │
│                  [HD Map]──► lane graph, topology, semantics, updates        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Sensor Suite & Roles

| Sensor         | Rate    | DoF        | Role                                  | Failure Mode            |
|----------------|---------|------------|---------------------------------------|-------------------------|
| GNSS-RTK       | 10 Hz   | $x,y,z$    | Absolute global anchor, drift bound   | Multipath, sky blockage |
| IMU (MEMS)     | 200 Hz  | $a,\omega$ | High-rate propagation, attitude       | Bias drift, vibration   |
| Mono/Stereo VO | 30 Hz   | rel $SE(3)$| Relative motion, scale (stereo)       | Textureless, motion blur|
| LiDAR (64-beam)| 10 Hz   | rel $SE(3)$| Geometric SLAM, map matching          | Rain/fog backscatter    |
| HD Map         | static  | lane graph | Absolute lane reference, semantics    | Staleness, construction |

The **IMU is the propagation backbone**: between any two measurement updates the pose
is integrated forward at 200 Hz from inertial readings, then corrected by GNSS/VO/LiDAR
through the ESKF innovation channel. This guarantees a smooth, high-rate pose even when
external sensors drop frames.

---

## 2. Coordinate Frames

We define a chain of frames used throughout the document.

```
ECEF (Earth) ──► ENU (local tangent) ──► Map frame (HD map) ──► Vehicle body ──► Sensor
   W                M                       m                      b                s
```

- **ECEF / ENU**: global geodetic; GNSS-RTK outputs in ECEF, converted to local ENU
  anchored at a map origin $\mathbf{p}_0 = (\lambda_0, \varphi_0, h_0)$.
- **Map frame $m$**: the HD map's metric frame (UTM or local ENU). All lane geometry
  is stored here. This is the frame the controller reasons in.
- **Body frame $b$**: origin at rear axle, $x$ forward, $y$ left, $z$ up. IMU mounted
  here (or extrinsics calibrated).
- **Sensor frame $s$**: each camera/LiDAR has a fixed extrinsic $\mathbf{T}_{b\leftarrow s}$.

The pose we estimate is $\mathbf{T}_{m\leftarrow b} \in SE(3)$, decomposed as
translation $\mathbf{r} \in \mathbb{R}^3$ and rotation $\mathbf{R} \in SO(3)$, with
the Lie-algebra error state $\delta\boldsymbol{\xi} = [\delta\boldsymbol{\rho}^\top,
\delta\boldsymbol{\theta}^\top]^\top \in \mathbb{R}^6$.

---

## 3. IMU Integration — Error-State Kalman Filter (ESKF)

### 3.1 Why error-state?

The full state is non-Euclidean (rotation lives on $SO(3)$). We propagate the **nominal**
state with the inertial equations and keep a **small error-state** that is approximately
linear and Gaussian. The error-state is reset to zero after every update, so it never
strays far from the linearization point — this is the standard, well-conditioned
formulation used by LIO-SAM and ORB-SLAM3 IMU modes.

### 3.2 Nominal propagation (200 Hz)

Given IMU specific force $\mathbf{f}_b$ and angular rate $\boldsymbol{\omega}_b$,
with gravity $\mathbf{g}$ in map frame:

$$
\dot{\mathbf{r}} = \mathbf{v}, \qquad
\dot{\mathbf{v}} = \mathbf{R}_{m\leftarrow b}\,\mathbf{f}_b + \mathbf{g}, \qquad
\dot{\mathbf{R}}_{m\leftarrow b} = \mathbf{R}_{m\leftarrow b}\,[\boldsymbol{\omega}_b]_\times
$$

Discretized with midpoint integration over $\Delta t$:

$$
\mathbf{v}_{k+1} = \mathbf{v}_k + \left(\mathbf{R}_k\,\mathbf{f}_b + \mathbf{g}\right)\Delta t
$$
$$
\mathbf{R}_{k+1} = \mathbf{R}_k\,\exp\left([\boldsymbol{\omega}_b\,\Delta t]_\times\right)
$$
$$
\mathbf{r}_{k+1} = \mathbf{r}_k + \mathbf{v}_k\,\Delta t + \tfrac{1}{2}\left(\mathbf{R}_k\,\mathbf{f}_b + \mathbf{g}\right)\Delta t^2
$$

### 3.3 Error-state dynamics

The error state $\delta\mathbf{x} = [\delta\mathbf{r}^\top, \delta\mathbf{v}^\top,
\delta\boldsymbol{\theta}^\top, \mathbf{b}_a^\top, \mathbf{b}_g^\top]^\top \in \mathbb{R}^{15}$
propagates as

$$
\delta\dot{\mathbf{x}} = \mathbf{F}\,\delta\mathbf{x} + \mathbf{G}\,\mathbf{n}
$$

with the key blocks (gravity-coupled attitude error):

$$
\delta\dot{\mathbf{v}} = -\mathbf{R}\,[\mathbf{f}_b]_\times\,\delta\boldsymbol{\theta}
- \mathbf{R}\,\delta\mathbf{b}_a + \mathbf{n}_a
$$
$$
\delta\dot{\boldsymbol{\theta}} = -[\boldsymbol{\omega}_b]_\times\,\delta\boldsymbol{\theta}
- \delta\mathbf{b}_g + \mathbf{n}_g
$$

Covariance propagates: $\mathbf{P}_{k+1} = \mathbf{F}_d\,\mathbf{P}_k\,\mathbf{F}_d^\top + \mathbf{Q}_d$.

### 3.4 Update (generic)

For a measurement $\mathbf{z}$ predicting $\hat{\mathbf{z}} = h(\mathbf{x}_{nom})$ with
Jacobian $\mathbf{H}$, the ESKF update is

$$
\mathbf{K} = \mathbf{P}\,\mathbf{H}^\top\left(\mathbf{H}\,\mathbf{P}\,\mathbf{H}^\top + \mathbf{R}_{obs}\right)^{-1}
$$
$$
\delta\hat{\mathbf{x}} = \mathbf{K}\,(\mathbf{z} - \hat{\mathbf{z}}), \qquad
\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\,\mathbf{H})\,\mathbf{P}
$$

The nominal state is then **injected** with $\delta\hat{\mathbf{x}}$ (rotation via
$\mathbf{R} \leftarrow \mathbf{R}\,\exp([\delta\boldsymbol{\theta}]_\times)$) and the
error state reset to zero. GNSS, VO, LiDAR, and map-matching all plug in as different
$\mathbf{H}$ matrices through this same channel.

---

## 4. GNSS-RTK Integration

RTK provides cm-level $\mathbf{r}^{gnss}_m$ at ~10 Hz in ENU. The measurement is the
GNSS antenna position expressed in map frame:

$$
\mathbf{z}_{gnss} = \mathbf{r}_m + \mathbf{R}_{m\leftarrow b}\,\mathbf{t}^{ant}_b + \mathbf{n}_{gnss}
$$

where $\mathbf{t}^{ant}_b$ is the lever arm from body origin to antenna. The Jacobian
w.r.t. the error state is

$$
\mathbf{H}_{gnss} = \begin{bmatrix} \mathbf{I}_3 & \mathbf{0} & -\mathbf{R}\,[\mathbf{t}^{ant}_b]_\times & \mathbf{0} & \mathbf{0} \end{bmatrix}
$$

RTK covariance inflates dynamically: when the fix drops to **float** or **DGPS**, we
scale $\mathbf{R}_{obs}$ by 10–100× so the filter trusts inertial/VO more. A **DOP
monitor** (PDOP > 5) gates out poor constellations entirely.

---

## 5. Visual Odometry & SLAM

### 5.1 ORB-SLAM3 (stereo-inertial mode)

ORB-SLAM3 is the chosen **visual-inertial SLAM** backend for texture-rich daylight
operation. It runs in **stereo-inertial** mode so scale is observable without scale
ambiguity, and its IMU term is tightly coupled (the same MEMS IMU feeds both our ESKF
and ORB-SLAM3's internal optimizer — we treat ORB-SLAM3's pose as a relative measurement
to our ESKF, not as a competing filter).

Key properties we rely on:
- **Bag-of-Words place recognition** for loop closure and relocalization.
- **Atlas multi-map** — handles map merging when relocalizing after a loss.
- **Local Bundle Adjustment** over keyframes, producing a drift-bounded trajectory.

ORB-SLAM3 emits a relative pose $\mathbf{T}_{k\leftarrow k-1}^{vo}$ at 30 Hz with
information matrix $\boldsymbol{\Lambda}^{vo}$. We feed this as a **relative-pose
factor** into the ESKF innovation:

$$
\mathbf{z}_{vo} = \log\left(\mathbf{T}_{k-1}^{-1}\,\mathbf{T}_k\right)^\vee, \qquad
\mathbf{R}_{obs}^{vo} = \boldsymbol{\Lambda}^{vo\,-1}
$$

### 5.2 SuperPoint + SuperGlue (visual relocalization)

When ORB-SLAM3 loses tracking (e.g., aggressive motion blur, sudden lighting change),
we fall back to a **deep feature relocalizer**:

- **SuperPoint** — self-supervised CNN detector/descriptor, 300–800 keypoints/frame at
  25 FPS on Jetson. Robust to viewpoint and moderate illumination change.
- **SuperGlue** — graph-neural-network matcher with optimal-transport assignment.
  Matches SuperPoint descriptors between the live frame and a **keyframe database**
  extracted offline from the HD map's geo-tagged reference images.

Pipeline:

```
Live frame ──► SuperPoint (kpts+desc) ──► SuperGlue vs DB ──► 2D-3D matches
                                                              │
                                            PnP + RANSAC ──► T_map<-body (6DoF)
                                                              │
                                            feed as ESKF update (H_vo)
```

The PnP solve uses the matched 2D pixels and their known 3D map landmarks to recover
$\mathbf{R}, \mathbf{t}$ via EPnP + RANSAC (inlier threshold 2 px). The resulting pose
is treated as an **absolute** measurement (unlike ORB-SLAM3's relative factor), which
lets the ESKF recover from large drift in a single update. Covariance is set from the
RANSAC inlier count: fewer inliers ⇒ larger $\mathbf{R}_{obs}$.

> **Note on RL boundary:** SuperPoint/SuperGlue are supervised/self-supervised CNNs and
> GNNs — they are **not** reinforcement learning. No policy gradient, no reward signal,
> no agent-environment loop. This keeps the project within its "NO RL" constraint while
> still using modern learned features for robust matching.

---

## 6. LiDAR SLAM — LIO-SAM

### 6.1 Why LIO-SAM?

LIO-SAM (Tightly-coupled Lidar-Inertial Odometry via Smoothing and Mapping) is the
LiDAR-inertial backbone. It factorizes the pose graph into four edge types:

1. **IMU preintegration** between LiDAR scans (uses our MEMS IMU).
2. **LiDAR odometry** — scan-to-map ICP on edge/planar features.
3. **GPS** — absolute position factors (we feed RTK here).
4. **Loop closure** — ICP on historical keyframes via distance/BoW trigger.

It runs a factor graph (GTSAM `iSAM2`) rather than a single Kalman filter, which gives
**smoother, consistent** trajectories over long horizons. We consume its optimized pose
at 10 Hz as a high-weight relative factor into the ESKF (or, alternatively, run the ESKF
*inside* LIO-SAM's GPS slot and let iSAM2 be the master — config selectable).

### 6.2 Feature extraction

Each LiDAR scan is split into **edge** (line) and **planar** (surface) features by
curvature threshold on the range image:

$$
c = \frac{1}{|S|}\left\|\sum_{j\in S}\mathbf{r}_j - |S|\,\mathbf{r}_i\right\|
$$

Points with $c > c_{edge}$ seed edge features; $c < c_{planar}$ seed planar features.
Scan-to-map matching minimizes point-to-line and point-to-plane distances:

$$
E(\mathbf{T}) = \sum_{i\in\mathcal{E}} d_{line}(\mathbf{T}\mathbf{p}_i, \ell_i)^2
              + \sum_{j\in\mathcal{P}} d_{plane}(\mathbf{T}\mathbf{p}_j, \pi_j)^2
$$

solved with Gauss-Newton on $SE(3)$.

### 6.3 LiDAR ↔ ESKF coupling

LIO-SAM pose $\mathbf{T}^{lio}_{m\leftarrow b}$ at 10 Hz enters the ESKF as an absolute
pose measurement with information from iSAM2's marginal covariance. Because LIO-SAM
already fuses IMU+GPS internally, we **down-weight** its factor when RTK is healthy
(to avoid double-counting the IMU) and **up-weight** it during GNSS denial.

---

## 7. HD Map Structure

### 7.1 Logical model

The HD map is a **lane-centric topological + geometric graph**:

```
                    ┌─────────────┐
        Lane A ────►│  Connector  │──── Lane B
        (s:[0,L_A]) │ (lane change│   (s:[0,L_B])
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Connector  │──── Lane C
                    └─────────────┘
```

Each **Lane** is a parametric centerline in map frame:

$$
\mathbf{c}(s) = \big(x(s),\, y(s),\, z(s),\, \kappa(s),\, \theta_{hdg}(s)\big),
\qquad s \in [0, L]
$$

stored as a piecewise cubic Hermite spline (C2-continuous, matches the controller's
curvature feed). Lane width $w(s)$, speed limit, road class, and boundary types
(solid/dashed/curb) are attributes per $s$-sample at 1 m resolution.

### 7.2 Storage format

We use a lightweight binary layering compatible with both CARLA exports and real-world
OpenDRIVE import:

| Layer        | Content                                  | Format            |
|--------------|------------------------------------------|-------------------|
| `geometry`   | centerline splines, boundaries, z-profile| binary sampled pts|
| `topology`   | lane connectors, successors, predecessors| graph (adjacency) |
| `semantics`  | signs, signals, crosswalks, speed limits | geo-tagged POIs   |
| `visual_db`  | SuperPoint descriptors + 3D landmarks    | HDF5 / LMDB       |
| `lidar_db`   | surfel/voxel map for ICP                 | PCD / voxel grid  |
| `metadata`   | version, bbox, timestamp, sensor calib   | JSON              |

### 7.3 Frenet projection

The controller works in Frenet coordinates $(s, e_y, e_\psi)$ relative to the matched
lane. Given pose $\mathbf{p} = (x, y, \psi)$ and lane centerline $\mathbf{c}(s)$:

$$
e_y = \hat{\mathbf{n}}(s)\cdot\big((x,y) - \mathbf{c}_{xy}(s)\big), \qquad
e_\psi = \psi - \theta_{hdg}(s)
$$

where $\hat{\mathbf{n}}(s) = [-\sin\theta_{hdg}(s),\, \cos\theta_{hdg}(s)]^\top$ is the
lane normal. The projection minimizes $s$ over the spline via Newton iteration on

$$
g(s) = (\mathbf{p}_{xy} - \mathbf{c}(s))\cdot \mathbf{c}'(s) = 0
$$

This mirrors the $(s, lat)$ vehicle-frame decomposition already implemented in
`src/carla_io.py::get_waypoints_all_lanes`, which returns ego/left/right lane paths as
$(s, lat)$ tuples — the same Frenet convention the HD map uses.

---

## 8. Map Matching — Particle Filter + ICP

### 8.1 Two-stage matching

Localization on the HD map is a **two-stage** process:

1. **Particle filter (global)** — maintains $N=512$ particles $\{s_i, e_{y,i}, e_{\psi,i}\}$
   in lane-Frenet space. Handles lane ambiguity, initialization, and recovery.
2. **ICP (local)** — refines the matched pose by aligning the live LiDAR scan to the
   map's local geometric layer around the particle-mode location.

### 8.2 Particle filter

Each particle carries a weight $w_i$. The proposal is the ESKF pose delta; the
observation likelihood combines lane-boundary consistency and LiDAR-to-map residual:

$$
w_i \propto \exp\!\left(-\frac{(e_{y,i} - \hat{e}_y)^2}{2\sigma_y^2}\right)
\cdot \exp\!\left(-\frac{(e_{\psi,i} - \hat{e}_\psi)^2}{2\sigma_\psi^2}\right)
\cdot \exp\!\left(-\frac{\text{ICP}_{res,i}^2}{2\sigma_{icp}^2}\right)
$$

Resampling uses **systematic resampling** with an effective-sample-size gate
$N_{eff} = 1/\sum w_i^2 < N/2$. Lane switches are handled by spawning particles on
adjacent lanes when $|e_y| > 0.4\,w$ — this is the same multi-lane awareness that
`get_waypoints_all_lanes` provides in simulation.

### 8.3 ICP refinement

Given the particle-mode pose as initialization, point-to-plane ICP aligns the live
LiDAR scan $\mathcal{S}$ to the local map patch $\mathcal{M}$:

$$
\mathbf{T}^* = \arg\min_{\mathbf{T}} \sum_{i}
\left(\mathbf{n}_i^\top\big(\mathbf{T}\mathbf{p}_i - \mathbf{q}_i\big)\right)^2
$$

with $\mathbf{p}_i \in \mathcal{S}$, $\mathbf{q}_i \in \mathcal{M}$ its closest map
point, $\mathbf{n}_i$ the map normal. Solved in closed form per iteration via:

$$
\mathbf{A} = \sum_i \mathbf{p}_i'\,{\mathbf{n}_i}^\top, \qquad
\mathbf{B} = \sum_i \mathbf{q}_i\,{\mathbf{n}_i}^\top
$$

converging in 10–20 iterations on a 30 m local patch. The ICP covariance (from the
Hessian) feeds back into the ESKF as the map-matching observation noise.

---

## 9. Tesla-Style Vision-Only Lane Matching

A core design goal is to support a **vision-only lane-matching** path that does not
depend on LiDAR or GNSS — mirroring Tesla's "vision is all you need" philosophy while
remaining RL-free.

### 9.1 Approach

The vision pipeline (see `02_vision_perception.md`) produces a BEV lane polynomial
$\hat{y}_{bev}(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3$ in vehicle frame. The HD map
provides candidate lane centerlines near the ESKF pose. Matching reduces to finding the
map lane whose BEV-rendered polynomial best fits the detected one:

$$
j^* = \arg\min_j \sum_{x_k} \left\| \hat{y}_{bev}(x_k) - y^{(j)}_{map\to bev}(x_k) \right\|^2_{\Sigma_k}
$$

where $y^{(j)}_{map\to bev}$ is the $j$-th map lane transformed into the vehicle BEV
using the current pose estimate. The residual is gated by the lane-detection covariance
$\Sigma_k$ (from the Kalman lane tracker in `src/perception/kalman_lane_tracker.py`).

### 9.2 Pose correction from lane match

Once the correct lane is identified, the lateral/heading residual is fed back as a
**soft** ESKF update — soft because vision lanes are relative and can be offset from
the map centerline by lane width or marking ambiguity. We only apply the update when:

- Detection confidence $> 0.7$,
- Residual is within $3\sigma$ of the tracker covariance,
- At least 40 m of lane is visible (long enough to observe heading).

This is the **degradation backbone**: when GNSS, LiDAR, and SuperGlue all fail, vision
lane matching alone keeps the vehicle lane-centered for tens of seconds.

---

## 10. Degradation Handling

### 10.1 Failure taxonomy

| Scenario        | GNSS | LiDAR SLAM | VO/SLAM | Vision Lane | Active Mode            |
|-----------------|------|------------|---------|-------------|------------------------|
| Open highway    | RTK  | OK         | OK      | OK          | Full fusion            |
| Urban canyon    | Float/Denied | OK | OK      | OK          | LiDAR+VO+vision        |
| Tunnel          | Denied | Degraded (few feats) | Degraded (dark) | OK (lit) | Vision lane + IMU dead-reckon |
| Long tunnel     | Denied | Degraded  | Degraded | Degraded    | IMU dead-reckon only   |
| Heavy rain/night| RTK  | Degraded (backscatter) | Degraded | Degraded | GNSS+IMU               |

### 10.2 Urban canyon

Multipath inflates GNSS covariance; the ESKF automatically down-weights RTK via the
DOP/fix-status scaling (§4). LIO-SAM's IMU preintegration carries the pose through;
SuperGlue relocalization against geo-tagged building-facade keyframes recovers absolute
pose at intersections. We pre-build the SuperPoint keyframe DB along mapped corridors
specifically for this case.

### 10.3 Tunnels

GNSS is fully denied. Strategy:
1. **Enter-tunnel snapshot**: lock the last good pose + a 60 s IMU bias estimate.
2. **Dead-reckoning**: IMU-only propagation with growing covariance
   $\sigma_y(t) \approx \sigma_{y,0} + \tfrac{1}{2} a_{bias}\,t^2$.
3. **Visual anchor**: if tunnel has lighting, vision lane matching (§9) bounds lateral
   drift even though absolute $x$ drifts.
4. **Exit-tunnel relocalization**: SuperGlue against the map's exit keyframes snaps
   pose back to absolute within one update.

A 60 s outage with bias noise $\sigma_{a,bias} \approx 10^{-3}\,\text{m/s}^2$ yields
lateral drift $\approx 1.8$ m — bounded by vision lane matching to $<0.3$ m when lane
markings are visible.

### 10.4 Confidence reporting

Every pose is tagged with a **localization confidence** $\in [0,1]$ derived from the
ESKF posterior trace and sensor-health flags. The behavior planner uses this to:

- Reduce target speed when confidence $< 0.6$,
- Force lane-keep (no lane change) when $< 0.4$,
- Trigger minimal-risk maneuver when $< 0.2$.

---

## 11. Map Updates

### 11.1 Offline vs online

- **Offline bulk rebuild**: full re-survey with RTK + LiDAR vehicle, weekly/monthly.
  Produces a new map version with refreshed geometry, semantics, and visual DB.
- **Online incremental**: the fleet (or single prototype) logs divergence events —
  places where live perception persistently disagrees with the map. These are queued
  as **change candidates** and merged into the next offline build.

### 11.2 Versioning & delta

Maps are content-addressed by version hash. A **delta layer** records only changed
tiles (1 km × 1 km grid) so vehicles download minimal patches. Each tile carries:

```
tile_id, version, bbox, geometry_hash, semantic_hash, visual_db_hash, ts, source
```

Conflict resolution: if a live change-candidate contradicts an offline update, the
offline (higher-fidelity survey) wins; the live candidate is re-queued for verification.

### 11.3 Change detection trigger

A divergence event is logged when, for $>5$ consecutive frames:

$$
\left\| \mathbf{z}_{vision} - h_{map}(\hat{\mathbf{x}}) \right\|_{\Sigma} > \chi^2_{0.99}
$$

i.e., the vision observation is a statistical outlier against the map given the current
pose. This catches new construction, re-striped lanes, and removed signs without
requiring an explicit "map is wrong" classifier.

---

## 12. CARLA Integration

### 12.1 Ground-truth vs estimated

In simulation, CARLA provides **ground-truth** pose and map via the Python API. The
localization stack is exercised in two modes:

1. **Ground-truth mode**: `carla_io.get_waypoints` / `get_waypoints_all_lanes` read the
   CARLA `Map` directly and return lane geometry in the vehicle frame. This is the
   "perfect localization" baseline used for controller tuning and behavior validation.
2. **Estimated mode**: we inject synthetic sensor noise into the ground-truth pose to
   emulate RTK/IMU/VO degradation, then run the full ESKF + map-matching pipeline
   against a CARLA-exported HD map. This validates the localization stack itself.

### 12.2 CARLA → HD map export

CARLA's `OpenDRIVE` map is converted to our HD-map format:

- Lane centerlines sampled from `carla.Waypoint` every `WP_STEP` meters (the same
  step used in `carla_io.py`).
- Topology from `waypoint.next()` / `get_left_lane()` / `get_right_lane()`.
- Boundaries from `carla.LaneType` and `carla.LaneMarkingType`.
- A synthetic SuperPoint keyframe DB is rendered offline by placing a virtual camera
  along each centerline and extracting descriptors — enabling SuperGlue relocalization
  experiments in simulation.

### 12.3 Sensor injection

```
CARLA ground-truth pose (x,y,z,roll,pitch,yaw)
        │
        ├──► add RTK noise (σ=2cm fix, 1m float) ──► GNSS channel
        ├──► add IMU noise (bias, σ_a, σ_g)      ──► IMU channel
        ├──► render cameras ──► ORB-SLAM3 / SuperPoint ──► VO channel
        └──► render LiDAR  ──► LIO-SAM              ──► LiDAR channel
                                                          │
                                              ESKF + Map Matching
                                                          │
                                          compare vs ground-truth pose
```

This lets us measure localization error against perfect truth, tune covariances, and
rehearse degradation scenarios (tunnel = toggle GNSS off, urban canyon = inflate RTK
noise) deterministically — impossible to do repeatably in the real world.

### 12.4 Reuse of existing code

`src/carla_io.py` already implements the core CARLA↔lane-geometry bridge:
- `get_waypoints` — forward lookahead waypoints on the ego lane (used by the controller
  as the reference path, equivalent to the HD map's centerline projection).
- `get_waypoints_all_lanes` — ego + left + right lane paths in vehicle-frame $(s, lat)$,
  which is exactly the Frenet representation the HD map matching produces.

The localization layer wraps these so that, in ground-truth mode, the controller
consumes the same $(s, e_y, e_\psi)$ interface whether the source is CARLA's perfect
map or the estimated HD-map-matched pose. This keeps the controller **localization-
agnostic** — a key architectural property.

---

## 13. Implementation Roadmap

| Phase | Deliverable                                          | Status      |
|-------|------------------------------------------------------|-------------|
| 1     | ESKF + IMU propagation + GNSS update (CARLA sim)     | Planned     |
| 2     | CARLA OpenDRIVE → HD-map export + Frenet projection  | Partial (carla_io) |
| 3     | Particle-filter map matching on CARLA map            | Planned     |
| 4     | LIO-SAM integration (real LiDAR)                     | Planned     |
| 5     | ORB-SLAM3 stereo-inertial factor                      | Planned     |
| 6     | SuperPoint+SuperGlue keyframe DB + relocalizer       | Planned     |
| 7     | Vision-only lane matching feedback (Tesla mode)      | Planned     |
| 8     | Degradation manager + confidence reporting           | Planned     |
| 9     | Map update / change-detection pipeline               | Future      |

---

## 14. Open Questions

1. **Single master filter vs federated**: should the ESKF own the state, with LIO-SAM
   and ORB-SLAM3 as measurement sources, or should LIO-SAM's iSAM2 graph be the master
   and the ESKF a lightweight propagator? Trade-off is consistency vs latency.
2. **SuperPoint DB density**: how many geo-tagged keyframes per km are needed for
   reliable urban-canyon relocalization without bloating the map?
3. **Map matching in parking lots / unmapped areas**: when the particle filter's map
   likelihood collapses everywhere, do we fall back to pure SLAM (build a local map
   on the fly) or to vision lane matching against a generic lane model?
4. **IMU grade**: automotive MEMS ($\sigma_a \sim 10^{-3}$) vs tactical-grade
   ($\sigma_a \sim 10^{-4}$) — the 60 s tunnel budget depends heavily on this choice.

---

*End of document.*
