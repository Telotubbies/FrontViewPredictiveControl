# 02 — Vision Perception Pipeline (Design Document)

> **Project:** FrontViewPredictiveControl — Tesla-like prototype
> **Scope:** Pure computer-vision perception stack (NO reinforcement learning)
> **Status:** Design / partial implementation
> **Owner:** Perception team
> **Related code:** `src/perception/` (UNet lanes, BEV pipeline, Kalman tracker, TSR)

---

## 0. Executive Summary

This document specifies the **Vision Perception Pipeline** that turns raw multi-camera
frames into a structured, temporally-consistent 3D scene representation consumed by the
predictive controller (MPC) and the behavior planner.

The pipeline is **geometry-first, learning-augmented, and RL-free**. It fuses 8 cameras
into a unified Bird's-Eye-View (BEV) world, detects and tracks dynamic agents, estimates
depth, segments drivable and non-drivable freespace, and recognizes traffic elements
(signs / lights / crosswalks). Every output is stabilized over time with classical
filters (Kalman, SORT/ByteTrack, optical-flow priors) so a single dropped frame never
resets the world state — a property already implemented for lane polynomials in
`src/perception/kalman_lane_tracker.py`.

Design targets: **30 FPS end-to-end, <100 ms perception latency, <30 W on a
Jetson-class SoC**, with graceful degradation at night and in adverse weather.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         VISION PERCEPTION PIPELINE                          │
│                                                                              │
│  [8 Cameras] ──► [ISP/HDR] ──► [Sync Buffer] ──► [Preproc]                  │
│                                          │                                   │
│         ┌────────────────────┬───────────┼────────────┬───────────────┐     │
│         ▼                    ▼           ▼            ▼               ▼     │
│   Object Det           Lane Det      Depth Est    Segmentation    TSR/TLR   │
│  (YOLO/DETR/BEVFormer) (UNet+Poly)  (MiDaS+Ster) (DeepLab/SAM)   (existing) │
│         │                    │           │            │               │     │
│         └────────────┬───────┴───────┬───┴────────────┘               │     │
│                      ▼               ▼                                │     │
│               [BEV Fusion]    [Temporal Tracking]◄────────────────────┘     │
│                      │          (Kalman / SORT / ByteTrack / OF)            │
│                      ▼               │                                      │
│              [Scene Model] ◄─────────┘                                      │
│              (agents, lanes, freespace, depth, traffic)                     │
│                      │                                                      │
│                      ▼                                                      │
│        [MPC / Behavior Planner / Visualization]                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Camera Setup

### 1.1 Rig topology (8 cameras)

A Tesla-like surround layout is used. Each camera has a dedicated role; the front
tri-camera cluster provides the long-range forward perception critical for highway
predictive control.

| # | Name              | Position            | Lens        | Resolution   | FPS | FOV (H×V)        | Role                                  |
|---|-------------------|---------------------|-------------|--------------|-----|------------------|---------------------------------------|
| 1 | **Main Forward**  | Top-center windshield | Narrow      | 1280×960     | 30  | 35° × 27°        | Long-range (250 m) object/lane        |
| 2 | **Wide Forward**  | Top-center (above main) | Fisheye | 1280×960     | 30  | 120° × 80°       | Near-field, intersections, cut-ins   |
| 3 | **Tele Forward**  | Top-center (below main) | Tele    | 1280×960     | 30  | 18° × 14°        | 500 m highway, sign/LR read           |
| 4 | **Left Pillar**   | B-pillar left       | Fisheye     | 1280×960     | 30  | 110° × 75°       | Left side, blind-spot, lane-merge     |
| 5 | **Right Pillar**  | B-pillar right      | Fisheye     | 1280×960     | 30  | 110° × 75°       | Right side, blind-spot                |
| 6 | **Fwd Left Side** | Front fender left   | Fisheye     | 1280×960     | 30  | 100° × 70°       | Front-left curb, crosswalk            |
| 7 | **Fwd Right Side**| Front fender right  | Fisheye     | 1280×960     | 30  | 100° × 70°       | Front-right curb, crosswalk           |
| 8 | **Rear**          | License-plate bar   | Fisheye     | 1280×960     | 30  | 130° × 90°       | Rear coverage, reverse, rear cut-ins  |

> **Note:** In the current CARLA simulation prototype only a single 640×480 front
> camera is wired (see `bev_lane_pipeline.py` `BEV_SRC` for the 640×480 transform).
> The 8-camera rig is the production target; the pipeline is written so cameras are
> **pluggable** (one `CameraChannel` per stream) and degrades to the single-front
> configuration without code changes.

### 1.2 Image Signal Processor (ISP)

Each raw Bayer frame passes through an on-sensor / SoC ISP:

1. **Black-level correction + defect-pixel map**
2. **Lens shading / vignetting compensation** (per-lens LUT)
3. **Demosaic** (AHD)
4. **White balance** (gray-world + per-camera gain lock)
5. **HDR fusion** — 3 exposures (short/medium/long) merged with a tone-mapping LUT
   to preserve highlights (headlights) and shadows (night road). See §9.
6. **Gamma + color-space** → YUV422 for the encoder, RGB888 for inference.
7. **Fisheye rectification** — per-camera undistortion LUT applied on the GPU before
   any detector consumes the image. Fisheye cameras are rectified to a virtual pinhole
   (120° kept, no severe cropping) so downstream geometry is consistent.

### 1.3 Time synchronization

- **Hardware sync:** a central sync generator emits a **global shutter trigger** at
  30 Hz; all 8 sensors expose on the same edge (±1 ms). Global-shutter sensors are
  preferred to avoid rolling-shutter shear during ego motion.
- **Software sync:** each frame is stamped with `t_capture` (PTP/monotonic). The sync
  buffer aligns the 8 streams into a **frame set** keyed by the nearest common
  timestamp; a camera is allowed to lag by ≤ 1 frame (33 ms) before the buffer emits a
  partial set and flags `degraded=True`.
- **Ego-motion compensation:** between capture and inference the ego pose is
  interpolated to `t_capture` so all BEV warps use a consistent ego frame.

### 1.4 CameraChannel abstraction

```python
@dataclass
class CameraConfig:
    name: str
    position: str           # "front_main", "left_pillar", ...
    lens: str               # "narrow" | "fisheye" | "tele"
    intrinsics: np.ndarray  # 3x3 K (post-rectification)
    extrinsics: np.ndarray  # 4x4 camera→ego
    resolution: tuple[int, int]
    fov_deg: tuple[float, float]
    fps: int

class CameraChannel:
    cfg: CameraConfig
    def grab(self) -> Frame:        # returns undistorted, time-stamped image
        ...
```

---

## 2. Object Detection

### 2.1 Model selection

| Detector        | Use case                                   | Notes                                  |
|-----------------|--------------------------------------------|----------------------------------------|
| **YOLOv8-X**    | Primary per-camera 2D detector             | Fast, mature, easy to export to TensorRT |
| **RT-DETR**     | Optional accuracy upgrade for main/tele    | Transformer, better small objects      |
| **BEVFormer**   | Multi-camera 3D detection (production)     | Queries BEV directly from 8 cams       |

For the prototype we run **YOLOv8-X per camera** and lift to 3D via depth (§4) +
BEV fusion (§7). BEVFormer is the upgrade path once 8-camera training data exists.

### 2.2 Classes & heads

- **Vehicles:** car, truck, bus, motorcycle
- **VRUs:** pedestrian, cyclist, wheelchair
- **Static obstacles:** cone, barrier, construction-zone
- **Misc:** animal (deer/dog), debris

Each detection yields:
```python
@dataclass
class Detection2D:
    box_xyxy: np.ndarray     # [x1,y1,x2,y2] in image px
    score: float
    cls: int
    feat: np.ndarray         # 128-d ReID embedding (for tracking)
    depth: float | None      # filled by §4
    age: int                 # frames since first seen
```

### 2.3 Inference details

- Input: **640×640** letterboxed (main/tele) or **960×960** (fisheye side/rear for
  wider horizontal coverage).
- Backbone: CSPDarknet, FPN-PAN neck, decoupled heads.
- **TensorRT FP16** export, dynamic batch = number of cameras (8) packed into one
  inference call to maximize GPU utilization.
- **Test-time augmentation:** horizontal flip only (for fisheye side cams the flip is
  skipped to preserve left/right semantics).
- **Confidence threshold:** 0.25 default, **0.15 for VRUs** (recall-prioritized;
  false positives are filtered by the tracker).
- **NMS:** per-class, IoU 0.45; class-agnostic NMS is avoided so a truck doesn't
  suppress a nearby pedestrian.

### 2.4 2D → 3D lift

For each 2D box we estimate 3D center by combining:
1. **Bottom-edge ground contact** → project to ground plane via known camera
   extrinsics → `(x, y, z)` in ego frame.
2. **Depth prior** from §4 (monocular + stereo) to disambiguate scale.
3. **Dimension regression head** (trained on the box) for length/width/height, used
   to validate the ground-plane projection.

---

## 3. Lane Detection

### 3.1 Existing UNet (implemented)

`src/perception/lane_detector.py` trains a **UNet** that outputs a 2-class
probability map (left-lane / right-lane pixels). The BEV pipeline in
`bev_lane_pipeline.py` then:

1. Warps the UNet probability map to BEV (`BEV_SRC` → `BEV_DST`, 640×640).
2. Vertical dilation (`DILATE_V=80`) to connect dashed lines.
3. Histogram peak detection to seed left/right lane bases.
4. **Bidirectional sliding windows** (`N_WINDOWS=25`, `WINDOW_MARGIN=40`).
5. **Width-constrained 2nd-order polynomial fit** `y = c2·x² + c1·x + c0`
   (`POLY_ORDER=2`).

This stays as the lane backbone. The design below extends it.

### 3.2 Polynomial fitting (extended)

- **Fit in BEV metric space**, not pixel space. Convert BEV pixels → meters using
  `m_per_px_x`, `m_per_px_y` calibrated per camera. Output polynomial is in ego
  coordinates: `Y_ego = a·X_ego² + b·X_ego + c`.
- **RANSAC** over inlier lane pixels before polyfit to reject outliers (curbs,
  tar seams, shadow edges that the UNet occasionally catches).
- **Curvature sanity:** reject fits with |a| > a_max (physically impossible turn
  radius at ego speed); fall back to previous frame's polynomial + Kalman predict.
- **Multi-lane support:** instead of only left/right, detect up to **4 lanes**
  (LL, L, R, RR) by running histogram peak detection with non-max suppression and
  seeding one sliding-window tracker per peak. Each lane gets its own polynomial
  and its own Kalman state (§3.3).

### 3.3 Temporal tracking — Kalman filter (existing, extended)

`src/perception/kalman_lane_tracker.py` already implements a per-polynomial
Kalman filter with:

- State `x = [c2, c1, c0]` (3-dim), constant-curvature dynamics `F = I`.
- `Q = 0.15`, `R = 0.05` (tuned: faster response, trust measurement).
- `MAX_FRAMES_STALE = 3` — when UNet misses a frame, the filter **predicts** up to
  3 frames from dynamics so the overlay/reference path stays smooth.
- Outlier gating (`outlier_threshold=3.0`) and optional adaptive `R`.

**Extensions:**

1. **Constant-curvature-rate model** — augment state to `[c2, c1, c0, ċ2]` with
   `F` modeling `ċ2` persistence. Improves prediction during lane changes where
   curvature is actively changing.
2. **Cross-correlation between left & right lanes** — jointly track the pair with a
   6-dim state and a process model that couples them through the **lane-width
   constraint** (≈3.7 m). This prevents the two polynomials from drifting apart
   when one side is occluded.
3. **Confidence-weighted update** — scale `R` per frame by `(1 - UNet_confidence)`,
   so a low-confidence UNet map widens the filter's uncertainty instead of being
   thrown away.
4. **Initialization from BEV histogram** — first frame seeds `x` from the polyfit;
   subsequent frames always update (never reset) unless `age > MAX_FRAMES_STALE`.

```
UNet prob-map ──► BEV warp ──► RANSAC polyfit ──┐
                                                ▼
              ┌─────────────────────────────────────┐
              │  KalmanLaneTracker (per lane)        │
              │  predict: x = F·x,  P = F·P·Fᵀ + Q   │
              │  gate:   |z - H·x| < 3·√(HPHᵀ+R)     │
              │  update: K = PHᵀ(HPHᵀ+R)⁻¹           │
              │          x = x + K(z - H·x)           │
              │          P = (I-KH)P                  │
              └──────────────┬──────────────────────┘
                             ▼
              smoothed lane polynomial (ego frame)
                             │
                             ▼
              reference path for MPC
```

### 3.4 Lane type & semantics

Classify each tracked lane as `solid | dashed | double-solid | bot-dots` using a
small CNN head on the UNet feature map. This feeds the planner (e.g. no lane-change
across double-solid).

---

## 4. Depth Estimation

### 4.1 Monocular depth — MiDaS / Depth Anything

- **MiDaS v3.1 (DPT-Large)** for robust relative depth; **Depth Anything V2** as the
  accuracy upgrade. Both run per-camera.
- Output: **relative inverse depth**. We convert to metric by anchoring to the
  ground plane: scale `s` and shift `t` solved so that the road region's depth
  matches the geometric ground-plane depth at the same pixels:
  `d_metric = s · d_relative + t`, with `s, t` from least-squares over road pixels
  (road mask from §5).
- Runs at **640×384**, TensorRT FP16, ≈8 ms on Orin.

### 4.2 Stereo fusion

- **Stereo baseline:** the **main + tele** front pair (≈22 cm baseline) form a
  pseudo-stereo rig. Disparity via **SGBM** or a learned stereo net
  (RAFT-Stereo-lite). Triangulated depth is metric and far more accurate than
  monocular at range.
- **Fusion rule:**
  - If stereo disparity valid (texture-rich, |disp| > disp_min) → use stereo depth.
  - Else → use monocular metric depth.
  - Blend with inverse-variance weighting in overlap regions.
- **Side/rear fisheye cams:** no stereo pair → monocular only, used for near-field
  (<20 m) where scale error is tolerable.

### 4.3 Depth → occupancy

Per-camera depth + camera extrinsics → project each pixel into a 3D point cloud in
ego frame → rasterize into a **BEV occupancy grid** (0.2 m × 0.2 m, 100 m × 50 m).
Cells with points above the ground plane (height > 0.1 m) are marked occupied. This
grid is the freespace complement used by the planner and by BEVFormer-style queries.

---

## 5. Segmentation

### 5.1 Model

- **DeepLabv3+ (MobileNetV3 backbone)** for real-time semantic segmentation, or
  **SAM2** in "everything" mode for promptable instance masks when accuracy matters
  more than latency (low-speed urban).
- Runs on the **wide-forward** and **main-forward** cameras primarily; side/rear
  run a lighter 4-class head only.

### 5.2 Classes

| Class        | Use                                            |
|--------------|------------------------------------------------|
| road         | drivable freespace, feeds §4 scale & occupancy |
| sidewalk     | non-drivable, VRU context                      |
| sky          | horizon, used to mask false detections         |
| vegetation   | ignore region, prevents depth outliers         |
| building     | ignore / static map context                    |
| crosswalk    | pedestrian priority, slow-down trigger         |
| stop-line    | feeds TSR/TLR stop logic                       |
| vehicle      | cross-check with §2 detector                   |
| person       | cross-check with §2 detector                   |

### 5.3 Output

```python
@dataclass
class SegResult:
    sem_mask: np.ndarray   # HxW int8 class ids
    road_mask: np.ndarray  # bool, drivable
    freespace_bev: np.ndarray  # occupancy-grid complement
```

The **road mask** is the single most important output: it defines where the car may
drive and is the geometric anchor for monocular-depth metric scaling (§4.1) and for
lane-polynomial ground-plane projection (§3.2).

---

## 6. Traffic Elements (existing TSR module)

`src/perception/traffic_sign_recognition.py` already defines:

- `TrafficLightState` enum: `UNKNOWN / RED / YELLOW / GREEN`
- `TrafficSignType` enum: `UNKNOWN / SPEED_LIMIT / STOP / YIELD / NO_ENTRY`
- CARLA-dependent helpers (`detect_from_carla`, `get_stop_line_distance`) wrapped
  in try/except so the module imports without CARLA.

### 6.1 Pipeline

```
wide/tele cam ──► YOLO (sign/light head) ──► crop ──► classifier ──► state
                                                                    │
              stop-line seg (§5) ──► distance ──────────────────────┤
                                                                    ▼
                                              TrafficElement list (ego frame)
```

- **Detection:** YOLO head with classes `sign, light, crosswalk, stop_line`.
- **Classification:**
  - Signs: small CNN (MobileNetV2) → `TrafficSignType` + value (speed-limit number
    via a digit OCR head).
  - Lights: color-state classifier (RGB histogram + tiny CNN) → `TrafficLightState`,
    with **temporal hysteresis** (a light must be RED for ≥2 consecutive frames
    before the planner brakes; GREEN→YELLOW needs 1 frame).
- **Distance:** from stop-line segmentation (§5) projected to BEV, cross-checked
  with depth (§4). This replaces the CARLA-only `get_stop_line_distance`.
- **Tracking:** each traffic element is tracked by centroid + class so a sign isn't
  re-detected as new each frame (avoids planner flicker).

### 6.2 Controller hooks

The existing TSR module exposes throttle/brake/target-speed overrides when
approaching lights. These remain; the design only adds the **vision-based distance**
so the module works outside CARLA.

---

## 7. Multi-Camera Fusion — BEV Transformation

### 7.1 BEV construction

For each camera `i` with extrinsics `T_i^ego` and intrinsics `K_i`:

1. Rectified image → (optional) depth map `D_i` (§4).
2. For every pixel `u` with depth `d`, back-project to 3D:
   `p_ego = T_i^ego · K_i⁻¹ · [u, 1]ᵀ · d`.
3. Rotate into ego-ground frame (pitch/roll compensated via IMU + road-plane fit).
4. Rasterize into the shared BEV grid (0.2 m res, 100 m forward × 50 m lateral).

Cameras overlap heavily in the near field; we **max-fuse** confidence and
**min-fuse** depth (closest valid) in overlap cells. The 8-camera BEV is the single
canonical world representation; all downstream consumers read from it.

### 7.2 BEV feature map (for BEVFormer path)

Instead of rasterizing points, BEVFormer-style: sample image features at projected
BEV query locations via deformable attention. This preserves resolution and is the
production target. For the prototype we use the explicit rasterization above
(cheaper, debuggable, no training required).

### 7.3 Ego-motion compensation

Between frame `t` and `t+1` the ego vehicle moves `(dx, dy, dθ)`. We **warp the
previous BEV grid** by `(-dx, -dy, -dθ)` before fusing the new frame, so static
structure aligns and only genuinely new information updates the grid. This is also
what makes temporal tracking (§8) stable.

```
cam1 ─┐
cam2 ─┤
cam3 ─┼─► per-cam BEV ─► ego-warp ─► fuse(max conf, min depth) ─► BEV grid(t)
...   │                                          ▲
cam8 ─┘                                          │
                                       ego-motion warp of BEV grid(t-1)
```

---

## 8. Temporal Tracking

### 8.1 Multi-object tracking — SORT / ByteTrack

- **ByteTrack** (preferred over vanilla SORT) because it associates **low-confidence
  detections** too — critical for VRUs and distant vehicles where the detector is
  marginal.
- State per track: `[cx, cy, cz, vx, vy, vz, w, l, h, θ]` (constant-velocity Kalman).
- Association: IoU in BEV + **ReID cosine distance** on the 128-d embedding from §2.
  Gating: Mahalanobis + cosine threshold.
- **Track lifecycle:** `tentative (3 hits) → confirmed → lost (30 frames) → deleted`.
- Output: each confirmed track is a `TrackedAgent` with a stable `track_id`,
  velocity, and a short motion forecast (constant-velocity + bicycle model for
  cars).

### 8.2 Optical-flow priors

- **RAFT-lite** (or Farnebäck for the cheap path) computes dense flow on the main
  forward camera at 15 Hz.
- Uses:
  1. **Detection propagation** between detector runs — when the 30 Hz detector
     skips a frame, flow warps last frame's boxes forward so the tracker always has
     a measurement.
  2. **Ego-motion estimation** — median flow over the road region gives ego
     translation/rotation, cross-checked against IMU/wheel odometry.
  3. **Moving-object segmentation** — flow inconsistent with ego-motion flags
     dynamic pixels, feeding the tracker's "is dynamic" bit.

### 8.3 Lane tracking recap

Lane polynomials are tracked by the Kalman filter of §3.3 (already implemented in
`kalman_lane_tracker.py`), not by SORT. The two trackers run in parallel and write
into the same `SceneModel`.

### 8.4 SceneModel

```python
@dataclass
class SceneModel:
    t: float                       # capture timestamp
    agents: list[TrackedAgent]     # from §8.1
    lanes: list[TrackedLane]       # from §3.3 (poly + type + conf)
    freespace_bev: np.ndarray      # from §5 + §4 occupancy
    traffic: list[TrafficElement]  # from §6
    depth_bev: np.ndarray          # from §4 fused
    ego_pose: SE2                  # at t_capture
    degraded: bool                 # partial camera set
```

This is the single structure handed to MPC / planner / viz.

---

## 9. Night & Adverse Weather

### 9.1 HDR

- 3-exposure bracketing per camera (e.g. 0.5 ms / 4 ms / 32 ms at night).
- Merge with a camera-specific **tone-mapping LUT** trained to preserve road
  markings and suppress headlight bloom. The LUT is selected by a **light-level
  estimator** (mean Y of the medium exposure) so day/night switch is seamless.
- **Anti-bloom:** specular highlights > 98th percentile are locally desaturated
  before detection to stop YOLO from hallucinating objects in glare.

### 9.2 Denoising

- **Sensor-level:** analog gain capped, multi-frame **temporal NR** (3-frame average
  with motion-compensated alignment via §8.2 flow) at night.
- **Network-level:** a lightweight **NafNet / Restormer-tiny** denoiser runs only
  when `light_level < threshold`, applied to the input before detectors. Cost ≈4 ms.

### 9.3 Domain adaptation

- Train detectors/segmenters with **adverse-weather augmentation**: rain streaks,
  fog (atmospheric scattering), snow, lens-water-droplet overlays, and night gamma
  curves. This closes the sim→real and day→night gap without collecting new data.
- **Fine-tune on unlabeled night data** with self-training: run the day model,
  keep high-confidence detections as pseudo-labels, retrain. Standard SSL recipe,
  no RL.
- **Fog-specific:** add a **dehaze** front-end (AOD-Net or dark-channel prior) that
  activates when the road-mask confidence drops below a threshold (fog collapses
  contrast).

### 9.4 Degradation policy

| Condition            | Action                                                |
|----------------------|-------------------------------------------------------|
| 1–2 cams missing     | `degraded=True`, continue with remaining BEV          |
| Heavy rain/night     | drop to 20 Hz, widen tracker gates, rely on Kalman    |
| All cams fail (>2s)  | safe-pull-over request to planner                     |

---

## 10. Performance Targets

### 10.1 Targets

| Metric                  | Target              | Measurement method                 |
|-------------------------|---------------------|------------------------------------|
| End-to-end FPS          | ≥ 30 Hz             | capture→SceneModel wall clock      |
| Perception latency      | < 100 ms (P95)      | t_SceneModel − t_capture           |
| Object detection mAP    | ≥ 0.55 (val)        | COCO-style on internal set         |
| Lane lateral err        | < 0.10 m @ 30 m     | vs. map / lidar GT                 |
| Depth RMSE              | < 0.8 m @ 30 m      | stereo GT                          |
| Power                   | < 30 W              | Jetson power rail                  |
| Tracker MOTA            | ≥ 0.80              | internal tracking benchmark        |

### 10.2 Budget breakdown (per frame, Orin-class SoC)

| Stage              | Time   | Power  | Notes                          |
|--------------------|--------|--------|--------------------------------|
| ISP + rectify      | 3 ms   | 2 W    | hardware ISP                   |
| YOLOv8-X ×8 cams   | 22 ms  | 12 W   | batched TensorRT FP16          |
| UNet lane (main)   | 6 ms   | 3 W    | existing model                 |
| Depth (MiDaS)      | 8 ms   | 4 W    | per main+tele                  |
| Segmentation       | 7 ms   | 3 W    | MobileNetV3 head               |
| TSR/TLR            | 3 ms   | 1 W    | small classifier               |
| BEV fuse + warp    | 4 ms   | 2 W    | CUDA kernel                    |
| Tracking (SORT/OF) | 3 ms   | 1 W    | CPU + CUDA flow                |
| **Total**          | **56 ms** | **28 W** | within 100 ms / 30 W budget |

The 56 ms compute leaves **~10 ms scheduling slack**; with pipelining (detector
frame N runs while fusion/tracking finishes frame N-1) the **steady-state
end-to-end latency is ~70 ms** at 30 FPS.

### 10.3 Optimization levers if budget is exceeded

1. Drop side/rear cameras to 15 Hz (they cover near-field, less motion).
2. Replace YOLOv8-X with YOLOv8-L or RT-DETR-R18 on side cams.
3. Quantize UNet/depth to INT8 (calibration on night + rain data).
4. Skip segmentation on alternate frames (it's quasi-static).
5. Run optical flow at 15 Hz instead of 30.

---

## 11. Module map → existing code

| Design section        | Existing file                              | Status        |
|-----------------------|--------------------------------------------|---------------|
| §3 Lane detection     | `src/perception/lane_detector.py`          | UNet trained  |
| §3 BEV + polyfit      | `src/perception/bev_lane_pipeline.py`      | Implemented   |
| §3.3 Kalman tracking  | `src/perception/kalman_lane_tracker.py`    | Implemented   |
| §3 spline fitting     | `src/perception/spline_lane_fitting.py`    | Implemented   |
| §3 ego lane           | `src/perception/ego_lane_tracker.py`       | Implemented   |
| §5 road perception    | `src/perception/road_perception.py`        | Partial       |
| §6 TSR/TLR            | `src/perception/traffic_sign_recognition.py` | Implemented |
| §7 BEV lane pipeline  | `src/perception/bev_lane_pipeline.py`      | Implemented   |
| §2 Object detection   | —                                          | **To build**  |
| §4 Depth              | —                                          | **To build**  |
| §5 Segmentation       | —                                          | **To build**  |
| §8 Tracking (SORT)    | —                                          | **To build**  |
| §8 Optical flow       | —                                          | **To build**  |
| §9 HDR/denoise/DA     | —                                          | **To build**  |

---

## 12. Open questions / risks

1. **8-camera data** — BEVFormer needs supervised multi-camera labels; for the
   prototype we fall back to per-camera YOLO + geometric BEV. Risk: scale errors in
   3D lift at range. Mitigation: stereo pair on main+tele.
2. **Night lane visibility** — UNet was trained on daytime CARLA. Mitigation: §9.3
   domain adaptation + the Kalman tracker's 3-frame stale prediction keeps lanes
   alive through brief dropouts.
3. **Fisheye rectification cost** — 8 LUT warps at 1280×960 may exceed ISP budget.
   Mitigation: warp on GPU into a pre-allocated buffer, fused with letterbox for
   inference input.
4. **Sync jitter in sim** — CARLA doesn't model real sensor sync; the sync buffer
   must be validated on hardware before trusting BEV fusion of fast-moving scenes.

---

*End of document.*
