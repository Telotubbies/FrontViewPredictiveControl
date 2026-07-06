# 10 — Data Pipeline & Logging (Design Document)

> **Project:** FrontViewPredictiveControl — Tesla-like prototype
> **Scope:** End-to-end data lifecycle: collection → storage → replay → labeling → curation → versioning → privacy
> **Constraint:** **NO reinforcement learning** — all data feeds supervised imitation / prediction models and offline evaluation only
> **Status:** Design / partial implementation
> **Owner:** Data & Infrastructure team
> **Related code:** `src/telemetry/influxdb_exporter.py`, `src/state.py`, `src/perception/`

---

## 0. Executive Summary

A Tesla-like prototype is only as good as the data it learns from. This document
specifies the **Data Pipeline & Logging** subsystem that governs every byte from
the moment a sensor photon hits silicon to the moment a curated, labeled,
privacy-compliant dataset is handed to the training cluster.

The pipeline is **RL-free**: there is no reward signal, no policy rollout, no
environment interaction during training. Instead, the system performs
**imitation / supervised learning** from human-driven logs and **offline
predictive evaluation** against held-out trajectories. Data quality, diversity,
and label fidelity are therefore the single most important determinants of
on-road performance.

Design targets:

| Metric | Target |
|--------|--------|
| On-vehicle write bandwidth | ≥ 800 MB/s sustained (8 cams + CAN + LiDAR) |
| Circular-buffer retention | 30 min rolling @ full rate, indefinite on trigger |
| Upload-to-cloud latency (parked, WiFi) | < 4 h for 1 TB segment |
| Replay clock accuracy | ±1 ms across all modalities |
| Auto-label IoU (lanes) | ≥ 0.90 vs. human gold set |
| Human review throughput | ≥ 200 frames / reviewer-hour |
| Privacy blur recall (faces + plates) | ≥ 0.998 |

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE — END TO END                          │
│                                                                              │
│  [Sensors] ──► [Edge Store] ──► [Upload] ──► [Replay] ──► [Auto-Label]      │
│  cams+CAN+IMU   NVMe circular    WiFi/5G      MCAP player   foundation models│
│       │              │              │             │              │            │
│       ▼              ▼              ▼             ▼              ▼            │
│  [Telemetry]    [Trigger]      [Cloud Lake]  [Sim/MPC eval]  [Human Review]  │
│  InfluxDB+Graf  event/safety   S3+DVC        offline metrics  CVAT/Labelbox  │
│                                                      │                        │
│                                                      ▼                        │
│                                              [Curated Dataset]               │
│                                              active learning + hard mining   │
│                                                      │                        │
│                                                      ▼                        │
│                                              [Trained Model] → OTA → Vehicle│
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Collection

### 1.1 Sensor Inventory

Every frame of driving produces a synchronized, timestamped multi-modal sample.

| Sensor | Rate | Interface | Payload | Est. BW |
|--------|------|-----------|---------|---------|
| Front camera (main) | 30 Hz | GMSL2 / FPD-Link | RGB 1920×1080 8-bit | ~190 MB/s |
| Side × 4 | 30 Hz | GMSL2 | RGB 1280×720 | ~440 MB/s |
| Rear camera | 30 Hz | GMSL2 | RGB 1920×1080 | ~190 MB/s |
| Narrow/long-range | 30 Hz | GMSL2 | RGB 1280×720 | ~110 MB/s |
| LiDAR (optional) | 10–20 Hz | Ethernet | PointCloud2 ~100k pts | ~50 MB/s |
| IMU | 200 Hz | CAN-FD | accel + gyro | <1 MB/s |
| GPS/GNSS | 10 Hz | NMEA | lat/lon/HDOP | <1 MB/s |
| Vehicle CAN | 50–100 Hz | CAN-FD | speed, steer, throttle, brake, signals | <1 MB/s |

### 1.2 FrameState as the Canonical Per-Frame Record

The prototype already defines a single per-frame state container in
`src/state.py` (`FrameState`). This dataclass is the **join key** for all
downstream data: every sensor packet, CAN sample, and label is aligned to the
same `frame_id` / `timestamp_ns` that produced a given `FrameState`.

Key fields reused by the data pipeline:

```python
# src/state.py (excerpt)
@dataclass
class FrameState:
    rgb: np.ndarray            # ← raw frame archived to MCAP
    speed_kmh: float           # ← CAN speed channel
    steer: float               # ← CAN steering channel
    throttle: float            # ← CAN throttle channel
    brake: float               # ← CAN brake channel
    cte_m: float               # ← derived lane geometry (label candidate)
    heading_rad: float         # ← derived lane geometry
    curvature: float           # ← derived lane geometry
    mode: str                  # ← driving mode (label / stratification key)
    lane_conf: float           # ← perception confidence (curation signal)
    solver_status: str         # ← "Solve_Succeeded" | "Fallback" (hard-example trigger)
    ...
```

### 1.3 Label Sources

Labels are **not** generated only by humans. The pipeline distinguishes four
label provenances:

1. **Human driver behavior** — steer / throttle / brake trajectories are implicit
   labels for the imitation predictor (expert demonstration).
2. **Geometric derivation** — `cte_m`, `heading_rad`, `curvature` from the lane
   pipeline serve as weak supervision for path-prediction heads.
3. **Auto-labeling** — foundation models (SAM, Depth-Anything, BEVFormer)
   produce dense pseudo-labels at 10× human throughput.
4. **Human review** — a reviewer corrects / confirms auto-labels on a sampled
   and hard-example subset (Section 5).

### 1.4 Collection Data Flow

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐
│  Cameras │──►│  ISP/HDR │──►│  Sync    │──►│  Frame Mux  │
│  (8×)    │   │  (SoC)   │   │  Buffer  │   │  (frame_id) │
└──────────┘   └──────────┘   └──────────┘   └──────┬──────┘
                                                    │
┌──────────┐   ┌──────────┐                        │
│   CAN    │──►│  CAN     │───────────────────────►│
│  bus     │   │  decoder │                        │
└──────────┘   └──────────┘                        ▼
                                          ┌────────────────┐
┌──────────┐                              │  FrameState     │
│  IMU/    │─────────────────────────────►│  (per frame)   │
│  GPS     │                              └────────┬───────┘
└──────────┘                                       │
                                                    ▼
                                          ┌────────────────┐
                                          │  MCAP Writer   │
                                          │  + Telemetry   │
                                          └────────────────┘
```

---

## 2. On-Vehicle Storage

### 2.1 Hardware

| Component | Spec | Role |
|-----------|------|------|
| NVMe SSD | 2 TB, PCIe Gen4, ≥ 1 GB/s sequential write | Primary ring buffer |
| Secondary SSD | 1 TB (optional) | Triggered-clip overflow + checksum mirror |
| RAM ring | 8 GB DDR | Pre-trigger burst buffer (sub-frame granularity) |

### 2.2 Circular Buffer Strategy

The SSD is partitioned into a **rolling ring buffer** so the vehicle never runs
out of space during untriggered driving. A background writer appends MCAP
chunks; a head pointer wraps around.

```
SSD layout (2 TB):
┌──────────────────────────────────────────────────────────┐
│  [Ring A: 1.6 TB rolling]  [Clip Store: 0.3 TB]  [Sys]  │
│   ← overwrites oldest      ← triggered snapshots         │
└──────────────────────────────────────────────────────────┘
        ▲ head/tail pointers managed by storage daemon
```

Rules:
- **Normal driving:** Ring A overwrites in 30-minute windows at full rate.
- **Trigger event** (AEB, MPC fallback, driver button, mode change): the
  surrounding ±60 s window is **promoted** to Clip Store and never overwritten.
  `solver_status == "Fallback"` in `FrameState` is one such trigger.
- **Clip Store full:** oldest promoted clips are evicted only after successful
  cloud upload (acknowledged by content hash).
- **Integrity:** each 256 MB chunk carries a CRC32; on boot the daemon scrubs
  the last written chunk and truncates on mismatch.

### 2.3 Write Path & Backpressure

```
Sensor ──► RAM ring (8 GB) ──► NVMe ring (1.6 TB) ──► Clip Store ──► Upload queue
            │ drop-oldest         │ wrap                │ promote      │
            └─ if full, drop      └─ if full, wrap      └─ on trigger  └─ WiFi/5G
```

If the NVMe write bandwidth falls below sensor rate for > 2 s, the RAM ring
drops the **oldest** frames (never the current frame) and emits a
`storage_backpressure` telemetry point so Grafana can flag it.

---

## 3. Data Formats

### 3.1 Format Selection Matrix

| Format | Container | Strengths | Weaknesses | Decision |
|--------|-----------|-----------|------------|----------|
| **ROS bag (bag2)** | sqlite3 / plaintext | Ecosystem, rosbag2 tooling | Schema-heavy, large | Supported for legacy replay only |
| **MCAP** | single file, indexed | Time-indexed, language-agnostic, ROS2-native, compact | Newer ecosystem | **Primary on-vehicle + cloud** |
| **KITTI** | directory of PNG/TXT | Simple, research-standard | No temporal index, no metadata | Export for academic benchmarks |
| **nuScenes** | JSON + blobs | Multi-sensor, canonical for prediction | Heavy schema, 12 tables | Export for model eval / sharing |

### 3.2 MCAP Channel Schema (Primary)

Every MCAP file is one driving **clip** (typically 20–120 s). Channels:

| Channel | Type | Frequency | Notes |
|---------|------|-----------|-------|
| `/cam/front` | `sensor_msgs/Image` (PNG-compressed) | 30 Hz | Main driving camera |
| `/cam/side_l`, `/cam/side_r`, ... | `sensor_msgs/Image` | 30 Hz | Surround |
| `/cam/rear` | `sensor_msgs/Image` | 30 Hz | Rear |
| `/lidar` (opt.) | `sensor_msgs/PointCloud2` | 10–20 Hz | Optional |
| `/can/vehicle` | custom `VehicleState` | 50 Hz | speed, steer, throttle, brake |
| `/imu` | `sensor_msgs/Imu` | 200 Hz | accel + gyro |
| `/gps` | `sensor_msgs/NavSatFix` | 10 Hz | lat/lon |
| `/frame_state` | custom `FrameState` (CBOR) | 30 Hz | Mirrors `src/state.py` dataclass |
| `/telemetry/adas` | custom `ADASTelemetry` | 30 Hz | Mirrors InfluxDB measurements |

### 3.3 KITTI / nuScenes Export

Export is a **derived, offline** step — the source of truth is always MCAP.

```
MCAP (source of truth)
   │
   ├──► KITTI export  (image_02/, velodyne/, calib.txt, labels/)   — for lane/depth papers
   └──► nuScenes export (scene-0001/, sweeps/, annotations.json)   — for prediction eval
```

---

## 4. Compression

### 4.1 Image / Video

| Data | Codec | Settings | Ratio | Notes |
|------|-------|----------|-------|-------|
| Camera streams (archive) | **H.265 (HEVC)** | CRF 23, 30 fps, NVENC hw | ~30:1 vs raw | Lossy; visually transparent for perception |
| Camera streams (gold set) | **PNG / lossless H.265** | per-frame | ~3:1 | Used for label review & model training |
| Depth maps | 16-bit PNG + zstd | level 19 | ~4:1 | Float32 → uint16 mm |
| BEV semantic maps | RLE + zstd | — | ~20:1 | Sparse label maps |

### 4.2 Point Clouds & Geometry

| Data | Codec | Notes |
|------|-------|-------|
| LiDAR scans | **DRACO** (Google) | Lossy 10-bit quantization, ~8:1 |
| Lane polynomials | CBOR + zstd | Tiny; stored alongside `/frame_state` |
| 3D bounding boxes | JSON + zstd | Human-readable in replay |

### 4.3 Compression Data Flow

```
[Raw sensor] ──► [HW encoder (NVENC/DRACO)] ──► [MCAP chunk] ──► [NVMe]
                         │
                         └─► [Lossless copy] ──► [Clip Store] (gold clips only)
```

Hardware encoding is mandatory on-vehicle to stay within the SoC power budget
(< 30 W for the entire data path). Software fallback exists but triggers
`storage_backpressure`.

---

## 5. Replay System

### 5.1 Requirements

- **Clock-accurate** multi-modal replay at ±1 ms across cameras, CAN, IMU, LiDAR.
- **Scrubable**: jump to any `frame_id` and resume real-time or fast-forward.
- **Pluggable sinks**: feed perception, planner, and MPC exactly as live driving
  would, so offline metrics are directly comparable to online behavior.
- **Telemetry parity**: replay must re-emit the same `ADASTelemetry` points that
  were originally sent to InfluxDB, enabling A/B comparison on Grafana.

### 5.2 Replay Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        MCAP REPLAYER                         │
│                                                              │
│  [MCAP File] ──► [Index] ──► [Clock Master] ──► [Demux]     │
│                                  │                  │        │
│                                  ▼                  ▼        │
│                          [Wall-clock /         [Topic Bus]   │
│                           step / fast]            │          │
│                                              ┌───┼───┐      │
│                                              ▼   ▼   ▼      │
│                                          [Perc][Plan][MPC]  │
│                                              │   │   │      │
│                                              └───┴───┘      │
│                                                    │         │
│                                                    ▼         │
│                                          [Metrics Recorder] │
│                                          + InfluxDB replay   │
└──────────────────────────────────────────────────────────────┘
```

### 5.3 Offline Evaluation Harness

The replay system feeds an **offline evaluation harness** that computes the
same KPIs used online (CTE, heading error, TTC, MPC solve time, fallback
count). Because `FrameState.solver_status` and the InfluxDB `mpc` measurement
share a schema, a replayed clip can be diffed against its original live run:

```
Grafana dashboard: "Live vs Replay — Clip 0x1A2B"
  Panel A: speed_kmh      (live) vs (replay)   — should overlap
  Panel B: cte_m          (live) vs (replay)
  Panel C: mpc.solve_time (live) vs (replay)
  Panel D: mpc.fallback_count delta            — nonzero ⇒ nondeterminism bug
```

---

## 6. Labeling Pipeline

### 6.1 Auto-Labeling Stage

Auto-labeling runs on the cloud over uploaded MCAP clips using a fleet of
foundation / specialized models. It is **non-causal** — it can use future
frames, multi-pass filtering, and heavy models that would never run online.

| Task | Model | Output |
|------|-------|--------|
| Lane instance + polyline | UNet + polynomial fit (existing `src/perception/`) | 4-polynomial lane set |
| Object detection | BEVFormer / DETR3D | 3D boxes + track IDs |
| Freespace / drivable | SAM + DeepLab | BEV binary mask |
| Depth | Depth-Anything-V2 | Dense depth map |
| Traffic signs / lights | Existing TSR module | Class + 2D box |
| Face / plate detection | RetinaFace + plate detector | Boxes for blurring (Section 9) |

### 6.2 Human Review Stage

Auto-labels are **not** trusted blindly. A sampled and hard-example subset is
routed to human reviewers in **CVAT** (self-hosted) or Labelbox.

```
[Auto-labels] ──► [Sampler] ──► [Review Queue] ──► [Reviewer] ──► [Gold labels]
                     │               │                              │
                     │               └─ stratified:                  │
                     │                 - random 5%                  │
                     │                 - low-confidence 20%         │
                     │                 - fallback clips 100%        │
                     │                 - novel-scene cluster 50%    │
                     │                                              │
                     └─► [Auto-accept 75%] ─────────────────────────┘
                          (high-confidence, non-novel)
```

### 6.3 Label Schema & Provenance

Every label carries provenance metadata so training can down-weight
pseudo-labels:

```json
{
  "frame_id": 1742893,
  "timestamp_ns": 1719000000123456789,
  "lane": { "polys": [...], "conf": 0.93 },
  "objects": [ { "box3d": [...], "class": "car", "track_id": 7 } ],
  "provenance": {
    "lane": "auto:unet-v2.3",
    "objects": "auto:bevformer-v1.1",
    "reviewed": true,
    "reviewer": "user_42",
    "review_time_ms": 11800
  }
}
```

---

## 7. Training Data Curation

### 7.1 Active Learning Loop

Because there is no RL reward, the only way the system improves is by being
shown **the right data next**. An active-learning loop selects clips for
review based on model uncertainty and disagreement.

```
┌──────────────────────────────────────────────────────────────┐
│                   ACTIVE LEARNING LOOP                       │
│                                                              │
│  [Current Model] ──► [Inference on unlabeled pool]           │
│                          │                                   │
│                          ▼                                   │
│                  [Uncertainty Score]                         │
│                   entropy + MC-dropout + ensemble var        │
│                          │                                   │
│                          ▼                                   │
│                  [Clip Selector] ──► [Review Queue]          │
│                          │                                   │
│                          ▼                                   │
│                  [New Gold Labels] ──► [Retrain] ──► back to top
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Hard Example Mining

Hard examples are explicitly flagged from multiple signals:

| Signal | Source | Threshold |
|--------|--------|-----------|
| `solver_status == "Fallback"` | `FrameState` / InfluxDB `mpc.status` | any fallback |
| `lane_conf < 0.6` | `FrameState.lane_conf` | < 0.6 for ≥ 5 frames |
| Large CTE spike | InfluxDB `lane.cte_m` | |Δcte| > 0.3 m in 1 s |
| AEB activation | InfluxDB `aeb.active == 1` | any activation |
| Driver takeover | CAN steering override while MPC active | torque > threshold |
| Model disagreement | ensemble variance on re-inference | top-5% per clip |

Mined hard clips are **always** promoted to Clip Store, prioritized for upload,
and routed 100% to human review (not the auto-accept path).

### 7.3 Diversity & Stratification

A curated dataset must cover the **operational design domain** (ODD) without
gaps. The curation service stratifies the pool along:

- **Scene type:** urban, highway, rural, parking, night, rain, fog, tunnel.
- **Speed bucket:** 0–20, 20–50, 50–80, 80–120 km/h.
- **Maneuver:** lane keep, lane change, turn, merge, stop-and-go.
- **`FrameState.mode`** (already present): used as a first-class stratification
  key so each driving mode is adequately represented.

A target distribution is defined per release; the curation service samples to
match it, over-sampling under-represented strata and down-sampling highway
cruising (which is over-abundant and low-information).

---

## 8. Existing Telemetry: InfluxDB + Grafana

### 8.1 Current Implementation

The prototype already streams per-frame ADAS telemetry to **InfluxDB v2** via
`src/telemetry/influxdb_exporter.py`. The `TelemetryExporter` class batches
points asynchronously (flush every 0.5 s) so the main control loop is never
blocked. Measurements currently exported:

| Measurement | Fields | Used by data pipeline as |
|-------------|--------|--------------------------|
| `vehicle` | speed_ms, speed_kmh, steering, throttle, brake | Driver-behavior labels, stratification |
| `lane` | cte_m, heading_err_rad, curvature, lane_conf | Hard-example mining, label QC |
| `aeb` | active, ttc, brake_override, warning_level | Safety-event trigger |
| `acc` | active, target_speed_ms, distance_m, closing_rate | ACC eval |
| `ldw` | state, warning_active, side, steering_assist | LDW eval |
| `bsw` | left_alert, right_alert, safe_left, safe_right | BSW eval |
| `tsr` | speed_limit_kmh, traffic_light, action | TSR label source |
| `tja` | state, active, target_speed_ms | TJA eval |
| `stop_and_go` | stopped, restart_ramp | Traffic-scene stratification |
| `lka_pro` | assist | LKA eval |
| `mpc` | solve_time_ms, status, fallback_count | **Hard-example trigger**, perf QC |
| `performance` | fps, loop_time_ms | Pipeline health |

### 8.2 Grafana Dashboards

| Dashboard | Panels | Audience |
|-----------|--------|----------|
| **Live Drive** | speed, steer, CTE, lane_conf, MPC status | Test driver |
| **Safety Events** | AEB/TTC, fallback count, takeover | Safety engineer |
| **Data Health** | storage BW, backpressure, upload queue, clip count | Data team |
| **Replay Diff** | live vs replay KPIs (Section 5.3) | ML / eval team |

### 8.3 Telemetry → Data Pipeline Bridge

```
[Vehicle] ──► [TelemetryExporter] ──► [InfluxDB] ──► [Grafana]
                     │
                     └─► same points embedded in MCAP `/telemetry/adas`
                                                           │
                                                           ▼
                                                   [Cloud Lake]
                                                   (queryable via Flux)
```

InfluxDB is the **live** observability layer; MCAP is the **durable** training
and replay layer. Both carry the same schema so a Grafana panel can be
reconstructed from an MCAP replay without re-instrumentation.

---

## 9. Data Versioning (DVC)

### 9.1 Why DVC

Model code lives in Git; multi-terabyte datasets do not. **DVC** (Data Version
Control) layers Git-like versioning over large files stored in S3 / MinIO,
keeping the Git repo small while recording exact dataset hashes.

### 9.2 Repository Layout

```
frontview-data/                 # DVC + Git repo
├── dvc.yaml                     # pipeline stages (raw → labeled → curated)
├── dvc.lock                     # pinned hashes
├── raw/
│   ├── 2026Q3_clip_0001.mcap.dvc
│   └── ...
├── labels/
│   ├── auto_v2.3.dvc
│   └── reviewed_v1.0.dvc
├── curated/
│   └── release_2026.09.dvc      # the dataset handed to training
└── models/
    └── predictor_v4.dvc         # model artifacts, also versioned
```

### 9.3 Pipeline Stages (dvc.yaml)

```yaml
stages:
  ingest:
    cmd: python -m datapipe ingest --from upload_queue --to raw/
    outs:
      - raw/
  auto_label:
    cmd: python -m datapipe label --in raw/ --out labels/auto_v2.3
    deps:
      - raw/
    outs:
      - labels/auto_v2.3
  review_merge:
    cmd: python -m datapipe merge_review --auto labels/auto_v2.3 --human labels/reviewed
    outs:
      - labels/reviewed_v1.0
  curate:
    cmd: python -m datapipe curate --labels labels/reviewed_v1.0 --out curated/release_2026.09
    deps:
      - labels/reviewed_v1.0
    outs:
      - curated/release_2026.09
  train:
    cmd: python -m train --data curated/release_2026.09 --out models/predictor_v4
    deps:
      - curated/release_2026.09
    outs:
      - models/predictor_v4
```

Every `dvc repro` run is reproducible from the locked hashes, giving full
**dataset ↔ model** traceability required for safety case documentation.

---

## 10. Privacy: Blurring & GDPR

### 10.1 Policy

All collected imagery may contain **personally identifiable information** (PII):
faces and license plates. Under GDPR (and equivalent Thai PDPA), these are
special-category data requiring explicit legal basis and minimization.

Policy decisions:

1. **Minimization:** collect only cameras needed for the current ODD; disable
   interior / cabin cameras by default.
2. **On-vehicle blur (edge):** faces and plates are blurred **before** the
   frame leaves the vehicle for any non-gold clip. Gold clips are blurred in
   the cloud under access control.
3. **Retention:** raw (unblurred) imagery retained at most 90 days in an
   access-controlled enclave; blurred imagery may be retained indefinitely for
   model development.
4. **Right to erasure:** a GDPR/PDPA deletion request is serviced by content
   hash across the lake + DVC index within 30 days.

### 10.2 Blurring Pipeline

```
[Frame] ──► [Face Detector (RetinaFace)] ──┐
        │                                  ├─► [Blur Mask] ──► [Blurred Frame] ──► [MCAP]
        └─► [Plate Detector (YOLO)] ───────┘
```

- Blur method: **Gaussian + pixelate** composite (strong enough to defeat
  re-identification, light enough to preserve scene structure for perception).
- Quality gate: a downstream verifier samples 0.1% of blurred frames and runs
  an independent face/plate detector; **recall must be ≥ 0.998**. Failures are
  quarantined and never released to the training pool.
- Metadata: every blurred region records `{"type":"face","bbox":...,"hash":...}`
  so erasure can target specific frames without re-scanning the lake.

### 10.3 Access Control

| Role | Raw enclave | Blurred lake | Labels | Models |
|------|-------------|--------------|--------|--------|
| Driver | own clips only | own clips | — | — |
| Labeler | — | ✓ | ✓ | — |
| ML engineer | case-by-case | ✓ | ✓ | ✓ |
| Safety reviewer | ✓ (audit) | ✓ | ✓ | ✓ |

All access is logged to an immutable audit table; GDPR data-subject access
requests are fulfilled from this log.

---

## 11. End-to-End Data Flow (Consolidated)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VEHICLE (EDGE)                                     │
│                                                                             │
│  Sensors ──► FrameState ──► MCAP Writer ──► NVMe Ring ──► Clip Store        │
│     │           │             │                                  │          │
│     │           └─► TelemetryExporter ──► InfluxDB ──► Grafana    │          │
│     │                                                             │          │
│     └─► [Edge Blur] ─────────────────────────────────────────────►│          │
│                                                                   │          │
└───────────────────────────────────────────────────────────────────┼──────────┘
                                                                     │ WiFi/5G
                                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CLOUD                                               │
│                                                                             │
│  [Upload Queue] ──► [S3 / MinIO Lake] ──► [MCAP Index]                      │
│                          │                       │                          │
│                          ▼                       ▼                          │
│                    [Auto-Label] ──► [Review] ──► [Curate] ──► [DVC]         │
│                          │            │            │            │           │
│                          └─► [Hard Example Mining] ◄─ InfluxDB queries      │
│                                                       (fallbacks, AEB)     │
│                                                                             │
│  [DVC release] ──► [Train] ──► [Eval via Replay] ──► [OTA] ──► Vehicle      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Open Items & Risks

| # | Item | Risk | Mitigation |
|---|------|------|------------|
| 1 | NVMe write bandwidth under winter cold | SSD slowdown → frame drops | Heated enclosure; RAM ring absorbs 60 s burst |
| 2 | Auto-label drift after model upgrade | New pseudo-labels inconsistent with old gold set | Versioned label schema; shadow-score before cutover |
| 3 | Reviewer bottleneck | 200 frames/h insufficient for long-tail scenes | Prioritized queue; pre-label everything; inter-annotator agreement ≥ 0.85 |
| 4 | GDPR erasure at PB scale | Scanning lake per request is infeasible | Content-hash index; blur metadata enables targeted deletion |
| 5 | Replay determinism | MPC solver nondeterminism breaks diff dashboards | Pin solver threads; record random seed in MCAP metadata |
| 6 | DVC lake growth | `dvc push` slow at > 100 TB | Sharded remotes; per-quarter snapshot releases |

---

## 13. References

- `src/state.py` — `FrameState` dataclass (canonical per-frame record).
- `src/telemetry/influxdb_exporter.py` — `TelemetryExporter` (InfluxDB v2
  batch writer, 0.5 s flush, measurements listed in Section 8).
- `docs/design/02_vision_perception.md` — perception outputs that become
  auto-label sources.
- `docs/design/05_path_planning.md` — MPC outputs that populate
  `mpc.*` telemetry and trigger hard-example mining.
- MCAP spec: https://mcap.dev
- DVC docs: https://dvc.org/doc
- nuScenes format: https://nuscenes.org/data-format
- KITTI format: http://www.cvlibs.net/datasets/kitti
