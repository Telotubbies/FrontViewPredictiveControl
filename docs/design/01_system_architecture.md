# 01 — System Architecture

**Project:** FrontViewPredictiveControl — Tesla-like Prototype Autonomous Vehicle
**Document Type:** Architecture Design Document
**Status:** Draft v1.0
**Constraint:** NO Reinforcement Learning (no end-to-end RL policy, no reward shaping, no RL training loop)

---

## 1. Purpose & Scope

This document defines the end-to-end system architecture for a Tesla-like
prototype autonomous vehicle built on top of the existing
`FrontViewPredictiveControl` (FVPC) CARLA MPC lane-keeping codebase. The
architecture follows a **classical, modular, deterministic** design philosophy
that mirrors the layered structure of production autonomy stacks
(Perception -> Planning -> Control -> Actuation) while deliberately excluding
reinforcement learning.

Scope of this document:

1. Layered design and data flow.
2. Module breakdown (20+ modules).
3. Compute architecture (NVIDIA Orin / Tesla FSD chip).
4. Justification for the non-RL approach.
5. Redundancy and fail-safe strategy.
6. Real-time timing constraints.
7. Integration with existing FVPC modules.

Out of scope: detailed algorithm internals (covered in per-module specs),
HIL/SIL test harness design (covered in `02_test_strategy.md`).

---

## 2. Design Philosophy: Why Not Reinforcement Learning?

A Tesla-like prototype must be **inspectable, debuggable, and certifiable**.
The team has made an explicit architectural decision to avoid reinforcement
learning for the driving policy. The rationale:

| Concern              | RL (end-to-end policy)            | Non-RL (modular MPC)                 |
|----------------------|-----------------------------------|--------------------------------------|
| Explainability       | Opaque black-box policy           | Each stage produces interpretable    |
|                      |                                   | signals (CTE, heading, curvature)    |
| Safety case          | Hard to argue correctness         | Constraints enforced in MPC cost +   |
|                      |                                   | explicit safety override             |
| Data efficiency      | Millions of driving km to train   | Hand-tuned + classical CV, runs in   |
|                      |                                   | simulation immediately               |
| Determinism          | Stochastic policy, hard to replay | Deterministic given inputs           |
| Failure isolation    | One network fails = whole car     | Module-level fallback (WP-only mode) |
| Certifiability       | No accepted standard              | ISO 26262-compatible decomposition   |
| Simulation cost      | Must retrain on sim-to-real gap   | Classical detectors generalize well  |

The existing codebase already implements this philosophy successfully:
`pipeline.py` runs Perception -> Fusion -> Reference Path -> MPC -> Safety,
producing `(steer, throttle, brake)` with no learned policy in the loop. The
UNet is used only for **lane segmentation** (a perception task), not for
control. This document formalizes and extends that pattern.

---

## 3. Layered Design

The system is decomposed into four primary layers plus two cross-cutting
subsystems (Safety / Telemetry). Each layer consumes the output of the layer
above and operates at its own frequency.

```
+--------------------------------------------------------------------------+
|                        MISSION / ROUTING LAYER                           |
|   (waypoints, global route, target speed profile)                        |
+--------------------------------------------------------------------------+
                                 |
                                 v
+--------------------------------------------------------------------------+
|  L1  PERCEPTION        (30 Hz)                                           |
|  Cameras, radar, IMU -> lane geometry, obstacles, road model, ego state  |
+--------------------------------------------------------------------------+
                                 |
                                 v
+--------------------------------------------------------------------------+
|  L2  PLANNING          (10 Hz)                                           |
|  Reference path, trajectory selection, speed planning, behavior arbiter  |
+--------------------------------------------------------------------------+
                                 |
                                 v
+--------------------------------------------------------------------------+
|  L3  CONTROL           (100 Hz)                                          |
|  MPC / Pure Pursuit, longitudinal controller, actuator setpoint gen      |
+--------------------------------------------------------------------------+
                                 |
                                 v
+--------------------------------------------------------------------------+
|  L4  ACTUATION         (100 Hz)                                          |
|  CAN/FBX bridge -> steering, throttle, brake commands to vehicle         |
+--------------------------------------------------------------------------+
        ^                              ^
        |                              |
+-------------------+        +------------------------------------------+
|  SAFETY (cross)   |        |  TELEMETRY / VIZ (cross)                 |
|  AEB, override,   |        |  InfluxDB, dashboard, logging            |
|  stuck recovery   |        |                                          |
+-------------------+        +------------------------------------------+
```

### 3.1 Layer Responsibilities

| Layer        | Input                          | Output                                | Rate    | Latency Budget |
|--------------|--------------------------------|---------------------------------------|---------|----------------|
| Perception   | Raw RGB, radar, IMU, waypoints | Lane geometry, obstacle list, ego pose| 30 Hz   | <= 33 ms       |
| Planning     | Perception output + mission    | Reference path (s,l), v_ref, behavior | 10 Hz   | <= 100 ms      |
| Control      | Reference path + ego state     | steer, throttle, brake setpoints      | 100 Hz  | <= 10 ms       |
| Actuation    | Setpoints                      | Vehicle actuator commands             | 100 Hz  | <= 5 ms        |

---

## 4. Data Flow (End-to-End ASCII Diagram)

```
  Camera(s)    Radar    IMU/Odometry    CARLA Waypoints (mission)
      |          |          |                   |
      v          v          v                   v
  +-----------------------------------------------+   30 Hz
  |            PERCEPTION FUSION BUS              |
  |  LaneDetector -> BEVLanePipeline -> Kalman    |
  |  TrafficSignRec -> ObstacleBridge -> EgoPose  |
  +-----------------------------------------------+
          |              |              |
      lane_geom      obstacles       ego_state
          |              |              |
          v              v              v
  +-----------------------------------------------+   10 Hz
  |              PLANNING CORE                    |
  |  FusionPolicy -> ReferencePath -> SpeedPlan   |
  |  BehaviorArbiter (lane keep / TJA / stop-go)  |
  +-----------------------------------------------+
          |
      ref_path, v_ref, behavior_mode
          |
          v
  +-----------------------------------------------+   100 Hz
  |              CONTROL CORE                     |
  |  LaneMPC (lateral) + LongitudinalCtrl         |
  |  PurePursuit (fallback)                       |
  +-----------------------------------------------+
          |
      steer, throttle, brake (raw)
          |
          v
  +-----------------------------------------------+   100 Hz
  |            SAFETY OVERRIDE                    |
  |  AEB/ACC, LDW, BSW/LCA, StuckRecovery         |
  |  -> clamp / emergency brake                   |
  +-----------------------------------------------+
          |
      steer, throttle, brake (safe)
          |
          v
  +-----------------------------------------------+
  |            ACTUATION BRIDGE                   |
  |  CarlaInterface / CAN bus                     |
  +-----------------------------------------------+
          |
          v
      VEHICLE (CARLA sim / real ECU)
          |
      telemetry -> InfluxDB + Dashboard
```

---

## 5. Module Breakdown (20+ Modules)

The table below maps each module to its layer, source file (where it already
exists in FVPC), and role. Modules marked **[NEW]** are proposed additions to
reach full Tesla-like capability; the rest already exist in `src/`.

### 5.1 Perception (L1)

| # | Module                    | Source (src/)                          | Role |
|---|---------------------------|----------------------------------------|------|
| 1 | LaneDetector              | `perception/lane_detector.py`          | Lane line detection (UNet or classical) |
| 2 | BEVLanePipeline           | `perception/bev_lane_pipeline.py`      | BEV transform + lane geometry extraction |
| 3 | LaneTrajectoryPipeline    | `perception/lane_trajectory.py`        | Full lane trajectory + confidence + v_ref |
| 4 | SplineLaneFitting         | `perception/spline_lane_fitting.py`    | Spline fit to lane points |
| 5 | KalmanLaneTracker         | `perception/kalman_lane_tracker.py`    | Temporal smoothing of lane params |
| 6 | EgoLaneMask / Tracker     | `perception/ego_lane_mask.py`          | Ego-lane segmentation mask |
| 7 | RoadPerception (BEV)      | `perception/road_perception.py`        | BEV road model aggregator |
| 8 | TrafficSignRecognition    | `perception/traffic_sign_recognition.py` | Speed limit / sign detection |
| 9 | ClassicalDetector         | `perception/classical/detector.py`     | Color-threshold fallback (no GPU) |
| 10| ObstacleBridge            | `bridge/obstacles.py`                  | CARLA traffic actors -> obstacle list |
| 11| SensorFusion              | `alg/fusion.py`                        | WP + vision fusion policy |
| 12| LaneTemporalSmoother      | `temporal/lane_lstm.py`                | EMA/LSTM temporal smoothing |

### 5.2 Planning (L2)

| # | Module                | Source (src/)                  | Role |
|---|-----------------------|--------------------------------|------|
| 13| ReferencePathGen      | `pipeline.py:get_reference_path` | Build (s, lateral) path from CTE/heading/curv |
| 14| SpeedPlanner          | `pipeline.py` (inline)         | v_ref from curvature + target speed |
| 15| BehaviorArbiter       | `adas/adas_manager.py`         | Lane keep / TJA / stop-go selection |
| 16| TJAStopGo             | `safety/tja_stop_go.py`        | Traffic jam assist / stop-and-go logic |

### 5.3 Control (L3)

| # | Module            | Source (src/)              | Role |
|---|-------------------|----------------------------|------|
| 17| LaneMPC           | `control/lane_mpc.py`      | Lateral MPC (N=10, dt=0.1) |
| 18| PurePursuit       | `control/pure_pursuit.py`  | Geometric fallback controller |
| 19| MPCRunner         | `core/mpc_runner.py`       | MPC solve orchestration |
| 20| ControlManager    | `managers/control_manager.py` | Controller lifecycle / mode switch |

### 5.4 Actuation (L4)

| # | Module            | Source (src/)              | Role |
|---|-------------------|----------------------------|------|
| 21| CarlaInterface    | `core/carla_interface.py`  | Sim vehicle command bridge |
| 22| ActuationBridge   | **[NEW]** `actuation/bridge.py` | Abstract CAN/FBX interface for real ECU |

### 5.5 Safety (cross-cutting)

| # | Module            | Source (src/)              | Role |
|---|-------------------|----------------------------|------|
| 23| SafetyOverride    | `safety/override.py`       | Global clamp + emergency brake |
| 24| AEB_ACC           | `safety/aeb_acc.py`        | Automatic emergency brake + ACC |
| 25| LDW               | `safety/ldw.py`            | Lane departure warning |
| 26| BSW_LCA           | `safety/bsw_lca.py`        | Blind-spot / lane change assist |
| 27| StuckRecovery     | `safety/stuck_recovery.py` | Unstick maneuver |

### 5.6 Telemetry / Visualization (cross-cutting)

| # | Module            | Source (src/)              | Role |
|---|-------------------|----------------------------|------|
| 28| TelemetryExporter | `telemetry/influxdb_exporter.py` | InfluxDB time-series export |
| 29| Dashboard         | `gui/dashboard.py`         | Real-time Qt visualization |
| 30| DisplayManager    | `managers/display_manager.py` | Render orchestration |

> Total: **30 modules** (22 existing, 1 new, 7 cross-cutting). Exceeds the
> 20-module requirement.

---

## 6. Compute Architecture

### 6.1 Target Hardware

The prototype targets two compute platforms. The software stack is
hardware-agnostic through a device-abstraction layer (`utils/device_utils.py`).

```
+---------------------------------------------------------------+
|                  APPLICATION LAYER (Python)                   |
|   Perception / Planning / Control / Safety / Telemetry        |
+---------------------------------------------------------------+
|                  ACCELERATION LAYER                           |
|   PyTorch (CUDA/TensorRT)  |  NumPy  |  CV (OpenCV)           |
+---------------------------------------------------------------+
|                  DEVICE ABSTRACTION (device_utils.py)         |
+---------------------------------------------------------------+
        |                                  |
+-------------------+          +-------------------------+
| NVIDIA Orin AGX   |          | Tesla FSD Chip (ref)    |
| - 275 TOPS (INT8) |          | - 144 TOPS (INT8)       |
| - 12-core ARM     |          | - Tri-core x86 + NPU    |
| - 64 GB LPDDR5    |          | - 32 GB LPDDR4          |
| - Ampere GPU      |          | - Custom NPU + ISP      |
| - Dev kit / prod  |          | - Production reference  |
+-------------------+          +-------------------------+
```

### 6.2 Compute Partitioning

| Subsystem        | Accelerator       | Budget        | Notes |
|------------------|-------------------|---------------|-------|
| Lane seg (UNet)  | GPU/NPU (INT8)    | ~8 ms / frame | TensorRT-optimized |
| Classical detect | CPU               | ~5 ms / frame | Fallback, no GPU |
| BEV transform    | CPU + GPU         | ~3 ms         | NumPy + torch |
| Kalman tracker   | CPU               | < 1 ms        | Lightweight |
| MPC solve        | CPU               | ~3 ms         | CasADi / scipy |
| Safety checks    | CPU               | < 1 ms        | Deterministic |
| Telemetry        | CPU (async)       | non-critical  | Decoupled thread |
| Dashboard        | CPU + GPU         | best-effort   | Must not starve control |

### 6.3 Memory & Bandwidth

- Perception frame buffer: 1280x720x3 RGB @ 30 Hz ~= 79 MB/s raw; compressed
  in shared memory ring buffer to avoid copies.
- MPC state: < 1 KB per solve; negligible.
- Telemetry: batched 1 Hz to InfluxDB to avoid I/O contention with control.

---

## 7. Redundancy & Fail-Safe

### 7.1 Redundancy Strategy

The system uses **graceful degradation**, not full hardware redundancy (single
compute node prototype). Each layer has a fallback path:

```
  Primary path:   UNet lane seg -> BEV -> Kalman -> MPC
                                  |
                  (on UNet failure / low confidence)
                                  v
  Fallback path:  Classical detector -> Spline -> MPC
                                  |
                  (on vision failure / geometry invalid)
                                  v
  Last resort:    Waypoint-only fusion -> PurePursuit
                                  |
                  (on controller failure / stuck)
                                  v
  Safe state:     SafetyOverride -> emergency brake -> stop
```

### 7.2 Fail-Safe States

| State            | Trigger                          | Action |
|------------------|----------------------------------|--------|
| NORMAL           | All modules healthy              | Full autonomous driving |
| DEGRADED_VISION  | Lane confidence < threshold      | Switch to classical detector |
| WAYPOINT_ONLY    | Vision geometry invalid          | WP fusion only, reduce speed |
| EMERGENCY_BRAKE  | AEB trigger / critical fault     | Max brake, steer hold |
| STUCK_RECOVERY   | Speed ~0 for N seconds           | Recovery maneuver, then retry |
| MANUAL_TAKEOVER  | Human override / watchdog timeout| Disengage autonomy, alert driver |

### 7.3 Watchdog

A heartbeat watchdog monitors each layer. If a layer misses its deadline by
> 2x, the system transitions to the next more conservative state. This is
implemented via the existing `SafetyOverride` plus a planned `Watchdog`
module (`safety/watchdog.py` [NEW]).

---

## 8. Timing Constraints

Real-time determinism is critical. The system runs three independent rate
groups, loosely coupled through lock-free shared state.

### 8.1 Rate Groups

| Rate Group   | Frequency | Period  | Jitter Budget | Modules |
|--------------|-----------|---------|---------------|---------|
| Perception   | 30 Hz     | 33.3 ms | +/- 2 ms      | LaneDetector, BEV, Kalman, ObstacleBridge |
| Planning     | 10 Hz     | 100 ms  | +/- 5 ms      | Fusion, ReferencePath, SpeedPlanner, Behavior |
| Control      | 100 Hz    | 10 ms   | +/- 1 ms      | LaneMPC, SafetyOverride, Actuation |

### 8.2 Timing Diagram

```
Time (ms):  0    10   20   30   40   50   60   70   80   90  100
            |    |    |    |    |    |    |    |    |    |    |
Control:    C----C----C----C----C----C----C----C----C----C----C   (100 Hz)
Perception: P---------P---------P---------P---------P---------P   (30 Hz)
Planning:   P-----------------------------P-------------------P   (10 Hz)
            ^                             ^
            plan uses latest perception   plan refreshes ref path
```

### 8.3 Deadline Enforcement

- Control loop is **strictly periodic**; it always uses the latest cached
  reference path and ego state. It never blocks on perception/planning.
- Perception and Planning run in separate threads/processes and publish
  results to shared state via atomic swaps.
- If Planning misses a deadline, Control continues using the last valid
  reference path for up to 200 ms, after which it commands a safe deceleration.

### 8.4 Current Codebase Alignment

The existing `config.py` already defines `CONTROL_HZ` and `MAIN_LOOP_SLEEP_S`,
and `pipeline.py` runs the full Perception->MPC->Safety chain in a single
step. The proposed architecture refactors this single-step loop into three
decoupled rate groups while preserving the existing module implementations.

---

## 9. Integration with Existing FVPC Modules

### 9.1 Current Architecture (as-is)

`src/main.py` defines `CARLAMPCSystem`, which:
1. Creates `CarlaInterface` (sim bridge).
2. Creates `LKAPipeline` (perception + fusion + MPC + safety in one object).
3. Runs a single main loop calling `pipeline.step(rgb, speed, wp_state, ...)`.
4. Applies `SafetyOverride`, `StuckRecovery`, `AEBACC`, `ADASManager`.
5. Updates `Dashboard` and `TelemetryExporter`.

```
  main.py
    |
    +-- CarlaInterface (core/carla_interface.py)
    +-- LKAPipeline (pipeline.py)
    |     +-- BEVRoadPerception / LaneTrajectoryPipeline  [Perception]
    |     +-- LaneTemporalSmoother (temporal/lane_lstm.py)
    |     +-- apply_fusion (alg/fusion.py)                [Planning]
    |     +-- get_reference_path (pipeline.py)
    |     +-- LaneMPC (control/lane_mpc.py)               [Control]
    |     +-- SafetyOverride (safety/override.py)         [Safety]
    +-- SafetyOverride, StuckRecovery, AEBACC, ADASManager
    +-- Dashboard, TelemetryExporter
```

### 9.2 Target Architecture (to-be)

The refactor splits the monolithic `LKAPipeline.step()` into three
independently-scheduled components, reusing the same module classes:

```
  main.py (orchestrator)
    |
    +-- PerceptionThread (30 Hz)
    |     +-- LaneTrajectoryPipeline.process(rgb)   [existing]
    |     +-- ObstacleBridge.get_traffic_obstacles() [existing]
    |     +-- publishes: LaneState, ObstacleList
    |
    +-- PlanningThread (10 Hz)
    |     +-- apply_fusion(wp, lane, ...)            [existing]
    |     +-- get_reference_path(cte, head, curv)    [existing]
    |     +-- SpeedPlanner (curv -> v_ref)           [existing inline]
    |     +-- BehaviorArbiter / ADASManager          [existing]
    |     +-- publishes: ReferencePath, BehaviorMode
    |
    +-- ControlThread (100 Hz)
    |     +-- LaneMPC.solve(...)                     [existing]
    |     +-- PurePursuit (fallback)                 [existing]
    |     +-- SafetyOverride.apply_safety_override() [existing]
    |     +-- AEBACC / StuckRecovery                 [existing]
    |     +-- publishes: ActuatorCommand
    |
    +-- ActuationBridge
    |     +-- CarlaInterface.send_control(...)       [existing]
    |
    +-- TelemetryThread (async)
          +-- TelemetryExporter                      [existing]
          +-- Dashboard                              [existing]
```

### 9.3 Integration Rules

1. **No rewrite of existing modules.** All current classes (`LaneMPC`,
   `LaneTrajectoryPipeline`, `SafetyOverride`, etc.) are reused as-is; only
   their invocation scheduling changes.
2. **Shared state via `state.py`.** The existing `FrameState` dataclass is
   extended into typed shared-state objects (`LaneState`, `PlanState`,
   `ActuatorCommand`) published between threads.
3. **Config-driven.** All tunables remain in `config.py`
   (`CONTROL_HZ`, `USE_TRAJECTORY_PIPELINE`, `USE_CLASSICAL_DETECTOR`, etc.).
4. **Backward compatibility.** A single-thread compatibility mode
   (`LKAPipeline.step`) is preserved for debugging and HIL benches where
   thread scheduling is undesirable.
5. **Device abstraction.** `utils/device_utils.py` selects CUDA/CPU; the
   refactor keeps this unchanged so the stack runs on Orin, FSD reference
   boards, or a dev laptop.

---

## 10. Open Items / Future Work

| Item | Description | Doc |
|------|-------------|-----|
| Watchdog module | Heartbeat monitor for each rate group | `safety/watchdog.py` [NEW] |
| ActuationBridge | Real ECU CAN interface (beyond CARLA) | `actuation/bridge.py` [NEW] |
| Multi-camera fusion | Extend beyond single front view | `02_sensor_fusion.md` |
| HIL test harness | dSPACE / Speedgoat integration | `02_test_strategy.md` |
| Formal timing analysis | WCET measurement on Orin | `03_timing_analysis.md` |
| ISO 26262 mapping | ASIL decomposition of modules | `04_safety_case.md` |

---

## 11. Appendix A: Module-to-File Cross-Reference

```
Layer         Module                   File
-----------  -----------------------  ------------------------------------
Perception    LaneDetector             src/perception/lane_detector.py
Perception    BEVLanePipeline          src/perception/bev_lane_pipeline.py
Perception    LaneTrajectoryPipeline   src/perception/lane_trajectory.py
Perception    SplineLaneFitting        src/perception/spline_lane_fitting.py
Perception    KalmanLaneTracker        src/perception/kalman_lane_tracker.py
Perception    EgoLaneMask              src/perception/ego_lane_mask.py
Perception    RoadPerception           src/perception/road_perception.py
Perception    TrafficSignRecognition   src/perception/traffic_sign_recognition.py
Perception    ClassicalDetector        src/perception/classical/detector.py
Perception    ObstacleBridge           src/bridge/obstacles.py
Perception    SensorFusion             src/alg/fusion.py
Perception    LaneTemporalSmoother     src/temporal/lane_lstm.py
Planning      ReferencePathGen         src/pipeline.py
Planning      SpeedPlanner             src/pipeline.py
Planning      BehaviorArbiter          src/adas/adas_manager.py
Planning      TJAStopGo                src/safety/tja_stop_go.py
Control       LaneMPC                  src/control/lane_mpc.py
Control       PurePursuit              src/control/pure_pursuit.py
Control       MPCRunner                src/core/mpc_runner.py
Control       ControlManager           src/managers/control_manager.py
Actuation     CarlaInterface           src/core/carla_interface.py
Actuation     ActuationBridge          src/actuation/bridge.py [NEW]
Safety        SafetyOverride           src/safety/override.py
Safety        AEB_ACC                  src/safety/aeb_acc.py
Safety        LDW                      src/safety/ldw.py
Safety        BSW_LCA                  src/safety/bsw_lca.py
Safety        StuckRecovery            src/safety/stuck_recovery.py
Telemetry     TelemetryExporter        src/telemetry/influxdb_exporter.py
Telemetry     Dashboard                src/gui/dashboard.py
Telemetry     DisplayManager           src/managers/display_manager.py
```

---

## 12. Appendix B: Glossary

| Term   | Meaning |
|--------|---------|
| AEB    | Automatic Emergency Braking |
| ACC    | Adaptive Cruise Control |
| BEV    | Bird's-Eye View |
| BSW    | Blind Spot Warning |
| CTE    | Cross-Track Error |
| FSD    | Full Self-Driving (Tesla chip) |
| FVPC   | FrontViewPredictiveControl (this project) |
| LCA    | Lane Change Assist |
| LDW    | Lane Departure Warning |
| LKA    | Lane Keeping Assist |
| MPC    | Model Predictive Control |
| Orin   | NVIDIA Orin AGX SoC |
| TJA    | Traffic Jam Assist |
| WCET   | Worst-Case Execution Time |
| WP     | Waypoint |

---

*End of document.*
