# Autonomous Vehicle Prototype Design Documents

> **Design Philosophy**: Tesla-like prototype — **NO reinforcement learning**
> All decisions use classical control, supervised learning, rule-based logic, and optimization.

## Documents

| # | Document | Topic | Lines |
|---|----------|-------|-------|
| 01 | [System Architecture](01_system_architecture.md) | โครงสร้างระบบทั้งหมด 30+ โมดูล | 533 |
| 02 | [Vision Perception](02_vision_perception.md) | 8 cameras, YOLO, UNet, BEV, depth | 400+ |
| 03 | [Sensor Fusion](03_sensor_fusion.md) | LiDAR+Radar+Camera fusion, EKF/UKF | 481 |
| 04 | [Localization & HD Maps](04_localization_hdmaps.md) | GNSS-RTK, SLAM, HD map matching | 622 |
| 05 | [Path Planning](05_path_planning.md) | A*, RRT*, FSM, lane change, overtake | 350+ |
| 06 | [Vehicle Control](06_vehicle_control.md) | MPC, pure pursuit, PID, bicycle model | 448 |
| 07 | [Safety & Redundancy](07_safety_redundancy.md) | ISO 26262, ASIL D, FMEA, fail-safe | 510 |
| 08 | [HMI & Dashboard](08_hmi_dashboard.md) | Touchscreen, HUD, alerts, DMS | 430 |
| 09 | [V2X & Connectivity](09_v2x_connectivity.md) | DSRC, C-V2X, fleet learning, OTA | 534 |
| 10 | [Data Pipeline](10_data_pipeline.md) | MCAP, labeling, DVC, GDPR, active learning | 531 |
| 14 | [ODD Definition](14_odd_definition.md) | Operational Design Domain boundaries | 600+ |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SENSORS (8 cameras, LiDAR, radar,             │
│                     ultrasonic, IMU, GNSS-RTK)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PERCEPTION (30 Hz)                                              │
│  ├─ Vision: YOLO + UNet + DepthAnything + SAM                   │
│  ├─ Sensor Fusion: EKF/UKF + Occupancy Grid                     │
│  ├─ Localization: SLAM + HD Map Matching                        │
│  └─ Traffic: TSR + Traffic Light Recognition                    │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PLANNING (10 Hz)                                                │
│  ├─ Behavior: FSM + Behavior Tree (non-RL)                      │
│  ├─ Route: A* on HD Map                                          │
│  ├─ Motion: RRT* + Trajectory Optimization                      │
│  └─ Prediction: Intent classification (supervised)              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  CONTROL (100 Hz)                                                │
│  ├─ Steering: MPC (IPOPT) + Pure Pursuit fallback               │
│  ├─ Speed: PID + Feedforward + ACC                              │
│  ├─ Safety: AEB + ESC + Jerk Limit                              │
│  └─ ADAS Manager: 9 features (AEB, ACC, LDW, LKA, BSW, LCA,     │
│                    TSR, TJA, Stop&Go)                            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ACTUATION (CAN FD)                                              │
│  ├─ Steering: EPS (drive-by-wire)                               │
│  ├─ Throttle: Electronic throttle body                          │
│  ├─ Brake: Primary + Secondary (redundant)                      │
│  └─ Display: HMI + Grafana + Telemetry                          │
└─────────────────────────────────────────────────────────────────┘
```

## Why NO Reinforcement Learning?

| Criterion | RL (end-to-end) | Classical + Supervised (our approach) |
|-----------|-----------------|---------------------------------------|
| **Explainability** | Black box | Full traceability per decision |
| **Safety case** | Cannot certify | ISO 26262 certifiable |
| **Data efficiency** | Millions of miles | Thousands of labeled examples |
| **Determinism** | Stochastic policy | Deterministic + reproducible |
| **Failure isolation** | Whole network fails | Module-level fallback |
| **Debugging** | Retrain | Fix specific module |
| **Regulatory** | No framework | ISO 21448 (SOTIF), ISO 26262 |

## Existing Code Integration

| Design Module | Existing Code |
|---------------|---------------|
| Vision Perception | `src/perception/` (UNet, Kalman tracker, classical) |
| Sensor Fusion | `src/algorithms/fusion.py` |
| Path Planning | `src/algorithms/reference.py`, `src/algorithms/lane_keep_assist_step.py` |
| Vehicle Control | `src/control/lane_mpc.py`, `src/control/pure_pursuit.py` |
| Safety | `src/safety/` (override, AEB, LDW, BSW, TJA, stuck recovery) |
| HMI | `src/gui/dashboard.py` (Pygame) |
| Telemetry | `src/telemetry/influxdb_exporter.py` + Grafana |
| ADAS Manager | `src/adas/adas_manager.py` (9 features integrated) |
| Localization | `src/carla_input_output.py` (waypoint system) |
| Testing | `tests/unit/` (625 tests), `tests/integration/` |

## Compute Targets

| Component | Power | Latency | Hardware |
|-----------|-------|---------|----------|
| Perception | 70W | <100ms | NVIDIA Orin GPU |
| Planning | 20W | <50ms | Orin CPU |
| Control | 10W | <10ms | Orin CPU (RT) |
| Safety | 5W | <5ms | Dual SoC watchdog |
| **Total** | **105W** | — | Liquid cooled |

## Key Design Decisions

1. **Modular over end-to-end** — each module independently testable and certifiable
2. **Supervised over RL** — human demonstrations + labeling, not trial-and-error
3. **Rule-based behavior** — deterministic FSM + behavior trees for planning
4. **Optimization-based control** — MPC with IPOPT, not learned policy
5. **Redundant safety** — dual compute, dual brake, sensor diversity
6. **Fleet learning without RL** — shadow mode + classical statistics aggregation
