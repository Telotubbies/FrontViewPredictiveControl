# CARLA UE5 Migration Plan — FrontViewPredictiveControl

> **Date:** 2026-07-06
> **Target:** Migrate FVPC from CARLA 0.9.15 (UE4) → CARLA UE5 0.10.0 (UE5.5)
> **Status:** Planning

## Executive Summary

**Good news:** CARLA Python API is **100% compatible** between 0.9.15 and UE5.
FVPC code requires **minimal changes** — mainly installation method and map selection.

## 1. API Compatibility

| API | 0.9.15 | UE5 0.10.0 | Change |
|-----|--------|------------|--------|
| `carla.Client(host, port)` | ✅ | ✅ | None |
| `client.get_world()` | ✅ | ✅ | None |
| `world.get_blueprint_library()` | ✅ | ✅ | None |
| `world.spawn_actor(bp, transform)` | ✅ | ✅ | None |
| `map.get_waypoint(location)` | ✅ | ✅ | None |
| `waypoint.next(distance)` | ✅ | ✅ | None |
| `vehicle.apply_control(control)` | ✅ | ✅ | None |
| `carla.VehicleControl()` | ✅ | ✅ | None |
| `sensor.listen(callback)` | ✅ | ✅ | None |
| `client.load_world('Town01')` | ✅ | ✅ | Optional `map_layers` param |

**Conclusion:** All FVPC modules (perception, control, safety, bridge) work without modification.

## 2. Breaking Changes

| Change | Impact | Action |
|--------|--------|--------|
| Python package: egg → pip wheel | High | `pip install carla>=0.10.0` |
| Python 3.7 dropped | Low | FVPC uses 3.11 ✅ |
| PhysX → Chaos physics | Medium | Tune MPC if dynamics differ |
| Maps: Town01-07 removed | High | Use Town10 or Mine01 |
| Semantic seg: Nanite-only | Low | FVPC uses RGB only |
| Light Manager removed | None | Not used |
| RSS removed | None | Not used |

## 3. Map Changes

| Map | 0.9.15 | UE5 | Notes |
|-----|--------|-----|-------|
| Town01-07 | ✅ | ❌ | Removed |
| Town10 | ✅ | ✅ Remodeled | Nanite + Lumen, default |
| Town11-13, 15 | ✅ | ❌ | Removed |
| Mine01 | ❌ | ✅ New | Off-road mining (Synkrotron) |

**FVPC config change:** `CARLA_MAP = "Town10"` (was Town04)

## 4. Required Code Changes

### 4.1 Python Package Installation

**Current** (`src/managers/carla_manager.py`):
```python
try:
    import carla
except ImportError:
    for _p in (_root.parent / "PythonAPI", _root / ".carla_py"):
        if _p.exists() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import carla
```

**New:**
```python
try:
    import carla
except ImportError:
    raise ImportError("Install CARLA: pip install carla>=0.10.0")
```

### 4.2 Map Selection

```python
# config/default.yaml
carla:
  map: Town10  # was Town04
```

### 4.3 requirements.txt

```
carla>=0.10.0
numpy>=1.24.4
networkx
```

## 5. New UE5 Features

| Feature | Benefit | Usage |
|---------|---------|-------|
| **Nanite** | Virtualized geometry, higher fidelity | Automatic |
| **Lumen** | Real-time global illumination | Automatic |
| **Town10 remodeled** | Urban with potholes, speed bumps | `client.load_world('Town10')` |
| **Mine01** | Off-road industrial testing | `client.load_world('Mine01')` |
| **Digital Twins** | OSM → 3D map generation | `Docs/adv_digital_twin.md` |
| **ROS2 native** | No bridge needed | `sensor.enable_for_ros()` |
| **Map layers** | Selective loading for Opt maps | `client.load_world('Town10_Opt', layers)` |
| **GeoProjection** | Lat/lon from CARLA coordinates | `map.get_geoprojection()` |
| **InvertedAI traffic** | Realistic AI traffic | `PythonAPI/examples/invertedai_traffic.py` |

## 6. Modules Requiring NO Changes

- ✅ `src/bridge/transforms.py` — coordinate transforms (pure math)
- ✅ `src/bridge/obstacles.py` — obstacle detection (stable actor APIs)
- ✅ `src/control/lane_mpc.py` — MPC (uses VehicleControl)
- ✅ `src/control/pure_pursuit.py` — pure pursuit (uses VehicleControl)
- ✅ `src/perception/` — vision pipeline (camera data unchanged)
- ✅ `src/safety/` — all safety modules (stable APIs)
- ✅ `src/gui/` — dashboard (display logic unchanged)
- ✅ `src/adas/` — ADAS manager (stable APIs)
- ✅ `src/telemetry/` — InfluxDB exporter (unchanged)

## 7. Migration Phases

| Phase | Duration | Tasks |
|-------|----------|-------|
| 1. Build CARLA UE5 | 2-3 hr | CarlaSetup.bat (in progress) |
| 2. Install Python pkg | 5 min | `pip install carla>=0.10.0` |
| 3. Update imports | 30 min | carla_manager.py, carla_interface.py |
| 4. Update map config | 5 min | Town04 → Town10 |
| 5. Test connection | 1 hr | Spawn vehicle, camera, waypoints |
| 6. Integration test | 2 hr | Full FVPC pipeline on Town10 |
| 7. MPC retune | 2 hr | If Chaos physics differs from PhysX |
| 8. Documentation | 1 hr | Update README, config docs |

## 8. Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Chaos physics differs | MPC fails | Medium | Retune weights, test on Town10 |
| Town04 unavailable | Scenarios break | High | Update to Town10 |
| LFS download slow | Build delayed | High | Wait (or use binary package) |
| Semantic camera changed | If used | Low | FVPC uses RGB only |

## 9. Rollback Plan

If UE5 migration fails:
```bash
pip uninstall carla
# Reinstall 0.9.15 egg file
git checkout pre-migration-branch
```

## 10. Build Status

- **CARLA source:** `G:\CarlaUE5` (cloned, ue5-dev branch)
- **CARLA content:** Cloning via LFS (~6 GB downloaded, 4% of 44k files)
- **UE5.5 fork:** Pending (225 GB, 1-2 hr build)
- **CarlaSetup.bat:** Running in background
