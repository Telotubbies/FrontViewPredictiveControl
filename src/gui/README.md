# ADAS Dashboard (PyQt5 / PySide6)

Modern dark-theme GUI for CARLA ADAS: UNet lane detection + waypoint fusion + MPC.

**Modular layout:** `theme.py`, `models.py`, `worker.py`, `panels/*`, `main_window.py`, `adas_dashboard.py` (entry).

## Run

From `CARLA_0.9.16` (with CARLA server on port 2000):

```bash
cd /path/to/CARLA_0.9.16
export PYTHONPATH="${PWD}/carla_mpc_classical:${PWD}/PythonAPI:${PYTHONPATH}"
python carla_mpc_classical/gui/adas_dashboard.py
```

Or as module from repo root:

```bash
cd /path/to/CARLA_0.9.16
export PYTHONPATH="${PWD}:${PWD}/PythonAPI:${PYTHONPATH}"
python -m carla_mpc_classical.gui.adas_dashboard
```

Or from `carla_mpc_classical`:

```bash
cd carla_mpc_classical
export PYTHONPATH="${PWD}:${PWD}/../PythonAPI:${PYTHONPATH}"
python gui/adas_dashboard.py
```

Optional: pass model path as first argument:

```bash
python gui/adas_dashboard.py model/lane_unet_final.pth
```

## Dependencies

- **PyQt5** (preferred) or **PySide6**
- **numpy**
- **pyqtgraph** (optional, for real-time charts): `pip install pyqtgraph`
- Project modules: `config`, `pipeline`, `carla_input_output`, `state` (from `carla_mpc_classical`)
- CARLA Python API (e.g. `../PythonAPI`)

Install Qt and optional chart library:

```bash
pip install PyQt5 numpy pyqtgraph
# or
pip install PySide6 numpy pyqtgraph
```

## Features

- **Status bar:** CARLA connection, Town, FPS, speed (km/h)
- **Left panel:** Mode (Classical/UNet), Start/Stop, Emergency brake, Town, fusion toggle, UNet weight slider, MPC horizon slider
- **Center:** Live camera + lane overlay, CTE / heading / curvature / mode
- **Right:** Rolling charts (steer, throttle, brake, CTE, confidence), adaptive weight monitor (w_e, w_psi, w_delta, w_jerk)
- **Bottom:** Log panel

Architecture and modular refactor plan: see [../docs/DASHBOARD_ARCHITECTURE.md](../docs/DASHBOARD_ARCHITECTURE.md).
