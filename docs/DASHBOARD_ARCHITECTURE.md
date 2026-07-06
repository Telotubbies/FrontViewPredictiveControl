# ADAS Dashboard — Architecture & Modular Refactor

## Current: Single-File Layout

`gui/adas_dashboard.py` is a single file containing:

| Component | Responsibility |
|-----------|----------------|
| **DARK_THEME** | Stylesheet (dark #121212, cyan accent) |
| **compute_adaptive_weights()** | Mirror of MPC scale factors for display |
| **FrameData** | Dataclass for one frame of telemetry |
| **CarlaWorker (QObject)** | Runs in QThread: CARLA connect, pipeline loop, emits frame_ready / fps / log / connection_changed |
| **VisualizationPanel** | Camera image + overlay, CTE/heading/curvature/mode labels |
| **DiagnosticsPanel** | Pyqtgraph rolling charts (steer, throttle, brake, CTE, confidence) |
| **AdaptiveWeightMonitor** | w_e, w_psi, w_delta, w_jerk; highlight when confidence low |
| **ControlPanel** | Mode, Start/Stop/Emergency, Town, fusion toggle, UNet weight & MPC horizon sliders |
| **LoggingPanel** | QTextEdit for system logs |
| **MainWindow** | Status bar, layout, connects worker signals to panels |

Signals/slots keep the UI responsive: worker runs in a separate thread and emits `frame_ready(FrameData)`; the main thread updates widgets from that data.

---

## Suggested Modular Refactor

Split into a small GUI package so each panel and the worker can be tested and styled independently.

```
carla_mpc_classical/gui/
├── __init__.py              # Re-export main entry
├── theme.py                 # DARK_THEME, colors, fonts
├── models.py                # FrameData, compute_adaptive_weights()
├── worker.py                # CarlaWorker (QObject), run() loop
├── panels/
│   ├── __init__.py
│   ├── control_panel.py     # ControlPanel
│   ├── visualization_panel.py
│   ├── diagnostics_panel.py
│   ├── weight_monitor.py    # AdaptiveWeightMonitor
│   └── logging_panel.py
├── main_window.py           # MainWindow, status bar, layout, signal wiring
└── adas_dashboard.py        # Thin entry: QApplication, MainWindow, main()
```

### 1. `theme.py`

- `DARK_THEME`: full stylesheet string.
- Optional: `ACCENT_COLOR`, `BG_COLOR`, `FONT_FAMILY` for reuse in custom widgets.

### 2. `models.py`

- `FrameData` dataclass.
- `compute_adaptive_weights(confidence, curvature) -> dict` (no Qt dependency).

### 3. `worker.py`

- `CarlaWorker(QObject)` with signals: `frame_ready`, `fps_updated`, `log_message`, `connection_changed`.
- Methods: `start_sim()`, `stop_sim()`, `request_emergency_brake()`, `set_town()`, `set_fusion_enabled()`, `set_unet_weight()`, `set_mpc_horizon()`.
- `run()`: CARLA connect, spawn, pipeline loop, emit frame_ready.

### 4. `panels/control_panel.py`

- `ControlPanel(QFrame)`.
- Signals: `mode_changed`, `start_clicked`, `stop_clicked`, `emergency_clicked`, `town_changed`, `fusion_toggled`, `unet_weight_changed`, `mpc_horizon_changed`.
- `set_running(bool)` to enable/disable Start/Stop.

### 5. `panels/visualization_panel.py`

- `VisualizationPanel(QFrame)`.
- Slot: `update_frame(FrameData)` — composite image + overlay, set labels.

### 6. `panels/diagnostics_panel.py`

- `DiagnosticsPanel(QFrame)`.
- Slot: `update_telemetry(FrameData, t_sec)`.
- Optional: fallback when pyqtgraph is missing (e.g. text-only or placeholder).

### 7. `panels/weight_monitor.py`

- `AdaptiveWeightMonitor(QFrame)`.
- Slot: `update_weights(weights_dict, confidence_low)`.

### 8. `panels/logging_panel.py`

- `LoggingPanel(QFrame)`.
- Method: `append(msg)`.

### 9. `main_window.py`

- `MainWindow(QMainWindow)`.
- Builds status bar, splitter (control | visualization | diagnostics+weight), log panel.
- Creates `CarlaWorker` and `QThread`, connects worker signals to panel slots.
- Connects control panel signals to `_on_start`, `_on_stop`, `_on_unet_weight`, `_on_fusion_toggle`, worker setters.
- No CARLA/pipeline imports; only Qt and panel/worker imports.

### 10. `adas_dashboard.py` (entry)

- Parse argv (optional model path).
- `QApplication`, apply style/font.
- `MainWindow(model_path)`, show, `app.exec_()`.

---

## Data Flow

1. User clicks **Start** → MainWindow calls `worker.start_sim()` and `thread.start()` → worker `run()` connects to CARLA, builds pipeline, enters loop.
2. Each frame: worker gets image, waypoints, calls `pipeline.step()` → gets `FrameState` → builds `FrameData` → emits `frame_ready(fd)`.
3. Main thread (Qt): `_on_frame(fd)` updates VisualizationPanel, DiagnosticsPanel, AdaptiveWeightMonitor, status speed; FPS from `fps_updated`.
4. User moves sliders → `_on_unet_weight` / `_on_fusion_toggle` update `config` (and worker setters for consistency) → next `pipeline.step()` in worker uses new config.
5. **Stop** → `worker.stop_sim()` → `_running = False` → worker loop exits → `connection_changed(False)`.

---

## Performance (20–30 FPS)

- CARLA + pipeline run in **worker thread**; only lightweight data (numpy arrays, floats) are passed via signals.
- UI updates in main thread: set pixmap, set curve data, set text. No heavy work in main thread.
- Pyqtgraph is used for charts (efficient for rolling updates). If FPS drops, reduce `DiagnosticsPanel.MAX_POINTS` or chart update frequency (e.g. every 2nd frame).

---

## Dependencies

- PyQt5 or PySide6
- numpy (for FrameData and pipeline)
- pyqtgraph (optional; for DiagnosticsPanel charts)
- Project: `carla_mpc_classical` (config, pipeline, carla_input_output, state) and CARLA PythonAPI (carla)
