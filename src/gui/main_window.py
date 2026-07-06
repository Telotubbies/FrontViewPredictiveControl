# Main window: layout, status bar, signal wiring (no CARLA/pipeline imports)
import time
from pathlib import Path
from typing import Optional

from .qt_compat import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QStatusBar,
    QSplitter,
    Qt,
    QThread,
)
from .theme import DARK_THEME
from .models import FrameData, compute_adaptive_weights
from .worker import CarlaWorker
from .panels import (
    ControlPanel,
    VisualizationPanel,
    DiagnosticsPanel,
    AdaptiveWeightMonitor,
    LoggingPanel,
)

# Project root for default model path (when main_window is used from gui package)
_GUI_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _GUI_DIR.parent


class MainWindow(QMainWindow):
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("CARLA ADAS — UNet + Waypoint Fusion + MPC")
        self.setMinimumSize(1280, 720)
        self.setStyleSheet(DARK_THEME)

        if model_path is None:
            model_path = str(_PROJECT_ROOT / "model" / "lane_unet_final.pth")
        self._model_path = model_path

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_connection = QLabel("CARLA: —")
        self._status_town = QLabel("Town: —")
        self._status_fps = QLabel("FPS: —")
        self._status_speed = QLabel("Speed: — km/h")
        self._status.addPermanentWidget(self._status_connection)
        self._status.addPermanentWidget(self._status_town)
        self._status.addPermanentWidget(self._status_fps)
        self._status.addPermanentWidget(self._status_speed)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self._control = ControlPanel()
        self._visual = VisualizationPanel()
        self._diagnostics = DiagnosticsPanel()
        self._weight_monitor = AdaptiveWeightMonitor()

        right_col = QVBoxLayout()
        right_col.addWidget(self._diagnostics)
        right_col.addWidget(self._weight_monitor)
        right_w = QWidget()
        right_w.setLayout(right_col)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._control)
        splitter.addWidget(self._visual)
        splitter.addWidget(right_w)
        splitter.setSizes([220, 680, 380])
        main_layout.addWidget(splitter)

        self._log_panel = LoggingPanel()
        main_layout.addWidget(self._log_panel)

        # Worker thread
        self._worker = CarlaWorker(self._model_path)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)

        self._worker.frame_ready.connect(self._on_frame)
        self._worker.fps_updated.connect(self._on_fps)
        self._worker.log_message.connect(self._log_panel.append)
        self._worker.connection_changed.connect(self._on_connection)
        self._worker.emergency_stop.connect(lambda: None)

        self._control.start_clicked.connect(self._on_start)
        self._control.stop_clicked.connect(self._on_stop)
        self._control.emergency_clicked.connect(self._worker.request_emergency_brake)
        self._control.town_changed.connect(self._worker.set_town)
        self._control.fusion_toggled.connect(self._on_fusion_toggle)
        self._control.unet_weight_changed.connect(self._on_unet_weight)
        self._control.mpc_horizon_changed.connect(self._worker.set_mpc_horizon)

        self._start_time = 0.0

    def _on_connection(self, connected: bool, town: str) -> None:
        if connected:
            self._status_connection.setText("CARLA: ● Connected")
            self._status_connection.setStyleSheet("color: #4caf50;")
            self._status_town.setText(f"Town: {town}")
        else:
            self._status_connection.setText("CARLA: ● Disconnected")
            self._status_connection.setStyleSheet("color: #f44336;")
            self._status_town.setText("Town: —")

    def _on_fps(self, fps: float) -> None:
        self._status_fps.setText(f"FPS: {fps:.1f}")

    def _on_frame(self, fd: FrameData) -> None:
        self._visual.update_frame(fd)
        t = time.time() - self._start_time
        self._diagnostics.update_telemetry(fd, t)
        weights = compute_adaptive_weights(fd.lane_conf, fd.curvature)
        self._weight_monitor.update_weights(weights, fd.lane_conf < 0.5)
        self._status_speed.setText(f"Speed: {fd.speed_kmh:.1f} km/h")

    def _on_start(self) -> None:
        self._control.set_running(True)
        self._start_time = time.time()
        self._worker.set_town(self._control._town_combo.currentText())
        self._worker.set_fusion_enabled(self._control._fusion_cb.isChecked())
        self._worker.set_unet_weight(self._control._unet_slider.value() / 100.0)
        self._worker.set_mpc_horizon(self._control._horizon_slider.value())
        self._on_unet_weight(self._control._unet_slider.value() / 100.0)
        self._on_fusion_toggle(self._control._fusion_cb.isChecked())
        self._worker.start_sim()
        self._thread.start()

    def _on_stop(self) -> None:
        self._worker.stop_sim()
        self._control.set_running(False)

    def _on_unet_weight(self, value: float) -> None:
        self._worker.set_unet_weight(value)
        try:
            import config as c
            c.UNET_WEIGHT_BASE = value
            if self._control._fusion_cb.isChecked():
                c.WP_DIRECTION_WEIGHT = 1.0 - value
        except ImportError:
            pass

    def _on_fusion_toggle(self, checked: bool) -> None:
        self._worker.set_fusion_enabled(checked)
        try:
            import config as c
            if not checked:
                c.WP_DIRECTION_WEIGHT = 0.0
                c.UNET_WEIGHT_BASE = 1.0
            else:
                c.UNET_WEIGHT_BASE = self._control._unet_slider.value() / 100.0
                c.WP_DIRECTION_WEIGHT = 1.0 - c.UNET_WEIGHT_BASE
        except ImportError:
            pass
