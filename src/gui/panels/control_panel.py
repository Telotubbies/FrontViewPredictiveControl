from ..qt_compatibility import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QCheckBox,
    pyqtSignal,
    Qt,
)


class ControlPanel(QFrame):
    """Left panel: mode, start/stop/emergency, town, fusion toggle, sliders."""
    mode_changed = pyqtSignal(str)
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    emergency_clicked = pyqtSignal()
    town_changed = pyqtSignal(str)
    fusion_toggled = pyqtSignal(bool)
    unet_weight_changed = pyqtSignal(float)
    mpc_horizon_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Mode"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Classical", "UNet"])
        self._mode_combo.currentTextChanged.connect(self.mode_changed.emit)
        layout.addWidget(self._mode_combo)

        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.setObjectName("startBtn")
        self._start_btn.clicked.connect(self.start_clicked.emit)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        btn_layout.addWidget(self._start_btn)
        btn_layout.addWidget(self._stop_btn)
        layout.addLayout(btn_layout)

        self._emergency_btn = QPushButton("EMERGENCY BRAKE")
        self._emergency_btn.setObjectName("emergencyBtn")
        self._emergency_btn.clicked.connect(self.emergency_clicked.emit)
        layout.addWidget(self._emergency_btn)

        layout.addWidget(QLabel("Town"))
        self._town_combo = QComboBox()
        self._town_combo.addItems(["Town01", "Town02", "Town03", "Town04", "Town05"])
        self._town_combo.currentTextChanged.connect(self.town_changed.emit)
        layout.addWidget(self._town_combo)

        self._fusion_cb = QCheckBox("Waypoint fusion on")
        self._fusion_cb.setChecked(True)
        self._fusion_cb.toggled.connect(self.fusion_toggled.emit)
        layout.addWidget(self._fusion_cb)

        layout.addWidget(QLabel("UNet weight (0–1)"))
        self._unet_slider = QSlider(Qt.Horizontal)
        self._unet_slider.setRange(0, 100)
        self._unet_slider.setValue(50)
        self._unet_slider.valueChanged.connect(lambda v: self.unet_weight_changed.emit(v / 100.0))
        layout.addWidget(self._unet_slider)

        layout.addWidget(QLabel("MPC horizon"))
        self._horizon_slider = QSlider(Qt.Horizontal)
        self._horizon_slider.setRange(5, 20)
        self._horizon_slider.setValue(10)
        self._horizon_slider.valueChanged.connect(self.mpc_horizon_changed.emit)
        layout.addWidget(self._horizon_slider)

        layout.addStretch()

    def set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
