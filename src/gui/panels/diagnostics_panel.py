from typing import Dict, List

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

from ..qt_compat import QFrame, QVBoxLayout, QLabel
from ..models import FrameData


class DiagnosticsPanel(QFrame):
    """Right panel: rolling charts for steer, throttle, brake, CTE, confidence."""
    MAX_POINTS = 300  # ~30 fps * 10 s

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        self._data: Dict[str, List[float]] = {
            "steer": [], "throttle": [], "brake": [], "cte": [], "confidence": [],
        }
        self._t: List[float] = []

        if HAS_PYQTGRAPH:
            pg.setConfigOptions(antialias=True, background="#1a1a1a", foreground="#b0b0b0")
            self._plot = pg.PlotWidget(title="Steer / Throttle / Brake")
            self._plot.setLabel("left", "value")
            self._plot.setLabel("bottom", "time (s)")
            self._plot.addLegend()
            self._curve_steer = self._plot.plot(pen=pg.mkPen("#00bcd4", width=2), name="Steer")
            self._curve_throttle = self._plot.plot(pen=pg.mkPen("#4caf50", width=2), name="Throttle")
            self._curve_brake = self._plot.plot(pen=pg.mkPen("#f44336", width=2), name="Brake")
            layout.addWidget(self._plot)
            self._plot2 = pg.PlotWidget(title="CTE / Confidence")
            self._curve_cte = self._plot2.plot(pen=pg.mkPen("#ff9800", width=2), name="CTE")
            self._curve_conf = self._plot2.plot(pen=pg.mkPen("#9c27b0", width=2), name="Confidence")
            layout.addWidget(self._plot2)
        else:
            self._plot = None
            self._plot2 = None
            layout.addWidget(QLabel("Install pyqtgraph for real-time charts:\n  pip install pyqtgraph"))

    def update_telemetry(self, fd: FrameData, t: float) -> None:
        self._t.append(t)
        self._data["steer"].append(fd.steer)
        self._data["throttle"].append(fd.throttle)
        self._data["brake"].append(fd.brake)
        self._data["cte"].append(fd.cte_m)
        self._data["confidence"].append(fd.lane_conf)
        for key in list(self._data.keys()):
            if len(self._data[key]) > self.MAX_POINTS:
                self._data[key].pop(0)
        if len(self._t) > self.MAX_POINTS:
            self._t.pop(0)
        if HAS_PYQTGRAPH and self._plot is not None and len(self._t) > 1:
            self._curve_steer.setData(self._t, self._data["steer"])
            self._curve_throttle.setData(self._t, self._data["throttle"])
            self._curve_brake.setData(self._t, self._data["brake"])
            self._curve_cte.setData(self._t, self._data["cte"])
            self._curve_conf.setData(self._t, self._data["confidence"])
