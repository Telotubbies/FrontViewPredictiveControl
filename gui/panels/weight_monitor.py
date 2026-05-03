from typing import Dict

from ..qt_compat import QFrame, QGridLayout, QLabel
from ..models import FrameData


class AdaptiveWeightMonitor(QFrame):
    """Shows current dynamic MPC weights; highlights when confidence is low."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        layout = QGridLayout(self)
        self._labels: Dict[str, QLabel] = {}
        for i, key in enumerate(["w_e", "w_psi", "w_delta", "w_jerk"]):
            lbl = QLabel(f"{key}: —")
            lbl.setStyleSheet("color: #b0b0b0;")
            layout.addWidget(QLabel(key), i, 0)
            layout.addWidget(lbl, i, 1)
            self._labels[key] = lbl
        self._highlight_style = "color: #ff9800; font-weight: bold;"
        self._normal_style = "color: #b0b0b0;"

    def update_weights(self, weights: Dict[str, float], confidence_low: bool) -> None:
        for k, v in weights.items():
            if k in self._labels:
                self._labels[k].setText(f"{v:.3f}")
                self._labels[k].setStyleSheet(self._highlight_style if confidence_low else self._normal_style)
