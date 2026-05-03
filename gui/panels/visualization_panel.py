import numpy as np
from ..qt_compat import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QImage, QPixmap, Qt
from ..models import FrameData


class VisualizationPanel(QFrame):
    """Center panel: live camera feed with overlays and key metrics."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        self._img_label = QLabel()
        self._img_label.setMinimumSize(640, 480)
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setStyleSheet("background-color: #1a1a1a; border-radius: 4px;")
        layout.addWidget(self._img_label)
        metrics = QHBoxLayout()
        self._cte_label = QLabel("CTE: — m")
        self._cte_label.setStyleSheet("color: #00bcd4; font-weight: bold;")
        self._head_label = QLabel("Heading: — rad")
        self._curv_label = QLabel("Curvature: —")
        self._mode_label = QLabel("Mode: —")
        for w in (self._cte_label, self._head_label, self._curv_label, self._mode_label):
            w.setStyleSheet("color: #b0b0b0;")
        metrics.addWidget(self._cte_label)
        metrics.addWidget(self._head_label)
        metrics.addWidget(self._curv_label)
        metrics.addWidget(self._mode_label)
        layout.addLayout(metrics)

    def update_frame(self, fd: FrameData) -> None:
        rgb = fd.rgb.copy()
        if fd.lane_overlay is not None and fd.lane_overlay.size > 0:
            try:
                overlay = fd.lane_overlay
                if overlay.shape[:2] == rgb.shape[:2]:
                    rgb = (rgb.astype(np.float32) * 0.6 + overlay.astype(np.float32) * 0.4).clip(0, 255).astype(np.uint8)
            except Exception:
                pass
        h, w = rgb.shape[:2]
        if len(rgb.shape) == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        if rgb.shape[2] == 3:
            bytes_per_line = w * 3
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            qimg = QImage(rgb.data, w, h, w * 4, QImage.Format_RGBA8888)
        self._img_label.setPixmap(QPixmap.fromImage(qimg))
        self._cte_label.setText(f"CTE: {fd.cte_m:+.3f} m")
        self._head_label.setText(f"Heading: {fd.heading_rad:+.3f} rad")
        self._curv_label.setText(f"Curvature: {fd.curvature:+.4f}")
        self._mode_label.setText(f"Mode: {fd.mode}")
