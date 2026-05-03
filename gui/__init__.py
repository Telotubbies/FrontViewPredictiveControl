# GUI package: ADAS dashboard (PyQt5/PySide6)
# Lightweight __init__; use "from gui.main_window import MainWindow" when needed.
from .models import FrameData, compute_adaptive_weights

__all__ = ["FrameData", "compute_adaptive_weights"]
