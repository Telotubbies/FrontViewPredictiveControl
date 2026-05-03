#!/usr/bin/env python3
"""
CARLA ADAS Dashboard — Entry point. Sets up path and launches MainWindow.

Run from CARLA_0.9.16 or carla_mpc_classical:
  python carla_mpc_classical/gui/adas_dashboard.py
  python -m carla_mpc_classical.gui.adas_dashboard [model_path]
"""
import sys
from pathlib import Path

# Project root and CARLA API on path before any project or carla imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_CARLA_API = _PROJECT_ROOT.parent / "PythonAPI"
if _CARLA_API.exists() and str(_CARLA_API) not in sys.path:
    sys.path.insert(0, str(_CARLA_API))

# Import from gui package (works when run as script or as -m carla_mpc_classical.gui.adas_dashboard)
try:
    from .qt_compat import QApplication, QFont
    from .theme import FONT_FAMILY
    from .main_window import MainWindow
except ImportError:
    from gui.qt_compat import QApplication, QFont
    from gui.theme import FONT_FAMILY
    from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont(FONT_FAMILY, 9))
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    win = MainWindow(model_path)
    win.show()
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())


if __name__ == "__main__":
    main()
