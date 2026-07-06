# Panels for ADAS dashboard
from ..qt_compatibility import QT_BINDING

if QT_BINDING is None:
    raise ImportError("PyQt5 or PySide6 is required for GUI panels")

from .control_panel import ControlPanel
from .visualization_panel import VisualizationPanel
from .diagnostics_panel import DiagnosticsPanel
from .weight_monitor import AdaptiveWeightMonitor
from .logging_panel import LoggingPanel

__all__ = [
    "ControlPanel",
    "VisualizationPanel",
    "DiagnosticsPanel",
    "AdaptiveWeightMonitor",
    "LoggingPanel",
]
