"""
Regression tests for bugs fixed during subagent sprint.

Covers:
1. UXColors missing aliases (TEXT, SUCCESS, WARNING, ERROR, BACKGROUND)
2. config_clean module not found in managers
3. Qt fallback when neither PyQt5 nor PySide6 is installed
4. ego_lane_tracker graceful degradation when lane_clustering is missing
5. lane_trajectory classical detector import path fix
6. BOM-free Python files in src/gui/panels/
"""

import unittest
import importlib
import sys
from pathlib import Path


class TestUXColorsAliases(unittest.TestCase):
    """Regression: UXColors was missing TEXT, SUCCESS, WARNING, ERROR, BACKGROUND."""

    def test_uxcolors_text_alias(self):
        from config import UXColors
        self.assertEqual(UXColors.TEXT, UXColors.TEXT_NORMAL)

    def test_uxcolors_success_alias(self):
        from config import UXColors
        self.assertEqual(UXColors.SUCCESS, UXColors.GREEN_VALID)

    def test_uxcolors_warning_defined(self):
        from config import UXColors
        self.assertIsInstance(UXColors.WARNING, tuple)
        self.assertEqual(len(UXColors.WARNING), 3)

    def test_uxcolors_error_alias(self):
        from config import UXColors
        self.assertEqual(UXColors.ERROR, UXColors.RED_INVALID)

    def test_uxcolors_background_defined(self):
        from config import UXColors
        self.assertIsInstance(UXColors.BACKGROUND, tuple)
        self.assertEqual(len(UXColors.BACKGROUND), 3)

    def test_dashboard_imports(self):
        """Dashboard should import without AttributeError."""
        from gui.dashboard import Dashboard
        self.assertTrue(hasattr(Dashboard, 'render_text'))


class TestConfigCleanReplaced(unittest.TestCase):
    """Regression: managers imported from nonexistent config_clean module."""

    def test_carla_manager_imports(self):
        from managers.carla_manager import CarlaManager, get_config
        cfg = get_config()
        self.assertIsNotNone(cfg)

    def test_perception_manager_imports(self):
        from managers.perception_manager import PerceptionManager, get_config
        cfg = get_config()
        self.assertIsNotNone(cfg)

    def test_display_manager_imports(self):
        from managers.display_manager import DisplayManager, get_config
        cfg = get_config()
        self.assertIsNotNone(cfg)


class TestQtFallback(unittest.TestCase):
    """Regression: qt_compatibility should not crash when Qt is not installed."""

    def test_qt_binding_value(self):
        from gui.qt_compatibility import QT_BINDING
        self.assertIn(QT_BINDING, ("PyQt5", "PySide6", None))

    def test_qt_objects_exist(self):
        """Even without Qt, dummy classes should be defined."""
        from gui.qt_compatibility import QObject, QThread, pyqtSignal
        self.assertIsNotNone(QObject)
        self.assertIsNotNone(QThread)
        self.assertIsNotNone(pyqtSignal)

    def test_worker_imports_without_qt(self):
        """worker.py should import even without PyQt5/PySide6."""
        from gui.worker import CarlaWorker
        self.assertIsNotNone(CarlaWorker)


class TestEgoLaneTrackerGracefulDegradation(unittest.TestCase):
    """Regression: ego_lane_tracker should handle missing lane_clustering."""

    def test_ego_lane_tracker_imports(self):
        from perception.ego_lane_tracker import EgoLaneTracker, EgoLaneResult
        self.assertIsNotNone(EgoLaneTracker)
        self.assertIsNotNone(EgoLaneResult)

    def test_ego_lane_tracker_instantiation(self):
        from perception.ego_lane_tracker import EgoLaneTracker
        tracker = EgoLaneTracker()
        tracker.reset()
        # update() should return EgoLaneResult even if lane_clustering is missing
        import numpy as np
        bev = np.zeros((320, 450), dtype=np.uint8)
        result = tracker.update(bev)
        self.assertIsNotNone(result)
        self.assertFalse(result.tracked)


class TestClassicalDetectorImport(unittest.TestCase):
    """Regression: lane_trajectory should import from classical.detector, not classical_lane_detector."""

    def test_classical_detector_module_exists(self):
        from perception.classical.detector import ClassicalLane
        self.assertIsNotNone(ClassicalLane)

    def test_lane_trajectory_classical_import(self):
        """LaneTrajectoryPipeline should be importable (classical path is lazy)."""
        from perception.lane_trajectory import LaneTrajectoryPipeline
        self.assertIsNotNone(LaneTrajectoryPipeline)


class TestNoBOMInPanels(unittest.TestCase):
    """Regression: panels/__init__.py should not have BOM."""

    def test_no_bom_in_panels_init(self):
        panels_init = Path(__file__).resolve().parent.parent.parent / "src" / "gui" / "panels" / "__init__.py"
        if not panels_init.exists():
            self.skipTest("panels/__init__.py not found")
        with open(panels_init, "rb") as f:
            first_bytes = f.read(3)
        self.assertNotEqual(first_bytes, b"\xef\xbb\xbf", "BOM found in panels/__init__.py")


if __name__ == "__main__":
    unittest.main()
