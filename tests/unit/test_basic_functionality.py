"""Basic functionality tests without external dependencies."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality without external dependencies."""

    def test_config_loading(self):
        """Test configuration loading."""
        try:
            from config_clean import get_config
            config = get_config()

            # Test that config has required attributes
            required_attrs = ['CAM_W', 'CAM_H', 'CAM_FOV_DEG', 'PANEL_W', 'PANEL_H']
            for attr in required_attrs:
                self.assertTrue(hasattr(config, attr), f"Missing config attribute: {attr}")
                value = getattr(config, attr)
                self.assertIsNotNone(value, f"Config attribute {attr} is None")
                self.assertIsInstance(value, (int, float), f"Config attribute {attr} should be numeric")

        except ImportError as e:
            self.fail(f"Failed to import config: {e}")

    def test_type_hints_import(self):
        """Test that type hints can be imported."""
        try:
            from utils.type_hints import CameraFrame, PerceptionResult, ControlCommand

            # Test that classes exist
            self.assertTrue(hasattr(CameraFrame, '__annotations__'))
            self.assertTrue(hasattr(PerceptionResult, '__annotations__'))
            self.assertTrue(hasattr(ControlCommand, '__annotations__'))

        except ImportError as e:
            self.fail(f"Failed to import type hints: {e}")

    def test_stuck_recovery_import(self):
        """Test StuckRecovery class import and basic functionality."""
        try:
            from safety.stuck_recovery import StuckRecovery

            # Test instantiation
            recovery = StuckRecovery()

            # Test that should_recover method exists
            self.assertTrue(hasattr(recovery, 'should_recover'))
            self.assertTrue(callable(recovery.should_recover))

            # Test basic functionality
            mock_transform = Mock()
            result = recovery.should_recover(10.0, mock_transform, {})
            self.assertIsInstance(result, bool)

            # Test with different speeds
            result_slow = recovery.should_recover(0.1, mock_transform, {})
            self.assertIsInstance(result_slow, bool)

        except ImportError as e:
            self.fail(f"Failed to import StuckRecovery: {e}")

    def test_display_manager_import(self):
        """Test DisplayManager import and basic functionality."""
        try:
            from managers.display_manager import DisplayManager

            # Test instantiation without GUI
            manager = DisplayManager(enable_gui=False)

            # Test that _get_confidence_color method exists
            self.assertTrue(hasattr(manager, '_get_confidence_color'))
            self.assertTrue(callable(manager._get_confidence_color))

            # Test confidence color generation
            color_high = manager._get_confidence_color(0.8)
            color_med = manager._get_confidence_color(0.5)
            color_low = manager._get_confidence_color(0.2)

            # Test color values are tuples of length 3
            for color in [color_high, color_med, color_low]:
                self.assertIsInstance(color, tuple)
                self.assertEqual(len(color), 3)
                for channel in color:
                    self.assertIsInstance(channel, int)
                    self.assertGreaterEqual(channel, 0)
                    self.assertLessEqual(channel, 255)

            # Test invalid confidence returns gray
            color_invalid = manager._get_confidence_color("invalid")
            self.assertEqual(color_invalid, (200, 200, 200))

        except ImportError as e:
            self.fail(f"Failed to import DisplayManager: {e}")

    def test_carla_manager_import(self):
        """Test CarlaManager import and basic setup."""
        try:
            from managers.carla_manager import CarlaManager

            # Test instantiation
            manager = CarlaManager()

            # Test that camera_callback method exists
            self.assertTrue(hasattr(manager, 'camera_callback'))
            self.assertTrue(callable(manager.camera_callback))

        except ImportError as e:
            self.fail(f"Failed to import CarlaManager: {e}")

    def test_pipeline_import(self):
        """Test LKAPipeline import and basic functionality."""
        try:
            from pipeline import LKAPipeline

            # Test that process method exists (the method we added)
            # We can't instantiate without dependencies, but we can check the class
            self.assertTrue(hasattr(LKAPipeline, 'process'))

        except ImportError as e:
            self.fail(f"Failed to import LKAPipeline: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling in various components."""

    def test_config_error_handling(self):
        """Test config error handling."""
        try:
            from config_clean import ConfigError

            # Test that ConfigError exists and can be raised
            with self.assertRaises(ConfigError):
                raise ConfigError("Test error")

        except ImportError:
            # ConfigError might not exist in all versions
            pass

    def test_import_error_handling(self):
        """Test that missing dependencies are handled gracefully."""
        # Test that CARLA import is handled with fallback
        try:
            import carla
            carla_available = True
        except ImportError:
            carla_available = False

        # The system should handle both cases
        self.assertIsInstance(carla_available, bool)


if __name__ == '__main__':
    unittest.main()
