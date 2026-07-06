"""Integration tests for the complete CARLA MPC system."""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestSystemIntegration:
    """Integration tests for system components working together."""

    def setup_method(self):
        """Setup test environment."""
        self.config_mock = Mock()
        self.config_mock.CAM_W = 640
        self.config_mock.CAM_H = 480
        self.config_mock.CAM_FOV_DEG = 90.0
        self.config_mock.PANEL_W = 800
        self.config_mock.PANEL_H = 600

    @patch('managers.carla_manager.get_config')
    def test_camera_to_display_pipeline(self, mock_get_config):
        """Test complete pipeline from camera callback to display rendering."""
        mock_get_config.return_value = self.config_mock

        # Setup components
        from managers.carla_manager import CarlaManager
        from managers.display_manager import DisplayManager

        carla_manager = CarlaManager()
        carla_manager.config = self.config_mock
        carla_manager.frame_queue = Mock()

        display_manager = DisplayManager(enable_gui=False)
        display_manager.config = self.config_mock

        # Create test image data
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock CARLA image
        mock_image = Mock()
        mock_image.raw_data = np.random.bytes(640 * 480 * 4)
        mock_image.width = 640
        mock_image.height = 480
        mock_image.timestamp = 123456789.0

        # Process through camera callback
        captured_frame = None
        def capture_callback(frame):
            nonlocal captured_frame
            captured_frame = frame

        carla_manager.camera_callback(mock_image, capture_callback)

        # Verify frame was created correctly
        assert captured_frame is not None
        assert captured_frame.width == 640
        assert captured_frame.height == 480
        assert captured_frame.fov == 90.0
        assert captured_frame.rgb.shape == (480, 640, 3)

    @patch('pipeline.get_config')
    def test_perception_pipeline_integration(self, mock_get_config):
        """Test perception pipeline integration with trajectory processing."""
        mock_get_config.return_value = self.config_mock

        with patch('pipeline.RoadPerception') as mock_perception, \
             patch('pipeline.LaneTemporalSmoother') as mock_smoother, \
             patch('pipeline.LaneMPC') as mock_mpc, \
             patch('pipeline.SafetyOverride') as mock_safety:

            from pipeline import LKAPipeline

            # Setup mocks
            mock_perception.return_value.process.return_value = (
                0.5, 0.1, 0.02, np.zeros((480, 640), dtype=np.uint8), 0.8, []
            )
            mock_smoother.return_value.update.return_value = [0.5, 0.1, 0.02]
            mock_mpc.return_value.solve.return_value = (0.1, 0.2, 0.0)
            mock_safety.return_value.apply.return_value = (0.1, 0.2, 0.0)

            pipeline = LKAPipeline(
                "dummy_model.pth",
                Mock(),  # device
                30.0,    # target speed
                use_trajectory_pipeline=False
            )

            # Test process method
            rgb_input = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = pipeline.process(rgb_input)

            # Verify pipeline execution
            assert len(result) == 4  # (steer, throttle, brake, frame_state)
            assert all(isinstance(x, (int, float)) for x in result[:3])

    def test_stuck_recovery_with_control_manager(self):
        """Test stuck recovery integration with control manager."""
        from safety.stuck_recovery import StuckRecovery

        recovery = StuckRecovery()

        # Simulate stuck condition
        recovery._n = recovery.STUCK_THRESHOLD + 10
        recovery._phase = "brake"

        # Test should_recover integration
        mock_transform = Mock()
        should_recover = recovery.should_recover(0.1, mock_transform, {})
        assert should_recover is True

        # Test recovery control generation
        control = recovery.update(0.1, 0.5)
        assert control is not None
        assert len(control) == 4  # (steer, throttle, brake, reverse)

    @patch('managers.display_manager.get_config')
    @patch('managers.display_manager.cv2')
    @patch('managers.display_manager.pygame')
    def test_display_manager_with_perception_data(self, mock_pygame, mock_cv2, mock_get_config):
        """Test display manager rendering with perception data."""
        mock_get_config.return_value = self.config_mock

        from managers.display_manager import DisplayManager
        from utils.type_hints import PerceptionResult

        display_manager = DisplayManager(enable_gui=False)
        display_manager.config = self.config_mock
        display_manager.dashboard_initialized = True

        # Mock pygame components
        mock_pygame.event.get.return_value = []
        display_manager.screen = Mock()
        display_manager.font = mock_pygame.font.Font.return_value

        # Create test perception data
        perception = PerceptionResult(
            cte=0.5,
            heading_error=0.1,
            curvature=0.02,
            confidence=0.8,
            geometry_valid=True,
            lane_overlay=None,
            bev_window_vis=None,
            mask_vis=None
        )

        # Test rendering (should not raise exceptions)
        with patch('managers.display_manager.logger'):
            result = display_manager._update_gui(None, perception, None, None)
            assert result is True

    def test_error_propagation_through_system(self):
        """Test that errors are properly handled and don't crash the system."""
        from managers.carla_manager import CarlaManager
        from managers.display_manager import DisplayManager

        # Test camera callback with invalid image data
        carla_manager = CarlaManager()
        carla_manager.config = self.config_mock
        carla_manager.frame_queue = Mock()

        # Mock image with invalid data
        mock_image = Mock()
        mock_image.raw_data = b"invalid_data"  # Too short
        mock_image.width = 640
        mock_image.height = 480
        mock_image.timestamp = 123456789.0

        # Should handle error gracefully
        callback_mock = Mock()
        try:
            carla_manager.camera_callback(mock_image, callback_mock)
        except Exception as e:
            # If exception occurs, it should be logged, not crash
            assert isinstance(e, (ValueError, TypeError))

        # Test display manager with invalid confidence
        display_manager = DisplayManager(enable_gui=False)
        display_manager.config = self.config_mock

        # Invalid confidence should return gray color
        color = display_manager._get_confidence_color("invalid")
        assert color == (200, 200, 200)


class TestConfigIntegration:
    """Test configuration integration across components."""

    def test_config_consistency(self):
        """Test that configuration values are consistent across components."""
        from config_clean import get_config

        config = get_config()

        # Verify required config keys exist
        required_keys = ['CAM_W', 'CAM_H', 'CAM_FOV_DEG', 'PANEL_W', 'PANEL_H']
        for key in required_keys:
            assert hasattr(config, key)
            assert getattr(config, key) is not None
            assert isinstance(getattr(config, key), (int, float))

        # Verify reasonable value ranges
        assert 100 <= config.CAM_W <= 2000
        assert 100 <= config.CAM_H <= 2000
        assert 30 <= config.CAM_FOV_DEG <= 180
        assert 200 <= config.PANEL_W <= 2000
        assert 200 <= config.PANEL_H <= 2000
