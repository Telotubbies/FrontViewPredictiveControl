"""Unit tests for DisplayManager image rendering and color handling."""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from managers.display_manager import DisplayManager
from utils.type_hints import CameraFrame, PerceptionResult, ControlCommand


class TestDisplayManager:
    """Test cases for DisplayManager class."""

    def setup_method(self):
        """Setup test environment."""
        with patch('managers.display_manager.get_config') as mock_config:
            mock_config.return_value.PANEL_W = 800
            mock_config.return_value.PANEL_H = 600
            mock_config.return_value.PANEL_H = 600
            self.manager = DisplayManager(enable_gui=False)
            self.manager.config = mock_config.return_value

    def test_get_confidence_color_valid_inputs(self):
        """Test confidence color generation with valid inputs."""
        # High confidence (> 0.7) should be green
        color = self.manager._get_confidence_color(0.8)
        assert color == (0, 255, 0)

        # Medium confidence (0.4-0.7) should be yellow
        color = self.manager._get_confidence_color(0.5)
        assert color == (255, 255, 0)

        # Low confidence (< 0.4) should be red
        color = self.manager._get_confidence_color(0.2)
        assert color == (255, 0, 0)

    def test_get_confidence_color_edge_cases(self):
        """Test confidence color with edge case values."""
        # Boundary values
        color = self.manager._get_confidence_color(0.7)
        assert color == (0, 255, 0)  # Exactly 0.7 should be green

        color = self.manager._get_confidence_color(0.4)
        assert color == (255, 255, 0)  # Exactly 0.4 should be yellow

        # Out of range values
        color = self.manager._get_confidence_color(1.5)
        assert color == (0, 255, 0)  # Above 1.0 should still be green

        color = self.manager._get_confidence_color(-0.5)
        assert color == (255, 0, 0)  # Below 0.0 should be red

    def test_get_confidence_color_invalid_inputs(self):
        """Test confidence color with invalid inputs returns gray."""
        # None input
        color = self.manager._get_confidence_color(None)
        assert color == (200, 200, 200)  # Gray for invalid

        # String input
        color = self.manager._get_confidence_color("invalid")
        assert color == (200, 200, 200)  # Gray for invalid

        # NaN input
        color = self.manager._get_confidence_color(float('nan'))
        assert color == (200, 200, 200)  # Gray for invalid

    @patch('managers.display_manager.cv2')
    @patch('managers.display_manager.pygame')
    def test_render_camera_frame_uses_cv2_resize(self, mock_pygame, mock_cv2):
        """Test that camera frame rendering uses cv2.resize instead of np.resize."""
        # Setup mocks
        mock_cv2.resize.return_value = np.random.randint(0, 255, (300, 400, 3))
        mock_pygame.surfarray.make_surface.return_value = Mock()

        # Create test frame
        test_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = CameraFrame(
            rgb=test_rgb,
            timestamp=123456789.0,
            width=640,
            height=480,
            fov=90.0
        )

        # Mock screen
        self.manager.screen = Mock()

        # Call render method
        self.manager._render_camera_frame(frame)

        # Verify cv2.resize was called with correct parameters
        mock_cv2.resize.assert_called_once()
        call_args = mock_cv2.resize.call_args
        assert call_args[0][0] is test_rgb  # Original RGB array
        assert call_args[0][1] == (400, 300)  # Target dimensions (PANEL_W//2, PANEL_H//2)

        # Verify np.resize was NOT used
        assert not hasattr(np.resize, 'called')  # np.resize should not be called

    @patch('managers.display_manager.cv2')
    @patch('managers.display_manager.pygame')
    def test_render_camera_frame_aspect_ratio(self, mock_pygame, mock_cv2):
        """Test that camera frame maintains proper aspect ratio."""
        mock_cv2.resize.return_value = np.random.randint(0, 255, (300, 400, 3))
        mock_pygame.surfarray.make_surface.return_value = Mock()

        # Test with different frame sizes
        test_cases = [
            (480, 640),  # Standard
            (720, 1280), # HD
            (360, 640),  # Tall
        ]

        for h, w in test_cases:
            test_rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            frame = CameraFrame(
                rgb=test_rgb,
                timestamp=123456789.0,
                width=w,
                height=h,
                fov=90.0
            )

            self.manager._render_camera_frame(frame)

            # Verify resize was called with consistent target dimensions
            mock_cv2.resize.assert_called_with(test_rgb, (400, 300))

    def test_render_text_color_validation(self):
        """Test text rendering with various color inputs."""
        with patch('managers.display_manager.pygame') as mock_pygame:
            mock_pygame.font.Font.return_value.render.return_value = Mock()
            self.manager.screen = Mock()
            self.manager.font = mock_pygame.font.Font.return_value

            # Valid color tuple
            self.manager._render_text("test", (10, 10), (255, 0, 0))
            mock_pygame.font.Font.return_value.render.assert_called_with("test", True, (255, 0, 0))

            # Default color
            self.manager._render_text("test", (10, 10))
            mock_pygame.font.Font.return_value.render.assert_called_with("test", True, (200, 200, 200))

    def test_render_text_exception_handling(self):
        """Test text rendering exception handling."""
        with patch('managers.display_manager.pygame') as mock_pygame:
            # Mock font.render to raise exception
            mock_pygame.font.Font.return_value.render.side_effect = ValueError("Invalid color")

            self.manager.screen = Mock()
            self.manager.font = mock_pygame.font.Font.return_value

            # Should not raise exception, should log error instead
            with patch('managers.display_manager.logger') as mock_logger:
                self.manager._render_text("test", (10, 10), (255, 0, 0))
                mock_logger.error.assert_called_once()

    def test_update_gui_exception_handling(self):
        """Test GUI update exception handling."""
        with patch('managers.display_manager.pygame') as mock_pygame:
            # Mock pygame to raise exception
            mock_pygame.event.get.side_effect = RuntimeError("Pygame error")

            self.manager.screen = Mock()
            self.manager.dashboard_initialized = True

            # Should handle exception gracefully
            with patch('managers.display_manager.logger') as mock_logger:
                result = self.manager._update_gui(None, None, None, None)
                assert result is True  # Should return True even on error
                mock_logger.error.assert_called_once()
