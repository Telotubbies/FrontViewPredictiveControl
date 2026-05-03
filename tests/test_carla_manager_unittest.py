"""Unit tests for CarlaManager using unittest framework."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from managers.carla_manager import CarlaManager
from utils.type_hints import CameraFrame


class TestCarlaManager(unittest.TestCase):
    """Test cases for CarlaManager class."""
    
    def setUp(self):
        """Setup test environment."""
        self.config_mock = Mock()
        self.config_mock.CAM_W = 640
        self.config_mock.CAM_H = 480
        self.config_mock.CAM_FOV_DEG = 90.0
        
        with patch('managers.carla_manager.get_config', return_value=self.config_mock):
            self.manager = CarlaManager()
            self.manager.config = self.config_mock
            self.manager.frame_queue = Mock()
            self.manager.frame_queue.put_nowait = Mock()
            self.manager.frame_queue.get_nowait = Mock()
            self.manager.frame_queue.Empty = Exception
    
    def test_camera_callback_image_conversion(self):
        """Test camera callback converts CARLA image to CameraFrame correctly."""
        # Mock CARLA image
        mock_image = Mock()
        mock_image.raw_data = np.random.bytes(640 * 480 * 4)  # RGBA data
        mock_image.width = 640
        mock_image.height = 480
        mock_image.timestamp = 123456789.0
        
        # Mock callback
        callback_mock = Mock()
        
        # Call camera callback
        self.manager.camera_callback(mock_image, callback_mock)
        
        # Verify callback was called with CameraFrame
        callback_mock.assert_called_once()
        frame_arg = callback_mock.call_args[0][0]
        self.assertIsInstance(frame_arg, CameraFrame)
        self.assertEqual(frame_arg.width, 640)
        self.assertEqual(frame_arg.height, 480)
        self.assertEqual(frame_arg.timestamp, 123456789.0)
        self.assertEqual(frame_arg.fov, 90.0)
        self.assertEqual(frame_arg.rgb.shape, (480, 640, 3))
    
    def test_camera_callback_fov_from_config(self):
        """Test that FOV comes from config, not image attributes."""
        # Set different FOV in config
        self.config_mock.CAM_FOV_DEG = 110.0
        
        mock_image = Mock()
        mock_image.raw_data = np.random.bytes(640 * 480 * 4)
        mock_image.width = 640
        mock_image.height = 480
        mock_image.timestamp = 123456789.0
        # Ensure image has no attributes (the bug we fixed)
        del mock_image.attributes
        
        callback_mock = Mock()
        self.manager.camera_callback(mock_image, callback_mock)
        
        # Verify FOV comes from config
        frame_arg = callback_mock.call_args[0][0]
        self.assertEqual(frame_arg.fov, 110.0)
    
    def test_camera_callback_queue_full_handling(self):
        """Test camera callback handles full queue gracefully."""
        mock_image = Mock()
        mock_image.raw_data = np.random.bytes(640 * 480 * 4)
        mock_image.width = 640
        mock_image.height = 480
        mock_image.timestamp = 123456789.0
        
        # Mock queue full scenario
        self.manager.frame_queue.put_nowait.side_effect = [Exception("Queue full"), None]
        self.manager.frame_queue.get_nowait.return_value = Mock()
        
        callback_mock = Mock()
        # Should not raise exception
        self.manager.camera_callback(mock_image, callback_mock)
        
        # Verify both put attempts were made
        self.assertEqual(self.manager.frame_queue.put_nowait.call_count, 2)
    
    def test_camera_callback_no_callback(self):
        """Test camera callback works without user callback."""
        mock_image = Mock()
        mock_image.raw_data = np.random.bytes(640 * 480 * 4)
        mock_image.width = 640
        mock_image.height = 480
        mock_image.timestamp = 123456789.0
        
        # Should not raise exception
        self.manager.camera_callback(mock_image, None)
        
        # Verify frame was still queued
        self.assertTrue(self.manager.frame_queue.put_nowait.called)


if __name__ == '__main__':
    unittest.main()
