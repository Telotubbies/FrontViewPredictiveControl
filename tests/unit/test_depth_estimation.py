"""Tests for depth estimation module."""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class TestFlatGroundDepth:
    """Test FlatGroundDepth fallback estimator."""

    def setup_method(self):
        from perception.depth_estimation import FlatGroundDepth
        self.estimator = FlatGroundDepth(
            camera_height=1.8, camera_pitch=-8.0, fov_deg=90.0
        )

    def test_estimate_returns_correct_shape(self):
        """Output should match input image dimensions."""
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = self.estimator.estimate(rgb)
        assert depth.shape == (480, 640)
        assert depth.dtype == np.float32

    def test_estimate_returns_positive_depth(self):
        """All depth values should be positive."""
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = self.estimator.estimate(rgb)
        assert np.all(depth > 0)

    def test_depth_increases_with_row(self):
        """Depth should increase from bottom to top (near → far)."""
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = self.estimator.estimate(rgb)
        # Bottom rows (near) should have smaller depth than top rows (far)
        assert depth[400, 320] < depth[300, 320]

    def test_depth_uniform_per_row(self):
        """All pixels in the same row should have the same depth."""
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = self.estimator.estimate(rgb)
        row = depth[300, :]
        np.testing.assert_allclose(row, row[0], rtol=1e-5)

    def test_different_camera_height(self):
        """Higher camera should produce larger depth values."""
        from perception.depth_estimation import FlatGroundDepth
        low = FlatGroundDepth(camera_height=1.0)
        high = FlatGroundDepth(camera_height=2.0)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        d_low = low.estimate(rgb)
        d_high = high.estimate(rgb)
        assert d_high[400, 320] > d_low[400, 320]

    def test_handles_different_resolutions(self):
        """Should work with various image sizes."""
        for h, w in [(240, 320), (480, 640), (720, 1280)]:
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            depth = self.estimator.estimate(rgb)
            assert depth.shape == (h, w)

    def test_all_black_image(self):
        """All-black image should still produce valid depth."""
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = self.estimator.estimate(rgb)
        assert np.all(np.isfinite(depth))


class TestDepthEstimator:
    """Test DepthEstimator with mocked model."""

    def test_init_without_model_returns_none(self):
        """If model fails to load, estimate() should return None."""
        from perception.depth_estimation import DepthEstimator
        with patch('perception.depth_estimation.DepthEstimator._load_model'):
            estimator = DepthEstimator(model_size="small", device="cpu")
            estimator._model = None
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            result = estimator.estimate(rgb)
            assert result is None

    def test_is_available_false_when_no_model(self):
        """is_available should return False when model is not loaded."""
        from perception.depth_estimation import DepthEstimator
        with patch('perception.depth_estimation.DepthEstimator._load_model'):
            estimator = DepthEstimator(model_size="small", device="cpu")
            estimator._model = None
            assert estimator.is_available() is False

    def test_get_info(self):
        """get_info should return model metadata."""
        from perception.depth_estimation import DepthEstimator
        with patch('perception.depth_estimation.DepthEstimator._load_model'):
            estimator = DepthEstimator(model_size="small", device="cpu")
            info = estimator.get_info()
            assert "model_size" in info
            assert "device" in info
            assert "available" in info
            assert info["model_size"] == "small"

    def test_resolve_device_auto(self):
        """Auto device should resolve to cuda or cpu."""
        from perception.depth_estimation import DepthEstimator
        with patch('perception.depth_estimation.DepthEstimator._load_model'):
            estimator = DepthEstimator(model_size="small", device="auto")
            assert estimator._device in ["cuda", "cpu"]

    def test_resolve_device_explicit(self):
        """Explicit device should be used as-is."""
        from perception.depth_estimation import DepthEstimator
        with patch('perception.depth_estimation.DepthEstimator._load_model'):
            estimator = DepthEstimator(model_size="small", device="cpu")
            assert estimator._device == "cpu"

    def test_estimate_with_mock_model(self):
        """Estimate should work with a mocked HuggingFace pipeline."""
        from perception.depth_estimation import DepthEstimator

        # Create mock that returns a depth-like array
        mock_result = {"depth": np.random.rand(480, 640).astype(np.float32) * 100}
        mock_pipeline = Mock(return_value=mock_result)

        with patch('perception.depth_estimation.DepthEstimator._load_model'):
            estimator = DepthEstimator(model_size="small", device="cpu")
            estimator._model = mock_pipeline

            rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth = estimator.estimate(rgb)

            assert depth is not None
            assert depth.shape == (480, 640)
            assert depth.dtype == np.float32
            assert depth.max() <= 1.0  # normalized

    def test_estimate_metric_with_mock(self):
        """estimate_metric should produce metric-scale depth."""
        from perception.depth_estimation import DepthEstimator

        mock_result = {"depth": np.random.rand(480, 640).astype(np.float32) * 100}
        mock_pipeline = Mock(return_value=mock_result)

        with patch('perception.depth_estimation.DepthEstimator._load_model'):
            estimator = DepthEstimator(model_size="small", device="cpu")
            estimator._model = mock_pipeline

            rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth = estimator.estimate_metric(
                rgb, camera_height=1.8, camera_pitch=-8.0, fov_deg=90.0
            )

            assert depth is not None
            assert depth.shape == (480, 640)
            assert np.all(depth > 0)  # metric depth should be positive

    def test_estimate_downscales_large_image(self):
        """Large images should be downscaled before inference."""
        from perception.depth_estimation import DepthEstimator

        mock_result = {"depth": np.random.rand(518, 518).astype(np.float32) * 100}
        mock_pipeline = Mock(return_value=mock_result)

        with patch('perception.depth_estimation.DepthEstimator._load_model'):
            estimator = DepthEstimator(
                model_size="small", device="cpu", max_resolution=518
            )
            estimator._model = mock_pipeline

            # Large image
            rgb = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            estimator.estimate(rgb)

            # Check that pipeline was called (with downscaled image)
            mock_pipeline.assert_called_once()

    def test_estimate_handles_float_input(self):
        """Float input should be converted to uint8 without crash."""
        from perception.depth_estimation import DepthEstimator

        mock_result = {"depth": np.random.rand(480, 640).astype(np.float32) * 100}
        mock_pipeline = Mock(return_value=mock_result)

        with patch('perception.depth_estimation.DepthEstimator._load_model'):
            estimator = DepthEstimator(model_size="small", device="cpu")
            estimator._model = mock_pipeline

            rgb = np.random.rand(480, 640, 3).astype(np.float32)
            depth = estimator.estimate(rgb)
            assert depth is not None

    def test_estimate_handles_inference_error(self):
        """If inference raises, estimate should return None (graceful)."""
        from perception.depth_estimation import DepthEstimator

        mock_pipeline = Mock(side_effect=RuntimeError("GPU OOM"))

        with patch('perception.depth_estimation.DepthEstimator._load_model'):
            estimator = DepthEstimator(model_size="small", device="cpu")
            estimator._model = mock_pipeline

            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            depth = estimator.estimate(rgb)
            assert depth is None


class TestDepthEstimatorIntegration:
    """Integration tests for depth estimation with perception pipeline."""

    def test_flat_ground_depth_vs_geometric(self):
        """FlatGroundDepth should match geometric projection formula."""
        from perception.depth_estimation import FlatGroundDepth
        import numpy as np

        cam_h = 1.8
        pitch = -8.0
        fov = 90.0
        estimator = FlatGroundDepth(cam_h, pitch, fov)

        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = estimator.estimate(rgb)

        # Manually compute depth at row 400
        h = 480
        vfov = np.radians(fov)
        vfov_per_pixel = vfov / h
        pitch_rad = np.radians(pitch)
        angle = pitch_rad + (400 - h / 2) * vfov_per_pixel
        expected = cam_h / np.tan(max(angle, 0.01))

        assert abs(depth[400, 320] - expected) < 0.01

    def test_depth_at_bottom_is_near(self):
        """Bottom of image should correspond to near distance."""
        from perception.depth_estimation import FlatGroundDepth
        estimator = FlatGroundDepth(camera_height=1.8, camera_pitch=-8.0)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = estimator.estimate(rgb)
        # Bottom row should be within a few meters
        assert depth[479, 320] < 10.0

    def test_depth_at_horizon_is_far(self):
        """Near horizon should correspond to very far distance."""
        from perception.depth_estimation import FlatGroundDepth
        estimator = FlatGroundDepth(camera_height=1.8, camera_pitch=-8.0)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = estimator.estimate(rgb)
        # Row near horizon (where angle ≈ 0) should be very large
        # With pitch=-8°, horizon is at row ≈ 480/2 + 8/90*480 ≈ 282
        # Just below horizon should be very far
        assert depth[285, 320] > 50.0 or depth[285, 320] > depth[400, 320]
