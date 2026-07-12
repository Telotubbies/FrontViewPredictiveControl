"""Unit tests for LKAPipeline process method."""
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import LKAPipeline
from utils.type_hints import CameraFrame
from state import FrameState


class TestLKAPipeline:
    """Test cases for LKAPipeline class."""

    def setup_method(self):
        """Setup test environment."""
        self.model_path = "dummy_model.pth"
        self.device = torch.device("cpu")
        self.target_speed_kmh = 30.0

        with patch('pipeline.BEVRoadPerception'), \
             patch('pipeline.LaneTemporalSmoother'), \
             patch('pipeline.LaneMPC'), \
             patch('pipeline.SafetyOverride'):
            self.pipeline = LKAPipeline(
                self.model_path,
                self.device,
                self.target_speed_kmh,
                use_trajectory_pipeline=False
            )

    def test_process_method_exists(self):
        """Test that process method exists and is callable."""
        assert hasattr(self.pipeline, 'process')
        assert callable(self.pipeline.process)

    def test_process_delegates_to_step(self):
        """Test that process method delegates to step with default values."""
        # Mock the step method
        mock_step_result = (0.1, 0.2, 0.3, Mock(spec=FrameState))
        self.pipeline.step = Mock(return_value=mock_step_result)

        # Create dummy RGB input
        rgb_input = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Call process
        result = self.pipeline.process(rgb_input)

        # Verify step was called with correct default values
        self.pipeline.step.assert_called_once_with(
            rgb=rgb_input,
            speed_ms=0.0,
            wp_state=None,
            prev_steer=0.0,
            prev_throttle=0.0
        )

        # Verify result is passed through
        assert result == mock_step_result

    def test_process_with_different_inputs(self):
        """Test process with various RGB input shapes."""
        mock_step_result = (0.0, 0.0, 0.0, Mock(spec=FrameState))
        self.pipeline.step = Mock(return_value=mock_step_result)

        # Test different input sizes
        test_shapes = [
            (480, 640, 3),
            (240, 320, 3),
            (720, 1280, 3)
        ]

        for shape in test_shapes:
            rgb_input = np.random.randint(0, 255, shape, dtype=np.uint8)
            result = self.pipeline.process(rgb_input)
            assert result == mock_step_result
            self.pipeline.step.assert_called()

    def test_process_step_exception_handling(self):
        """Test that process propagates exceptions from step."""
        # Mock step to raise exception
        self.pipeline.step = Mock(side_effect=ValueError("Test error"))

        rgb_input = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Should raise the same exception
        with pytest.raises(ValueError, match="Test error"):
            self.pipeline.process(rgb_input)


class TestLKAPipelineWithTrajectory:
    """Test LKAPipeline with trajectory pipeline enabled."""

    def setup_method(self):
        """Setup test environment with trajectory pipeline."""
        self.model_path = "dummy_model.pth"
        self.device = torch.device("cpu")
        self.target_speed_kmh = 30.0

        import perception.lane_trajectory as _lt
        with patch('pipeline.BEVRoadPerception'), \
             patch('pipeline.LaneTemporalSmoother'), \
             patch('pipeline.LaneMPC'), \
             patch('pipeline.SafetyOverride'), \
             patch.object(_lt, 'LaneTrajectoryPipeline') as mock_trajectory:
            self.pipeline = LKAPipeline(
                self.model_path,
                self.device,
                self.target_speed_kmh,
                use_trajectory_pipeline=True
            )
            self.mock_trajectory = mock_trajectory

    def test_process_with_trajectory_pipeline(self):
        """Test process method works with trajectory pipeline enabled."""
        # Mock trajectory pipeline process output
        mock_trajectory_output = Mock()
        mock_trajectory_output.cte = 0.5
        mock_trajectory_output.heading_err = 0.1
        mock_trajectory_output.curvature = 0.02
        mock_trajectory_output.confidence = 0.8
        mock_trajectory_output.waypoint_only = False
        mock_trajectory_output.geometry_valid = True
        mock_trajectory_output.v_ref_at_ego = 8.0
        mock_trajectory_output.lane_overlay = None
        mock_trajectory_output.bev_window_vis = None
        mock_trajectory_output.mask_vis = None
        mock_trajectory_output.left_px_img = None
        mock_trajectory_output.right_px_img = None
        mock_trajectory_output.reference_path = None
        mock_trajectory_output.left_curvature_m = 9999.0
        mock_trajectory_output.right_curvature_m = 9999.0
        mock_trajectory_output.bev_binary = None

        self.mock_trajectory.return_value.process.return_value = mock_trajectory_output

        # Mock other components
        self.pipeline._mpc.solve = Mock(return_value=(0.1, 0.2, 0.0, None))
        self.pipeline._mpc.steer_to_carla = Mock(return_value=0.1)
        self.pipeline._mpc.accel_to_carla = Mock(return_value=(0.5, 0.0))
        self.pipeline._safety.apply = Mock(return_value=(0.1, 0.2, 0.0))
        self.pipeline._safety.apply_safety_override = Mock(return_value=(0.1, 0.5, 0.0))

        rgb_input = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Should not raise exception
        result = self.pipeline.process(rgb_input)

        # Verify trajectory pipeline was used (process called with rgb + world/vehicle kwargs)
        self.mock_trajectory.return_value.process.assert_called_once()
        call_args = self.mock_trajectory.return_value.process.call_args
        assert call_args.args[0] is rgb_input or np.array_equal(call_args.args[0], rgb_input)

        # Verify result structure
        assert len(result) == 4  # (steer, throttle, brake, frame_state)
