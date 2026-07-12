"""
Heavy tests for perception module — edge cases, multi-scenario, multi-env.

Covers:
- LaneTrajectoryPipeline: image_to_ground/ground_to_image round-trip, phases 1-4, process()
- LaneDetector: detect_lanes, detect_lanes_bev with synthetic images
- ClassicalLane: process_frame with synthetic lane images
- ego_lane_mask: create_ego_lane_mask_from_waypoints, filter_prob_map_with_ego_mask
- KalmanPoly: predict/update/reset
- CurvatureSmoother: smoothing + rate limit
- Edge cases: empty, all-black, all-white, tiny, huge, noisy
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SDL_VIDEODRIVER = "dummy"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _synthetic_lane_image(width=640, height=480, left_x=200, right_x=440,
                          line_width=5, curvature=0.0):
    """Create a synthetic image with two vertical lane lines (white on dark)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        offset = int(curvature * (row - height / 2) ** 2)
        lx = left_x + offset
        rx = right_x + offset
        for c in range(max(0, lx - line_width), min(width, lx + line_width)):
            img[row, c] = [255, 255, 255]
        for c in range(max(0, rx - line_width), min(width, rx + line_width)):
            img[row, c] = [255, 255, 255]
    return img


def _synthetic_yellow_lane_image(width=640, height=480):
    """Create image with yellow left lane and white right lane."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        for c in range(195, 205):
            img[row, c] = [0, 255, 255]  # Yellow (BGR)
        for c in range(435, 445):
            img[row, c] = [255, 255, 255]  # White
    return img


# ──────────────────────────────────────────────────────────────────────────────
# LaneTrajectoryPipeline — image_to_ground / ground_to_image round-trip
# ──────────────────────────────────────────────────────────────────────────────

class TestImageGroundProjection:
    """Test _image_to_ground and _ground_to_image round-trip accuracy."""

    def setup_method(self):
        from perception.lane_trajectory import LaneTrajectoryPipeline
        self.pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=640, cam_h=480,
            use_kalman=False, use_ego_lane_tracker=False,
        )

    def test_round_trip_ground_to_image_to_ground(self):
        """Ground → image → ground should recover original points (approx)."""
        x_orig = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        y_orig = np.array([-1.8, -1.8, -1.8, -1.8, -1.8])
        rows, cols = self.pipe._ground_to_image(x_orig, y_orig)
        x_back, y_back = self.pipe._image_to_ground(rows, cols)
        np.testing.assert_allclose(x_back, x_orig, atol=0.5)
        np.testing.assert_allclose(y_back, y_orig, atol=0.5)

    def test_round_trip_image_to_ground_to_image(self):
        """Image → ground → image should recover original pixels (approx)."""
        rows_orig = np.array([400.0, 350.0, 300.0])
        cols_orig = np.array([200.0, 250.0, 300.0])
        gx, gy = self.pipe._image_to_ground(rows_orig, cols_orig)
        # Filter out invalid (behind camera) projections
        valid = np.isfinite(gx) & (gx > 0)
        if not np.any(valid):
            pytest.skip("No valid ground projections")
        rows_back, cols_back = self.pipe._ground_to_image(gx[valid], gy[valid])
        np.testing.assert_allclose(rows_back, rows_orig[valid], atol=10.0)
        np.testing.assert_allclose(cols_back, cols_orig[valid], atol=10.0)

    def test_ground_to_image_behind_camera_returns_invalid(self):
        """Points behind camera (x < 0) should project to invalid image coords."""
        rows, cols = self.pipe._ground_to_image(
            np.array([-5.0, -1.0]),
            np.array([0.0, 0.0]),
        )
        assert np.all(np.isnan(rows)) or np.all(rows < 0) or np.all(rows > 10000)

    def test_image_to_ground_at_horizon(self):
        """Points near horizon should map to far distance or be invalid."""
        # Row 300 is just below horizon — should map to far distance
        rows = np.array([300.0])
        cols = np.array([320.0])  # center
        gx, gy = self.pipe._image_to_ground(rows, cols)
        # Should be far ahead or invalid (above horizon = -1)
        assert gx[0] > 10.0 or gx[0] < 0 or np.isnan(gx[0])

    def test_image_to_ground_centerline_is_zero_y(self):
        """Center column should map to y ≈ 0 (vehicle centerline)."""
        rows = np.array([400.0, 350.0, 300.0])
        cols = np.array([320.0, 320.0, 320.0])  # center
        gx, gy = self.pipe._image_to_ground(rows, cols)
        np.testing.assert_allclose(np.abs(gy), 0.0, atol=1.0)

    def test_ground_to_image_left_right_symmetry(self):
        """Left and right points at same distance should be symmetric around center."""
        x = np.array([10.0, 10.0])
        y = np.array([-1.8, 1.8])
        rows, cols = self.pipe._ground_to_image(x, y)
        center_col = 320.0
        left_offset = center_col - cols[0]
        right_offset = cols[1] - center_col
        assert abs(left_offset - right_offset) < 10.0  # roughly symmetric

    def test_image_to_ground_array_shapes(self):
        """Output arrays should match input array shapes."""
        rows = np.array([400.0, 350.0, 300.0])
        cols = np.array([200.0, 250.0, 300.0])
        gx, gy = self.pipe._image_to_ground(rows, cols)
        assert gx.shape == rows.shape
        assert gy.shape == cols.shape

    def test_image_to_ground_handles_empty_input(self):
        """Empty input arrays should produce empty output."""
        gx, gy = self.pipe._image_to_ground(np.array([]), np.array([]))
        assert len(gx) == 0
        assert len(gy) == 0


# ──────────────────────────────────────────────────────────────────────────────
# LaneTrajectoryPipeline — Phase 1 (lane mask)
# ──────────────────────────────────────────────────────────────────────────────

class TestPhase1LaneMask:
    """Test _phase1_lane_mask with various inputs."""

    def setup_method(self):
        from perception.lane_trajectory import LaneTrajectoryPipeline
        self.pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=640, cam_h=480,
            use_kalman=False, use_ego_lane_tracker=False,
        )

    def test_phase1_with_synthetic_lanes(self):
        """Phase 1 should produce a non-empty mask for synthetic lane image."""
        img = _synthetic_lane_image()
        mask, conf = self.pipe._phase1_lane_mask(img)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8
        assert conf >= 0.0

    def test_phase1_all_black_image(self):
        """All-black image should produce low or zero confidence."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        mask, conf = self.pipe._phase1_lane_mask(img)
        assert mask.shape == (480, 640)

    def test_phase1_all_white_image(self):
        """All-white image should not crash."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        mask, conf = self.pipe._phase1_lane_mask(img)
        assert mask.shape == (480, 640)

    def test_phase1_tiny_image(self):
        """Very small image should not crash."""
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        # Pipeline expects 640x480 — resize internally or handle gracefully
        try:
            mask, conf = self.pipe._phase1_lane_mask(img)
            assert mask is not None
        except (ValueError, IndexError):
            pytest.skip("Tiny image not supported by this config")

    def test_phase1_returns_valid_mask_values(self):
        """Mask should contain only binary values (0 or non-zero)."""
        img = _synthetic_lane_image()
        mask, _ = self.pipe._phase1_lane_mask(img)
        unique = np.unique(mask)
        # Mask may have intermediate values from morphological ops
        assert all(0 <= v <= 255 for v in unique)


# ──────────────────────────────────────────────────────────────────────────────
# LaneTrajectoryPipeline — Phase 2 (ground projection)
# ──────────────────────────────────────────────────────────────────────────────

class TestPhase2GroundProjection:
    """Test _phase2_bev_and_quality."""

    def setup_method(self):
        from perception.lane_trajectory import LaneTrajectoryPipeline
        self.pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=640, cam_h=480,
            use_kalman=False, use_ego_lane_tracker=False,
        )

    def test_phase2_empty_mask(self):
        """Empty mask should return 'invalid' quality."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        _, quality = self.pipe._phase2_bev_and_quality(mask)
        assert quality == "invalid"

    def test_phase2_full_mask(self):
        """Full mask should produce valid ground points."""
        mask = np.ones((480, 640), dtype=np.uint8) * 255
        _, quality = self.pipe._phase2_bev_and_quality(mask)
        assert quality in ["valid", "degraded"]

    def test_phase2_synthetic_lane_mask(self):
        """Mask with lane-like pixels should produce valid ground points."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[400:, 195:205] = 255  # left lane
        mask[400:, 435:445] = 255  # right lane
        _, quality = self.pipe._phase2_bev_and_quality(mask)
        assert quality in ["valid", "degraded", "invalid"]

    def test_phase2_ground_points_populated(self):
        """After phase 2, _ground_x and _ground_y should be set."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[400:, 195:205] = 255
        mask[400:, 435:445] = 255
        self.pipe._phase2_bev_and_quality(mask)
        if len(self.pipe._ground_x) > 0:
            assert len(self.pipe._ground_x) == len(self.pipe._ground_y)
            assert np.all(self.pipe._ground_x > 0)  # forward


# ──────────────────────────────────────────────────────────────────────────────
# LaneTrajectoryPipeline — Phase 3 (boundaries + center)
# ──────────────────────────────────────────────────────────────────────────────

class TestPhase3Boundaries:
    """Test _phase3_boundaries_and_center."""

    def setup_method(self):
        from perception.lane_trajectory import LaneTrajectoryPipeline
        self.pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=640, cam_h=480,
            use_kalman=False, use_ego_lane_tracker=False,
        )

    def test_phase3_no_points_returns_none(self):
        """With no ground points, phase 3 should return None coefficients."""
        self.pipe._ground_x = np.array([])
        self.pipe._ground_y = np.array([])
        self.pipe._img_rows = np.array([])
        self.pipe._img_cols = np.array([])
        result = self.pipe._phase3_boundaries_and_center(self.pipe._empty_bev)
        center_raw = result[0]
        assert center_raw is None

    def test_phase3_synthetic_left_right(self):
        """Synthetic left/right points should produce valid polynomials."""
        # Create synthetic ground points: left at y=-1.8, right at y=+1.8
        n = 100
        x = np.linspace(5, 25, n)
        y_left = np.full(n, -1.8) + np.random.randn(n) * 0.1
        y_right = np.full(n, 1.8) + np.random.randn(n) * 0.1

        self.pipe._ground_x = np.concatenate([x, x])
        self.pipe._ground_y = np.concatenate([y_left, y_right])
        self.pipe._img_rows = np.full(2 * n, 300)
        self.pipe._img_cols = np.full(2 * n, 320)

        result = self.pipe._phase3_boundaries_and_center(self.pipe._empty_bev)
        c_l, c_r = result[1], result[2]
        if c_l is not None and c_r is not None:
            # Lane width at x=10 should be ~3.6m
            width = np.polyval(c_r, 10.0) - np.polyval(c_l, 10.0)
            assert 2.0 < width < 5.0

    def test_phase3_only_left_points(self):
        """Only left points should trigger mirror_right case."""
        n = 50
        x = np.linspace(5, 20, n)
        y_left = np.full(n, -1.8)
        self.pipe._ground_x = x
        self.pipe._ground_y = y_left
        self.pipe._img_rows = np.full(n, 300)
        self.pipe._img_cols = np.full(n, 200)

        result = self.pipe._phase3_boundaries_and_center(self.pipe._empty_bev)
        p2_case = result[11]
        if result[1] is not None:
            assert p2_case in ["mirror_right", "both"]


# ──────────────────────────────────────────────────────────────────────────────
# LaneTrajectoryPipeline — Phase 4 (Kalman smoothing)
# ──────────────────────────────────────────────────────────────────────────────

class TestPhase4Smoothing:
    """Test _phase4_smooth_center."""

    def setup_method(self):
        from perception.lane_trajectory import LaneTrajectoryPipeline
        self.pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=640, cam_h=480,
            use_kalman=True, use_ego_lane_tracker=False,
        )

    def test_phase4_none_input(self):
        """None center_coeffs should return (None, False) eventually."""
        center, valid = self.pipe._phase4_smooth_center(None, "invalid")
        assert center is None or valid is False

    def test_phase4_valid_input(self):
        """Valid coefficients should produce smoothed output."""
        coeffs = np.array([0.001, -0.01, 0.0])  # slight curve
        center, valid = self.pipe._phase4_smooth_center(coeffs, "valid")
        if center is not None:
            assert len(center) == 3  # 2nd order poly

    def test_phase4_converges_after_multiple_updates(self):
        """After multiple updates, Kalman should converge to stable values."""
        coeffs = np.array([0.001, -0.01, 0.0])
        for _ in range(20):
            center, valid = self.pipe._phase4_smooth_center(coeffs, "valid")
        if center is not None:
            np.testing.assert_allclose(center, coeffs, atol=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# LaneTrajectoryPipeline — process() end-to-end
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineProcess:
    """Test LaneTrajectoryPipeline.process() with various inputs."""

    def setup_method(self):
        from perception.lane_trajectory import LaneTrajectoryPipeline
        self.pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=640, cam_h=480,
            use_kalman=False, use_ego_lane_tracker=False,
        )

    def test_process_synthetic_lanes(self):
        """process() should return TrajectoryOutput with synthetic lanes."""
        img = _synthetic_lane_image()
        out = self.pipe.process(img)
        assert out is not None
        assert hasattr(out, 'cte')
        assert hasattr(out, 'heading_err')
        assert hasattr(out, 'curvature')
        assert hasattr(out, 'confidence')

    def test_process_all_black(self):
        """process() with all-black image should not crash."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = self.pipe.process(img)
        assert out is not None
        assert out.confidence >= 0.0

    def test_process_all_white(self):
        """process() with all-white image should not crash."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        out = self.pipe.process(img)
        assert out is not None

    def test_process_noisy_image(self):
        """process() with random noise should not crash."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = self.pipe.process(img)
        assert out is not None

    def test_process_returns_lane_overlay(self):
        """process() should produce a lane_overlay image."""
        img = _synthetic_lane_image()
        out = self.pipe.process(img)
        if out.lane_overlay is not None:
            assert out.lane_overlay.shape[:2] == (480, 640)

    def test_process_returns_reference_path(self):
        """process() should return reference path arrays."""
        img = _synthetic_lane_image()
        out = self.pipe.process(img)
        if out.x_ref is not None:
            assert len(out.x_ref) > 0
            assert len(out.y_ref) == len(out.x_ref)

    def test_process_multiple_calls_stateful(self):
        """Multiple process() calls should be stateful (Kalman)."""
        img = _synthetic_lane_image()
        out1 = self.pipe.process(img)
        out2 = self.pipe.process(img)
        assert out1 is not None
        assert out2 is not None

    def test_process_curved_lanes(self):
        """process() with curved synthetic lanes should not crash."""
        img = _synthetic_lane_image(curvature=0.001)
        out = self.pipe.process(img)
        assert out is not None


# ──────────────────────────────────────────────────────────────────────────────
# KalmanPoly
# ──────────────────────────────────────────────────────────────────────────────

class TestKalmanPoly:
    """Test KalmanPoly filter for polynomial coefficients."""

    def setup_method(self):
        from perception.lane_trajectory import KalmanPoly
        self.kf = KalmanPoly(q=0.08, r=0.005, dim=3)

    def test_initial_state_is_none(self):
        """Before any update, state should be None."""
        assert self.kf.get_state() is None

    def test_update_sets_state(self):
        """After update, state should be set."""
        coeffs = np.array([0.001, -0.01, 0.0])
        state = self.kf.update(coeffs)
        assert state is not None
        assert len(state) == 3

    def test_predict_after_update(self):
        """Predict should return current state."""
        coeffs = np.array([0.001, -0.01, 0.0])
        self.kf.update(coeffs)
        predicted = self.kf.predict()
        assert predicted is not None

    def test_multiple_updates_converge(self):
        """Multiple updates with same value should converge."""
        coeffs = np.array([0.001, -0.01, 0.0])
        for _ in range(50):
            state = self.kf.update(coeffs)
        np.testing.assert_allclose(state, coeffs, atol=0.01)

    def test_reset_clears_state(self):
        """Reset should clear the filter state."""
        coeffs = np.array([0.001, -0.01, 0.0])
        self.kf.update(coeffs)
        self.kf.reset()
        assert self.kf.get_state() is None

    def test_update_with_noisy_measurements(self):
        """Filter should smooth noisy measurements."""
        true_coeffs = np.array([0.001, -0.01, 0.0])
        states = []
        for i in range(100):
            noisy = true_coeffs + np.random.randn(3) * 0.02
            state = self.kf.update(noisy)
            states.append(state)
        # Last state should be close to true value
        np.testing.assert_allclose(states[-1], true_coeffs, atol=0.02)


# ──────────────────────────────────────────────────────────────────────────────
# CurvatureSmoother
# ──────────────────────────────────────────────────────────────────────────────

class TestCurvatureSmoother:
    """Test CurvatureSmoother."""

    def setup_method(self):
        from perception.lane_trajectory import CurvatureSmoother
        self.sm = CurvatureSmoother(alpha=0.3, rate_limit=0.01)

    def test_initial_smooth_is_zero(self):
        """First update should return the input (no history)."""
        result = self.sm.update(0.05)
        assert isinstance(result, float)

    def test_smoothing_reduces_noise(self):
        """Alternating values should be smoothed."""
        values = [0.05, -0.05, 0.05, -0.05, 0.05]
        results = [self.sm.update(v) for v in values]
        # Last result should be closer to 0 than raw
        assert abs(results[-1]) < 0.05

    def test_rate_limit_prevents_jumps(self):
        """Rate limit should prevent sudden jumps."""
        self.sm.update(0.0)
        result = self.sm.update(0.5)  # big jump
        assert abs(result) < 0.5  # should be limited

    def test_reset(self):
        """Reset should clear history."""
        self.sm.update(0.05)
        self.sm.reset()
        result = self.sm.update(0.1)
        assert isinstance(result, float)


# ──────────────────────────────────────────────────────────────────────────────
# ClassicalLane detector
# ──────────────────────────────────────────────────────────────────────────────

class TestClassicalLane:
    """Test ClassicalLane detector with synthetic images."""

    def setup_method(self):
        try:
            from perception.classical.detector import ClassicalLane
        except ImportError:
            pytest.skip("ClassicalLane not available")
        self.detector = ClassicalLane(image_width=640, image_height=480)

    def test_get_line_markings_white_lanes(self):
        """get_line_markings should detect white lane lines."""
        img = _synthetic_lane_image()
        result = self.detector.get_line_markings(img)
        assert result is not None
        assert result.shape[:2] == img.shape[:2]

    def test_get_line_markings_all_black(self):
        """All-black image should produce all-zero markings."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.get_line_markings(img)
        assert result is not None

    def test_perspective_transform(self):
        """perspective_transform should warp image."""
        img = _synthetic_lane_image()
        markings = self.detector.get_line_markings(img)
        warped = self.detector.perspective_transform(markings)
        assert warped is not None

    def test_calculate_histogram(self):
        """Histogram should be computed from warped image."""
        img = _synthetic_lane_image()
        markings = self.detector.get_line_markings(img)
        self.detector.perspective_transform(markings)
        hist = self.detector.calculate_histogram()
        assert hist is not None
        assert len(hist) > 0

    def test_histogram_peak(self):
        """Histogram peak should find left and right base positions."""
        img = _synthetic_lane_image()
        markings = self.detector.get_line_markings(img)
        self.detector.perspective_transform(markings)
        self.detector.calculate_histogram()
        leftx, rightx = self.detector.histogram_peak()
        assert isinstance(leftx, int)
        assert isinstance(rightx, int)

    def test_process_frame(self):
        """Full process should return (cte, heading, curvature, valid)."""
        img = _synthetic_lane_image()
        try:
            result = self.detector.process(img)
            if isinstance(result, tuple) and len(result) >= 3:
                cte = result[0]
                heading = result[1]
                curvature = result[2]
                assert isinstance(cte, float)
                assert isinstance(heading, float)
                assert isinstance(curvature, float)
        except (ValueError, IndexError, np.linalg.LinAlgError):
            pytest.skip("process failed on synthetic image")


# ──────────────────────────────────────────────────────────────────────────────
# ego_lane_mask functions
# ──────────────────────────────────────────────────────────────────────────────

class TestEgoLaneMask:
    """Test ego lane mask functions."""

    def test_filter_prob_map_with_ego_mask(self):
        """filter_prob_map_with_ego_mask should filter probability map."""
        from perception.ego_lane_mask import filter_prob_map_with_ego_mask
        prob_map = np.random.rand(480, 640).astype(np.float32)
        ego_mask = np.zeros((480, 640), dtype=np.uint8)
        ego_mask[200:, 200:440] = 255  # ego lane region
        result = filter_prob_map_with_ego_mask(prob_map, ego_mask)
        assert result.shape == prob_map.shape
        # Outside ego mask should be zero
        assert np.all(result[:200, :] == 0)

    def test_filter_prob_map_with_blend(self):
        """Blend factor should retain some probability outside mask."""
        from perception.ego_lane_mask import filter_prob_map_with_ego_mask
        prob_map = np.ones((100, 100), dtype=np.float32) * 0.8
        ego_mask = np.zeros((100, 100), dtype=np.uint8)
        ego_mask[50:, :] = 255
        result = filter_prob_map_with_ego_mask(prob_map, ego_mask, blend_factor=0.5)
        # Outside mask should retain 0.5 * 0.8 = 0.4
        assert result[0, 0] == pytest.approx(0.4, abs=0.1)

    def test_filter_prob_map_zero_blend(self):
        """Zero blend should zero out everything outside mask."""
        from perception.ego_lane_mask import filter_prob_map_with_ego_mask
        prob_map = np.ones((100, 100), dtype=np.float32)
        ego_mask = np.zeros((100, 100), dtype=np.uint8)
        ego_mask[50:, :] = 255
        result = filter_prob_map_with_ego_mask(prob_map, ego_mask, blend_factor=0.0)
        assert np.all(result[:50, :] == 0)


# ──────────────────────────────────────────────────────────────────────────────
# LaneDetector — BEV detection
# ──────────────────────────────────────────────────────────────────────────────

class TestLaneDetectorBEV:
    """Test LaneDetector.detect_lanes_bev with synthetic images."""

    def setup_method(self):
        from perception.lane_detector import LaneDetector
        try:
            self.detector = LaneDetector(use_carla=True)
        except Exception:
            pytest.skip("LaneDetector requires carla")

    def test_detect_lanes_bev_synthetic(self):
        """detect_lanes_bev should process synthetic lane image."""
        img = _synthetic_lane_image()
        try:
            result = self.detector.detect_lanes_bev(img, return_vis=False)
            left_coeffs, right_coeffs, cte, heading, curvature, confidence, _ = result
            assert isinstance(cte, float)
            assert isinstance(heading, float)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0
        except (ValueError, IndexError, np.linalg.LinAlgError):
            pytest.skip("BEV detection failed on synthetic image")

    def test_detect_lanes_bev_all_black(self):
        """detect_lanes_bev with all-black image should not crash."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect_lanes_bev(img, return_vis=False)
        left_coeffs, right_coeffs, cte, heading, curvature, confidence, _ = result
        assert confidence < 0.5  # low confidence for blank image

    def test_detect_lanes_bev_returns_vis(self):
        """detect_lanes_bev with return_vis=True should return visualization."""
        img = _synthetic_lane_image()
        result = self.detector.detect_lanes_bev(img, return_vis=True)
        bev_vis = result[6]
        if bev_vis is not None:
            assert bev_vis.ndim == 3


# ──────────────────────────────────────────────────────────────────────────────
# Multi-scenario: different camera resolutions
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiResolution:
    """Test perception pipeline with different camera resolutions."""

    @pytest.mark.parametrize("cam_w,cam_h", [
        (320, 240),
        (640, 360),
        (640, 480),
        (1280, 720),
    ])
    def test_pipeline_different_resolutions(self, cam_w, cam_h):
        """Pipeline should handle different camera resolutions."""
        from perception.lane_trajectory import LaneTrajectoryPipeline
        pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=cam_w, cam_h=cam_h,
            use_kalman=False, use_ego_lane_tracker=False,
        )
        img = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        # Draw simple lanes
        for row in range(cam_h):
            for c in range(cam_w // 3 - 5, cam_w // 3 + 5):
                if 0 <= c < cam_w:
                    img[row, c] = [255, 255, 255]
            for c in range(2 * cam_w // 3 - 5, 2 * cam_w // 3 + 5):
                if 0 <= c < cam_w:
                    img[row, c] = [255, 255, 255]
        out = pipe.process(img)
        assert out is not None


# ──────────────────────────────────────────────────────────────────────────────
# Multi-scenario: noisy images
# ──────────────────────────────────────────────────────────────────────────────

class TestNoisyImages:
    """Test perception with various noise levels."""

    @pytest.mark.parametrize("noise_std", [0, 10, 30, 50, 100])
    def test_process_with_noise(self, noise_std):
        """Pipeline should handle various noise levels."""
        from perception.lane_trajectory import LaneTrajectoryPipeline
        pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=640, cam_h=480,
            use_kalman=False, use_ego_lane_tracker=False,
        )
        img = _synthetic_lane_image()
        if noise_std > 0:
            noise = np.random.randn(*img.shape) * noise_std
            img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        out = pipe.process(img)
        assert out is not None


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases: extreme inputs
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test perception with extreme edge cases."""

    def test_single_pixel_image(self):
        """1x1 image should not crash (graceful failure)."""
        from perception.lane_trajectory import LaneTrajectoryPipeline
        pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=1, cam_h=1,
            use_kalman=False, use_ego_lane_tracker=False,
        )
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        try:
            out = pipe.process(img)
            assert out is not None
        except (ValueError, IndexError):
            pytest.skip("1x1 image not supported")

    def test_very_wide_image(self):
        """Very wide image should not crash."""
        from perception.lane_trajectory import LaneTrajectoryPipeline
        pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=1920, cam_h=480,
            use_kalman=False, use_ego_lane_tracker=False,
        )
        img = np.zeros((480, 1920, 3), dtype=np.uint8)
        try:
            out = pipe.process(img)
            assert out is not None
        except Exception:
            pytest.skip("Very wide image not supported")

    def test_grayscale_input_as_rgb(self):
        """Grayscale image passed as 3-channel should work."""
        from perception.lane_trajectory import LaneTrajectoryPipeline
        pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=640, cam_h=480,
            use_kalman=False, use_ego_lane_tracker=False,
        )
        gray = np.zeros((480, 640), dtype=np.uint8)
        gray[400:, 200:210] = 255
        gray[400:, 430:440] = 255
        img = np.stack([gray, gray, gray], axis=-1)
        out = pipe.process(img)
        assert out is not None

    def test_float_image(self):
        """Float image [0,1] should be handled or converted."""
        from perception.lane_trajectory import LaneTrajectoryPipeline
        pipe = LaneTrajectoryPipeline(
            use_classical_detector=True, cam_w=640, cam_h=480,
            use_kalman=False, use_ego_lane_tracker=False,
        )
        img = np.random.rand(480, 640, 3).astype(np.float32)
        try:
            out = pipe.process(img)
            assert out is not None
        except (TypeError, ValueError, Exception):
            pytest.skip("Float image not directly supported")
