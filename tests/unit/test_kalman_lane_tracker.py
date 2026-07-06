"""Production tests for Kalman lane tracker (perception/kalman_lane_tracker.py)."""
import pytest
import numpy as np

from perception.kalman_lane_tracker import (
    KalmanLaneTracker, _KalmanPoly,
    POLY_DIM, DEFAULT_Q, DEFAULT_R, MAX_FRAMES_STALE,
)


# ── _KalmanPoly (internal single-polynomial filter) ──────────────────────────


class TestKalmanPolyInit:
    """Test _KalmanPoly constructor."""

    def test_default_params(self):
        kf = _KalmanPoly()
        assert kf.dim == POLY_DIM
        assert kf.x.shape == (POLY_DIM,)
        assert kf.P.shape == (POLY_DIM, POLY_DIM)
        assert kf._initialized is False

    def test_custom_q_r(self):
        kf = _KalmanPoly(q=0.5, r=0.1)
        assert kf.Q[0, 0] == pytest.approx(0.5)
        assert kf.R[0, 0] == pytest.approx(0.1)

    def test_initial_state_zero(self):
        kf = _KalmanPoly()
        assert np.all(kf.x == 0.0)

    def test_initial_covariance(self):
        kf = _KalmanPoly()
        # P should be 0.1 * I
        assert kf.P[0, 0] == pytest.approx(0.1)
        assert kf.P[1, 0] == pytest.approx(0.0)


class TestKalmanPolyUpdate:
    """Test _KalmanPoly.update()."""

    def test_first_update_sets_state(self):
        """First update should set state directly (no prediction)."""
        kf = _KalmanPoly()
        z = np.array([0.1, 0.2, 0.3])
        result = kf.update(z)
        assert np.allclose(result, z)
        assert kf._initialized is True

    def test_subsequent_update_blends(self):
        """Second update should blend prediction with measurement."""
        kf = _KalmanPoly()
        kf.update(np.array([0.1, 0.2, 0.3]))
        result = kf.update(np.array([0.15, 0.25, 0.35]))
        # Should be between previous and new measurement
        assert 0.1 <= result[0] <= 0.15

    def test_wrong_dim_returns_current(self):
        """Wrong dimension measurement should be ignored."""
        kf = _KalmanPoly()
        kf.update(np.array([0.1, 0.2, 0.3]))
        result = kf.update(np.array([0.1, 0.2]))  # dim=2, wrong
        assert np.allclose(result, [0.1, 0.2, 0.3])

    def test_update_returns_copy(self):
        """update() should return a copy, not internal state."""
        kf = _KalmanPoly()
        result = kf.update(np.array([0.1, 0.2, 0.3]))
        result[0] = 999.0
        assert kf.x[0] == pytest.approx(0.1)


class TestKalmanPolyPredict:
    """Test _KalmanPoly.predict()."""

    def test_predict_before_init_returns_zeros(self):
        """Predict before initialization should return zeros."""
        kf = _KalmanPoly()
        result = kf.predict()
        assert np.allclose(result, 0.0)

    def test_predict_after_init(self):
        """Predict after init should return current state (F=I)."""
        kf = _KalmanPoly()
        kf.update(np.array([0.1, 0.2, 0.3]))
        result = kf.predict()
        assert np.allclose(result, [0.1, 0.2, 0.3])

    def test_predict_increases_covariance(self):
        """Predict should increase uncertainty (P += Q)."""
        kf = _KalmanPoly()
        kf.update(np.array([0.1, 0.2, 0.3]))
        P_before = kf.P[0, 0]
        kf.predict()
        assert kf.P[0, 0] > P_before


class TestKalmanPolyGetState:
    """Test _KalmanPoly.get_state()."""

    def test_before_init_returns_none(self):
        kf = _KalmanPoly()
        assert kf.get_state() is None

    def test_after_init_returns_state(self):
        kf = _KalmanPoly()
        kf.update(np.array([0.1, 0.2, 0.3]))
        state = kf.get_state()
        assert state is not None
        assert np.allclose(state, [0.1, 0.2, 0.3])


class TestKalmanPolyReset:
    """Test _KalmanPoly.reset()."""

    def test_reset_clears_state(self):
        kf = _KalmanPoly()
        kf.update(np.array([0.1, 0.2, 0.3]))
        kf.reset()
        assert kf._initialized is False
        assert np.all(kf.x == 0.0)
        assert kf.get_state() is None


# ── KalmanLaneTracker (public API) ────────────────────────────────────────────


class TestKalmanLaneTrackerInit:
    """Test KalmanLaneTracker constructor."""

    def test_default_params(self):
        tracker = KalmanLaneTracker()
        assert tracker._max_frames_stale == MAX_FRAMES_STALE
        assert tracker._frames_without_left == 0
        assert tracker._frames_without_right == 0

    def test_custom_params(self):
        tracker = KalmanLaneTracker(q=0.3, r=0.1, max_frames_stale=5)
        assert tracker._max_frames_stale == 5


class TestKalmanLaneTrackerUpdate:
    """Test KalmanLaneTracker.update()."""

    def test_both_lanes_present(self):
        """Both lanes observed → both returned."""
        tracker = KalmanLaneTracker()
        left, right = tracker.update(
            np.array([0.001, 0.1, 0.5]),
            np.array([0.001, -0.1, -0.5]),
        )
        assert left is not None
        assert right is not None
        assert len(left) == POLY_DIM
        assert len(right) == POLY_DIM

    def test_left_missing_predicts(self):
        """Left missing → predict from previous (if initialized)."""
        tracker = KalmanLaneTracker()
        # First frame: both present
        tracker.update(np.array([0.001, 0.1, 0.5]), np.array([0.001, -0.1, -0.5]))
        # Second frame: left missing
        left, right = tracker.update(None, np.array([0.001, -0.1, -0.5]))
        assert left is not None  # predicted
        assert right is not None

    def test_right_missing_predicts(self):
        """Right missing → predict from previous (if initialized)."""
        tracker = KalmanLaneTracker()
        tracker.update(np.array([0.001, 0.1, 0.5]), np.array([0.001, -0.1, -0.5]))
        left, right = tracker.update(np.array([0.001, 0.1, 0.5]), None)
        assert left is not None
        assert right is not None

    def test_both_missing_before_init_returns_none(self):
        """Both missing before any observation → (None, None)."""
        tracker = KalmanLaneTracker()
        left, right = tracker.update(None, None)
        assert left is None
        assert right is None

    def test_stale_resets(self):
        """After max_frames_stale missing frames, tracker resets."""
        tracker = KalmanLaneTracker(max_frames_stale=3)
        tracker.update(np.array([0.001, 0.1, 0.5]), np.array([0.001, -0.1, -0.5]))
        # Miss 3 frames
        for _ in range(3):
            tracker.update(None, None)
        # After reset, get_state should be None
        left, right = tracker.get_left_right()
        assert left is None
        assert right is None

    def test_partial_coeffs_truncated(self):
        """Coeffs longer than POLY_DIM should be truncated."""
        tracker = KalmanLaneTracker()
        left, right = tracker.update(
            np.array([0.001, 0.1, 0.5, 0.99, 0.88]),  # 5 elements
            np.array([0.001, -0.1, -0.5]),
        )
        assert len(left) == POLY_DIM


class TestKalmanLaneTrackerGetLeftRight:
    """Test KalmanLaneTracker.get_left_right()."""

    def test_before_update_returns_none(self):
        tracker = KalmanLaneTracker()
        left, right = tracker.get_left_right()
        assert left is None
        assert right is None

    def test_after_update_returns_state(self):
        tracker = KalmanLaneTracker()
        tracker.update(np.array([0.001, 0.1, 0.5]), np.array([0.001, -0.1, -0.5]))
        left, right = tracker.get_left_right()
        assert left is not None
        assert right is not None


class TestKalmanLaneTrackerReset:
    """Test KalmanLaneTracker.reset()."""

    def test_reset_clears_all(self):
        tracker = KalmanLaneTracker()
        tracker.update(np.array([0.001, 0.1, 0.5]), np.array([0.001, -0.1, -0.5]))
        tracker.reset()
        left, right = tracker.get_left_right()
        assert left is None
        assert right is None
        assert tracker._frames_without_left == 0
        assert tracker._frames_without_right == 0


class TestKalmanLaneTrackerConvergence:
    """Test that tracker converges to true value over multiple updates."""

    def test_converges_to_noisy_measurement(self):
        """With noisy measurements, tracker should converge near true value."""
        tracker = KalmanLaneTracker(q=0.01, r=0.1)
        true_coeffs = np.array([0.001, 0.1, 0.5])
        np.random.seed(42)
        for _ in range(50):
            noisy = true_coeffs + np.random.randn(POLY_DIM) * 0.05
            tracker.update(noisy, np.array([0.0, 0.0, 0.0]))
        left, _ = tracker.get_left_right()
        # Should be close to true value
        assert np.allclose(left, true_coeffs, atol=0.1)
