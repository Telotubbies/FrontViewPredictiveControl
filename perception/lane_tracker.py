#!/usr/bin/env python3
"""
Lane Tracker using Kalman Filter and Line Class.

Tracks lane polynomial coefficients over time to handle:
- Dashed lines (intermittent detection)
- Noisy detections
- Temporary occlusions

Based on research:
- "Robust Lane Detection and Tracking with RANSAC and Kalman Filter"
- GitHub: aidanscannell/lane-detection
- GitHub: uranus4ever/Advanced-Lane-Detection (Line Class)
"""

import numpy as np
from typing import Optional, Tuple, List
from collections import deque


class Line:
    """
    Line class to track lane line properties across frames.
    Based on Udacity Advanced Lane Detection project.
    """
    
    def __init__(self, history_size: int = 10):
        # Was the line detected in the last iteration?
        self.detected = False
        
        # Polynomial coefficients history
        self.recent_fits = deque(maxlen=history_size)
        
        # Polynomial coefficients averaged over last n iterations
        self.best_fit = None
        
        # X values for detected line pixels (recent)
        self.recent_xfitted = deque(maxlen=history_size)
        
        # Radius of curvature
        self.radius_of_curvature = None
        
        # Distance from center
        self.line_base_pos = None
        
        # Difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype=np.float64)
        
        # Number of consecutive frames without detection
        self.frames_without_detection = 0
        self.max_frames_without_detection = 5
        
    def add_fit(self, fit: np.ndarray) -> np.ndarray:
        """Add new polynomial fit and return smoothed result."""
        if fit is None:
            self.detected = False
            self.frames_without_detection += 1
            
            if self.frames_without_detection > self.max_frames_without_detection:
                # Too many frames without detection - reset
                return None
            
            # Return best fit from history
            return self.best_fit
        
        # Check if fit is reasonable (not too different from previous)
        if self.best_fit is not None:
            self.diffs = np.abs(fit - self.best_fit)
            # If coefficients changed too much, it might be wrong lane
            if self.diffs[0] > 100 or self.diffs[1] > 1.0 or self.diffs[2] > 0.01:
                # Suspicious - use weighted average with history
                fit = 0.3 * fit + 0.7 * self.best_fit
        
        self.detected = True
        self.frames_without_detection = 0
        self.recent_fits.append(fit)
        
        # Update best fit as average of recent fits
        if len(self.recent_fits) > 0:
            self.best_fit = np.mean(self.recent_fits, axis=0)
        else:
            self.best_fit = fit
            
        return self.best_fit
    
    def reset(self):
        """Reset line state."""
        self.detected = False
        self.recent_fits.clear()
        self.best_fit = None
        self.frames_without_detection = 0


def ransac_polyfit(x: np.ndarray, y: np.ndarray, degree: int = 2, 
                   n_iterations: int = 50, threshold: float = 5.0,
                   min_inliers: int = 10) -> Optional[np.ndarray]:
    """
    RANSAC polynomial fitting - robust to outliers.
    
    Args:
        x, y: Data points
        degree: Polynomial degree
        n_iterations: Number of RANSAC iterations
        threshold: Inlier threshold (pixels)
        min_inliers: Minimum inliers for valid fit
        
    Returns:
        Polynomial coefficients or None if failed
    """
    if len(x) < min_inliers:
        return None
    
    best_fit = None
    best_inliers = 0
    
    n_samples = degree + 1
    
    for _ in range(n_iterations):
        # Random sample
        idx = np.random.choice(len(x), min(n_samples, len(x)), replace=False)
        sample_x, sample_y = x[idx], y[idx]
        
        try:
            # Fit polynomial to sample
            fit = np.polyfit(sample_y, sample_x, degree)
            
            # Count inliers
            predicted_x = np.polyval(fit, y)
            errors = np.abs(x - predicted_x)
            inliers = np.sum(errors < threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_fit = fit
        except:
            continue
    
    if best_fit is not None and best_inliers >= min_inliers:
        # Refit using all inliers
        predicted_x = np.polyval(best_fit, y)
        errors = np.abs(x - predicted_x)
        inlier_mask = errors < threshold
        
        if np.sum(inlier_mask) >= min_inliers:
            try:
                best_fit = np.polyfit(y[inlier_mask], x[inlier_mask], degree)
            except:
                pass
    
    return best_fit


class LaneKalmanFilter:
    """
    Kalman Filter for tracking lane polynomial coefficients.
    
    State: [c0, c1, c2, dc0, dc1, dc2] (coefficients + velocities)
    Measurement: [c0, c1, c2] (polynomial coefficients)
    
    For polynomial: x = c0 + c1*y + c2*y^2
    """
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        # State dimension: 6 (3 coefficients + 3 velocities)
        # Measurement dimension: 3 (coefficients only)
        self.n_states = 6
        self.n_meas = 3
        
        # State vector [c0, c1, c2, dc0, dc1, dc2]
        self.x = np.zeros(self.n_states)
        
        # State covariance
        self.P = np.eye(self.n_states) * 1.0
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(self.n_states)
        self.F[0, 3] = 1.0  # c0 += dc0
        self.F[1, 4] = 1.0  # c1 += dc1
        self.F[2, 5] = 1.0  # c2 += dc2
        
        # Measurement matrix
        self.H = np.zeros((self.n_meas, self.n_states))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        
        # Process noise
        self.Q = np.eye(self.n_states) * process_noise
        self.Q[3, 3] = process_noise * 0.1  # Velocity noise lower
        self.Q[4, 4] = process_noise * 0.1
        self.Q[5, 5] = process_noise * 0.1
        
        # Measurement noise
        self.R = np.eye(self.n_meas) * measurement_noise
        
        self._initialized = False
        self._frames_without_measurement = 0
        self._max_frames_predict = 10  # Max frames to predict without measurement
        
    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]  # Return coefficients
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with measurement."""
        if not self._initialized:
            # Initialize state with first measurement
            self.x[:3] = measurement
            self._initialized = True
            self._frames_without_measurement = 0
            return measurement
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - self.H @ self.x  # Innovation
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P
        
        self._frames_without_measurement = 0
        return self.x[:3]
    
    def get_coefficients(self, measurement: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Get filtered coefficients.
        
        Args:
            measurement: New measurement (None if no detection)
            
        Returns:
            Filtered coefficients or None if too many frames without measurement
        """
        if measurement is not None:
            self.predict()
            return self.update(measurement)
        else:
            self._frames_without_measurement += 1
            if self._frames_without_measurement > self._max_frames_predict:
                return None
            return self.predict()
    
    def reset(self):
        """Reset filter state."""
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 1.0
        self._initialized = False
        self._frames_without_measurement = 0


class LaneTracker:
    """
    Tracks left and right lane lines using Line class + Kalman filters.
    
    Uses multiple strategies:
    1. Line class for history averaging
    2. Kalman filter for prediction
    3. Width validation for sanity checking
    """
    
    def __init__(
        self,
        process_noise: float = 0.005,
        measurement_noise: float = 0.05,
        history_size: int = 10,
    ):
        # Line class for history-based smoothing
        self.left_line = Line(history_size=history_size)
        self.right_line = Line(history_size=history_size)
        
        # Kalman filters for prediction
        self.left_kf = LaneKalmanFilter(process_noise, measurement_noise)
        self.right_kf = LaneKalmanFilter(process_noise, measurement_noise)
        
        # History for sanity checking
        self.left_history = deque(maxlen=history_size)
        self.right_history = deque(maxlen=history_size)
        
        # Lane width tracking
        self._expected_width = None
        self._width_tolerance = 0.3  # 30% tolerance
        
        # CARLA lane_id tracking for "cheating"
        self._locked_lane_id = None
        self._lane_id_frames = 0
        
    def update(
        self,
        left_coeffs: Optional[np.ndarray],
        right_coeffs: Optional[np.ndarray],
        bev_height: int = 480,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Update tracker with new detections.
        
        Args:
            left_coeffs: Left lane polynomial coefficients (or None)
            right_coeffs: Right lane polynomial coefficients (or None)
            bev_height: BEV image height for width calculation
            
        Returns:
            (filtered_left, filtered_right) coefficients
        """
        # Sanity check: lane width
        if left_coeffs is not None and right_coeffs is not None:
            # Calculate width at bottom of BEV
            y = bev_height - 1
            left_x = np.polyval(left_coeffs[::-1], y)
            right_x = np.polyval(right_coeffs[::-1], y)
            width = right_x - left_x
            
            if self._expected_width is None:
                self._expected_width = width
            else:
                # Check if width is reasonable
                if abs(width - self._expected_width) / self._expected_width > self._width_tolerance:
                    # Width changed too much - likely wrong lane
                    # Use history average instead
                    if len(self.left_history) > 0:
                        left_coeffs = np.mean(self.left_history, axis=0)
                    if len(self.right_history) > 0:
                        right_coeffs = np.mean(self.right_history, axis=0)
                else:
                    # Update expected width slowly
                    self._expected_width = 0.95 * self._expected_width + 0.05 * width
        
        # Step 1: Line class smoothing (history-based)
        smoothed_left = self.left_line.add_fit(left_coeffs)
        smoothed_right = self.right_line.add_fit(right_coeffs)
        
        # Step 2: Kalman filter for prediction when detection fails
        filtered_left = self.left_kf.get_coefficients(smoothed_left)
        filtered_right = self.right_kf.get_coefficients(smoothed_right)
        
        # Update history
        if filtered_left is not None:
            self.left_history.append(filtered_left.copy())
        if filtered_right is not None:
            self.right_history.append(filtered_right.copy())
        
        return filtered_left, filtered_right
    
    def set_lane_id(self, lane_id: int):
        """Set CARLA lane_id for validation."""
        if self._locked_lane_id is None:
            self._locked_lane_id = lane_id
            self._lane_id_frames = 1
        elif lane_id == self._locked_lane_id:
            self._lane_id_frames += 1
        else:
            # Lane changed - might be intentional lane change
            if self._lane_id_frames > 30:  # ~1 second at 30fps
                # Established in previous lane, this is a change
                self._locked_lane_id = lane_id
                self._lane_id_frames = 1
    
    def is_lane_change_detected(self, current_lane_id: int) -> bool:
        """Check if current detection suggests wrong lane."""
        if self._locked_lane_id is None:
            return False
        return current_lane_id != self._locked_lane_id
    
    def reset(self):
        """Reset tracker state."""
        self.left_line.reset()
        self.right_line.reset()
        self.left_kf.reset()
        self.right_kf.reset()
        self.left_history.clear()
        self.right_history.clear()
        self._expected_width = None
        self._locked_lane_id = None
        self._lane_id_frames = 0
