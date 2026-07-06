"""
Unit tests for improved lane fitting with B-splines and geometric constraints.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.improved_lane_fitting import (
    BSplineLaneFitter,
    GeometricLaneValidator,
    convert_bspline_to_polynomial,
)


class TestBSplineLaneFitter:
    """Test B-spline lane fitting."""
    
    def test_straight_lane_fitting(self):
        """Test fitting straight parallel lanes."""
        # Create straight lanes (3.5m apart)
        x = np.linspace(0, 50, 30)
        left_xy = np.column_stack([x, np.ones_like(x) * -1.75])
        right_xy = np.column_stack([x, np.ones_like(x) * 1.75])
        
        fitter = BSplineLaneFitter(degree=3, num_control_points=6)
        left_spline, right_spline, info = fitter.fit_lane_pair(
            left_xy, right_xy,
            enforce_parallel=True,
            enforce_width=True,
        )
        
        assert info["valid"] is True
        assert info["width_valid"] is True
        assert 3.0 < info["avg_width_m"] < 4.0
        assert info["width_std_m"] < 0.5
        assert info["parallel_score"] > 0.8
    
    def test_curved_lane_fitting(self):
        """Test fitting curved lanes."""
        # Create curved lanes (parabolic)
        x = np.linspace(0, 50, 30)
        curvature = 0.02
        left_y = -1.75 + 0.5 * curvature * x**2
        right_y = 1.75 + 0.5 * curvature * x**2
        
        left_xy = np.column_stack([x, left_y])
        right_xy = np.column_stack([x, right_y])
        
        fitter = BSplineLaneFitter(degree=3, num_control_points=8)
        left_spline, right_spline, info = fitter.fit_lane_pair(
            left_xy, right_xy,
            enforce_parallel=True,
            enforce_width=False,  # Width changes in curves
        )
        
        assert info["valid"] is True
        assert info["left_curvature_max"] < 0.1
        assert info["right_curvature_max"] < 0.1
    
    def test_insufficient_points(self):
        """Test handling of insufficient points."""
        left_xy = np.array([[0, -1.5], [10, -1.5]])  # Only 2 points
        right_xy = np.array([[0, 1.5], [10, 1.5]])
        
        fitter = BSplineLaneFitter()
        left_spline, right_spline, info = fitter.fit_lane_pair(left_xy, right_xy)
        
        assert info["valid"] is False
        assert "insufficient_points" in info.get("reason", "")
    
    def test_evaluate_at_positions(self):
        """Test evaluating spline at specific positions."""
        x = np.linspace(0, 50, 30)
        left_xy = np.column_stack([x, np.ones_like(x) * -1.75])
        right_xy = np.column_stack([x, np.ones_like(x) * 1.75])
        
        fitter = BSplineLaneFitter()
        fitter.fit_lane_pair(left_xy, right_xy)
        
        # Evaluate at new positions
        x_eval = np.array([0, 10, 20, 30, 40, 50])
        left_y, right_y = fitter.evaluate_at(x_eval)
        
        assert left_y is not None
        assert right_y is not None
        assert len(left_y) == len(x_eval)
        assert np.all(left_y < 0)  # Left lane should be negative
        assert np.all(right_y > 0)  # Right lane should be positive


class TestGeometricLaneValidator:
    """Test geometric lane validation."""
    
    def test_valid_lane_pair(self):
        """Test validation of valid lane pair."""
        x = np.linspace(0, 50, 30)
        left_xy = np.column_stack([x, np.ones_like(x) * -1.75])
        right_xy = np.column_stack([x, np.ones_like(x) * 1.75])
        
        validator = GeometricLaneValidator()
        valid, info = validator.validate_lane_pair(left_xy, right_xy)
        
        assert valid == True
        assert info["checks"]["sufficient_points"] == True
        assert info["checks"]["lane_width"] == True
        assert info["checks"]["width_consistent"] == True
        assert info["checks"]["no_crossing"] == True
    
    def test_invalid_width(self):
        """Test detection of invalid lane width."""
        x = np.linspace(0, 50, 30)
        left_xy = np.column_stack([x, np.ones_like(x) * -0.5])  # Too narrow
        right_xy = np.column_stack([x, np.ones_like(x) * 0.5])
        
        validator = GeometricLaneValidator()
        valid, info = validator.validate_lane_pair(left_xy, right_xy)
        
        assert valid is False
        assert info["checks"]["lane_width"] is False
        assert info["avg_width_m"] < 2.5
    
    def test_crossing_lanes(self):
        """Test detection of crossing lanes."""
        x = np.linspace(0, 50, 30)
        # Lanes that cross each other
        left_xy = np.column_stack([x, -1.75 + 0.1 * x])
        right_xy = np.column_stack([x, 1.75 - 0.1 * x])
        
        validator = GeometricLaneValidator()
        valid, info = validator.validate_lane_pair(left_xy, right_xy)
        
        # Should detect crossing at some point
        assert info["checks"]["width_consistent"] is False
    
    def test_insufficient_points(self):
        """Test handling of insufficient points."""
        left_xy = np.array([[0, -1.5]])  # Only 1 point
        right_xy = np.array([[0, 1.5]])
        
        validator = GeometricLaneValidator()
        valid, info = validator.validate_lane_pair(left_xy, right_xy, min_points=10)
        
        assert valid is False
        assert info["checks"]["sufficient_points"] is False


class TestPolynomialConversion:
    """Test B-spline to polynomial conversion."""
    
    def test_conversion(self):
        """Test converting B-spline to polynomial."""
        # Create simple straight lane
        x = np.linspace(0, 50, 30)
        left_xy = np.column_stack([x, np.ones_like(x) * -1.75])
        
        fitter = BSplineLaneFitter()
        left_spline, _, _ = fitter.fit_lane_pair(
            left_xy,
            np.column_stack([x, np.ones_like(x) * 1.75]),
        )
        
        # Convert to polynomial
        coeffs = convert_bspline_to_polynomial(left_spline, (0, 50), poly_order=2)
        
        assert coeffs is not None
        assert len(coeffs) == 3  # 2nd order = 3 coefficients
        
        # For straight line, should be approximately [0, 0, -1.75]
        assert abs(coeffs[-1] - (-1.75)) < 0.5  # Constant term


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_noisy_lane_detection(self):
        """Test fitting with noisy lane points."""
        np.random.seed(42)
        
        # Create lanes with noise
        x = np.linspace(0, 50, 30)
        noise = np.random.normal(0, 0.1, len(x))
        left_xy = np.column_stack([x, -1.75 + noise])
        right_xy = np.column_stack([x, 1.75 + noise])
        
        fitter = BSplineLaneFitter()
        left_spline, right_spline, info = fitter.fit_lane_pair(
            left_xy, right_xy,
            enforce_parallel=True,
            enforce_width=True,
        )
        
        assert info["valid"] is True
        # Constraints should reduce noise effect
        assert info["width_std_m"] < 0.3
    
    def test_sharp_curve(self):
        """Test fitting sharp curve."""
        # Create sharp S-curve
        x = np.linspace(0, 50, 50)
        curvature = 0.05
        left_y = -1.75 + curvature * np.sin(x / 10)
        right_y = 1.75 + curvature * np.sin(x / 10)
        
        left_xy = np.column_stack([x, left_y])
        right_xy = np.column_stack([x, right_y])
        
        fitter = BSplineLaneFitter(degree=3, num_control_points=12)
        left_spline, right_spline, info = fitter.fit_lane_pair(
            left_xy, right_xy,
            enforce_parallel=True,
        )
        
        assert info["valid"] is True
        # B-spline should handle curves well
        assert info["left_rmse"] < 0.5
        assert info["right_rmse"] < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
