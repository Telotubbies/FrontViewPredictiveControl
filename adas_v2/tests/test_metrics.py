"""Unit tests for PerformanceMetrics."""

import pytest
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adas_v2.metrics import PerformanceMetrics, MetricsSnapshot


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        snap = MetricsSnapshot()
        assert snap.cte_mean == 0.0
        assert snap.cte_rms == 0.0
        assert snap.speed_mean_kmh == 0.0
        assert snap.mode_lane_pct == 0.0
        assert snap.elapsed_s == 0.0
        assert snap.total_frames == 0


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""
    
    def test_init(self):
        """Test initialization."""
        metrics = PerformanceMetrics()
        assert metrics.window_size == 500
        
    def test_init_custom_window(self):
        """Test initialization with custom window size."""
        metrics = PerformanceMetrics(window_size=100)
        assert metrics.window_size == 100
        
    def test_reset(self):
        """Test reset functionality."""
        metrics = PerformanceMetrics()
        metrics.update(cte=1.0, speed_ms=10.0, target_speed_ms=15.0,
                      steer=0.1, heading_rad=0.05, mode="LANE")
        metrics.reset()
        snap = metrics.get_snapshot()
        assert snap.total_frames == 0
        
    def test_update_single_frame(self):
        """Test single frame update."""
        metrics = PerformanceMetrics()
        metrics.update(
            cte=0.5,
            speed_ms=10.0,
            target_speed_ms=15.0,
            steer=0.1,
            heading_rad=0.05,
            mode="LANE",
        )
        snap = metrics.get_snapshot()
        assert snap.total_frames == 1
        assert snap.cte_current == 0.5
        assert snap.speed_current_kmh == pytest.approx(36.0, abs=0.1)
        
    def test_update_multiple_frames(self):
        """Test multiple frame updates."""
        metrics = PerformanceMetrics()
        
        for i in range(10):
            metrics.update(
                cte=0.1 * i,
                speed_ms=10.0,
                target_speed_ms=15.0,
                steer=0.01 * i,
                heading_rad=0.01 * i,
                mode="LANE",
            )
            
        snap = metrics.get_snapshot()
        assert snap.total_frames == 10
        
    def test_cte_statistics(self):
        """Test CTE statistics calculation."""
        metrics = PerformanceMetrics()
        
        # Add known CTE values
        ctes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for cte in ctes:
            metrics.update(cte=cte, speed_ms=10.0, target_speed_ms=10.0,
                          steer=0.0, heading_rad=0.0, mode="LANE")
            
        snap = metrics.get_snapshot()
        
        # Mean should be 0.5
        assert snap.cte_mean == pytest.approx(0.5, abs=0.01)
        # Max should be 1.0
        assert snap.cte_max == pytest.approx(1.0, abs=0.01)
        # RMS should be sqrt(mean(x^2))
        import numpy as np
        expected_rms = np.sqrt(np.mean(np.array(ctes)**2))
        assert snap.cte_rms == pytest.approx(expected_rms, abs=0.01)
        
    def test_speed_statistics(self):
        """Test speed statistics calculation."""
        metrics = PerformanceMetrics()
        
        # Add constant speed
        for _ in range(10):
            metrics.update(cte=0.0, speed_ms=10.0, target_speed_ms=15.0,
                          steer=0.0, heading_rad=0.0, mode="LANE")
            
        snap = metrics.get_snapshot()
        
        assert snap.speed_mean_kmh == pytest.approx(36.0, abs=0.1)
        assert snap.speed_target_mean_kmh == pytest.approx(54.0, abs=0.1)
        assert snap.speed_error_mean_kmh == pytest.approx(18.0, abs=0.1)
        
    def test_mode_distribution(self):
        """Test mode distribution calculation."""
        metrics = PerformanceMetrics()
        
        # 5 LANE, 3 FUSION, 2 WP
        for _ in range(5):
            metrics.update(cte=0.0, speed_ms=10.0, target_speed_ms=10.0,
                          steer=0.0, heading_rad=0.0, mode="LANE")
        for _ in range(3):
            metrics.update(cte=0.0, speed_ms=10.0, target_speed_ms=10.0,
                          steer=0.0, heading_rad=0.0, mode="FUSION")
        for _ in range(2):
            metrics.update(cte=0.0, speed_ms=10.0, target_speed_ms=10.0,
                          steer=0.0, heading_rad=0.0, mode="WP")
            
        snap = metrics.get_snapshot()
        
        assert snap.mode_lane_pct == pytest.approx(50.0, abs=1.0)
        assert snap.mode_fusion_pct == pytest.approx(30.0, abs=1.0)
        assert snap.mode_wp_pct == pytest.approx(20.0, abs=1.0)
        
    def test_steering_smoothness(self):
        """Test steering smoothness calculation."""
        metrics = PerformanceMetrics()
        
        # Constant steering (smooth)
        for _ in range(10):
            metrics.update(cte=0.0, speed_ms=10.0, target_speed_ms=10.0,
                          steer=0.1, heading_rad=0.0, mode="LANE")
            
        snap = metrics.get_snapshot()
        
        # Constant steering should have low rate
        assert snap.steer_rate_mean == pytest.approx(0.0, abs=0.01)
        
    def test_steering_not_smooth(self):
        """Test steering with oscillation."""
        metrics = PerformanceMetrics()
        
        # Oscillating steering (not smooth)
        for i in range(10):
            steer = 0.1 if i % 2 == 0 else -0.1
            metrics.update(cte=0.0, speed_ms=10.0, target_speed_ms=10.0,
                          steer=steer, heading_rad=0.0, mode="LANE")
            
        snap = metrics.get_snapshot()
        
        # Oscillating steering should have high rate
        assert snap.steer_rate_mean > 0.1
        
    def test_distance_calculation(self):
        """Test distance calculation."""
        metrics = PerformanceMetrics()
        
        # Simulate 10 frames at 10 m/s with 0.1s between frames
        for i in range(10):
            metrics.update(cte=0.0, speed_ms=10.0, target_speed_ms=10.0,
                          steer=0.0, heading_rad=0.0, mode="LANE")
            time.sleep(0.01)  # Small delay
            
        snap = metrics.get_snapshot()
        
        # Distance should be positive
        assert snap.distance_m >= 0
        
    def test_fps_calculation(self):
        """Test FPS calculation."""
        metrics = PerformanceMetrics()
        
        for _ in range(10):
            metrics.update(cte=0.0, speed_ms=10.0, target_speed_ms=10.0,
                          steer=0.0, heading_rad=0.0, mode="LANE")
            
        snap = metrics.get_snapshot()
        
        # FPS should be positive
        assert snap.fps > 0
        
    def test_window_trimming(self):
        """Test that history is trimmed to window size."""
        metrics = PerformanceMetrics(window_size=10)
        
        # Add more than window size
        for i in range(20):
            metrics.update(cte=float(i), speed_ms=10.0, target_speed_ms=10.0,
                          steer=0.0, heading_rad=0.0, mode="LANE")
            
        # History should be trimmed
        assert len(metrics._cte_history) <= 10
        
    def test_get_summary(self):
        """Test get_summary method."""
        metrics = PerformanceMetrics()
        
        for _ in range(5):
            metrics.update(cte=0.5, speed_ms=10.0, target_speed_ms=10.0,
                          steer=0.0, heading_rad=0.0, mode="LANE")
            
        summary = metrics.get_summary()
        
        assert "cte_rms_m" in summary
        assert "speed_mean_kmh" in summary
        assert "mode_lane_pct" in summary
        assert "distance_m" in summary
        
    def test_print_summary(self, capsys):
        """Test print_summary method."""
        metrics = PerformanceMetrics()
        
        for _ in range(5):
            metrics.update(cte=0.5, speed_ms=10.0, target_speed_ms=10.0,
                          steer=0.0, heading_rad=0.0, mode="LANE")
            
        metrics.print_summary()
        
        captured = capsys.readouterr()
        assert "ADAS v2 Performance Summary" in captured.out
        assert "CTE RMS" in captured.out
        assert "Mode Distribution" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
