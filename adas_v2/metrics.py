"""
Performance Metrics for ADAS v2

Real-time and post-run metrics:
    - CTE statistics (mean, std, RMS, max)
    - Speed tracking accuracy
    - Steering smoothness
    - Mode distribution
    - Lap time (if applicable)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import time


@dataclass
class MetricsSnapshot:
    """Snapshot of current metrics."""
    # CTE metrics
    cte_mean: float = 0.0
    cte_std: float = 0.0
    cte_rms: float = 0.0
    cte_max: float = 0.0
    cte_current: float = 0.0
    
    # Speed metrics
    speed_mean_kmh: float = 0.0
    speed_target_mean_kmh: float = 0.0
    speed_error_mean_kmh: float = 0.0
    speed_current_kmh: float = 0.0
    
    # Steering metrics
    steer_mean: float = 0.0
    steer_std: float = 0.0
    steer_rate_mean: float = 0.0  # Smoothness indicator
    steer_current: float = 0.0
    
    # Heading metrics
    heading_rms_deg: float = 0.0
    heading_current_deg: float = 0.0
    
    # Mode distribution (%)
    mode_lane_pct: float = 0.0
    mode_wp_pct: float = 0.0
    mode_fusion_pct: float = 0.0
    
    # Time
    elapsed_s: float = 0.0
    total_frames: int = 0
    fps: float = 0.0
    
    # Distance
    distance_m: float = 0.0


class PerformanceMetrics:
    """
    Real-time performance metrics tracker.
    
    Computes running statistics for ADAS evaluation.
    """
    
    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self._cte_history: List[float] = []
        self._speed_history: List[float] = []
        self._target_speed_history: List[float] = []
        self._steer_history: List[float] = []
        self._heading_history: List[float] = []
        self._mode_history: List[str] = []
        self._time_history: List[float] = []
        
        self._start_time = time.time()
        self._frame_count = 0
        self._total_distance = 0.0
        self._last_speed = 0.0
        self._last_time = None
        
    def update(
        self,
        cte: float,
        speed_ms: float,
        target_speed_ms: float,
        steer: float,
        heading_rad: float,
        mode: str,
    ):
        """Update metrics with new frame data."""
        current_time = time.time()
        
        # Update distance
        if self._last_time is not None:
            dt = current_time - self._last_time
            avg_speed = (speed_ms + self._last_speed) / 2
            self._total_distance += avg_speed * dt
            
        self._last_speed = speed_ms
        self._last_time = current_time
        
        # Add to history
        self._cte_history.append(cte)
        self._speed_history.append(speed_ms * 3.6)  # Convert to km/h
        self._target_speed_history.append(target_speed_ms * 3.6)
        self._steer_history.append(steer)
        self._heading_history.append(np.degrees(heading_rad))
        self._mode_history.append(mode.split()[0])  # Extract mode name
        self._time_history.append(current_time)
        
        # Trim to window size
        if len(self._cte_history) > self.window_size:
            self._cte_history.pop(0)
            self._speed_history.pop(0)
            self._target_speed_history.pop(0)
            self._steer_history.pop(0)
            self._heading_history.pop(0)
            self._mode_history.pop(0)
            self._time_history.pop(0)
            
        self._frame_count += 1
        
    def get_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        if not self._cte_history:
            return MetricsSnapshot()
            
        cte_arr = np.array(self._cte_history)
        speed_arr = np.array(self._speed_history)
        target_arr = np.array(self._target_speed_history)
        steer_arr = np.array(self._steer_history)
        heading_arr = np.array(self._heading_history)
        
        # Compute steering rate
        steer_rate = 0.0
        if len(steer_arr) > 1:
            steer_diff = np.diff(steer_arr)
            steer_rate = float(np.mean(np.abs(steer_diff)))
            
        # Compute mode distribution
        mode_counts = {}
        for m in self._mode_history:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        total = len(self._mode_history)
        
        # Compute FPS
        elapsed = time.time() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        
        return MetricsSnapshot(
            # CTE
            cte_mean=float(np.mean(cte_arr)),
            cte_std=float(np.std(cte_arr)),
            cte_rms=float(np.sqrt(np.mean(cte_arr**2))),
            cte_max=float(np.max(np.abs(cte_arr))),
            cte_current=float(cte_arr[-1]),
            
            # Speed
            speed_mean_kmh=float(np.mean(speed_arr)),
            speed_target_mean_kmh=float(np.mean(target_arr)),
            speed_error_mean_kmh=float(np.mean(np.abs(speed_arr - target_arr))),
            speed_current_kmh=float(speed_arr[-1]),
            
            # Steering
            steer_mean=float(np.mean(steer_arr)),
            steer_std=float(np.std(steer_arr)),
            steer_rate_mean=steer_rate,
            steer_current=float(steer_arr[-1]),
            
            # Heading
            heading_rms_deg=float(np.sqrt(np.mean(heading_arr**2))),
            heading_current_deg=float(heading_arr[-1]),
            
            # Mode distribution
            mode_lane_pct=mode_counts.get("LANE", 0) / total * 100 if total > 0 else 0,
            mode_wp_pct=mode_counts.get("WP", 0) / total * 100 if total > 0 else 0,
            mode_fusion_pct=mode_counts.get("FUSION", 0) / total * 100 if total > 0 else 0,
            
            # Time
            elapsed_s=elapsed,
            total_frames=self._frame_count,
            fps=fps,
            
            # Distance
            distance_m=self._total_distance,
        )
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary dict for logging/saving."""
        snap = self.get_snapshot()
        return {
            "cte_rms_m": snap.cte_rms,
            "cte_max_m": snap.cte_max,
            "speed_mean_kmh": snap.speed_mean_kmh,
            "speed_error_kmh": snap.speed_error_mean_kmh,
            "steer_smoothness": 1.0 - min(snap.steer_rate_mean * 10, 1.0),  # 0-1, higher is smoother
            "heading_rms_deg": snap.heading_rms_deg,
            "mode_lane_pct": snap.mode_lane_pct,
            "distance_m": snap.distance_m,
            "elapsed_s": snap.elapsed_s,
            "fps": snap.fps,
        }
    
    def print_summary(self):
        """Print formatted summary."""
        snap = self.get_snapshot()
        
        print("\n" + "="*50)
        print("ADAS v2 Performance Summary")
        print("="*50)
        print(f"Duration: {snap.elapsed_s:.1f}s | Frames: {snap.total_frames} | FPS: {snap.fps:.1f}")
        print(f"Distance: {snap.distance_m:.0f}m ({snap.distance_m/1000:.2f}km)")
        print("-"*50)
        print("Lane Keeping:")
        print(f"  CTE RMS: {snap.cte_rms:.3f}m | Max: {snap.cte_max:.3f}m")
        print(f"  Heading RMS: {snap.heading_rms_deg:.2f}°")
        print("-"*50)
        print("Speed Control:")
        print(f"  Mean: {snap.speed_mean_kmh:.1f} km/h | Target: {snap.speed_target_mean_kmh:.1f} km/h")
        print(f"  Error: {snap.speed_error_mean_kmh:.1f} km/h")
        print("-"*50)
        print("Steering:")
        print(f"  Mean: {snap.steer_mean:+.3f} | Std: {snap.steer_std:.3f}")
        print(f"  Smoothness: {(1.0 - min(snap.steer_rate_mean * 10, 1.0))*100:.0f}%")
        print("-"*50)
        print("Mode Distribution:")
        print(f"  LANE: {snap.mode_lane_pct:.1f}% | FUSION: {snap.mode_fusion_pct:.1f}% | WP: {snap.mode_wp_pct:.1f}%")
        print("="*50 + "\n")
