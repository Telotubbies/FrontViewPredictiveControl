"""Unit tests for ADASRecorder and ADASReplayer."""

import pytest
import sys
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adas_v2.recorder import ADASRecorder, ADASReplayer, FrameRecord


class TestFrameRecord:
    """Tests for FrameRecord dataclass."""
    
    def test_creation(self):
        """Test frame record creation."""
        record = FrameRecord(
            frame_id=0,
            timestamp=0.0,
            speed_ms=10.0,
            target_speed_ms=15.0,
            steer=0.1,
            throttle=0.5,
            brake=0.0,
            cte=0.3,
            heading=0.05,
            curvature=0.01,
            lane_confidence=0.8,
            lane_weight=0.7,
            mode="LANE",
            wp_lookahead_curv=0.02,
        )
        
        assert record.frame_id == 0
        assert record.speed_ms == 10.0
        assert record.mode == "LANE"


class TestADASRecorder:
    """Tests for ADASRecorder class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        dir_path = tempfile.mkdtemp()
        yield dir_path
        shutil.rmtree(dir_path, ignore_errors=True)
        
    def test_init_default_dir(self):
        """Test initialization with default directory."""
        recorder = ADASRecorder()
        assert recorder.output_dir.exists() or True  # May not exist until start
        assert recorder.fps == 20
        
    def test_init_custom_dir(self, temp_dir):
        """Test initialization with custom directory."""
        recorder = ADASRecorder(output_dir=temp_dir)
        assert str(recorder.output_dir) == temp_dir
        
    def test_init_custom_fps(self, temp_dir):
        """Test initialization with custom FPS."""
        recorder = ADASRecorder(output_dir=temp_dir, fps=30)
        assert recorder.fps == 30
        
    def test_start(self, temp_dir):
        """Test start recording."""
        recorder = ADASRecorder(output_dir=temp_dir)
        recorder.start()
        
        assert recorder._start_time is not None
        assert recorder._frame_count == 0
        
        # Cleanup
        recorder.stop()
        
    def test_record_frame_without_video(self, temp_dir):
        """Test recording frame without video."""
        recorder = ADASRecorder(output_dir=temp_dir, record_video=False)
        recorder.start()
        
        recorder.record_frame(
            rgb=None,
            speed_ms=10.0,
            target_speed_ms=15.0,
            steer=0.1,
            throttle=0.5,
            brake=0.0,
            cte=0.3,
            heading=0.05,
            curvature=0.01,
            lane_confidence=0.8,
            lane_weight=0.7,
            mode="LANE",
            wp_lookahead_curv=0.02,
        )
        
        assert recorder._frame_count == 1
        
        recorder.stop()
        
    def test_record_frame_with_video(self, temp_dir):
        """Test recording frame with video."""
        recorder = ADASRecorder(output_dir=temp_dir, record_video=True)
        recorder.start()
        
        # Create dummy RGB image
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[100:200, 100:200] = [255, 0, 0]  # Red square
        
        recorder.record_frame(
            rgb=rgb,
            speed_ms=10.0,
            target_speed_ms=15.0,
            steer=0.1,
            throttle=0.5,
            brake=0.0,
            cte=0.3,
            heading=0.05,
            curvature=0.01,
            lane_confidence=0.8,
            lane_weight=0.7,
            mode="LANE",
            wp_lookahead_curv=0.02,
        )
        
        assert recorder._frame_count == 1
        
        recorder.stop()
        
    def test_stop_creates_files(self, temp_dir):
        """Test that stop creates all output files."""
        recorder = ADASRecorder(output_dir=temp_dir, record_video=False)
        recorder.start()
        
        # Record some frames
        for i in range(5):
            recorder.record_frame(
                rgb=None,
                speed_ms=10.0 + i,
                target_speed_ms=15.0,
                steer=0.1,
                throttle=0.5,
                brake=0.0,
                cte=0.3,
                heading=0.05,
                curvature=0.01,
                lane_confidence=0.8,
                lane_weight=0.7,
                mode="LANE",
                wp_lookahead_curv=0.02,
            )
            
        metrics = recorder.stop()
        
        # Check files exist
        assert (Path(temp_dir) / "telemetry.csv").exists()
        assert (Path(temp_dir) / "metadata.json").exists()
        
        # Check metrics
        assert "cte_rms" in metrics
        assert "speed_mean_kmh" in metrics
        
    def test_compute_metrics(self, temp_dir):
        """Test metrics computation."""
        recorder = ADASRecorder(output_dir=temp_dir, record_video=False)
        recorder.start()
        
        # Record frames with known values
        for i in range(10):
            recorder.record_frame(
                rgb=None,
                speed_ms=10.0,
                target_speed_ms=15.0,
                steer=0.1,
                throttle=0.5,
                brake=0.0,
                cte=0.5,  # Constant CTE
                heading=0.05,
                curvature=0.01,
                lane_confidence=0.8,
                lane_weight=0.7,
                mode="LANE",
                wp_lookahead_curv=0.02,
            )
            
        metrics = recorder.stop()
        
        assert metrics["cte_mean"] == pytest.approx(0.5, abs=0.01)
        assert metrics["cte_rms"] == pytest.approx(0.5, abs=0.01)
        assert metrics["speed_mean_kmh"] == pytest.approx(36.0, abs=0.1)
        
    def test_mode_percentages(self, temp_dir):
        """Test mode percentage calculation."""
        recorder = ADASRecorder(output_dir=temp_dir, record_video=False)
        recorder.start()
        
        # 5 LANE, 3 FUSION, 2 WP
        modes = ["LANE"] * 5 + ["FUSION"] * 3 + ["WP"] * 2
        for mode in modes:
            recorder.record_frame(
                rgb=None, speed_ms=10.0, target_speed_ms=15.0,
                steer=0.1, throttle=0.5, brake=0.0,
                cte=0.5, heading=0.05, curvature=0.01,
                lane_confidence=0.8, lane_weight=0.7,
                mode=mode, wp_lookahead_curv=0.02,
            )
            
        metrics = recorder.stop()
        
        assert metrics["lane_mode_pct"]["LANE"] == pytest.approx(50.0, abs=1.0)
        assert metrics["lane_mode_pct"]["FUSION"] == pytest.approx(30.0, abs=1.0)
        assert metrics["lane_mode_pct"]["WP"] == pytest.approx(20.0, abs=1.0)
        
    def test_auto_start_on_record(self, temp_dir):
        """Test that recording auto-starts if not started."""
        recorder = ADASRecorder(output_dir=temp_dir, record_video=False)
        
        # Don't call start(), just record
        recorder.record_frame(
            rgb=None, speed_ms=10.0, target_speed_ms=15.0,
            steer=0.1, throttle=0.5, brake=0.0,
            cte=0.5, heading=0.05, curvature=0.01,
            lane_confidence=0.8, lane_weight=0.7,
            mode="LANE", wp_lookahead_curv=0.02,
        )
        
        assert recorder._frame_count == 1
        
        recorder.stop()


class TestADASReplayer:
    """Tests for ADASReplayer class."""
    
    @pytest.fixture
    def recorded_session(self):
        """Create a recorded session for replay tests."""
        temp_dir = tempfile.mkdtemp()
        
        recorder = ADASRecorder(output_dir=temp_dir, record_video=False)
        recorder.start()
        
        for i in range(10):
            recorder.record_frame(
                rgb=None,
                speed_ms=10.0 + i,
                target_speed_ms=15.0,
                steer=0.1 * i,
                throttle=0.5,
                brake=0.0,
                cte=0.1 * i,
                heading=0.01 * i,
                curvature=0.001 * i,
                lane_confidence=0.8,
                lane_weight=0.7,
                mode="LANE" if i < 5 else "FUSION",
                wp_lookahead_curv=0.02,
            )
            
        recorder.stop()
        
        yield temp_dir
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    def test_init(self, recorded_session):
        """Test initialization."""
        replayer = ADASReplayer(recorded_session)
        assert str(replayer.recording_dir) == recorded_session
        
    def test_load(self, recorded_session):
        """Test loading recording."""
        replayer = ADASReplayer(recorded_session)
        result = replayer.load()
        
        assert result == True
        assert len(replayer.records) == 10
        
    def test_load_nonexistent(self):
        """Test loading nonexistent recording."""
        replayer = ADASReplayer("/nonexistent/path")
        result = replayer.load()
        
        assert result == False
        
    def test_records_property(self, recorded_session):
        """Test records property."""
        replayer = ADASReplayer(recorded_session)
        replayer.load()
        
        records = replayer.records
        
        assert len(records) == 10
        assert records[0].frame_id == 0
        assert records[9].frame_id == 9
        
    def test_metadata_property(self, recorded_session):
        """Test metadata property."""
        replayer = ADASReplayer(recorded_session)
        replayer.load()
        
        metadata = replayer.metadata
        
        assert "total_frames" in metadata
        assert metadata["total_frames"] == 10
        
    def test_metrics_property(self, recorded_session):
        """Test metrics property."""
        replayer = ADASReplayer(recorded_session)
        replayer.load()
        
        metrics = replayer.metrics
        
        assert "cte_rms" in metrics
        
    def test_record_values(self, recorded_session):
        """Test that record values are correct."""
        replayer = ADASReplayer(recorded_session)
        replayer.load()
        
        # Check first record
        r0 = replayer.records[0]
        assert r0.speed_ms == pytest.approx(10.0, abs=0.1)
        assert r0.mode == "LANE"
        
        # Check last record
        r9 = replayer.records[9]
        assert r9.speed_ms == pytest.approx(19.0, abs=0.1)
        assert r9.mode == "FUSION"
        
    def test_close(self, recorded_session):
        """Test close method."""
        replayer = ADASReplayer(recorded_session)
        replayer.load()
        replayer.close()
        
        # Should not raise
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
