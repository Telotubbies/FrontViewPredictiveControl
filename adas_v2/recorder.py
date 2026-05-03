"""
Recording & Replay System for ADAS v2

Records:
    - Camera frames (compressed video)
    - Control outputs (CSV)
    - Perception state (CSV)
    - Performance metrics

Replay:
    - Load recorded session
    - Visualize with dashboard
    - Analyze performance
"""

import csv
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameRecord:
    """Single frame record."""
    frame_id: int
    timestamp: float
    speed_ms: float
    target_speed_ms: float
    steer: float
    throttle: float
    brake: float
    cte: float
    heading: float
    curvature: float
    lane_confidence: float
    lane_weight: float
    mode: str
    wp_lookahead_curv: float


class ADASRecorder:
    """
    Records ADAS session for replay and analysis.
    
    Saves:
        - video.mp4: Camera feed
        - telemetry.csv: Control and perception data
        - metadata.json: Session info
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        fps: int = 20,
        video_codec: str = "mp4v",
        record_video: bool = True,
    ):
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"recordings/adas_v2_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.video_codec = video_codec
        self.record_video = record_video
        
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._csv_file = None
        self._csv_writer = None
        self._frame_count = 0
        self._start_time = None
        self._records: List[FrameRecord] = []
        
        logger.info(f"Recorder initialized: {self.output_dir}")
        
    def start(self):
        """Start recording session."""
        self._start_time = time.time()
        self._frame_count = 0
        self._records = []
        
        # Open CSV file
        csv_path = self.output_dir / "telemetry.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "frame_id", "timestamp", "speed_ms", "target_speed_ms",
                "steer", "throttle", "brake", "cte", "heading", "curvature",
                "lane_confidence", "lane_weight", "mode", "wp_lookahead_curv"
            ]
        )
        self._csv_writer.writeheader()
        
        logger.info("Recording started")
        
    def record_frame(
        self,
        rgb: np.ndarray,
        speed_ms: float,
        target_speed_ms: float,
        steer: float,
        throttle: float,
        brake: float,
        cte: float,
        heading: float,
        curvature: float,
        lane_confidence: float,
        lane_weight: float,
        mode: str,
        wp_lookahead_curv: float,
    ):
        """Record single frame."""
        if self._start_time is None:
            self.start()
            
        timestamp = time.time() - self._start_time
        
        # Create record
        record = FrameRecord(
            frame_id=self._frame_count,
            timestamp=timestamp,
            speed_ms=speed_ms,
            target_speed_ms=target_speed_ms,
            steer=steer,
            throttle=throttle,
            brake=brake,
            cte=cte,
            heading=heading,
            curvature=curvature,
            lane_confidence=lane_confidence,
            lane_weight=lane_weight,
            mode=mode,
            wp_lookahead_curv=wp_lookahead_curv,
        )
        
        # Write to CSV
        self._csv_writer.writerow(asdict(record))
        self._records.append(record)
        
        # Write video frame
        if self.record_video and rgb is not None:
            if self._video_writer is None:
                h, w = rgb.shape[:2]
                video_path = str(self.output_dir / "video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
                self._video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
            
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self._video_writer.write(bgr)
        
        self._frame_count += 1
        
    def stop(self) -> Dict[str, Any]:
        """Stop recording and return summary."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
            
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Save metadata
        metadata = {
            "start_time": datetime.now().isoformat(),
            "duration_s": time.time() - self._start_time if self._start_time else 0,
            "total_frames": self._frame_count,
            "fps": self.fps,
            "metrics": metrics,
        }
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Recording stopped: {self._frame_count} frames, {metadata['duration_s']:.1f}s")
        logger.info(f"Saved to: {self.output_dir}")
        
        return metrics
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute performance metrics from recorded data."""
        if not self._records:
            return {}
            
        ctes = [r.cte for r in self._records]
        speeds = [r.speed_ms for r in self._records]
        headings = [r.heading for r in self._records]
        
        return {
            "cte_mean": float(np.mean(ctes)),
            "cte_std": float(np.std(ctes)),
            "cte_max": float(np.max(np.abs(ctes))),
            "cte_rms": float(np.sqrt(np.mean(np.array(ctes)**2))),
            "speed_mean_kmh": float(np.mean(speeds) * 3.6),
            "speed_std_kmh": float(np.std(speeds) * 3.6),
            "heading_rms_deg": float(np.sqrt(np.mean(np.array(headings)**2)) * 180 / np.pi),
            "lane_mode_pct": self._mode_percentages(),
        }
    
    def _mode_percentages(self) -> Dict[str, float]:
        """Compute percentage of time in each mode."""
        if not self._records:
            return {}
            
        mode_counts: Dict[str, int] = {}
        for r in self._records:
            mode = r.mode.split()[0]  # Extract mode name
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
        total = len(self._records)
        return {k: v / total * 100 for k, v in mode_counts.items()}


class ADASReplayer:
    """
    Replays recorded ADAS session.
    
    Loads telemetry and video for analysis.
    """
    
    def __init__(self, recording_dir: str):
        self.recording_dir = Path(recording_dir)
        self._records: List[FrameRecord] = []
        self._metadata: Dict[str, Any] = {}
        self._video_cap: Optional[cv2.VideoCapture] = None
        
    def load(self) -> bool:
        """Load recording."""
        # Load metadata
        metadata_path = self.recording_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)
                
        # Load telemetry
        csv_path = self.recording_dir / "telemetry.csv"
        if not csv_path.exists():
            logger.error(f"Telemetry not found: {csv_path}")
            return False
            
        self._records = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._records.append(FrameRecord(
                    frame_id=int(row["frame_id"]),
                    timestamp=float(row["timestamp"]),
                    speed_ms=float(row["speed_ms"]),
                    target_speed_ms=float(row["target_speed_ms"]),
                    steer=float(row["steer"]),
                    throttle=float(row["throttle"]),
                    brake=float(row["brake"]),
                    cte=float(row["cte"]),
                    heading=float(row["heading"]),
                    curvature=float(row["curvature"]),
                    lane_confidence=float(row["lane_confidence"]),
                    lane_weight=float(row["lane_weight"]),
                    mode=row["mode"],
                    wp_lookahead_curv=float(row["wp_lookahead_curv"]),
                ))
                
        # Open video
        video_path = self.recording_dir / "video.mp4"
        if video_path.exists():
            self._video_cap = cv2.VideoCapture(str(video_path))
            
        logger.info(f"Loaded {len(self._records)} frames from {self.recording_dir}")
        return True
    
    @property
    def records(self) -> List[FrameRecord]:
        return self._records
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata
    
    @property
    def metrics(self) -> Dict[str, float]:
        return self._metadata.get("metrics", {})
    
    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get video frame by ID."""
        if self._video_cap is None:
            return None
            
        self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self._video_cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def close(self):
        """Close video capture."""
        if self._video_cap:
            self._video_cap.release()
            self._video_cap = None
