"""
DSUNet Driving Evaluator — real-time evaluation of lane detection + driving.

Measures DSUNet performance while the car is actually driving in CARLA:
- Lane detection: IoU, precision, recall vs CARLA ground truth mask
- Driving quality: CTE, heading error, lane departures, speed tracking
- Comparison: DSUNet-only vs DSUNet+CARLA fallback vs CARLA-only

Usage:
    from evaluation.dsunet_driving_eval import DSUNetDrivingEvaluator

    evaluator = DSUNetDrivingEvaluator(
        model_path="dsunet_carla_20260302_031021/best_model_iou.pth",
        world=world, vehicle=vehicle,
    )
    evaluator.run(duration_s=60)
    report = evaluator.generate_report()
"""
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    frame_idx: int
    timestamp: float

    # Lane detection metrics
    dsunet_iou: float = 0.0
    dsunet_precision: float = 0.0
    dsunet_recall: float = 0.0
    dsunet_mask_pixels: int = 0
    gt_mask_pixels: int = 0
    dsunet_used: bool = True  # True if DSUNet was primary, False if CARLA fallback

    # Driving metrics
    cte_m: float = 0.0
    heading_error_rad: float = 0.0
    curvature: float = 0.0
    confidence: float = 0.0
    speed_ms: float = 0.0
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    # Lane departure
    lane_departure: bool = False

    # Mode
    mode: str = "unknown"  # "dsunet" | "carla_fallback" | "hybrid"


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""
    # Summary statistics
    total_frames: int = 0
    duration_s: float = 0.0
    distance_m: float = 0.0

    # Lane detection (averaged)
    mean_iou: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    iou_std: float = 0.0
    dsunet_usage_rate: float = 0.0  # fraction of frames DSUNet was primary

    # Driving quality
    mean_cte: float = 0.0
    max_cte: float = 0.0
    cte_std: float = 0.0
    mean_heading_error: float = 0.0
    max_heading_error: float = 0.0
    mean_speed: float = 0.0
    speed_tracking_rmse: float = 0.0

    # Lane departures
    lane_departure_count: int = 0
    lane_departure_rate: float = 0.0  # departures per km

    # Completion
    completed: bool = False
    completion_reason: str = ""

    # Per-frame data (for plotting)
    frame_metrics: List[FrameMetrics] = field(default_factory=list)


class DSUNetDrivingEvaluator:
    """Real-time DSUNet evaluation during driving.

    Runs the car with DSUNet as primary lane detector and measures:
    1. Lane mask quality (IoU vs CARLA ground truth)
    2. Driving performance (CTE, heading, speed)
    3. Lane departures
    4. DSUNet vs CARLA fallback usage
    """

    def __init__(
        self,
        model_path: str,
        world,
        vehicle,
        cam_w: int = 640,
        cam_h: int = 480,
        target_speed_kmh: float = 30.0,
        lane_departure_threshold_m: float = 0.9,
    ):
        """Initialize the evaluator.

        Args:
            model_path: Path to DSUNet checkpoint
            world: carla.World
            vehicle: carla.Vehicle
            cam_w, cam_h: Camera resolution
            target_speed_kmh: Target speed for speed tracking metric
            lane_departure_threshold_m: CTE threshold for lane departure
        """
        self.world = world
        self.vehicle = vehicle
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.target_speed_ms = target_speed_kmh / 3.6
        self.lane_departure_threshold = lane_departure_threshold_m

        # Load DSUNet detector
        from perception.lane_detector import LaneDetector
        self.detector = LaneDetector(
            model_path=model_path,
            use_carla=True,
            model_type="dsunet",
        )
        self.dsunet_available = self.detector.model is not None

        # Load trajectory pipeline
        from perception.lane_trajectory import LaneTrajectoryPipeline
        self.pipeline = LaneTrajectoryPipeline(
            detector=self.detector,
            use_classical_detector=not self.dsunet_available,
            cam_w=cam_w, cam_h=cam_h,
            use_kalman=True,
            use_ego_lane_tracker=False,
        )

        # Metrics storage
        self.frame_metrics: List[FrameMetrics] = []
        self._start_time = 0.0
        self._start_location = None
        self._prev_location = None

        logger.info("DSUNetDrivingEvaluator initialized (DSUNet available: %s)",
                    self.dsunet_available)

    def evaluate_frame(
        self,
        rgb: np.ndarray,
        frame_idx: int,
        steer: float = 0.0,
        throttle: float = 0.0,
        brake: float = 0.0,
    ) -> FrameMetrics:
        """Evaluate a single frame.

        Args:
            rgb: RGB image from camera (H, W, 3)
            frame_idx: Frame index
            steer, throttle, brake: Current control commands

        Returns:
            FrameMetrics for this frame
        """
        timestamp = time.time()

        # 1. Run DSUNet inference
        dsunet_mask = self.detector._infer_model(rgb)
        dsunet_used = dsunet_mask is not None and np.count_nonzero(dsunet_mask) > 50

        # 2. Get CARLA ground truth mask
        gt_mask = self._get_carla_gt_mask(rgb.shape[:2])

        # 3. Compute IoU
        if dsunet_mask is not None and gt_mask is not None:
            iou, precision, recall = self._compute_iou(dsunet_mask, gt_mask)
        else:
            iou, precision, recall = 0.0, 0.0, 0.0

        # 4. Run trajectory pipeline for CTE/heading
        try:
            out = self.pipeline.process(rgb, world=self.world, vehicle=self.vehicle)
            cte = float(out.cte)
            heading = float(out.heading_err)
            curvature = float(out.curvature)
            confidence = float(out.confidence)
        except Exception as e:
            logger.debug("Pipeline error: %s", e)
            cte, heading, curvature, confidence = 0.0, 0.0, 0.0, 0.0

        # 5. Get vehicle speed
        velocity = self.vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        # 6. Check lane departure
        lane_departure = abs(cte) > self.lane_departure_threshold

        # 7. Determine mode
        if dsunet_used:
            mode = "dsunet"
        elif gt_mask is not None:
            mode = "carla_fallback"
        else:
            mode = "unknown"

        # 8. Create metrics
        metrics = FrameMetrics(
            frame_idx=frame_idx,
            timestamp=timestamp,
            dsunet_iou=iou,
            dsunet_precision=precision,
            dsunet_recall=recall,
            dsunet_mask_pixels=int(np.count_nonzero(dsunet_mask)) if dsunet_mask is not None else 0,
            gt_mask_pixels=int(np.count_nonzero(gt_mask)) if gt_mask is not None else 0,
            dsunet_used=dsunet_used,
            cte_m=cte,
            heading_error_rad=heading,
            curvature=curvature,
            confidence=confidence,
            speed_ms=speed_ms,
            steer=steer,
            throttle=throttle,
            brake=brake,
            lane_departure=lane_departure,
            mode=mode,
        )

        self.frame_metrics.append(metrics)
        return metrics

    def _get_carla_gt_mask(self, img_shape) -> Optional[np.ndarray]:
        """Get CARLA ground truth lane mask for current vehicle position."""
        try:
            from perception.lane_detector import LaneDetector
            # Use CARLA waypoint detection for ground truth
            mask = self.detector.detect_lanes_carla(
                np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8),
                self.world, self.vehicle,
            )
            return mask
        except Exception as e:
            logger.debug("CARLA GT mask error: %s", e)
            return None

    def _compute_iou(self, pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
        """Compute IoU, precision, recall between predicted and GT masks."""
        pred_bin = (pred > 127).astype(np.uint8)
        gt_bin = (gt > 127).astype(np.uint8)

        tp = np.logical_and(pred_bin, gt_bin).sum()
        fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
        fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum()

        iou = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        return float(iou), float(precision), float(recall)

    def generate_report(self) -> EvaluationReport:
        """Generate aggregated evaluation report."""
        if not self.frame_metrics:
            return EvaluationReport()

        report = EvaluationReport()
        report.frame_metrics = self.frame_metrics.copy()
        report.total_frames = len(self.frame_metrics)
        report.duration_s = self.frame_metrics[-1].timestamp - self.frame_metrics[0].timestamp

        # Distance traveled
        if self._start_location and self._prev_location:
            try:
                report.distance_m = self._start_location.distance(self._prev_location)
            except Exception:
                pass

        # Lane detection metrics
        ious = [m.dsunet_iou for m in self.frame_metrics if m.dsunet_used]
        precisions = [m.dsunet_precision for m in self.frame_metrics if m.dsunet_used]
        recalls = [m.dsunet_recall for m in self.frame_metrics if m.dsunet_used]

        if ious:
            report.mean_iou = float(np.mean(ious))
            report.iou_std = float(np.std(ious))
            report.mean_precision = float(np.mean(precisions))
            report.mean_recall = float(np.mean(recalls))

        # DSUNet usage rate
        dsunet_frames = sum(1 for m in self.frame_metrics if m.dsunet_used)
        report.dsunet_usage_rate = dsunet_frames / max(report.total_frames, 1)

        # Driving metrics
        ctes = [m.cte_m for m in self.frame_metrics]
        headings = [abs(m.heading_error_rad) for m in self.frame_metrics]
        speeds = [m.speed_ms for m in self.frame_metrics]

        report.mean_cte = float(np.mean(ctes))
        report.max_cte = float(np.max(np.abs(ctes)))
        report.cte_std = float(np.std(ctes))
        report.mean_heading_error = float(np.mean(headings))
        report.max_heading_error = float(np.max(headings))
        report.mean_speed = float(np.mean(speeds))

        # Speed tracking RMSE
        speed_errors = [(s - self.target_speed_ms) for s in speeds]
        report.speed_tracking_rmse = float(np.sqrt(np.mean(np.square(speed_errors))))

        # Lane departures
        report.lane_departure_count = sum(1 for m in self.frame_metrics if m.lane_departure)
        if report.distance_m > 0:
            report.lane_departure_rate = report.lane_departure_count / (report.distance_m / 1000.0)

        return report

    def print_summary(self, report: EvaluationReport):
        """Print evaluation summary to console."""
        print("\n" + "=" * 70)
        print("  DSUNet Driving Evaluation Report")
        print("=" * 70)
        print(f"  Frames: {report.total_frames}  Duration: {report.duration_s:.1f}s  "
              f"Distance: {report.distance_m:.1f}m")
        print()
        print("  Lane Detection (DSUNet vs CARLA GT):")
        print(f"    Mean IoU:       {report.mean_iou:.3f} ± {report.iou_std:.3f}")
        print(f"    Mean Precision: {report.mean_precision:.3f}")
        print(f"    Mean Recall:    {report.mean_recall:.3f}")
        print(f"    DSUNet usage:   {report.dsunet_usage_rate*100:.1f}% "
              f"({sum(1 for m in report.frame_metrics if m.dsunet_used)}/{report.total_frames} frames)")
        print()
        print("  Driving Quality:")
        print(f"    Mean CTE:       {report.mean_cte:.3f}m (max: {report.max_cte:.3f}m)")
        print(f"    CTE std:        {report.cte_std:.3f}m")
        print(f"    Mean heading:   {report.mean_heading_error:.3f}rad "
              f"({math.degrees(report.mean_heading_error):.2f}°)")
        print(f"    Mean speed:     {report.mean_speed:.2f}m/s "
              f"({report.mean_speed*3.6:.1f}km/h)")
        print(f"    Speed RMSE:     {report.speed_tracking_rmse:.2f}m/s")
        print()
        print("  Safety:")
        print(f"    Lane departures: {report.lane_departure_count} "
              f"({report.lane_departure_rate:.2f}/km)")
        print("=" * 70)

    def save_report(self, report: EvaluationReport, output_path: str):
        """Save report to file."""
        import json
        data = {
            "total_frames": report.total_frames,
            "duration_s": report.duration_s,
            "distance_m": report.distance_m,
            "mean_iou": report.mean_iou,
            "iou_std": report.iou_std,
            "mean_precision": report.mean_precision,
            "mean_recall": report.mean_recall,
            "dsunet_usage_rate": report.dsunet_usage_rate,
            "mean_cte": report.mean_cte,
            "max_cte": report.max_cte,
            "cte_std": report.cte_std,
            "mean_heading_error": report.mean_heading_error,
            "mean_speed": report.mean_speed,
            "speed_tracking_rmse": report.speed_tracking_rmse,
            "lane_departure_count": report.lane_departure_count,
            "lane_departure_rate": report.lane_departure_rate,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Report saved to %s", output_path)
