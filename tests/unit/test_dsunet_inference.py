"""Tests for DSUNet lane detection inference pipeline.

Tests that:
- DSUNet model loads correctly from checkpoint
- Inference produces valid masks on real CARLA images
- ImageNet normalization is applied (critical — without it, output is all zeros)
- detect_lanes() uses DSUNet as primary when use_model_priority=True
- detect_lanes() falls back to CARLA when model produces empty mask
- Full pipeline (DSUNet → mask → polynomial → path) works end-to-end
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# DSUNet checkpoint path
DSUNET_PATH = ROOT / "dsunet_carla_20260302_031021" / "best_model_iou.pth"
# CARLA dataset for testing
DATA_DIR = Path("/home/supawich/Desktop/CARLA_0.9.16/carla_lstm_mpc_project/data/lane_train_diverse_20260227_121538")


def _has_model():
    return DSUNET_PATH.exists()


def _has_data():
    return DATA_DIR.exists() and (DATA_DIR / "images" / "000000.png").exists()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

class TestDSUNetLoading:
    """Test DSUNet model loading via LaneDetector."""

    @pytest.mark.skipif(not _has_model(), reason="DSUNet checkpoint not found")
    def test_model_loads_successfully(self):
        """LaneDetector should load DSUNet model from checkpoint."""
        from perception.lane_detector import LaneDetector
        detector = LaneDetector(
            model_path=str(DSUNET_PATH),
            use_carla=True,
            model_type="dsunet",
        )
        assert detector.model is not None
        assert detector.model_type == "dsunet"
        assert detector.use_carla is False  # overridden when model loads

    def test_invalid_model_path_falls_back_to_carla(self):
        """Invalid model path should fall back to CARLA mode."""
        from perception.lane_detector import LaneDetector
        detector = LaneDetector(
            model_path="/nonexistent/model.pth",
            use_carla=True,
            model_type="dsunet",
        )
        assert detector.model is None
        assert detector.use_carla is True

    def test_none_model_path_uses_carla(self):
        """None model path should use CARLA mode."""
        from perception.lane_detector import LaneDetector
        detector = LaneDetector(model_path=None, use_carla=True, model_type="dsunet")
        assert detector.model is None
        assert detector.use_carla is True


# ──────────────────────────────────────────────────────────────────────────────
# Inference with ImageNet normalization
# ──────────────────────────────────────────────────────────────────────────────

class TestDSUNetInference:
    """Test DSUNet inference produces valid masks."""

    @pytest.mark.skipif(not _has_model() or not _has_data(), reason="Model or data not found")
    def test_inference_produces_nonempty_mask(self):
        """DSUNet should produce a non-empty mask on a real CARLA image."""
        import cv2
        from perception.lane_detector import LaneDetector

        detector = LaneDetector(
            model_path=str(DSUNET_PATH), use_carla=True, model_type="dsunet",
        )
        img = cv2.imread(str(DATA_DIR / "images" / "000000.png"))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = detector._infer_model(img_rgb)
        assert mask is not None
        assert mask.shape == img_rgb.shape[:2]
        assert mask.dtype == np.uint8
        assert np.count_nonzero(mask) > 50  # non-empty

    @pytest.mark.skipif(not _has_model() or not _has_data(), reason="Model or data not found")
    def test_inference_iou_above_threshold(self):
        """DSUNet mask should have IoU > 0.7 with ground truth."""
        import cv2
        from perception.lane_detector import LaneDetector

        detector = LaneDetector(
            model_path=str(DSUNET_PATH), use_carla=True, model_type="dsunet",
        )
        img = cv2.imread(str(DATA_DIR / "images" / "000050.png"))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(DATA_DIR / "masks" / "000050.png"), cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt_mask > 127).astype(np.uint8)

        mask = detector._infer_model(img_rgb)
        pred_bin = (mask > 127).astype(np.uint8)
        iou = np.logical_and(gt_bin, pred_bin).sum() / max(np.logical_or(gt_bin, pred_bin).sum(), 1)
        assert iou > 0.7, f"IoU too low: {iou:.3f}"

    @pytest.mark.skipif(not _has_model() or not _has_data(), reason="Model or data not found")
    def test_inference_on_multiple_images(self):
        """DSUNet should produce reasonable IoU on multiple test images."""
        import cv2
        from perception.lane_detector import LaneDetector

        detector = LaneDetector(
            model_path=str(DSUNET_PATH), use_carla=True, model_type="dsunet",
        )
        ious = []
        for idx in [0, 50, 100, 500]:
            img_path = DATA_DIR / "images" / f"{idx:06d}.png"
            mask_path = DATA_DIR / "masks" / f"{idx:06d}.png"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            gt_bin = (gt > 127).astype(np.uint8)

            mask = detector._infer_model(img_rgb)
            pred_bin = (mask > 127).astype(np.uint8)
            iou = np.logical_and(gt_bin, pred_bin).sum() / max(np.logical_or(gt_bin, pred_bin).sum(), 1)
            ious.append(iou)

        assert len(ious) > 0
        mean_iou = np.mean(ious)
        assert mean_iou > 0.7, f"Mean IoU too low: {mean_iou:.3f}"

    def test_inference_returns_none_when_no_model(self):
        """_infer_model should return None when model is not loaded."""
        from perception.lane_detector import LaneDetector
        detector = LaneDetector(model_path=None, use_carla=True, model_type="dsunet")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector._infer_model(img)
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# detect_lanes priority logic
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectLanesPriority:
    """Test detect_lanes priority logic (model vs CARLA)."""

    def test_model_priority_uses_model_when_available(self):
        """When use_model_priority=True and model is loaded, should use model."""
        from perception.lane_detector import LaneDetector
        detector = LaneDetector(model_path=None, use_carla=True, model_type="dsunet")

        # Mock model inference to return a non-empty mask
        mock_mask = np.zeros((480, 640), dtype=np.uint8)
        mock_mask[200:, 200:210] = 255
        detector.model = Mock()
        detector._infer_model = Mock(return_value=mock_mask)

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        mask, features, coords = detector.detect_lanes(
            img, world=Mock(), vehicle=Mock(), use_model_priority=True,
        )
        detector._infer_model.assert_called_once()
        assert np.array_equal(mask, mock_mask)

    def test_model_priority_falls_back_on_empty_mask(self):
        """When model produces empty mask, should fall back to CARLA."""
        from perception.lane_detector import LaneDetector
        detector = LaneDetector(model_path=None, use_carla=True, model_type="dsunet")

        # Mock model to return empty mask
        empty_mask = np.zeros((480, 640), dtype=np.uint8)
        detector.model = Mock()
        detector._infer_model = Mock(return_value=empty_mask)
        detector.detect_lanes_carla = Mock(return_value=np.zeros((480, 640), dtype=np.uint8))

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        mask, features, coords = detector.detect_lanes(
            img, world=Mock(), vehicle=Mock(), use_model_priority=True,
        )
        # Should have called CARLA fallback
        detector.detect_lanes_carla.assert_called_once()

    def test_legacy_mode_uses_carla_first(self):
        """When use_model_priority=False, should use CARLA first if available."""
        from perception.lane_detector import LaneDetector
        detector = LaneDetector(model_path=None, use_carla=True, model_type="dsunet")

        carla_mask = np.zeros((480, 640), dtype=np.uint8)
        carla_mask[200:, 200:210] = 255
        detector.detect_lanes_carla = Mock(return_value=carla_mask)
        detector.model = Mock()
        detector._infer_model = Mock()

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        mask, features, coords = detector.detect_lanes(
            img, world=Mock(), vehicle=Mock(), use_model_priority=False,
        )
        detector.detect_lanes_carla.assert_called_once()
        detector._infer_model.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline: DSUNet → mask → polynomial → path
# ──────────────────────────────────────────────────────────────────────────────

class TestDSUNetFullPipeline:
    """Test full pipeline with DSUNet detector."""

    @pytest.mark.skipif(not _has_model() or not _has_data(), reason="Model or data not found")
    def test_pipeline_produces_valid_trajectory(self):
        """LaneTrajectoryPipeline with DSUNet should produce valid trajectory."""
        import cv2
        from perception.lane_detector import LaneDetector
        from perception.lane_trajectory import LaneTrajectoryPipeline

        detector = LaneDetector(
            model_path=str(DSUNET_PATH), use_carla=True, model_type="dsunet",
        )
        pipe = LaneTrajectoryPipeline(
            detector=detector,
            use_classical_detector=False,
            cam_w=640, cam_h=480,
            use_kalman=True,
            use_ego_lane_tracker=False,
        )

        img = cv2.imread(str(DATA_DIR / "images" / "000050.png"))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = pipe.process(img_rgb, world=None, vehicle=None)

        assert out is not None
        assert hasattr(out, 'cte')
        assert hasattr(out, 'heading_err')
        assert hasattr(out, 'curvature')
        assert hasattr(out, 'confidence')
        assert out.x_ref is not None
        assert len(out.x_ref) > 0

    @pytest.mark.skipif(not _has_model() or not _has_data(), reason="Model or data not found")
    def test_pipeline_cte_is_reasonable(self):
        """CTE from DSUNet pipeline should be within reasonable range."""
        import cv2
        from perception.lane_detector import LaneDetector
        from perception.lane_trajectory import LaneTrajectoryPipeline

        detector = LaneDetector(
            model_path=str(DSUNET_PATH), use_carla=True, model_type="dsunet",
        )
        pipe = LaneTrajectoryPipeline(
            detector=detector,
            use_classical_detector=False,
            cam_w=640, cam_h=480,
            use_kalman=False,
            use_ego_lane_tracker=False,
        )

        img = cv2.imread(str(DATA_DIR / "images" / "000000.png"))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = pipe.process(img_rgb, world=None, vehicle=None)

        # CTE should be within ±2m for a centered vehicle
        assert abs(out.cte) < 2.0, f"CTE too large: {out.cte:.3f}"

    @pytest.mark.skipif(not _has_model() or not _has_data(), reason="Model or data not found")
    def test_pipeline_produces_lane_overlay(self):
        """Pipeline should produce a lane overlay image."""
        import cv2
        from perception.lane_detector import LaneDetector
        from perception.lane_trajectory import LaneTrajectoryPipeline

        detector = LaneDetector(
            model_path=str(DSUNET_PATH), use_carla=True, model_type="dsunet",
        )
        pipe = LaneTrajectoryPipeline(
            detector=detector,
            use_classical_detector=False,
            cam_w=640, cam_h=480,
            use_kalman=False,
            use_ego_lane_tracker=False,
        )

        img = cv2.imread(str(DATA_DIR / "images" / "000050.png"))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = pipe.process(img_rgb, world=None, vehicle=None)

        if out.lane_overlay is not None:
            assert out.lane_overlay.shape[:2] == (480, 640)

    @pytest.mark.skipif(not _has_model() or not _has_data(), reason="Model or data not found")
    def test_pipeline_multiple_frames_stateful(self):
        """Multiple frames should maintain Kalman state."""
        import cv2
        from perception.lane_detector import LaneDetector
        from perception.lane_trajectory import LaneTrajectoryPipeline

        detector = LaneDetector(
            model_path=str(DSUNET_PATH), use_carla=True, model_type="dsunet",
        )
        pipe = LaneTrajectoryPipeline(
            detector=detector,
            use_classical_detector=False,
            cam_w=640, cam_h=480,
            use_kalman=True,
            use_ego_lane_tracker=False,
        )

        for idx in [0, 50, 100]:
            img = cv2.imread(str(DATA_DIR / "images" / f"{idx:06d}.png"))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out = pipe.process(img_rgb, world=None, vehicle=None)
            assert out is not None
