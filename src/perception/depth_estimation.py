"""
Monocular Depth Estimation module using Depth Anything V2.

Provides depth maps from a single RGB camera image, replacing the flat-ground
assumption in BEV projection. This enables more accurate ground-plane projection
on slopes, hills, and non-flat terrain.

Usage:
    from perception.depth_estimation import DepthEstimator

    estimator = DepthEstimator(model_size="small", device="cuda")
    depth_map = estimator.estimate(rgb_image)  # (H, W) float32 in meters

If the model is unavailable or fails, falls back to flat-ground depth (None),
allowing the pipeline to degrade gracefully.
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2.

    Args:
        model_size: "small" (25M), "base" (97M), or "large" (335M)
        device: "cuda", "cpu", or "auto" (detect best available)
        max_resolution: Max input resolution for inference (downscale if larger)
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        max_resolution: int = 518,
    ):
        self.model_size = model_size
        self.max_resolution = max_resolution
        self._model = None
        self._processor = None
        self._device = self._resolve_device(device)
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        """Resolve device string, auto-detecting if needed."""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device

    def _load_model(self):
        """Load Depth Anything V2 model from HuggingFace."""
        model_map = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf",
        }
        model_id = model_map.get(self.model_size, model_map["small"])

        try:
            from transformers import pipeline as hf_pipeline

            self._model = hf_pipeline(
                task="depth-estimation",
                model=model_id,
                device=self._device,
            )
            logger.info(
                "DepthEstimator: loaded %s on %s", model_id, self._device
            )
        except Exception as e:
            logger.warning(
                "DepthEstimator: failed to load model (%s) — "
                "will return None (flat-ground fallback)",
                e,
            )
            self._model = None

    def estimate(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth map from RGB image.

        Args:
            rgb: RGB image (H, W, 3) uint8

        Returns:
            Depth map (H, W) float32 in relative depth units,
            or None if model is unavailable.
            Note: Depth Anything V2 produces *relative* depth (not metric).
            For metric depth, use the metric_depth variant.
        """
        if self._model is None:
            return None

        try:
            from PIL import Image

            # Ensure uint8 RGB
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            # Convert to PIL Image
            if rgb.shape[2] == 3:
                pil_img = Image.fromarray(rgb)
            else:
                pil_img = Image.fromarray(rgb[:, :, :3])

            # Downscale if too large
            w, h = pil_img.size
            if max(w, h) > self.max_resolution:
                scale = self.max_resolution / max(w, h)
                new_size = (int(w * scale), int(h * scale))
                pil_img = pil_img.resize(new_size, Image.LANCZOS)

            # Run inference
            result = self._model(pil_img)
            depth = np.array(result["depth"], dtype=np.float32)

            # Normalize to [0, 1] range (relative depth)
            if depth.max() > 0:
                depth = depth / depth.max()

            return depth

        except Exception as e:
            logger.warning("DepthEstimator: inference failed (%s)", e)
            return None

    def estimate_metric(
        self,
        rgb: np.ndarray,
        camera_height: float = 1.8,
        camera_pitch: float = -8.0,
        fov_deg: float = 90.0,
    ) -> Optional[np.ndarray]:
        """Estimate metric depth using flat-ground assumption as fallback.

        This combines Depth Anything V2 (relative depth) with geometric
        constraints to produce approximate metric depth.

        Args:
            rgb: RGB image (H, W, 3) uint8
            camera_height: Camera height above ground (meters)
            camera_pitch: Camera pitch angle (degrees, negative = looking down)
            fov_deg: Camera vertical FOV (degrees)

        Returns:
            Depth map (H, W) float32 in meters, or None if unavailable.
        """
        rel_depth = self.estimate(rgb)
        if rel_depth is None:
            return None

        h, w = rel_depth.shape
        # Use geometric projection for metric depth
        # At each row v, the distance to the ground point is:
        #   d = camera_height / tan(pitch + (v - h/2) * vfov_per_pixel)
        vfov = np.radians(fov_deg)
        vfov_per_pixel = vfov / h
        pitch_rad = np.radians(camera_pitch)

        rows = np.arange(h, dtype=np.float32)
        # Angle below horizontal for each row
        angles = pitch_rad + (rows - h / 2) * vfov_per_pixel

        # Avoid division by zero / negative angles (above horizon)
        angles = np.maximum(angles, 0.01)

        geometric_depth = camera_height / np.tan(angles)  # (H,)

        # Blend: use geometric depth as scale, relative depth as shape
        # Normalize geometric depth to [0, 1]
        geo_norm = geometric_depth / max(geometric_depth.max(), 1.0)

        # Weight: trust geometric depth more at bottom (near ground)
        # Trust relative depth more at top (far away, geometry unreliable)
        weight_geo = np.linspace(0.8, 0.2, h, dtype=np.float32)  # (H,)

        # Blend
        blended = weight_geo[:, None] * geo_norm[:, None] + \
                  (1 - weight_geo[:, None]) * rel_depth

        # Scale back to metric using geometric depth
        scale = geometric_depth.max() / max(blended.max(), 1e-6)
        metric_depth = blended * scale

        return metric_depth

    def is_available(self) -> bool:
        """Check if the depth model is loaded and ready."""
        return self._model is not None

    def get_info(self) -> dict:
        """Return model info."""
        return {
            "model_size": self.model_size,
            "device": self._device,
            "available": self.is_available(),
            "max_resolution": self.max_resolution,
        }


class FlatGroundDepth:
    """Fallback depth estimator using flat-ground geometry.

    This is the original approach used by the pipeline before Depth Anything V2.
    It assumes the ground is perfectly flat and uses camera intrinsics to
    compute depth at each pixel row.
    """

    def __init__(
        self,
        camera_height: float = 1.8,
        camera_pitch: float = -8.0,
        fov_deg: float = 90.0,
    ):
        self.camera_height = camera_height
        self.camera_pitch = np.radians(camera_pitch)
        self.fov_deg = fov_deg

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """Compute flat-ground depth map.

        Args:
            rgb: RGB image (H, W, 3) — only shape is used

        Returns:
            Depth map (H, W) float32 in meters
        """
        h, w = rgb.shape[:2]
        vfov = np.radians(self.fov_deg)
        vfov_per_pixel = vfov / h

        rows = np.arange(h, dtype=np.float32)
        angles = self.camera_pitch + (rows - h / 2) * vfov_per_pixel
        angles = np.maximum(angles, 0.01)

        depth_per_row = self.camera_height / np.tan(angles)  # (H,)
        depth_map = np.tile(depth_per_row[:, None], (1, w))

        return depth_map.astype(np.float32)
