"""
Lane Detection Model using U-Net architecture.

Detects lane markings from RGB images and extracts lane features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Tuple, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    UNET_INPUT_W_INFER,
    UNET_INPUT_H_INFER,
    CAM_FOV_DEG,
)
from utils.device_utils import get_device
from perception.birds_eye_view_lane_pipeline import BEVLanePipeline

logger = logging.getLogger(__name__)


class UNetBlock(nn.Module):
    """U-Net building block."""

    def __init__(self, in_channels: int, out_channels: int):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LaneUNet(nn.Module):
    """
    U-Net for lane detection.

    Architecture:
    - Encoder: Downsampling path
    - Decoder: Upsampling path with skip connections
    - Output: Lane segmentation mask
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super(LaneUNet, self).__init__()

        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)

        # Bottleneck
        self.bottleneck = UNetBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNetBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

        self.device = get_device()
        self.to(self.device)

        logger.info(f"✅ LaneUNet initialized (device={self.device})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)

        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)

        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2)

        e4 = self.enc4(p3)
        p4 = F.max_pool2d(e4, 2)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


class LaneDetector:
    """
    Lane detection using U-Net model or Ultra-Fast-Lane-Detection-v2.
    """

    def __init__(self, model_path: Optional[str] = None, use_carla: bool = True,
                 model_type: str = "unet"):
        self.use_carla = use_carla
        self.model = None
        self.model_type = model_type
        self.ultra_fast_detector = None

        if model_path and Path(model_path).exists():
            if model_type == "ultra_fast":
                try:
                    from perception.ultra_fast_lane_detector import UltraFastLaneDetector
                    dataset = "tusimple" if "tusimple" in model_path.lower() else \
                              "culane" if "culane" in model_path.lower() else "tusimple"
                    backbone = "18"
                    self.ultra_fast_detector = UltraFastLaneDetector(
                        model_path=model_path, dataset=dataset, backbone=backbone
                    )
                    self.use_carla = False
                    logger.info(f"✅ Loaded Ultra-Fast-Lane-Detection-v2 from {model_path}")
                except Exception as e:
                    logger.warning(
                        "Failed to load Ultra-Fast model: %s, fallback to CARLA",
                        e,
                        exc_info=True,
                    )
                    self.use_carla = True
            else:
                try:
                    self.model = LaneUNet()
                    self.model.load_state_dict(
                    torch.load(model_path, map_location=get_device(), weights_only=True)
                )
                    self.model.eval()
                    self.use_carla = False
                    logger.info(f"✅ Loaded U-Net lane model from {model_path}")
                except Exception as e:
                    logger.warning(
                        "Failed to load U-Net model: %s, fallback to CARLA",
                        e,
                        exc_info=True,
                    )
                    self.use_carla = True
        else:
            if use_carla:
                logger.info("Using CARLA built-in lane detection")
            else:
                logger.warning("No model provided, switching to CARLA detection")
                self.use_carla = True

        # Initialize BEV lane pipeline
        self.bev_pipeline = BEVLanePipeline()

    # ------------------------------------------------------------------
    # BEV Lane Detection (robust polynomial fitting)
    # ------------------------------------------------------------------
    def detect_lanes_bev(self, image: np.ndarray, return_vis: bool = True, ego_mask: np.ndarray = None):
        """
        Detect lanes using BEV pipeline with polynomial fitting.

        Args:
            image: RGB image (H, W, 3)
            return_vis: If True, return BEV visualization
            ego_mask: Optional ego-lane mask to filter adjacent lanes

        Returns:
            left_coeffs: Left lane polynomial coefficients in BEV
            right_coeffs: Right lane polynomial coefficients in BEV
            cte: Cross-track error [-1, 1]
            heading: Heading error in radians
            curvature: Road curvature
            confidence: Detection confidence [0, 1]
            bev_vis: (optional) BEV visualization image
        """
        if self.model is None:
            return None, None, 0.0, 0.0, 0.0, 0.0, None

        # Get probability map from UNet
        with torch.no_grad():
            img_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(self.model.device)
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            prob_map = probs[0, 1].cpu().numpy()

        # Apply ego-lane mask to filter adjacent lanes (completely suppress non-ego)
        if ego_mask is not None:
            from perception.ego_lane_mask import filter_prob_map_with_ego_mask
            prob_map = filter_prob_map_with_ego_mask(prob_map, ego_mask, blend_factor=0.0)

        # Process through BEV pipeline with visualization
        left_coeffs, right_coeffs, l_active, r_active, bev_vis = \
            self.bev_pipeline.process(prob_map, return_vis=return_vis)

        # Compute CTE, heading, curvature
        cte, heading, curvature = self.bev_pipeline.compute_cte_heading(
            left_coeffs, right_coeffs, image.shape[0])

        # Confidence based on active window counts
        max_active = 25  # N_WINDOWS (updated)
        confidence = min(1.0, (l_active + r_active) / (2 * max_active) + 0.35)
        if left_coeffs is None or right_coeffs is None:
            confidence *= 0.6

        return left_coeffs, right_coeffs, cte, heading, curvature, confidence, bev_vis

    # ------------------------------------------------------------------
    # CARLA Lane Detection  (ปรับปรุงใหม่)
    # ------------------------------------------------------------------
    def detect_lanes_carla(
        self,
        image: np.ndarray,
        world,
        vehicle,
        camera_transform=None,
        fov: float = 90.0,
        detection_radius: float = 50.0,
        waypoint_spacing: float = 1.5,
        lane_half_width: float = 1.8,
    ) -> np.ndarray:
        """
        Detect lanes using CARLA waypoints + proper 3-D → 2-D projection.

        Improvements vs. original:
        • Sort waypoints by **forward distance along camera Z-axis** instead of
          raw image-v — avoids zigzag lines when multiple lanes are visible.
        • Project left / right lane-edge points (±lane_half_width in road-right
          direction) so the mask shows actual lane boundaries, not just centers.
        • Skip degenerate segments more robustly (angle + distance guard).
        • Configurable FOV, detection radius, and waypoint spacing.
        • Cleaner coordinate-system comment.

        Args:
            image            : RGB image (H, W, 3)
            world            : carla.World
            vehicle          : carla.Vehicle
            camera_transform : carla.Transform of camera relative to vehicle.
                               Defaults to a standard front-camera pose.
            fov              : Camera horizontal FOV in degrees (match sensor).
            detection_radius : Max distance [m] from vehicle to search waypoints.
            waypoint_spacing : Waypoint granularity [m] from generate_waypoints().
            lane_half_width  : Half-width [m] used to project lane edges.

        Returns:
            lane_mask : uint8 binary mask (H, W), 255 = lane marking.
        """
        try:
            import carla

            h, w = image.shape[:2]
            lane_mask = np.zeros((h, w), dtype=np.uint8)

            # ── Camera intrinsic ──────────────────────────────────────────────
            f = w / (2.0 * np.tan(np.radians(fov / 2.0)))
            cx, cy = w / 2.0, h / 2.0
            K = np.array([[f, 0, cx],
                          [0, f, cy],
                          [0, 0,  1]], dtype=np.float64)

            # ── Camera world transform ────────────────────────────────────────
            vehicle_transform = vehicle.get_transform()

            if camera_transform is None:
                camera_transform = carla.Transform(
                    carla.Location(x=1.5, z=2.0),
                    carla.Rotation(pitch=-8.0, yaw=0.0, roll=0.0),
                )

            # CARLA supports operator* for Transform composition
            try:
                cam_world_tf = vehicle_transform * camera_transform
            except TypeError:
                # Fallback manual compose (handles older CARLA Python API)
                vl = vehicle_transform.location
                vr = vehicle_transform.rotation
                cl = camera_transform.location
                cr = camera_transform.rotation
                cam_world_tf = carla.Transform(
                    carla.Location(x=vl.x + cl.x, y=vl.y + cl.y, z=vl.z + cl.z),
                    carla.Rotation(
                        pitch=vr.pitch + cr.pitch,
                        yaw=vr.yaw + cr.yaw,
                        roll=vr.roll + cr.roll,
                    ),
                )

            # World → Camera matrix  (4×4, CARLA's get_inverse_matrix is world→local)
            W2C = np.array(cam_world_tf.get_inverse_matrix(), dtype=np.float64)  # (4,4)

            # ── Collect nearby waypoints ──────────────────────────────────────
            carla_map = world.get_map()
            all_waypoints = carla_map.generate_waypoints(distance=waypoint_spacing)
            vehicle_loc = vehicle_transform.location

            nearby = [
                wp for wp in all_waypoints
                if vehicle_loc.distance(wp.transform.location) < detection_radius
            ]

            if not nearby:
                logger.debug("No nearby waypoints found")
                return lane_mask

            # ── Helper: project a single 3-D world point → (u, v) or None ────
            def project(wx: float, wy: float, wz: float):
                """Return (u, v) pixel or None if behind camera / out of bounds."""
                pw = np.array([wx, wy, wz, 1.0], dtype=np.float64)
                pc = W2C @ pw                    # camera-local (UE4 axes: X-fwd, Y-right, Z-up)

                # UE4 → OpenCV camera axes:
                #   forward = X  →  depth  = pc[0]
                #   right   = Y  →  right  = pc[1]
                #   up      = Z  →  down   = -pc[2]
                depth  =  pc[0]
                x_cam  =  pc[1]
                y_cam  = -pc[2]

                if depth < 0.5:          # behind or too close to camera
                    return None

                u = int(K[0, 0] * x_cam / depth + K[0, 2])
                v = int(K[1, 1] * y_cam / depth + K[1, 2])

                if 0 <= u < w and 0 <= v < h:
                    return (u, v)
                return None

            # ── Helper: compute lane-edge offsets for a waypoint ─────────────
            def edge_points(wp):
                """
                Return (left_pt, right_pt) projected pixels for lane edges.
                Uses waypoint's right vector scaled by lane_half_width.
                Returns (None, None) if both edges are off-screen.
                """
                loc = wp.transform.location
                # CARLA right vector (unit, points to road's right)
                right_vec = wp.transform.get_right_vector()

                lx = loc.x - right_vec.x * lane_half_width
                ly = loc.y - right_vec.y * lane_half_width
                lz = loc.z

                rx = loc.x + right_vec.x * lane_half_width
                ry = loc.y + right_vec.y * lane_half_width
                rz = loc.z

                return project(lx, ly, lz), project(rx, ry, rz)

            # ── Sort waypoints by camera-space depth (closest first) ──────────
            def cam_depth(wp) -> float:
                loc = wp.transform.location
                pc = W2C @ np.array([loc.x, loc.y, loc.z, 1.0], dtype=np.float64)
                return pc[0]  # X = forward in UE4

            nearby.sort(key=cam_depth)

            # ── Group waypoints by road_id + lane_id for per-lane polylines ───
            lane_groups: dict = {}
            for wp in nearby:
                key = (wp.road_id, wp.lane_id)
                lane_groups.setdefault(key, []).append(wp)

            MAX_PIXEL_GAP = 250   # pixels — skip segment if points jump too far
            LINE_THICKNESS = 5

            for wps in lane_groups.values():
                # Sort each group by depth so lines go near → far
                wps.sort(key=cam_depth)

                left_prev: Optional[Tuple[int, int]] = None
                right_prev: Optional[Tuple[int, int]] = None

                for wp in wps:
                    left_pt, right_pt = edge_points(wp)

                    # Draw left edge segment
                    if left_pt is not None and left_prev is not None:
                        dist = np.hypot(left_pt[0] - left_prev[0],
                                        left_pt[1] - left_prev[1])
                        if dist < MAX_PIXEL_GAP:
                            cv2.line(lane_mask, left_prev, left_pt, 255, LINE_THICKNESS)

                    # Draw right edge segment
                    if right_pt is not None and right_prev is not None:
                        dist = np.hypot(right_pt[0] - right_prev[0],
                                        right_pt[1] - right_prev[1])
                        if dist < MAX_PIXEL_GAP:
                            cv2.line(lane_mask, right_prev, right_pt, 255, LINE_THICKNESS)

                    left_prev  = left_pt  if left_pt  is not None else left_prev
                    right_prev = right_pt if right_pt is not None else right_prev

            # ── Post-process: close small gaps, remove isolated noise ─────────
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel_close)
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN,  kernel_open)

            return lane_mask

        except Exception as e:
            logger.warning("CARLA lane detection failed: %s", e, exc_info=True)
            return np.zeros(image.shape[:2], dtype=np.uint8)

    # ------------------------------------------------------------------
    def detect_lanes(
        self,
        image: np.ndarray,
        world=None,
        vehicle=None,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[int, int]]]]:
        """
        Detect lanes in image.

        Returns:
            lane_mask     : Binary lane mask (H, W)
            lane_features : Feature vector (128,)
            lane_coords   : List of lane coordinate lists (for visualization)
        """
        if self.ultra_fast_detector is not None:
            return self.ultra_fast_detector.detect_lanes(image, use_post_processing=True)

        # CARLA waypoint detection is primary when world/vehicle available (ground truth)
        if world is not None and vehicle is not None:
            mask = self.detect_lanes_carla(image, world, vehicle, fov=CAM_FOV_DEG)
            return mask, self._extract_lane_features(mask), []

        # UNet fallback when no CARLA access
        if self.model is not None:
            with torch.no_grad():
                img_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
                img_tensor = img_tensor.to(self.model.device)
                img_tensor = F.interpolate(img_tensor, size=(UNET_INPUT_H_INFER, UNET_INPUT_W_INFER),
                                           mode='bilinear', align_corners=False)
                output = self.model(img_tensor)

                probs = torch.softmax(output, dim=1)
                lane_prob = probs[0, 1].cpu().numpy()
                lane_prob = cv2.resize(lane_prob, (image.shape[1], image.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)

                mask_bin = (lane_prob > 0.02).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
                return (mask * 255).astype(np.uint8), self._extract_lane_features(mask), []

        # Last resort: Canny edge in lower half
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 2:, :] = edges[h // 2:, :]
        return mask, self._extract_lane_features(mask), []

    def _extract_lane_features(self, lane_mask: np.ndarray) -> np.ndarray:
        h, w = lane_mask.shape
        features = []

        lane_pixels = np.where(lane_mask > 0)
        if len(lane_pixels[0]) > 0:
            features.extend([np.mean(lane_pixels[1]) / w,
                              np.mean(lane_pixels[0]) / h,
                              np.std(lane_pixels[1]) * 2 / w])
        else:
            features.extend([0.5, 0.5, 0.0])

        if len(lane_pixels[0]) > 10 and len(np.unique(lane_pixels[0])) > 3:
            coeffs = np.polyfit(lane_pixels[0], lane_pixels[1], 2)
            features.extend(coeffs.tolist())
        else:
            features.extend([0.0, 0.0, 0.0])

        for y1, y2 in [(0, h // 3), (h // 3, 2 * h // 3), (2 * h // 3, h)]:
            region = lane_mask[y1:y2, :]
            features.append(np.sum(region > 0) / (region.size + 1e-6))

        while len(features) < 128:
            features.append(0.0)

        return np.array(features[:128], dtype=np.float32)


# ---------------------------------------------------------------------------
def create_lane_labels_from_carla(images_dir: str, output_dir: str, world, vehicle):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    detector = LaneDetector(use_carla=True)
    image_files = list(images_dir.glob("*.png"))
    logger.info(f"Processing {len(image_files)} images…")

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask, _, _ = detector.detect_lanes(image, world, vehicle)
        cv2.imwrite(str(output_dir / f"{img_path.stem}_lane.png"), mask)

    logger.info(f"✅ Created {len(image_files)} lane masks")
