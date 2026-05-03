#!/usr/bin/env python3
"""
Phase 1 only — การจับเลนระดับพิกเซล (Lane mask + confidence).

Input:  RGB image (จากไฟล์ หรือสร้าง dummy)
Output: binary mask, confidence, P1_ok
ใช้เมื่อ: ลุย P1 ก่อน รอดูผล แล้วค่อยไป P2

Usage:
  # จากภาพในไฟล์
  python scripts/run_phase1_only.py --image path/to/image.png
  # สร้างภาพเทส (สีเทา) แล้วรัน P1
  python scripts/run_phase1_only.py
  # บันทึก mask ออกไฟล์
  python scripts/run_phase1_only.py --image frame.png --out-dir out_p1
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CAM_W, CAM_H, DEFAULT_MODEL


def run_p1(rgb: np.ndarray, model_path: Path, device: str = "cuda") -> tuple:
    """
    Phase 1: RGB → mask, confidence.
    Returns: (mask, confidence, p1_ok)
    """
    import torch
    from perception.lane_detector import LaneDetector

    detector = LaneDetector(
        model_path=str(model_path),
        use_carla=False,
        model_type="unet",
    )
    img = cv2.resize(rgb, (CAM_W, CAM_H))
    mask_uint8, _, _ = detector.detect_lanes(img, world=None, vehicle=None)
    mask = (mask_uint8 > 0).astype(np.uint8)

    road_top_ratio = 0.10
    mask[: int(mask.shape[0] * road_top_ratio), :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    roi = mask[int(mask.shape[0] * 0.5) :, :]
    conf = min(1.0, np.sum(roi > 0) / max(roi.size * 0.05, 1))
    p1_ok = conf > 0.01

    return mask, float(conf), bool(p1_ok)


def main():
    ap = argparse.ArgumentParser(description="Run Phase 1 only: Lane mask + confidence")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="UNet .pth path")
    ap.add_argument("--image", type=Path, default=None, help="Input RGB image (optional)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Save mask + overlay here")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    a = ap.parse_args()

    if a.image and a.image.exists():
        rgb = cv2.imread(str(a.image))
        if rgb is None:
            logger.error("Failed to load image: %s", a.image)
            sys.exit(1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        # Dummy: gray image 320x240
        rgb = np.full((CAM_H, CAM_W, 3), 128, dtype=np.uint8)
        if a.image:
            logger.warning("Image not found, using dummy: %s", a.image)

    model_path = a.model if a.model.exists() else ROOT / "model" / "lane_unet_final.pth"
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    mask, confidence, p1_ok = run_p1(rgb, model_path, a.device)

    logger.info("── Phase 1 (Lane mask + confidence) ──")
    logger.info("  confidence = %s", round(confidence, 4))
    logger.info("  P1_ok      = %s (confidence > 0.01)", p1_ok)
    logger.info("  mask shape = %s | lane pixels = %d", mask.shape, int(np.sum(mask > 0)))

    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        mask_vis = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(a.out_dir / "mask_p1.png"), mask_vis)
        overlay = rgb.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
        overlay = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(a.out_dir / "overlay_p1.png"), overlay)
        logger.info("  saved mask + overlay to %s", a.out_dir)

    return 0 if p1_ok else 1


if __name__ == "__main__":
    sys.exit(main())
