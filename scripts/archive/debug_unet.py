#!/usr/bin/env python3
"""
Debug UNet - ตรวจสอบว่า UNet model ทำงานหรือไม่
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from perception.lane_detector import LaneDetector
from config_clean import get_config

config = get_config()

print("=" * 60)
print("🔍 UNet Debug Tool")
print("=" * 60)

# Check model file
model_path = ROOT / "model" / "lane_unet_final.pth"
print(f"\n1. Model file: {model_path}")
print(f"   Exists: {model_path.exists()}")
if model_path.exists():
    print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n2. Device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Load model
print(f"\n3. Loading UNet model...")
try:
    detector = LaneDetector(
        model_path=str(model_path),
        use_carla=False,
        model_type="unet"
    )
    print("   ✅ Model loaded successfully")
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    sys.exit(1)

# Test with dummy image
print(f"\n4. Testing with dummy image...")
dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

try:
    mask, _, _ = detector.detect_lanes(dummy_img, world=None, vehicle=None)
    print(f"   ✅ Inference successful")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Mask dtype: {mask.dtype}")
    print(f"   Mask range: [{mask.min()}, {mask.max()}]")
    print(f"   Non-zero pixels: {np.sum(mask > 0)}")
    
    # Save mask for inspection
    debug_path = ROOT / "debug_mask_dummy.png"
    cv2.imwrite(str(debug_path), mask)
    print(f"   Saved to: {debug_path}")
    
except Exception as e:
    print(f"   ❌ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with real CARLA image (if available)
print(f"\n5. Testing with CARLA screenshot...")
screenshot_dirs = [
    ROOT / "screenshots_debug",
    ROOT / "gui_screenshots",
]

test_image = None
for sdir in screenshot_dirs:
    if sdir.exists():
        images = list(sdir.glob("*.png"))
        if images:
            test_image = images[0]
            break

if test_image:
    print(f"   Found: {test_image}")
    img = cv2.imread(str(test_image))
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to expected input size
        img_resized = cv2.resize(img_rgb, (640, 480))
        
        try:
            mask, _, _ = detector.detect_lanes(img_resized, world=None, vehicle=None)
            print(f"   ✅ Inference successful")
            print(f"   Mask shape: {mask.shape}")
            print(f"   Non-zero pixels: {np.sum(mask > 0)}")
            print(f"   Coverage: {np.sum(mask > 0) / mask.size * 100:.2f}%")
            
            # Save results
            debug_img_path = ROOT / "debug_input.png"
            debug_mask_path = ROOT / "debug_mask.png"
            
            cv2.imwrite(str(debug_img_path), img)
            cv2.imwrite(str(debug_mask_path), mask)
            
            print(f"   Input saved to: {debug_img_path}")
            print(f"   Mask saved to: {debug_mask_path}")
            
            # Overlay mask on image
            overlay = img.copy()
            overlay[mask > 0] = [0, 255, 0]  # Green
            blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
            
            debug_overlay_path = ROOT / "debug_overlay.png"
            cv2.imwrite(str(debug_overlay_path), blended)
            print(f"   Overlay saved to: {debug_overlay_path}")
            
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
else:
    print(f"   ⚠️  No CARLA screenshots found")

print("\n" + "=" * 60)
print("✅ Debug complete")
print("=" * 60)
print("\nNext steps:")
print("1. Check debug_mask.png - should show white pixels where lanes are detected")
print("2. If mask is empty, model may not be trained properly")
print("3. If mask has noise, check preprocessing")
print("4. Compare with debug_overlay.png to see if detection makes sense")
