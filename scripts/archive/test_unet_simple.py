#!/usr/bin/env python3
"""
Simple UNet test - create test images and check if UNet can detect lanes
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from perception.lane_detector import LaneDetector

print("=" * 70)
print("🔍 Simple UNet Test")
print("=" * 70)

# Load UNet
print("\n1. Loading UNet model...")
model_path = ROOT / "model" / "lane_unet_final.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    detector = LaneDetector(
        model_path=str(model_path),
        use_carla=False,
        model_type="unet"
    )
    print(f"   ✅ UNet loaded (device={device})")
except Exception as e:
    print(f"   ❌ Failed to load UNet: {e}")
    sys.exit(1)

# Create synthetic test images
print("\n2. Creating test images...")

def create_lane_image(width=640, height=480):
    """Create synthetic image with white lane markings"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gray road
    img[:, :] = [80, 80, 80]
    
    # White lane markings
    # Left lane
    left_x = width // 3
    cv2.line(img, (left_x, height), (left_x - 50, 0), (255, 255, 255), 10)
    
    # Right lane
    right_x = 2 * width // 3
    cv2.line(img, (right_x, height), (right_x + 50, 0), (255, 255, 255), 10)
    
    # Center dashed line
    center_x = width // 2
    for y in range(0, height, 40):
        cv2.line(img, (center_x, y), (center_x, y + 20), (255, 255, 255), 5)
    
    return img

# Test images
test_cases = [
    ("synthetic_lanes", create_lane_image()),
    ("black_image", np.zeros((480, 640, 3), dtype=np.uint8)),
    ("white_image", np.ones((480, 640, 3), dtype=np.uint8) * 255),
    ("gray_road", np.ones((480, 640, 3), dtype=np.uint8) * 100),
]

output_dir = ROOT / "unet_test_simple"
output_dir.mkdir(exist_ok=True)

print(f"\n3. Testing UNet on {len(test_cases)} images...")
print("-" * 70)

for name, img in test_cases:
    print(f"\n   Test: {name}")
    
    try:
        # Run UNet
        mask, _, _ = detector.detect_lanes(img, world=None, vehicle=None)
        
        # Stats
        non_zero = np.sum(mask > 0)
        coverage = non_zero / mask.size * 100
        
        print(f"   - Non-zero pixels: {non_zero:,}")
        print(f"   - Coverage: {coverage:.2f}%")
        
        # Save
        cv2.imwrite(str(output_dir / f"{name}_input.png"), img)
        cv2.imwrite(str(output_dir / f"{name}_mask.png"), mask)
        
        # Overlay
        overlay = img.copy()
        overlay[mask > 0] = [0, 255, 0]
        blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        cv2.imwrite(str(output_dir / f"{name}_overlay.png"), blended)
        
        # Verdict
        if non_zero > 1000:
            print(f"   ✅ DETECTED: {non_zero:,} pixels")
        elif non_zero > 0:
            print(f"   ⚠️  WEAK: Only {non_zero} pixels")
        else:
            print(f"   ❌ NOTHING: 0 pixels detected")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Check model output directly
print("\n4. Checking raw model output...")
try:
    img = create_lane_image()
    img_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = detector.model(img_tensor)
        
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Check class probabilities
    probs = torch.softmax(output, dim=1)
    class0_mean = probs[0, 0].mean().item()
    class1_mean = probs[0, 1].mean().item()
    
    print(f"   Class 0 (background) mean prob: {class0_mean:.4f}")
    print(f"   Class 1 (lane) mean prob: {class1_mean:.4f}")
    
    if class1_mean < 0.01:
        print(f"   ❌ Model always predicts background!")
        print(f"   ⚠️  Model may not be trained properly")
    else:
        print(f"   ✅ Model can predict lanes")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ Test Complete")
print("=" * 70)
print(f"\nResults: {output_dir}")
print("\nConclusion:")
print("  If all tests show 0 pixels detected:")
print("  → UNet model is NOT working (needs retraining)")
print("\n  If synthetic lanes are detected:")
print("  → UNet works but may need better preprocessing for CARLA images")
