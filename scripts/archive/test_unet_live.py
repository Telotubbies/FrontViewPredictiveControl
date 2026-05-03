#!/usr/bin/env python3
"""
Test UNet with live CARLA images - capture frames and test lane detection
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import time

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Add CARLA to path
try:
    import carla
except ImportError:
    carla_paths = [
        ROOT.parent / "PythonAPI",
        ROOT / ".carla_py",
    ]
    for p in carla_paths:
        if p.exists():
            sys.path.insert(0, str(p))
    import carla

from perception.lane_detector import LaneDetector
from config_clean import get_config

config = get_config()

print("=" * 70)
print("🔍 UNet Live Test with CARLA Images")
print("=" * 70)

# Connect to CARLA
print("\n1. Connecting to CARLA...")
try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"   ✅ Connected to CARLA")
    print(f"   Map: {world.get_map().name}")
except Exception as e:
    print(f"   ❌ Failed to connect: {e}")
    sys.exit(1)

# Find vehicle
print("\n2. Finding vehicle...")
vehicles = world.get_actors().filter('vehicle.*')
if not vehicles:
    print("   ❌ No vehicle found")
    sys.exit(1)

vehicle = vehicles[0]
print(f"   ✅ Found vehicle: {vehicle.type_id}")

# Setup camera
print("\n3. Setting up camera...")
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
camera_bp.set_attribute('fov', '110')

camera_transform = carla.Transform(
    carla.Location(x=2.0, z=1.4),
    carla.Rotation(pitch=0.0)
)

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
print(f"   ✅ Camera spawned")

# Load UNet
print("\n4. Loading UNet model...")
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
    camera.destroy()
    sys.exit(1)

# Capture images
print("\n5. Capturing and testing images...")
print("   Will capture 5 frames and test UNet on each")
print("-" * 70)

captured_images = []
frame_count = 0
max_frames = 5

def camera_callback(image):
    global frame_count
    if frame_count >= max_frames:
        return
    
    # Convert to numpy
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3]
    
    captured_images.append(rgb.copy())
    frame_count += 1
    print(f"   📸 Captured frame {frame_count}/{max_frames}")

camera.listen(camera_callback)

# Wait for frames
while frame_count < max_frames:
    time.sleep(0.2)

camera.stop()
camera.destroy()

print(f"\n   ✅ Captured {len(captured_images)} frames")

# Test UNet on each frame
print("\n6. Testing UNet on captured frames...")
print("-" * 70)

output_dir = ROOT / "unet_test_results"
output_dir.mkdir(exist_ok=True)

for i, rgb in enumerate(captured_images, 1):
    print(f"\n   Frame {i}:")
    
    try:
        # Run UNet
        mask, features, coords = detector.detect_lanes(rgb, world=None, vehicle=None)
        
        # Stats
        non_zero = np.sum(mask > 0)
        coverage = non_zero / mask.size * 100
        
        print(f"   - Mask shape: {mask.shape}")
        print(f"   - Non-zero pixels: {non_zero:,}")
        print(f"   - Coverage: {coverage:.2f}%")
        print(f"   - Mask range: [{mask.min()}, {mask.max()}]")
        
        # Save original
        img_path = output_dir / f"frame_{i:02d}_input.png"
        cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # Save mask
        mask_path = output_dir / f"frame_{i:02d}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        
        # Create overlay
        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        overlay[mask > 0] = [0, 255, 0]  # Green
        blended = cv2.addWeighted(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.7,
            overlay, 0.3, 0
        )
        
        overlay_path = output_dir / f"frame_{i:02d}_overlay.png"
        cv2.imwrite(str(overlay_path), blended)
        
        # Create side-by-side comparison
        comparison = np.hstack([
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB),
            blended
        ])
        
        comparison_path = output_dir / f"frame_{i:02d}_comparison.png"
        cv2.imwrite(str(comparison_path), comparison)
        
        print(f"   ✅ Saved to {output_dir}/frame_{i:02d}_*.png")
        
        # Verdict
        if non_zero > 1000:
            print(f"   ✅ GOOD: UNet detected lanes!")
        elif non_zero > 100:
            print(f"   ⚠️  WEAK: UNet detected some pixels but coverage is low")
        else:
            print(f"   ❌ FAIL: UNet did not detect lanes")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("✅ Test Complete")
print("=" * 70)
print(f"\nResults saved to: {output_dir}")
print("\nFiles created:")
print("  - frame_XX_input.png     : Original CARLA image")
print("  - frame_XX_mask.png      : UNet output mask (white = lane)")
print("  - frame_XX_overlay.png   : Mask overlaid on image (green)")
print("  - frame_XX_comparison.png: Side-by-side comparison")
print("\nTo view results:")
print(f"  cd {output_dir}")
print(f"  eog *.png")
print("\nOr open in file manager:")
print(f"  xdg-open {output_dir}")
