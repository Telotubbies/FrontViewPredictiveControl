#!/usr/bin/env python3
"""
Capture screenshots from ADAS GUI display for debugging.

Takes screenshots of the pygame GUI window showing:
- Camera feed with lane overlay
- BEV visualization
- Perception data
- Control commands
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
import subprocess

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).parent / "gui_screenshots" / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

print(f"📸 GUI Screenshot Capture Tool")
print(f"Output directory: {output_dir}")
print(f"=" * 60)

# Instructions
print("""
วิธีใช้งาน:
1. รันสคริปต์นี้ในเทอร์มินัลหนึ่ง
2. รัน ADAS ในอีกเทอร์มินัล: ./run_adas.sh --town Town04 --speed 25
3. สคริปต์จะ capture หน้าจอ GUI อัตโนมัติทุก 2 วินาที
4. กด Ctrl+C เพื่อหยุด

กำลังเริ่ม capture ใน 5 วินาที...
""")

time.sleep(5)

# Use scrot or gnome-screenshot to capture
screenshot_num = 0
interval = 2  # seconds

try:
    print("🎬 เริ่ม capture screenshots...")
    print("กด Ctrl+C เพื่อหยุด\n")
    
    while True:
        screenshot_num += 1
        output_file = output_dir / f"gui_frame_{screenshot_num:04d}.png"
        
        # Try different screenshot tools
        try:
            # Try gnome-screenshot first (works with Wayland)
            result = subprocess.run(
                ["gnome-screenshot", "-w", "-f", str(output_file)],
                capture_output=True,
                timeout=3
            )
            if result.returncode == 0:
                print(f"✅ Captured frame {screenshot_num}: {output_file.name}")
            else:
                # Try scrot (works with X11)
                result = subprocess.run(
                    ["scrot", "-u", str(output_file)],
                    capture_output=True,
                    timeout=3
                )
                if result.returncode == 0:
                    print(f"✅ Captured frame {screenshot_num}: {output_file.name}")
                else:
                    # Try import from ImageMagick
                    result = subprocess.run(
                        ["import", "-window", "root", str(output_file)],
                        capture_output=True,
                        timeout=3
                    )
                    if result.returncode == 0:
                        print(f"✅ Captured frame {screenshot_num}: {output_file.name}")
                    else:
                        print(f"⚠️  Screenshot tool failed, trying alternative...")
        except Exception as e:
            print(f"❌ Error capturing screenshot: {e}")
        
        time.sleep(interval)

except KeyboardInterrupt:
    print(f"\n\n🛑 Stopped capture")
    print(f"📊 Total screenshots: {screenshot_num}")
    print(f"📁 Saved to: {output_dir}")
    print(f"\nคุณสามารถดูภาพได้ที่:")
    print(f"  cd {output_dir}")
    print(f"  eog *.png  # หรือใช้ image viewer อื่น")

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
