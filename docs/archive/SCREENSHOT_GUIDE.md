# 📸 วิธีเก็บภาพหน้าจอ GUI และ Log

## วิธีที่ 1: ใช้ Pygame Screenshot (แนะนำ)

### ขั้นตอน:

1. **รัน ADAS ปกติ**
```bash
cd /home/supawich/Desktop/CARLA_0.9.16
./run_adas.sh --town Town04 --speed 25
```

2. **กด S เพื่อบันทึกภาพหน้าจอ**
   - กด `S` = บันทึก screenshot ทันที
   - ภาพจะถูกบันทึกที่ `carla_mpc_classical/screenshots/`

3. **ดู Log แบบละเอียด**
```bash
# Log จะแสดงใน terminal อยู่แล้ว
# ถ้าต้องการเก็บ log ไว้:
./run_adas.sh --town Town04 --speed 25 2>&1 | tee adas_debug.log
```

---

## วิธีที่ 2: บันทึกอัตโนมัติทุก N frames

### แก้ไขโค้ดใน `managers/display_manager.py`:

เพิ่มในส่วน `_update_gui()`:

```python
# เพิ่มที่ต้นไฟล์
import os
from datetime import datetime

# ใน __init__
self.screenshot_dir = Path("screenshots") / datetime.now().strftime("%Y%m%d_%H%M%S")
self.screenshot_dir.mkdir(parents=True, exist_ok=True)
self.screenshot_interval = 10  # บันทึกทุก 10 frames
self.frame_count = 0

# ใน _update_gui() หลัง pygame.display.flip()
self.frame_count += 1
if self.frame_count % self.screenshot_interval == 0:
    screenshot_path = self.screenshot_dir / f"frame_{self.frame_count:04d}.png"
    pygame.image.save(self.screen, str(screenshot_path))
    logger.info(f"Screenshot saved: {screenshot_path}")
```

---

## วิธีที่ 3: ใช้ External Screenshot Tool

### ติดตั้ง tool:
```bash
sudo apt install scrot  # หรือ
sudo apt install gnome-screenshot
```

### รันใน terminal แยก:
```bash
# Terminal 1: รัน ADAS
./run_adas.sh --town Town04 --speed 25

# Terminal 2: Capture screenshots ทุก 2 วินาที
cd /home/supawich/Desktop/CARLA_0.9.16/carla_mpc_classical
mkdir -p screenshots
for i in {1..100}; do
    scrot -u "screenshots/frame_$(printf %04d $i).png"
    sleep 2
done
```

---

## วิธีที่ 4: Record Video แล้ว Extract Frames

### ใช้ SimpleScreenRecorder:
```bash
sudo apt install simplescreenrecorder
# เปิดโปรแกรม record หน้าจอ GUI
# หลังจากนั้น extract frames:
ffmpeg -i recording.mp4 -vf fps=5 screenshots/frame_%04d.png
```

---

## 📊 วิเคราะห์ปัญหาจาก Log

### ดู Log แบบละเอียด:

```bash
# รัน ADAS พร้อม debug logging
cd /home/supawich/Desktop/CARLA_0.9.16
./run_adas.sh --town Town04 --speed 25 2>&1 | tee debug_$(date +%Y%m%d_%H%M%S).log
```

### สิ่งที่ต้องดูใน Log:

1. **Lane Detection Quality:**
```
INFO   phases: P1✓ P2✓ P3✓ P4✓ P5✓  # ครบทุก phase = ดี
INFO   mode=WP+UNET_CENTER          # ใช้ UNet ช่วย
INFO   lane_conf=0.XXX              # confidence สูง = ดี
```

2. **Control Oscillation:**
```
WARNING ⚠️  STEERING LIMIT EXCEEDED: 0.570 > 0.450
# ถ้าเห็นบ่อย = steering เหวี่ยง
```

3. **Perception Issues:**
```
INFO   P2 Case: mirror_left  # ใช้ mirror = จับเลนข้างเดียว
INFO   P3 Case: completion   # ใช้ completion = เลนขาด
```

---

## 🔍 วิเคราะห์ปัญหาที่คุณพบ

### ปัญหา 1: "Raw UNet ไม่จับเส้นเลย"

**ตรวจสอบ:**
- ดู `P1` phase: ถ้า `P1✗` = UNet ไม่จับเลน
- ดู `lane_conf`: ถ้า < 0.1 = confidence ต่ำมาก
- ดู `mode`: ถ้าเป็น `WP_PRIMARY` ตลอด = ไม่ใช้ UNet เลย

**แก้ไข:**
- ตรวจสอบ model: `model/lane_unet_final.pth` มีหรือไม่
- ตรวจสอบ input image: ขนาดถูกต้องหรือไม่ (640x480)
- ลอง visualize mask: ดู `state.mask` ว่ามีอะไรหรือไม่

### ปัญหา 2: "การ control เหวี่ยง"

**ตรวจสอบ:**
- ดู steering values: ถ้า oscillate -0.5 ↔ +0.5 = เหวี่ยง
- ดู `STEERING LIMIT EXCEEDED`: ถ้าเห็นบ่อย = MPC คำนวณ steering สูงเกินไป

**แก้ไข:**
- เพิ่ม steering smoothing (ปรับ `STEER_SMOOTH_ALPHA` ใน config)
- ลด MPC weights (ลด `MPC_W_CTE`, เพิ่ม `MPC_W_STEER_RATE`)
- ใช้ B-spline fitting ที่เราสร้างไว้ (smooth กว่า polynomial)

---

## 🎯 Quick Start: เก็บผลเลย

```bash
# 1. สร้างโฟลเดอร์
mkdir -p ~/Desktop/CARLA_0.9.16/carla_mpc_classical/debug_output

# 2. รัน ADAS พร้อมเก็บ log
cd ~/Desktop/CARLA_0.9.16
./run_adas.sh --town Town04 --speed 25 2>&1 | tee carla_mpc_classical/debug_output/run_$(date +%H%M%S).log

# 3. ขณะที่รัน ให้กด S เพื่อ screenshot (ถ้า pygame รองรับ)
#    หรือใช้ scrot ใน terminal อื่น:
#    watch -n 2 'scrot -u carla_mpc_classical/debug_output/screenshot_$(date +%H%M%S).png'

# 4. หลังจากรันเสร็จ ดู log:
cat carla_mpc_classical/debug_output/run_*.log | grep -E "P1|P2|P3|P4|P5|STEERING|mode="
```

---

## 📝 Template สำหรับรายงานปัญหา

```
=== ADAS Debug Report ===
Date: $(date)
Town: Town04
Speed: 25 km/h
Frames: XXX

Lane Detection:
- P1 (UNet Mask): ✓/✗
- P2 (BEV): ✓/✗
- P3 (Boundaries): ✓/✗
- P4 (Smooth): ✓/✗
- P5 (Trajectory): ✓/✗

Mode: WP_PRIMARY / WP+UNET_CENTER
Lane Confidence: 0.XXX
CTE: ±X.XX m
Heading Error: ±XX°
Curvature: 0.XXXX

Control:
- Steering range: -0.XX to +0.XX
- Steering limit exceeded: XX times
- Average speed: XX km/h

Screenshots: XX files
Log file: debug_output/run_XXXXXX.log
```
