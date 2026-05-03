# 🔴 ปัญหา: UNet Model ไม่ทำงาน

## 🔍 การวินิจฉัย

จาก debug พบว่า:
```
Mask range: [0, 0]
Non-zero pixels: 0
```

**UNet model ไม่ detect lane เลย!**

### สาเหตุ:
```python
# lane_detector.py:403
mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
```

Model output ทั้งหมดเป็น **class 0 (background)** ไม่มี **class 1 (lane)** เลย

นี่หมายความว่า:
1. ❌ Model ไม่ได้ train
2. ❌ Model train ไม่ดีพอ (accuracy ต่ำมาก)
3. ❌ Model architecture ไม่ตรงกับ weights

---

## ✅ วิธีแก้ไข

### Option 1: ใช้ CARLA Waypoint แทน UNet (แนะนำ - ใช้งานได้ทันที)

แก้ไขใน `config.yaml`:
```yaml
# ปิดการใช้ trajectory pipeline (ที่ต้องการ UNet)
reference_mpc:
  USE_TRAJECTORY_PIPELINE: false
```

หรือรันด้วย flag:
```bash
# แก้ใน run_adas.sh หรือ run_adas_full.py
# เพิ่ม --no-trajectory-pipeline
```

**ผลลัพธ์:**
- ✅ ใช้ CARLA waypoint เป็นหลัก (WP_PRIMARY mode)
- ✅ ไม่ต้องพึ่ง UNet
- ✅ ทำงานได้ทันที
- ⚠️ แต่อาจไม่แม่นยำเท่า UNet ที่ดี

---

### Option 2: Train UNet Model ใหม่

**ขั้นตอน:**

1. **เตรียมข้อมูล:**
   ```bash
   # Collect CARLA images with lane labels
   # ใช้ CARLA waypoint เป็น ground truth
   ```

2. **Train model:**
   ```python
   # ใช้ script ใน carla_lstm_mpc_project/
   # หรือสร้าง training script ใหม่
   ```

3. **Test model:**
   ```bash
   python debug_unet.py
   # ตรวจสอบว่า Non-zero pixels > 0
   ```

---

### Option 3: ใช้ Pre-trained Model อื่น

**ทางเลือก:**

1. **SCNN (Spatial CNN)**
   - ดีสำหรับ lane detection
   - มี pre-trained weights

2. **Ultra-Fast Lane Detection**
   - เร็วมาก
   - Accuracy ดี

3. **LaneNet**
   - Instance segmentation
   - แยก lane แต่ละเส้นได้

**การติดตั้ง:**
```bash
# Download pre-trained weights
# แก้ไข lane_detector.py ให้รองรับ
```

---

### Option 4: ใช้ Classical Computer Vision

แทนที่ UNet ด้วย:

```python
def detect_lanes_classical(image):
    """Classical lane detection without deep learning."""
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 2. Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # 4. Region of interest (lower half)
    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, h), (w, h), (w, h//2), (0, h//2)]])
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 5. Hough line detection
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50,
                            minLineLength=50, maxLineGap=150)
    
    # 6. Draw lines on mask
    lane_mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_mask, (x1, y1), (x2, y2), 255, 10)
    
    return lane_mask
```

---

## 🚀 Quick Fix: ปิด UNet ใช้ Waypoint

**แก้ไขทันที:**

```bash
cd /home/supawich/Desktop/CARLA_0.9.16/carla_mpc_classical

# แก้ config.yaml
sed -i 's/USE_TRAJECTORY_PIPELINE: true/USE_TRAJECTORY_PIPELINE: false/' config.yaml

# รันใหม่
cd ..
./run_adas.sh --town Town04 --speed 25
```

**ผลลัพธ์ที่คาดหวัง:**
```
mode=WP_PRIMARY  # ใช้ waypoint เป็นหลัก
P1✗ P2✗ P3✗     # UNet phases ไม่ผ่าน (ปกติ)
P4✓ P5✓         # Fusion + MPC ทำงาน
```

---

## 📊 ตารางเปรียบเทียบ

| Method | Accuracy | Speed | Setup | Status |
|--------|----------|-------|-------|--------|
| UNet (current) | ❌ 0% | Fast | ❌ Broken | ไม่ทำงาน |
| CARLA Waypoint | ⭐⭐⭐ 70% | Fast | ✅ Ready | **แนะนำ** |
| Classical CV | ⭐⭐ 50% | Fast | Easy | ทางเลือก |
| SCNN | ⭐⭐⭐⭐ 85% | Medium | Hard | ต้อง train |
| Ultra-Fast | ⭐⭐⭐⭐ 90% | Very Fast | Medium | ต้อง setup |

---

## 💡 คำแนะนำ

**สำหรับตอนนี้:**
1. ✅ **ปิด USE_TRAJECTORY_PIPELINE**
2. ✅ **ใช้ WP_PRIMARY mode**
3. ✅ **ปรับ MPC weights ที่เราแก้ไว้แล้ว**

**สำหรับระยะยาว:**
1. Train UNet model ใหม่ด้วย CARLA data
2. หรือใช้ pre-trained model อื่น (Ultra-Fast, SCNN)
3. หรือพัฒนา classical CV method ให้ดีขึ้น

---

**ตอนนี้ให้ปิด UNet แล้วใช้ waypoint ก่อนครับ จะได้ระบบทำงาน!** 🚗
