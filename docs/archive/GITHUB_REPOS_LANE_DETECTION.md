# 🔗 GitHub Repositories: Lane Detection + MPC Control

รวม GitHub repositories ที่น่าสนใจสำหรับนำมาปรับใช้กับโปรเจกต์

---

## 🎯 CARLA + MPC Control

### 1. **Lane-Keeping-Assist-on-CARLA** ⭐⭐⭐⭐⭐
**Repo:** https://github.com/paulyehtw/Lane-Keeping-Assist-on-CARLA

**Description:**
- Lane Keeping Assist บน CARLA simulator
- ใช้ Pure Pursuit, Stanley, และ **MPC** สำหรับ lateral control
- PID controller สำหรับ longitudinal control
- Pre-defined waypoints

**Technologies:**
- Python
- CARLA Simulator
- MPC Controller
- Pure Pursuit / Stanley

**สิ่งที่นำมาใช้ได้:**
```python
# MPC Controller implementation
class MPCController:
    def __init__(self):
        self.horizon = 10
        self.dt = 0.1
        # Bicycle model parameters
        
    def solve(self, state, reference_path):
        # Optimize steering and throttle
        # Minimize CTE, heading error, speed error
        return steering, throttle
```

**การปรับใช้:**
- ✅ ดู MPC implementation
- ✅ เรียนรู้ lateral control methods
- ✅ ใช้ waypoint tracking approach

---

### 2. **Lane-detection-in-Carla-simulator**
**Repo:** https://github.com/nickos97/Lane-detection-in-Carla-simulator

**Description:**
- Lane detection ใน CARLA
- Semantic segmentation บน RGB camera
- Pre-processing pipeline

**Technologies:**
- Python
- CARLA API
- Semantic Segmentation

**สิ่งที่นำมาใช้ได้:**
- Image preprocessing techniques
- CARLA camera setup
- Lane segmentation approach

---

### 3. **Real-Time-Lane-Detection-using-CARLA-Simulator**
**Repo:** https://github.com/smitsavaliya-1258/Real-Time-Lane-Detection-using-CARLA-Simulator

**Description:**
- Real-time lane detection
- CARLA integration
- Manual/Auto control

**Technologies:**
- Python
- CARLA
- Real-time processing

---

## 🧠 UNet Lane Detection

### 4. **Lane-Detection-UNet** ⭐⭐⭐⭐
**Repo:** https://github.com/AnshChoudhary/Lane-Detection-UNet

**Description:**
- UNet model สำหรับ lane detection
- ใช้ BDD100K dataset
- PyTorch implementation
- Training + Evaluation scripts

**Technologies:**
- PyTorch
- UNet architecture
- BDD100K dataset

**สิ่งที่นำมาใช้ได้:**
```python
# UNet Training Pipeline
class UNetTrainer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        
    def train(self, epochs):
        for epoch in range(epochs):
            for batch in self.dataloader:
                # Forward pass
                output = self.model(batch['image'])
                
                # Loss (Dice + BCE)
                loss = dice_loss(output, batch['mask']) + \
                       bce_loss(output, batch['mask'])
                
                # Backward pass
                loss.backward()
                optimizer.step()
```

**การปรับใช้:**
- ✅ ใช้ training pipeline
- ✅ เรียนรู้ loss functions (Dice, BCE)
- ✅ Data augmentation techniques

**Files to check:**
- `train.py` - Training script
- `model.py` - UNet architecture
- `dataset.py` - Data loading
- `inference.py` - Testing

---

### 5. **Lane_Detection_Using_UNet**
**Repo:** https://github.com/GirishChowdary0208/Lane_Detection_Using_UNet

**Description:**
- UNet สำหรับ ADAS
- Semantic segmentation
- CNN-based approach

**Technologies:**
- Deep Learning
- UNet
- ADAS

---

### 6. **Lane-Detection-with-Semantic-Segmentation**
**Repo:** https://github.com/yudhisteer/Lane-Detection-with-Semantic-Segmentation

**Description:**
- Semantic segmentation สำหรับ lane detection
- สำหรับ autonomous shuttles
- Controlled environment

**Technologies:**
- Semantic Segmentation
- Deep Learning

---

## 🚀 Advanced Lane Detection

### 7. **LaneNet** ⭐⭐⭐⭐⭐
**Repo:** https://github.com/MaybeShewill-CV/lanenet-lane-detection

**Description:**
- **Instance Segmentation** approach
- TensorFlow implementation
- Based on paper: "Towards End-to-End Lane Detection"
- Encoder-Decoder architecture

**Technologies:**
- TensorFlow
- Instance Segmentation
- Discriminative Loss Function

**Key Features:**
- แยก lane แต่ละเส้นได้ (instance segmentation)
- ใช้ discriminative loss
- End-to-end learning

**Architecture:**
```python
# LaneNet Architecture
class LaneNet:
    def __init__(self):
        # Encoder: Extract features
        self.encoder = Encoder()
        
        # Binary Segmentation Branch
        self.binary_seg = BinarySegmentation()
        
        # Instance Segmentation Branch
        self.instance_seg = InstanceSegmentation()
        
    def forward(self, image):
        features = self.encoder(image)
        
        # Binary: Lane vs Background
        binary_mask = self.binary_seg(features)
        
        # Instance: Separate each lane
        instance_embedding = self.instance_seg(features)
        
        return binary_mask, instance_embedding
```

**การปรับใช้:**
- ✅ ใช้ instance segmentation แทน semantic
- ✅ แยก lane แต่ละเส้นได้ชัดเจน
- ✅ Discriminative loss function

---

### 8. **LaneNet (PyTorch)** ⭐⭐⭐⭐
**Repo:** https://github.com/harryhan618/LaneNet

**Description:**
- PyTorch implementation ของ LaneNet
- Instance segmentation
- Discriminative loss

**Technologies:**
- PyTorch
- Instance Segmentation

**สิ่งที่นำมาใช้ได้:**
- PyTorch implementation (ง่ายกว่า TensorFlow)
- Training scripts
- Evaluation metrics

---

### 9. **Lane-Detection (Multiple Algorithms)** ⭐⭐⭐⭐
**Repo:** https://github.com/sakshamjindal/Lane-Detection

**Description:**
- เปรียบเทียบ 4 algorithms:
  1. **DeepLabV3**
  2. **UNet**
  3. **FCN**
  4. **SegNet**
- PyTorch implementation
- BDD100K dataset

**Technologies:**
- PyTorch
- Multiple architectures
- Comparative study

**สิ่งที่นำมาใช้ได้:**
```python
# Compare different architectures
models = {
    'deeplabv3': DeepLabV3(),
    'unet': UNet(),
    'fcn': FCN(),
    'segnet': SegNet()
}

# Evaluate each
for name, model in models.items():
    metrics = evaluate(model, test_loader)
    print(f"{name}: IoU={metrics['iou']}, Acc={metrics['acc']}")
```

**การปรับใช้:**
- ✅ ทดสอบหลาย architectures
- ✅ เลือก model ที่ดีที่สุด
- ✅ Benchmark performance

---

## 🛠️ Useful Tools & Libraries

### 10. **Simulanes - CARLA Dataset Generator**
**Repo:** https://github.com/anita-hu/simulanes

**Description:**
- CARLA synthetic data generator
- สำหรับ lane detection training
- Sim-to-Real domain adaptation

**Technologies:**
- CARLA
- Dataset Generation
- Domain Adaptation

**สิ่งที่นำมาใช้ได้:**
```python
# Generate training data from CARLA
class CARLADataGenerator:
    def __init__(self, carla_client):
        self.client = carla_client
        
    def generate_dataset(self, num_samples):
        for i in range(num_samples):
            # Spawn vehicle
            vehicle = spawn_vehicle()
            
            # Attach camera
            camera = attach_camera(vehicle)
            
            # Capture image + ground truth
            image = camera.capture()
            ground_truth = get_lane_ground_truth()
            
            # Save
            save_sample(image, ground_truth)
```

**การปรับใช้:**
- ✅ สร้าง training data จาก CARLA
- ✅ Ground truth จาก CARLA API
- ✅ Domain adaptation techniques

---

## 📊 Comparison Table

| Repository | Stars | Language | Features | Difficulty | Usefulness |
|------------|-------|----------|----------|------------|------------|
| **Lane-Keeping-Assist-on-CARLA** | ⭐⭐⭐ | Python | MPC, CARLA | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Lane-Detection-UNet** | ⭐⭐⭐ | PyTorch | UNet, BDD100K | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **LaneNet (TensorFlow)** | ⭐⭐⭐⭐⭐ | TensorFlow | Instance Seg | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **LaneNet (PyTorch)** | ⭐⭐⭐⭐ | PyTorch | Instance Seg | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Lane-Detection (Multi)** | ⭐⭐⭐ | PyTorch | 4 Algorithms | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Simulanes** | ⭐⭐ | Python | Data Gen | ⭐⭐ | ⭐⭐⭐ |

---

## 🎯 Recommendations

### **ควรนำมาใช้ทันที:**

1. **Lane-Keeping-Assist-on-CARLA** → MPC implementation
   ```bash
   git clone https://github.com/paulyehtw/Lane-Keeping-Assist-on-CARLA
   # ดู module_7.py สำหรับ MPC controller
   ```

2. **Lane-Detection-UNet** → Training pipeline
   ```bash
   git clone https://github.com/AnshChoudhary/Lane-Detection-UNet
   # ใช้ train.py, model.py
   ```

### **ควรศึกษาและปรับใช้:**

3. **LaneNet (PyTorch)** → Instance segmentation
   ```bash
   git clone https://github.com/harryhan618/LaneNet
   # แยก lane แต่ละเส้นได้ชัดเจน
   ```

4. **Lane-Detection (Multiple)** → Compare architectures
   ```bash
   git clone https://github.com/sakshamjindal/Lane-Detection
   # ทดสอบ DeepLabV3, UNet, FCN, SegNet
   ```

### **สำหรับ Advanced:**

5. **Simulanes** → Generate training data
   ```bash
   git clone https://github.com/anita-hu/simulanes
   # สร้าง dataset จาก CARLA
   ```

---

## 💡 Integration Ideas

### **Idea 1: ใช้ LaneNet แทน UNet**

```python
# Current: UNet (semantic segmentation)
mask = unet_model(image)  # Output: binary mask

# Improved: LaneNet (instance segmentation)
binary_mask, instance_embedding = lanenet_model(image)
# Output: แยก lane แต่ละเส้น

# Benefits:
# - แยก left/right lane ได้ชัดเจน
# - ไม่ต้อง sliding window
# - Robust กว่า
```

### **Idea 2: ใช้ MPC จาก Lane-Keeping-Assist**

```python
# Clone their MPC implementation
from lane_keeping_assist import MPCController

# Integrate with your system
mpc = MPCController(
    horizon=10,
    dt=0.1,
    wheelbase=2.875
)

# Use with your lane detection
steering, throttle = mpc.solve(
    current_state=vehicle_state,
    reference_path=lane_center_from_unet
)
```

### **Idea 3: Train Better UNet with BDD100K**

```python
# Use their training pipeline
from lane_detection_unet import UNetTrainer, BDD100KDataset

# Load BDD100K dataset
dataset = BDD100KDataset(root='./bdd100k')

# Train
trainer = UNetTrainer(model=your_unet, dataset=dataset)
trainer.train(epochs=100)

# Benefits:
# - More training data
# - Better generalization
# - Real-world scenarios
```

### **Idea 4: Generate CARLA Dataset**

```python
# Use Simulanes to generate training data
from simulanes import CARLADataGenerator

# Generate 10,000 samples
generator = CARLADataGenerator(carla_client)
generator.generate_dataset(
    num_samples=10000,
    towns=['Town01', 'Town02', 'Town03'],
    weather_conditions=['Clear', 'Cloudy', 'Rain']
)

# Train your model
model.train(carla_dataset)
```

---

## 📝 Action Items

### **Immediate (ทำได้เลย):**

- [ ] Clone **Lane-Keeping-Assist-on-CARLA**
- [ ] ดู MPC implementation ใน `module_7.py`
- [ ] เปรียบเทียบกับ MPC ของคุณ
- [ ] นำ best practices มาใช้

### **Short-term (1-2 สัปดาห์):**

- [ ] Clone **Lane-Detection-UNet**
- [ ] ศึกษา training pipeline
- [ ] Train UNet ใหม่ด้วย CARLA data
- [ ] ทดสอบ performance

### **Medium-term (1 เดือน):**

- [ ] Implement **LaneNet** (instance segmentation)
- [ ] เปรียบเทียบกับ UNet
- [ ] Integrate กับ MPC
- [ ] Evaluate performance

### **Long-term (2-3 เดือน):**

- [ ] Generate large CARLA dataset
- [ ] Train multiple architectures
- [ ] Benchmark และเลือก best model
- [ ] Deploy to production

---

## 🔗 Quick Links

**CARLA + MPC:**
- https://github.com/paulyehtw/Lane-Keeping-Assist-on-CARLA

**UNet Implementations:**
- https://github.com/AnshChoudhary/Lane-Detection-UNet
- https://github.com/GirishChowdary0208/Lane_Detection_Using_UNet

**Advanced (LaneNet):**
- https://github.com/MaybeShewill-CV/lanenet-lane-detection (TensorFlow)
- https://github.com/harryhan618/LaneNet (PyTorch)

**Multi-Algorithm:**
- https://github.com/sakshamjindal/Lane-Detection

**Dataset Generation:**
- https://github.com/anita-hu/simulanes

---

**ทั้งหมดนี้สามารถนำมาปรับใช้กับโปรเจกต์ของคุณได้เลยครับ!** 🚀
