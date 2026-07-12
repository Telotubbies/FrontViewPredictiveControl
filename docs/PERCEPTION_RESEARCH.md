# Perception Module — Research Summary (Phase 2)

## เปรียบเทียบ Tesla FSD Stack กับ FrontViewPredictiveControl

### สิ่งที่เรามี (Current State)

| Feature | ของเรา | Tesla FSD |
|---------|--------|-----------|
| Lane Detection | CARLA waypoint-based + classical CV (Canny/Sobel) | HydraNet backbone + lane head |
| Lane Fitting | Polynomial (2nd order) in vehicle frame | Vectorized lane segments in 3D |
| Lane Topology | None (single ego lane only) | Autoregressive transformer for lane connectivity |
| BEV Projection | Homography (flat ground assumption) | Transformer-based Lift-Splat-Shoot |
| Depth Estimation | None | Implicit in BEV transformer |
| Object Detection | None (CARLA ground truth only) | HydraNet detection head |
| Traffic Sign/Light | Basic HOG + template matching | HydraNet traffic head |
| Multi-Camera | Single front camera | 8 cameras → unified vector space |
| Temporal Fusion | Kalman filter on poly coeffs | Temporal self-attention in BEV |
| Sensor Fusion | None | Camera-only (vision) |

### สิ่งที่ขาด (Gaps)

1. **Monocular Depth Estimation** — ไม่มี depth จากกล้อง ทำให้ BEV projection ใช้ flat-ground assumption ซึ่งผิดบนทางลาด/โค้ง
2. **Lane Topology / Connectivity** — ไม่เข้าใจว่า lane ไหนเชื่อมกับ lane ไหนที่ intersection (สำคัญมากสำหรับ planning)
3. **Object Detection** — ไม่มี detector ของจริง (ใช้ CARLA ground truth) จำเป็นสำหรับ obstacle avoidance ในโลกจริง
4. **BEV from Camera** — ใช้ homography ไม่ใช่ learned projection → ไม่ robust กับ road gradient
5. **Multi-Task Backbone** — รันแยกทุก task (lane, traffic sign) ไม่ได้ share features

---

## Top 5 Papers/Projects น่า Implement (เรียงตามความสำคัญ)

### 1. Depth Anything V2 — Monocular Depth Estimation
- **Source**: [github.com/DepthAnything/Depth-Anything-V2](https://github.com/depthanything/depth-anything-v2) (NeurIPS 2024)
- **ทำไมสำคัญ**: ปัจจุบัน BEV projection ของเราใช้ flat-ground assumption → ผิดบนทางลาด ถ้ามี depth map จะ project ถูกต้องขึ้น
- **ยาก/ง่าย**: ง่าย — มี pretrained model พร้อมใช้, รองรับ CUDA/CPU, มี HuggingFace integration
- **Implementation**: โหลด Depth-Anything-V2-Small (25M params) → รัน inference → ใช้ depth map แทน flat-ground ใน `_image_to_ground`
- **Priority**: **สูงสุด** — แก้ปัญหา BEV projection โดยตรง

### 2. CLRNet / CLRerNet — Lane Detection with LaneIoU
- **Source**: [CLRerNet (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Honda_CLRerNet_Improving_Confidence_of_Lane_Detection_With_LaneIoU_WACV_2024_paper.pdf) + [CLRNet](https://github.com/Turoad/CLRNet)
- **ทำไมสำคัญ**: ตรวจจับเลนแบบ anchor-based ที่เร็วและแม่นยำ รองรับโค้งและ intersection ดีกว่า polynomial fitting
- **ยาก/ง่าย**: ปานกลาง — ต้อง train หรือ fine-tune บน CARLA dataset
- **Implementation**: ใช้ pretrained CLRNet บน CULane → fine-tune บน CARLA → แทนที่ classical detector
- **Priority**: **สูง** — ปรับปรุง lane detection โดยรวม

### 3. BEVFormer / Lift-Splat-Shoot — Camera-Only BEV
- **Source**: [BEVFormer (ECCV 2022, TPAMI 2024)](https://github.com/fundamentalvision/BEVFormer) + [Camera-Only BEV (2025)](https://arxiv.org/html/2505.06113v1)
- **ทำไมสำคัญ**: แทน homography ด้วย learned BEV projection → ไม่ต้องสมมติ flat ground
- **ยาก/ง่าย**: ยาก — ต้อง multi-camera setup และ train บน large dataset
- **Implementation**: เริ่มจาก Lift-Splat-Shoot (ง่ายกว่า BEVFormer) → ใช้ Depth-Anything สำหรับ depth estimation → project to BEV
- **Priority**: **ปานกลาง** — ใช้เวลานาน แต่ impact สูง

### 4. OpenLane-V2 — Lane Topology Reasoning
- **Source**: [OpenLane-V2 (NeurIPS 2023)](https://github.com/OpenDriveLab/OpenLane-V2)
- **ทำไมสำคัญ**: ทำให้รถเข้าใจว่า lane ไหนเชื่อมกับ lane ไหนที่ intersection → สำคัญสำหรับ lane change และ intersection handling
- **ยาก/ง่าย**: ยาก — เป็น reasoning task ต้องการ graph neural network
- **Implementation**: ใช้ scene graph จาก lane detection → topology prediction head
- **Priority**: **ปานกลาง** — สำคัญสำหรับ planning แต่ซับซ้อน

### 5. HydraNet — Multi-Task Perception Backbone
- **Source**: [Tesla-HydraNet](https://github.com/sushantgagneja1310/Tesla-HydraNet) (educational)
- **ทำไมสำคัญ**: Share backbone ระหว่าง lane detection, object detection, traffic sign → ลด latency และเพิ่ม feature quality
- **ยาก/ง่าย**: ยาก — ต้อง train multi-task พร้อมกัน
- **Implementation**: ResNet-50 backbone + BiFPN neck + multiple heads (lane, object, traffic)
- **Priority**: **ต่ำ** — ดีแต่ไม่จำเป็นสำหรับ single-camera CARLA setup

---

## ลำดับ Implementation (Phase 3)

1. **Depth Anything V2** → แทน flat-ground assumption ใน BEV projection
2. **CLRNet** → แทน classical lane detector
3. **Object Detection (YOLOv8/v11)** → เพิ่ม real object detection (ไม่ใช้ CARLA ground truth)
4. **Lane Topology** → เพิ่ม lane connectivity reasoning
5. **BEVFormer** → ถ้ามีเวลา, เปลี่ยนเป็น learned BEV

---

## References

1. Depth Anything V2 — Yang et al., NeurIPS 2024. [GitHub](https://github.com/depthanything/depth-anything-v2)
2. BEVFormer — Li et al., ECCV 2022 / TPAMI 2024. [GitHub](https://github.com/fundamentalvision/BEVFormer)
3. CLRerNet — Honda, WACV 2024. [Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Honda_CLRerNet_Improving_Confidence_of_Lane_Detection_With_LaneIoU_WACV_2024_paper.pdf)
4. OpenLane-V2 — OpenDriveLab, NeurIPS 2023. [GitHub](https://github.com/OpenDriveLab/OpenLane-V2)
5. Camera-Only BEV — Zhang et al., 2025. [arXiv](https://arxiv.org/html/2505.06113v1)
6. Tesla HydraNet — Sushant Gagneja. [GitHub](https://github.com/sushantgagneja1310/Tesla-HydraNet)
7. Tesla FSD Lane Connectivity Patent — US 2026/0170852 A1
