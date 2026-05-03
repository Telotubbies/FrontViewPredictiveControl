# แนะนำ: carla_mpc_classical vs carla_lstm_mpc_project

## เปรียบเทียบสั้นๆ

| หัวข้อ | carla_mpc_classical | carla_lstm_mpc_project |
|--------|----------------------|------------------------|
| **Perception** | UFLDv2 (pretrained) → RANSAC polyfit → vector | U-Net/ResNet (ต้อง train) → mask → features |
| **Temporal** | Kalman Filter (ไม่ต้อง train) | LSTM (ต้อง train จาก data) |
| **Control** | MPC + slip term | MPC |
| **ต้อง train ไหม** | **ไม่ต้อง** — ใช้ weight UFLDv2 อย่างเดียว | ต้อง: เก็บ data → สร้าง lane mask → train U-Net (+ LSTM, ResNet ถ้าใช้) |
| **ความซับซ้อน** | น้อยกว่า โฟลเดอร์เล็ก ไฟล์ชัด | มากกว่า หลาย phase, หลาย model |
| **ผลลัพธ์ (จาก results_matrix)** | ยังไม่ได้รันครบตามเกณฑ์ C1–C5 | S1 FAIL; S2–S5 PENDING |

---

## คำแนะนำ: **ใช้ carla_mpc_classical เป็นหลัก**

### เหตุผล

1. **ไม่ต้อง train**  
   แค่มี weight UFLDv2 (.pth) ก็รันได้เลย ปรับแค่ RANSAC / Kalman / MPC ไม่ต้องเก็บข้อมูลหรือ train U-Net/LSTM

2. **เหมาะกับเป้า lane keeping**  
   ทาง rule (Incremental Upgrade A) ออกแบบมาให้ UFLD → RANSAC → Kalman → MPC ครบใน classical pipeline และมี automation test loop (C1–C5) ให้ไล่ tune ได้

3. **ดูแลง่าย**  
   โครงสร้างน้อย ชัดเจน แก้ perception/temporal/control แยกกันได้ ไม่ต้องพึ่ง training pipeline

4. **carla_lstm_mpc_project ยังไม่ผ่านเกณฑ์**  
   results_matrix บอก S1 ยัง FAIL ต้องลงทุนเทสและ tune หลายจุด (data, mask, U-Net, LSTM) กว่าจะได้ lane keeping ที่มั่นคง

---

## เมื่อไหร่ถึงค่อยใช้ carla_lstm_mpc_project

- อยากได้ **lane mask โดยตรง** (เช่น ใช้ mask ไปทำ task อื่น นอกเหนือจาก lane keeping)
- อยากลอง **LSTM ทำ temporal** หรือ **ResNet ทำ feature** เป็นงานวิจัย/ทดลอง
- มี **ข้อมูลและเวลาพร้อม** จะเก็บ data, สร้าง mask, train U-Net (+ LSTM) และไล่เทสจนผ่าน C1–C5

---

## สรุปหนึ่งบรรทัด

**แนะนำ carla_mpc_classical** สำหรับ lane keeping ตอนนี้ เพราะไม่ต้อง train ใช้ได้ทันที และโฟกัส tune RANSAC + Kalman + MPC ให้ผ่าน C1–C5 ได้ตรงกับ rule; ใช้ carla_lstm_mpc_project เมื่อต้องการ mask/model หลายตัวและพร้อมลงทุนใน data + training
