# Bridge (จาก carla_apollo_bridge)

Adapter จาก [carla_apollo_bridge](https://github.com/guardstrikelab/carla_apollo_bridge) — **ไม่มี Apollo/Cyber dependency**

## ใช้ทำอะไร

- **transforms**: แปลงพิกัด CARLA (Unreal left-handed) ↔ right-handed (ENU-style)
- **obstacles**: แปลง CARLA actor (vehicle/walker) → `Obstacle` (Apollo-style) สำหรับใช้ใน pipeline

## ตัวอย่าง

```python
from bridge import get_traffic_obstacles, ObstacleType

# ใน world (CARLA)
obstacles = get_traffic_obstacles(world, exclude_actor_id=ego_vehicle.id)
for obs in obstacles:
    print(obs.id, obs.position, obs.velocity, obs.type)  # ObstacleType.VEHICLE / PEDESTRIAN
```

## โครงสร้าง

- `transforms.py` — คำนวณพิกัด/ความเร็ว/heading (ไม่มี protobuf)
- `obstacles.py` — `Obstacle` dataclass, `carla_actor_to_obstacle()`, `get_traffic_obstacles()`

## Integration

- `run_unet_mpc.py` เรียก `get_traffic_obstacles(w, exclude_actor_id=veh.id)` ทุก frame และส่งไป Dashboard เพื่อแสดง "Obstacles: N"
