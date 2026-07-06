# ADAS Telemetry Dashboard — Grafana + InfluxDB

## สถาปัยการณ์

```
CARLA → main.py → ADASManager → TelemetryExporter
                                        ↓
                                   InfluxDB v2 (port 8086)
                                        ↓
                                   Grafana (port 3000)
                                        ↓
                                   Web Browser — ADAS Dashboard
```

## วิธีรัน

### 1. ติดตั้ง Docker (ถ้ายังไม่มี)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)

### 2. รัน InfluxDB + Grafana

```bash
cd telemetry
docker-compose up -d
```

รอ 30 วินาทีให้ services พร้อม ตรวจสถานะ:
```bash
docker-compose ps
```

### 3. ติดตั้ง Python dependency

```bash
pip install influxdb-client
```

### 4. รัน CARLA + ADAS

```bash
# Terminal 1: รัน CARLA server
./CarlaUE4-Win64-Shipping.exe

# Terminal 2: รัน ADAS system
python src/main.py --classical
```

### 5. เปิด Grafana Dashboard

เปิดเบราว์เซอร์: **http://localhost:3000**

- Username: `admin`
- Password: `admin`

Dashboard "ADAS Full Suite Dashboard" จะปรากฏในโฟลเดอร์ ADAS โดยอัตโนมัติ

## Measurements ใน InfluxDB

| Measurement | Fields | คำอธิบาย |
|-------------|--------|----------|
| `vehicle` | speed_ms, speed_kmh, steering, throttle, brake | สถานะรถ |
| `lane` | cte_m, heading_err_rad, curvature, lane_conf | lane detection |
| `aeb` | active, ttc, brake_override | AEB status |
| `acc` | active, target_speed_ms, distance_m | ACC status |
| `ldw` | state, state_str, warning_active, steering_assist | LDW + LKA Pro |
| `bsw` | left_alert, right_alert, safe_left, safe_right | Blind Spot |
| `tsr` | speed_limit_kmh, traffic_light, traffic_light_str | TSR + Traffic Light |
| `tja` | state, state_str, active, target_speed_ms | Traffic Jam Assist |
| `stop_and_go` | stopped | Stop & Go |
| `lka_pro` | assist | LKA Pro steering assist |
| `mpc` | solve_time_ms, status, fallback_count | MPC solver |
| `performance` | fps, loop_time_ms | Performance metrics |

## Dashboard Panels

1. **Gauges**: Speed, Steering, Throttle, Brake
2. **Status**: AEB active, AEB TTC, LDW state, BSW left/right, Traffic Light, TJA state, Speed Limit
3. **Time-series**: Speed vs Target, CTE & Heading, ACC Distance, LKA Pro Assist, MPC Solve Time, FPS

## การปรับแต่ง

### เปลี่ยน InfluxDB credentials
แก้ `telemetry/docker-compose.yml`:
```yaml
environment:
  - DOCKER_INFLUXDB_INIT_PASSWORD=your_password
  - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=your_token
```

แล้วแก้ `src/telemetry/influxdb_exporter.py`:
```python
TelemetryExporter(token="your_token", org="adas", bucket="adas_telemetry")
```

### ปิด telemetry (กรณีไม่มี Docker)
```python
# ใน main.py
self.telemetry = TelemetryExporter(enabled=False)
```

หรือถ้าไม่ติดตั้ง `influxdb-client` telemetry จะปิดอัตโนมัติ

## การหยุด

```bash
cd telemetry
docker-compose down          # หยุด containers
docker-compose down -v       # หยุด + ลบ data
```
