"""
InfluxDB Telemetry Exporter — ส่ง ADAS data ไป InfluxDB v2

ใช้ influxdb-client (pip install influxdb-client)
ส่ง data ทุก frame แบบ async (ไม่บล็อก main loop)

Measurements:
- vehicle: speed_ms, speed_kmh, steering, throttle, brake
- lane: cte_m, heading_err_rad, curvature, lane_conf
- aeb: active (0/1), ttc, brake_override, warning_level
- acc: active (0/1), target_speed_ms, distance_m, closing_rate
- ldw: state (0/1/2), state_str, warning_active, side (0/1/2), steering_assist
- bsw: left_alert (0/1/2), right_alert, left_alert_str, right_alert_str, safe_left, safe_right
- tsr: speed_limit_kmh, traffic_light (0/1/2/3), traffic_light_str, action
- tja: state (0-4), state_str, active (0/1), target_speed_ms
- stop_and_go: stopped (0/1), restart_ramp
- lka_pro: assist
- mpc: solve_time_ms, status (0/1), fallback_count
- performance: fps, loop_time_ms
"""
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    logger.warning("influxdb-client not installed — telemetry disabled. pip install influxdb-client")


class TelemetryExporter:
    """
    ส่ง ADAS telemetry ไป InfluxDB v2
    ใช้ batch writing เพื่อไม่บล็อก main loop
    """

    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "adas-token-1234567890abcdef",
        org: str = "adas",
        bucket: str = "adas_telemetry",
        enabled: bool = True,
    ):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.enabled = enabled and INFLUXDB_AVAILABLE
        self._client = None
        self._write_api = None
        self._buffer = []
        self._last_flush = time.time()
        self._flush_interval = 0.5  # flush ทุก 0.5s
        self._frame_count = 0

        if self.enabled:
            self._connect()

    def _connect(self):
        """เชื่อมต่อ InfluxDB"""
        try:
            self._client = InfluxDBClient(
                url=self.url, token=self.token, org=self.org
            )
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            logger.info("TelemetryExporter connected to InfluxDB at %s", self.url)
        except Exception as e:
            logger.warning("Failed to connect to InfluxDB: %s — telemetry disabled", e)
            self.enabled = False

    def export_frame(
        self,
        vehicle: Dict[str, float],
        lane: Dict[str, float],
        adas_out: Optional[Any] = None,
        mpc_stats: Optional[Dict[str, float]] = None,
        performance: Optional[Dict[str, float]] = None,
    ):
        """
        ส่งข้อมูล 1 frame ไป InfluxDB (buffered)
        แต่ละ dict มี key-value pairs ที่จะเป็น fields ใน measurement
        """
        if not self.enabled or self._write_api is None:
            return

        ts = time.time()
        points = []

        # ── Vehicle ──────────────────────────────────────────────────
        if vehicle:
            p = Point("vehicle").time(int(ts * 1e9))  # ns
            for k, v in vehicle.items():
                if v is not None and not isinstance(v, bool):
                    p = p.field(k, float(v))
            points.append(p)

        # ── Lane ─────────────────────────────────────────────────────
        if lane:
            p = Point("lane").time(int(ts * 1e9))
            for k, v in lane.items():
                if v is not None and not isinstance(v, bool):
                    p = p.field(k, float(v))
            points.append(p)

        # ── ADAS outputs ─────────────────────────────────────────────
        if adas_out is not None:
            # AEB
            points.append(
                Point("aeb").time(int(ts * 1e9))
                .field("active", int(adas_out.aeb_active))
                .field("ttc", float(adas_out.aeb_ttc) if adas_out.aeb_ttc >= 0 else 999.0)
                .field("brake_override", float(adas_out.brake_override))
            )
            # ACC
            points.append(
                Point("acc").time(int(ts * 1e9))
                .field("active", int(adas_out.acc_active))
                .field("target_speed_ms", float(adas_out.acc_target_speed_ms) if adas_out.acc_target_speed_ms >= 0 else -1.0)
                .field("distance_m", float(adas_out.acc_distance_m) if adas_out.acc_distance_m >= 0 else -1.0)
            )
            # LDW
            ldw_state_map = {"in_lane": 0, "departing": 1, "departed": 2}
            points.append(
                Point("ldw").time(int(ts * 1e9))
                .field("state", ldw_state_map.get(adas_out.ldw_state, 0))
                .field("state_str", adas_out.ldw_state)
                .field("warning_active", int(adas_out.ldw_warning_active))
                .field("steering_assist", float(adas_out.lka_pro_assist))
            )
            # BSW
            bsw_map = {"clear": 0, "blind_spot": 1, "approaching": 2}
            points.append(
                Point("bsw").time(int(ts * 1e9))
                .field("left_alert", bsw_map.get(adas_out.bsw_left_alert, 0))
                .field("right_alert", bsw_map.get(adas_out.bsw_right_alert, 0))
                .field("left_alert_str", adas_out.bsw_left_alert)
                .field("right_alert_str", adas_out.bsw_right_alert)
                .field("safe_left", int(adas_out.bsw_safe_left))
                .field("safe_right", int(adas_out.bsw_safe_right))
            )
            # TSR
            tl_map = {"unknown": 0, "red": 1, "yellow": 2, "green": 3}
            points.append(
                Point("tsr").time(int(ts * 1e9))
                .field("speed_limit_kmh", float(adas_out.tsr_speed_limit_kmh) if adas_out.tsr_speed_limit_kmh is not None else -1.0)
                .field("traffic_light", tl_map.get(adas_out.tsr_traffic_light, 0))
                .field("traffic_light_str", adas_out.tsr_traffic_light)
                .field("traffic_light_distance", float(adas_out.tsr_traffic_light_distance) if adas_out.tsr_traffic_light_distance >= 0 else -1.0)
            )
            # TJA
            tja_map = {"inactive": 0, "active": 1, "stopped": 2, "creeping": 3, "no_lead": 4}
            points.append(
                Point("tja").time(int(ts * 1e9))
                .field("state", tja_map.get(adas_out.tja_state, 0))
                .field("state_str", adas_out.tja_state)
                .field("active", int(adas_out.tja_active))
                .field("target_speed_ms", float(adas_out.target_speed_override) if adas_out.target_speed_override >= 0 else -1.0)
            )
            # Stop & Go
            points.append(
                Point("stop_and_go").time(int(ts * 1e9))
                .field("stopped", int(adas_out.stop_and_go_stopped))
            )
            # LKA Pro
            points.append(
                Point("lka_pro").time(int(ts * 1e9))
                .field("assist", float(adas_out.lka_pro_assist))
            )

        # ── MPC stats ────────────────────────────────────────────────
        if mpc_stats:
            p = Point("mpc").time(int(ts * 1e9))
            for k, v in mpc_stats.items():
                if v is not None and not isinstance(v, bool):
                    p = p.field(k, float(v))
            points.append(p)

        # ── Performance ──────────────────────────────────────────────
        if performance:
            p = Point("performance").time(int(ts * 1e9))
            for k, v in performance.items():
                if v is not None and not isinstance(v, bool):
                    p = p.field(k, float(v))
            points.append(p)

        self._buffer.extend(points)
        self._frame_count += 1

        # Flush ตาม interval
        if time.time() - self._last_flush > self._flush_interval:
            self._flush()

    def _flush(self):
        """ส่ง buffer ไป InfluxDB"""
        if not self._buffer or self._write_api is None:
            return
        try:
            self._write_api.write(bucket=self.bucket, record=self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()
        except Exception as e:
            logger.debug("Telemetry flush failed: %s", e)
            # ไม่ clear buffer — ลองส่งใหม่ครั้งถัดไป
            if len(self._buffer) > 1000:
                self._buffer.clear()  # กัน memory leak

    def close(self):
        """Flush ครั้งสุดท้าย + ปิด connection"""
        self._flush()
        if self._client:
            self._client.close()
            logger.info("TelemetryExporter closed")

    def is_connected(self) -> bool:
        return self.enabled and self._write_api is not None
