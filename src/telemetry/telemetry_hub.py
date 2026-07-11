"""
TelemetryHub — Facade ที่รวมทุก telemetry component ไว้ในจุดเดียว

Main control loop เรียกแค่ TelemetryHub.update() ทุก frame แล้ว hub จะกระจายงาน
ไปยัง sub-components ทั้งหมด (BehaviorLogger, LaneLossDetector, BehaviorContextLogger,
CarlaDataSampler, RunLogger, AiReportGenerator) โดยอัตโนมัติ

การออกแบบ:
- ทุก sub-component import แบบ lazy + try/except → ถ้า module ไหนยังไม่มี จะข้ามไป
  และ log warning แต่ hub ยังทำงานต่อได้
- 1 method ต่อ frame: update() → คืน dict ของ immediate alerts
- stop() → สรุปผล + สร้าง AI report แล้วคืน dict ให้ caller
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TelemetryHub:
    """
    Facade สำหรับรวม telemetry ทุกตัว

    Usage ใน main loop::

        hub = TelemetryHub(run_dir="runs/exp1", snapshot_dir="runs/exp1/snapshots")
        hub.start(run_meta={"map": "Town04", "target_speed_kmh": 30})
        for frame_idx in range(N):
            steer, throttle, brake, frame_state = pipeline.step()
            ctx = {"frame_idx": frame_idx, "speed_ms": ..., "loop_time_ms": ...}
            alerts = hub.update(frame_state, ctx, vehicle=vehicle, world=world)
            if alerts["lane_lost"]:
                ...
        report = hub.stop()
    """

    def __init__(
        self,
        run_dir: str,
        snapshot_dir: Optional[str] = None,
        enable_carla_sampler: bool = True,
        enable_context_logger: bool = True,
        map_name: str = "unknown",
        vehicle_type: str = "unknown",
        target_speed_kmh: float = 0.0,
    ) -> None:
        """
        Args:
            run_dir: directory สำหรับเก็บผลลัพธ์ของ run นี้
            snapshot_dir: directory สำหรับ snapshot metadata (default = run_dir/snapshots)
            enable_carla_sampler: เปิด/ปิด CarlaDataSampler
            enable_context_logger: เปิด/ปิด BehaviorContextLogger
            map_name: ชื่อแผนที่ (ส่งต่อให้ RunLogger)
            vehicle_type: ชื่อรถ (ส่งต่อให้ RunLogger)
            target_speed_kmh: ความเร็วเป้าหมาย (ส่งต่อให้ RunLogger)
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if snapshot_dir is None:
            snapshot_dir = str(self.run_dir / "snapshots")
        self.snapshot_dir = snapshot_dir

        self._enable_carla_sampler = enable_carla_sampler
        self._enable_context_logger = enable_context_logger

        # ── Sub-components (lazy import + graceful fallback) ───────────────
        self.behavior_logger = None
        self.lane_loss_detector = None
        self.context_logger = None
        self.carla_sampler = None
        self.run_logger = None
        self.ai_report_generator = None

        self._init_behavior_logger()
        self._init_lane_loss_detector()
        self._init_context_logger()
        self._init_carla_sampler()
        self._init_run_logger(map_name, vehicle_type, target_speed_kmh)
        self._init_ai_report_generator()

        # Runtime state
        self._frame_idx: int = 0
        self._active_warnings: List[str] = []
        self._lane_lost: bool = False
        self._last_incident: Optional[Dict[str, Any]] = None
        self._run_meta: Dict[str, Any] = {}

    # ── Sub-component initializers (each isolated) ────────────────────────

    def _init_behavior_logger(self) -> None:
        try:
            from telemetry.behavior_logger import BehaviorLogger
            self.behavior_logger = BehaviorLogger(snapshot_dir=self.snapshot_dir)
            logger.info("TelemetryHub: BehaviorLogger loaded")
        except Exception as exc:  # noqa: BLE001
            logger.warning("TelemetryHub: BehaviorLogger unavailable — %s", exc)

    def _init_lane_loss_detector(self) -> None:
        try:
            from telemetry.lane_loss_detector import LaneLossDetector
            self.lane_loss_detector = LaneLossDetector()
            logger.info("TelemetryHub: LaneLossDetector loaded")
        except Exception as exc:  # noqa: BLE001
            logger.warning("TelemetryHub: LaneLossDetector unavailable — %s", exc)

    def _init_context_logger(self) -> None:
        if not self._enable_context_logger:
            return
        try:
            from telemetry.behavior_context_logger import BehaviorContextLogger
            self.context_logger = BehaviorContextLogger()
            logger.info("TelemetryHub: BehaviorContextLogger loaded")
        except Exception as exc:  # noqa: BLE001
            logger.warning("TelemetryHub: BehaviorContextLogger unavailable — %s", exc)

    def _init_carla_sampler(self) -> None:
        if not self._enable_carla_sampler:
            return
        try:
            from telemetry.carla_data_sampler import CarlaDataSampler
            self.carla_sampler = CarlaDataSampler()
            logger.info("TelemetryHub: CarlaDataSampler loaded")
        except Exception as exc:  # noqa: BLE001
            logger.warning("TelemetryHub: CarlaDataSampler unavailable — %s", exc)

    def _init_run_logger(
        self, map_name: str, vehicle_type: str, target_speed_kmh: float
    ) -> None:
        try:
            from metrics.run_logger import RunLogger
            self.run_logger = RunLogger(
                run_dir=str(self.run_dir),
                map_name=map_name,
                vehicle_type=vehicle_type,
                target_speed_kmh=target_speed_kmh,
            )
            logger.info("TelemetryHub: RunLogger loaded")
        except Exception as exc:  # noqa: BLE001
            logger.warning("TelemetryHub: RunLogger unavailable — %s", exc)

    def _init_ai_report_generator(self) -> None:
        try:
            from telemetry.ai_report_generator import AiReportGenerator
            self.ai_report_generator = AiReportGenerator()
            logger.info("TelemetryHub: AiReportGenerator loaded")
        except Exception as exc:  # noqa: BLE001
            logger.warning("TelemetryHub: AiReportGenerator unavailable — %s", exc)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self, run_meta: Optional[Dict[str, Any]] = None) -> None:
        """
        เริ่มทุก sub-component ก่อนเข้า main loop

        Args:
            run_meta: metadata ของ run (map, vehicle, target speed, timestamp, ...)
        """
        self._run_meta = run_meta or {}
        if self.run_logger:
            try:
                self.run_logger.start()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TelemetryHub: RunLogger.start() failed — %s", exc)
        logger.info("TelemetryHub started → %s", self.run_dir)

    def update(
        self,
        frame_state,
        context: Dict[str, Any],
        vehicle: Any = None,
        world: Any = None,
    ) -> Dict[str, Any]:
        """
        เรียกทุก frame — กระจายงานไปยัง sub-components ทั้งหมด

        Args:
            frame_state: FrameState จาก pipeline.step()
            context: dict ที่มีอย่างน้อย 'frame_idx' (และ speed_ms, loop_time_ms, ...)
            vehicle: CARLA vehicle actor (optional, ใช้กับ CarlaDataSampler)
            world: CARLA world object (optional, ใช้กับ CarlaDataSampler)

        Returns:
            dict ของ immediate alerts::

                {
                    "lane_lost": bool,
                    "incident": Optional[dict],   # lane loss incident ถ้ามี
                    "warnings": List[str],        # active warnings ปัจจุบัน
                }
        """
        frame_idx = int(context.get("frame_idx", self._frame_idx))
        self._frame_idx = frame_idx
        warnings: List[str] = []
        incident_dict: Optional[Dict[str, Any]] = None
        lane_lost = False

        # 1) BehaviorLogger
        if self.behavior_logger:
            try:
                self.behavior_logger.update(frame_state, context)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"behavior_logger error: {exc}")
                logger.debug("TelemetryHub: behavior_logger.update error — %s", exc)

        # 2) LaneLossDetector → ถ้ามี incident ให้ record_event ใน context logger
        if self.lane_loss_detector:
            try:
                incident = self.lane_loss_detector.update(frame_state, context)
                if incident is not None:
                    lane_lost = True
                    self._lane_lost = True
                    incident_dict = self._incident_to_dict(incident)
                    self._last_incident = incident_dict
                    warnings.append(
                        f"Lane loss detected at frame {frame_idx}"
                    )
                    # ส่งต่อให้ BehaviorContextLogger บันทึก snapshot
                    if self.context_logger:
                        try:
                            self.context_logger.record_event(
                                event_type=getattr(incident, "event_type", "lane_loss"),
                                severity=getattr(incident, "severity", "critical"),
                                message=getattr(incident, "message", "lane loss"),
                                frame_idx=frame_idx,
                                data=incident_dict,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "TelemetryHub: context_logger.record_event error — %s",
                                exc,
                            )
                else:
                    # กู้คืน lane
                    if self._lane_lost:
                        self._lane_lost = False
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"lane_loss_detector error: {exc}")
                logger.debug("TelemetryHub: lane_loss_detector.update error — %s", exc)

        # 3) BehaviorContextLogger (per-frame update)
        if self.context_logger:
            try:
                self.context_logger.update(frame_state, frame_idx)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"context_logger error: {exc}")
                logger.debug("TelemetryHub: context_logger.update error — %s", exc)

        # 4) CarlaDataSampler (ต้องมี vehicle + world)
        if self.carla_sampler and vehicle is not None and world is not None:
            try:
                self.carla_sampler.sample(vehicle, world, frame_idx, frame_state)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"carla_sampler error: {exc}")
                logger.debug("TelemetryHub: carla_sampler.sample error — %s", exc)

        # 5) RunLogger (low-level CSV)
        if self.run_logger:
            try:
                self.run_logger.log_frame(frame_state, context)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"run_logger error: {exc}")
                logger.debug("TelemetryHub: run_logger.log_frame error — %s", exc)

        # รวม warnings ที่ active อยู่
        self._active_warnings = warnings

        return {
            "lane_lost": lane_lost,
            "incident": incident_dict,
            "warnings": warnings,
        }

    def log_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        frame_idx: int = 0,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        บันทึก event สำคัญ (AEB, stuck, fallback, ...) ไปยัง RunLogger และ
        BehaviorContextLogger พร้อมกัน

        Args:
            event_type: ประเภท event เช่น 'aeb', 'stuck', 'safety'
            severity: 'info' | 'warning' | 'error' | 'critical'
            message: ข้อความอธิบาย
            frame_idx: frame ที่เกิด event
            data: metadata เพิ่มเติม
        """
        if self.run_logger:
            try:
                self.run_logger.log_event(
                    event_type=event_type,
                    severity=severity,
                    message=message,
                    frame_idx=frame_idx,
                    data=data,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("TelemetryHub: run_logger.log_event error — %s", exc)

        if self.context_logger:
            try:
                self.context_logger.record_event(
                    event_type=event_type,
                    severity=severity,
                    message=message,
                    frame_idx=frame_idx,
                    data=data or {},
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("TelemetryHub: context_logger.record_event error — %s", exc)

    def stop(self) -> Dict[str, Any]:
        """
        หยุดทุก sub-component และสร้าง final AI report

        Returns:
            dict::

                {
                    "summary": dict,              # BehaviorSummary.to_dict()
                    "report_text": str,           # AI readable text report
                    "report_json": dict,          # structured JSON report
                    "lane_loss_incidents": list,  # รายการ lane loss incidents
                    "behavior_snapshots": list,   # behavior context snapshots
                }
        """
        # หยุด RunLogger (flush CSV)
        if self.run_logger:
            try:
                self.run_logger.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TelemetryHub: run_logger.stop() failed — %s", exc)

        # สรุป BehaviorSummary
        summary_dict: Dict[str, Any] = {}
        behavior_summary = None
        if self.behavior_logger:
            try:
                behavior_summary = self.behavior_logger.get_summary()
                summary_dict = behavior_summary.to_dict()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TelemetryHub: get_summary() failed — %s", exc)

        # รวบรวม events
        events_list: List[Dict[str, Any]] = []
        if self.behavior_logger:
            try:
                events_list = [e.to_dict() for e in self.behavior_logger.events]
            except Exception as exc:  # noqa: BLE001
                logger.debug("TelemetryHub: events collect error — %s", exc)

        # รวบรวม lane loss incidents
        lane_loss_incidents: List[Dict[str, Any]] = []
        if self.lane_loss_detector:
            try:
                for inc in self.lane_loss_detector.get_incidents():
                    lane_loss_incidents.append(self._incident_to_dict(inc))
            except Exception as exc:  # noqa: BLE001
                logger.debug("TelemetryHub: lane loss incidents collect error — %s", exc)

        # รวบรวม behavior context snapshots
        behavior_snapshots: List[Dict[str, Any]] = []
        if self.context_logger:
            try:
                for snap in self.context_logger.get_snapshots():
                    behavior_snapshots.append(self._snapshot_to_dict(snap))
            except Exception as exc:  # noqa: BLE001
                logger.debug("TelemetryHub: snapshots collect error — %s", exc)

        # สร้าง AI report
        report_text = ""
        report_json: Dict[str, Any] = {}
        if self.ai_report_generator:
            try:
                report_text = self.ai_report_generator.generate_report(
                    behavior_summary=behavior_summary,
                    events=events_list,
                    lane_loss_incidents=lane_loss_incidents,
                    run_meta=self._run_meta,
                )
                report_json = self.ai_report_generator.generate_json_report(
                    behavior_summary=behavior_summary,
                    events=events_list,
                    lane_loss_incidents=lane_loss_incidents,
                    run_meta=self._run_meta,
                )
                # บันทึกลงไฟล์
                self.ai_report_generator.save_report(
                    report_text, str(self.run_dir / "ai_report.txt")
                )
                with open(self.run_dir / "ai_report.json", "w", encoding="utf-8") as f:
                    json.dump(report_json, f, indent=2, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("TelemetryHub: AI report generation failed — %s", exc)
        else:
            # Fallback: เขียน summary อย่างน้อยเป็น JSON
            try:
                with open(self.run_dir / "ai_report.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "summary": summary_dict,
                            "events": events_list,
                            "lane_loss_incidents": lane_loss_incidents,
                            "behavior_snapshots": behavior_snapshots,
                            "run_meta": self._run_meta,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("TelemetryHub: fallback report write failed — %s", exc)

        logger.info("TelemetryHub stopped → %s", self.run_dir)

        return {
            "summary": summary_dict,
            "report_text": report_text,
            "report_json": report_json,
            "lane_loss_incidents": lane_loss_incidents,
            "behavior_snapshots": behavior_snapshots,
        }

    # ── Dashboard helpers ─────────────────────────────────────────────────

    def get_realtime_alerts(self) -> List[str]:
        """
        คืน active warnings ปัจจุบันสำหรับแสดงใน dashboard

        Returns:
            List[str] ของ warning messages
        """
        alerts: List[str] = list(self._active_warnings)

        # เพิ่ม lane loss text จาก detector ถ้ามี
        if self.lane_loss_detector:
            try:
                text = self.lane_loss_detector.to_ai_readable_text()
                if text:
                    alerts.append(text)
            except Exception:  # noqa: BLE001
                pass

        # เพิ่ม context logger text ถ้ามี
        if self.context_logger:
            try:
                text = self.context_logger.to_ai_readable_text()
                if text:
                    alerts.append(text)
            except Exception:  # noqa: BLE001
                pass

        return alerts

    def get_lane_health(self) -> Dict[str, Any]:
        """
        คืน snapshot สุขภาพ lane detection ปัจจุบันสำหรับ dashboard

        Returns:
            dict::

                {
                    "conf": float,
                    "geometry_valid": bool,
                    "phase_status": dict,   # P1..P5
                    "mode": str,
                    "lane_lost": bool,
                    "cte": float,
                }
        """
        # ค่า default
        health: Dict[str, Any] = {
            "conf": 0.0,
            "geometry_valid": False,
            "phase_status": {
                "p1": False,
                "p2": False,
                "p3": False,
                "p4": False,
                "p5": False,
            },
            "mode": "unknown",
            "lane_lost": self._lane_lost,
            "cte": 0.0,
        }

        # ดึงจาก LaneLossDetector summary ถ้ามี
        if self.lane_loss_detector:
            try:
                det_summary = self.lane_loss_detector.get_summary()
                if isinstance(det_summary, dict):
                    health["conf"] = float(det_summary.get("conf", health["conf"]))
                    health["geometry_valid"] = bool(
                        det_summary.get("geometry_valid", health["geometry_valid"])
                    )
                    health["mode"] = str(det_summary.get("mode", health["mode"]))
                    health["lane_lost"] = bool(
                        det_summary.get("lane_lost", health["lane_lost"])
                    )
                    health["cte"] = float(det_summary.get("cte", health["cte"]))
            except Exception:  # noqa: BLE001
                pass

        return health

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _incident_to_dict(incident: Any) -> Dict[str, Any]:
        """แปลง LaneLossIncident (dataclass หรือ object) → dict อย่างปลอดภัย"""
        if isinstance(incident, dict):
            return incident
        if hasattr(incident, "to_dict"):
            try:
                return incident.to_dict()
            except Exception:  # noqa: BLE001
                pass
        # Fallback: ดึง attributes ที่มี
        result: Dict[str, Any] = {}
        for key in (
            "timestamp", "frame_idx", "event_type", "severity",
            "message", "conf", "cte", "mode", "duration_frames",
        ):
            val = getattr(incident, key, None)
            if val is not None:
                result[key] = val
        return result

    @staticmethod
    def _snapshot_to_dict(snapshot: Any) -> Dict[str, Any]:
        """แปลง BehaviorContextSnapshot (dataclass หรือ object) → dict อย่างปลอดภัย"""
        if isinstance(snapshot, dict):
            return snapshot
        if hasattr(snapshot, "to_dict"):
            try:
                return snapshot.to_dict()
            except Exception:  # noqa: BLE001
                pass
        # Fallback: ดึง attributes ที่มี
        result: Dict[str, Any] = {}
        for key in (
            "timestamp", "frame_idx", "event_type", "severity",
            "message", "context", "data",
        ):
            val = getattr(snapshot, key, None)
            if val is not None:
                result[key] = val
        return result
