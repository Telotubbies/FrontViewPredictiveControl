#!/usr/bin/env python3
"""
CARLA MPC Lane Keeping System - Main Entry Point

Refactored main module that provides clean entry point for:
- UNet + MPC lane keeping
- Classical + MPC lane keeping
- Real-time visualization
- Safety systems
"""

import argparse
import json
import logging
import math
import queue
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Setup CARLA paths
from core import setup_carla_paths
setup_carla_paths()

# Import core modules
from core.carla_interface import CarlaInterface
from gui.dashboard import Dashboard
from gui.gui_process_bridge import GuiProcessBridge

# Import system components
from pipeline import LKAPipeline
from safety.override import SafetyOverride
from safety.stuck_recovery import StuckRecovery
from safety.emergency_braking_adaptive_cruise_control import AEBACC
from safety.obstacle_avoidance import ObstacleAvoidancePlanner
from adas.adas_manager import ADASManager
from control.arbitrator import ControlArbitrator
from bridge.obstacles import get_traffic_obstacles
from carla_input_output import get_waypoints, waypoints_to_cte_heading
from telemetry.influxdb_exporter import TelemetryExporter
from telemetry.metrics_collector import MetricsCollector
from telemetry.behavior_logger import BehaviorLogger
from telemetry.behavior_context_logger import BehaviorContextLogger
from telemetry.lane_loss_detector import LaneLossDetector
from telemetry.ai_report_generator import AiReportGenerator
from metrics.run_logger import RunLogger
from metrics.realtime_stats import RealTimeStats
from metrics.run_analyzer import RunAnalyzer
from config import (
    TARGET_SPEED_KMH, USE_TRAJECTORY_PIPELINE,
    CONTROL_HZ, MAIN_LOOP_SLEEP_S
)

# Pipeline orchestrator + nodes
from orchestrator import PipelineOrchestrator
from nodes import (
    SensorNode,
    PerceptionNode,
    ActuatorNode,
    TelemetryNode,
    DashboardNode,
)
try:
    from nodes import ArbitrationNode
except Exception:
    ArbitrationNode = None

logger = logging.getLogger(__name__)


class CARLAMPCSystem:
    """Main CARLA MPC system orchestrator"""

    def __init__(self,
                 model_path: str,
                 target_speed_kmh: float = TARGET_SPEED_KMH,
                 use_classical: bool = False,
                 no_gui: bool = False,
                 metrics_dir: str = "metrics_output",
                 model_type: str = "unet"):
        self.model_path = model_path
        self.target_speed_kmh = target_speed_kmh
        self.use_classical = use_classical
        self.no_gui = no_gui
        self.metrics_dir = metrics_dir
        self.model_type = model_type

        # System components
        self.carla: Optional[CarlaInterface] = None
        self.dashboard: Optional[Dashboard] = None
        self.gui_bridge: Optional[GuiProcessBridge] = None
        self.pipeline: Optional[LKAPipeline] = None
        self.safety: Optional[SafetyOverride] = None
        self.stuck_recovery: Optional[StuckRecovery] = None
        self.aeb_acc: Optional[AEBACC] = None
        self.adas: Optional[ADASManager] = None
        self.arbitrator: Optional[ControlArbitrator] = None
        self.telemetry: Optional[TelemetryExporter] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.run_logger: Optional[RunLogger] = None
        self.realtime_stats: Optional[RealTimeStats] = None

        # Behavior + lane-loss telemetry (wired into main loop)
        self.behavior_logger: Optional[BehaviorLogger] = None
        self.context_logger: Optional[BehaviorContextLogger] = None
        self.lane_loss_detector: Optional[LaneLossDetector] = None
        self.ai_report_gen: Optional[AiReportGenerator] = None

        # Control state (persisted between frames for smoothing)
        self._prev_steer = 0.0
        self._prev_throttle = 0.0

        # Background control refresher — ส่ง control ซ้ำทุก 50ms เพื่อให้
        # CARLA 0.9.x async mode ไม่หยุดรถเมื่อ main loop ช้า
        self._control_thread = None
        self._latest_control = None
        self._control_thread_running = False

        # Device setup — supports CUDA (NVIDIA) and ROCm (AMD)
        from utils.device_utils import get_device
        self.device = get_device()

        # System state
        self.running = False
        self.frame_count = 0
        self._loop_times = []

    def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("Initializing CARLA MPC System...")

            # Initialize CARLA interface
            self.carla = CarlaInterface()
            if not self.carla.connect():
                return False

            # Spawn vehicle
            if not self.carla.spawn_vehicle():
                return False

            # Setup camera
            if not self.carla.setup_camera():
                return False

            # Initialize dashboard if not disabled
            # GUI runs in a SEPARATE PROCESS to avoid blocking the control loop
            if not self.no_gui:
                self.gui_bridge = GuiProcessBridge(target_speed_kmh=self.target_speed_kmh)
                if not self.gui_bridge.start():
                    logger.warning("Failed to start GUI process, continuing without GUI")
                    self.gui_bridge = None

            # Initialize perception pipeline
            self.pipeline = LKAPipeline(
                model_path=self.model_path,
                device=self.device,
                target_speed_kmh=self.target_speed_kmh,
                use_trajectory_pipeline=USE_TRAJECTORY_PIPELINE,
                model_type=self.model_type,
            )
            logger.info(f"Pipeline initialized: {'Classical' if self.use_classical else self.model_type.upper()}")

            # Initialize safety systems
            self.safety = SafetyOverride({
                "max_speed_kmh": self.target_speed_kmh + 5.0,
                "max_steering_angle": 0.45,
                "emergency_brake_enabled": True,
            })
            self.stuck_recovery = StuckRecovery()
            self.aeb_acc = AEBACC()
            self.obstacle_avoidance = ObstacleAvoidancePlanner()
            # TSR ปิดชั่วคราว — CARLA 0.9.16 มี traffic light แดงตลอดเวลา
            # ทำให้รถไม่สามารถขับได้ สามารถเปิดได้ใน CARLA เวอร์ชั่นที่ traffic light เป็น green บ้าง
            self.adas = ADASManager(aeb_acc=self.aeb_acc, enable_tsr=False)
            self.arbitrator = ControlArbitrator(
                safety_override=self.safety,
                adas_manager=self.adas,
                stuck_recovery=self.stuck_recovery,
                obstacle_avoidance=self.obstacle_avoidance,
            )
            self.telemetry = TelemetryExporter(enabled=True)
            self.metrics_collector = MetricsCollector(output_dir=self.metrics_dir)

            # Initialize run logger + realtime stats
            import time as _time
            run_timestamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
            map_name = getattr(self.carla, 'map_name', 'unknown') if self.carla else 'unknown'
            vehicle_type = getattr(self.carla, 'vehicle_type', 'unknown') if self.carla else 'unknown'
            run_dir = f"runs/{run_timestamp}_{map_name}"
            self.run_logger = RunLogger(
                run_dir=run_dir,
                map_name=map_name,
                vehicle_type=vehicle_type,
                target_speed_kmh=self.target_speed_kmh,
            )
            self.run_logger.start()

            self.realtime_stats = RealTimeStats(
                print_interval=50,
                target_speed_kmh=self.target_speed_kmh,
            )
            logger.info(f"Run logging → {run_dir}/")

            # ── Behavior + Lane Loss telemetry ───────────────────────
            # บันทึกพฤติกรรมรถ + root-cause analysis ของ lane loss
            self.behavior_logger = BehaviorLogger(snapshot_dir=run_dir)
            self.context_logger = BehaviorContextLogger()
            self.lane_loss_detector = LaneLossDetector()
            self.ai_report_gen = AiReportGenerator()
            logger.info("Behavior telemetry initialized (BehaviorLogger + ContextLogger + LaneLossDetector)")

            # ── เริ่ม background control refresher ──────────────────────
            # ส่ง control ซ้ำทุก 50ms เพื่อให้รถขยับใน async mode
            # โดยเฉพาะในโหมด GUI ที่ main loop ช้ากว่า 20 FPS
            self._start_control_thread()

            logger.info("System initialization completed — Full ADAS suite active")
            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False

    def _start_control_thread(self):
        """เริ่ม background thread ที่ส่ง control ซ้ำทุก 50ms.
        CARLA 0.9.x async mode: ถ้า apply_control ไม่ถูกเรียกบ่อยพอ รถจะหยุด
        """
        import threading
        self._control_thread_running = True

        def _refresh_loop():
            while self._control_thread_running:
                ctrl = self._latest_control
                if ctrl is not None and self.carla and self.carla.vehicle:
                    try:
                        self.carla.vehicle.apply_control(ctrl)
                    except Exception:
                        pass
                time.sleep(0.05)

        self._control_thread = threading.Thread(target=_refresh_loop, daemon=True)
        self._control_thread.start()
        logger.info("Control refresher thread started (50ms interval)")

    def _stop_control_thread(self):
        """หยุด background control thread"""
        self._control_thread_running = False
        if self._control_thread:
            self._control_thread.join(timeout=1.0)
            self._control_thread = None

    def run(self) -> int:
        """Main system loop"""
        if not self.initialize():
            return 1

        self.running = True
        logger.info("Starting main control loop...")

        try:
            # Camera callback setup
            rgb_frame = None
            camera_queue = queue.Queue(maxsize=2)

            def camera_callback(image):
                nonlocal rgb_frame
                try:
                    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (image.height, image.width, 4))
                    array = array[:, :, :3]  # Remove alpha channel
                    array = array[:, :, ::-1]  # BGR to RGB

                    # Put in queue for MPC runner
                    try:
                        camera_queue.put_nowait(array)
                    except queue.Full:
                        try:
                            camera_queue.get_nowait()
                            camera_queue.put_nowait(array)
                        except queue.Empty:
                            pass

                    rgb_frame = array
                    self._cam_frame_count = getattr(self, '_cam_frame_count', 0) + 1
                except Exception as e:
                    logger.error(f"Camera callback error: {e}")

            # Start camera
            self.carla.camera.listen(camera_callback)

            # Main control loop (no MPCRunner — use LKAPipeline.step() directly)
            while self.running:
                loop_start_time = time.time()

                # Handle GUI events (check if GUI process requested quit)
                if self.gui_bridge and self.gui_bridge.should_quit():
                    break

                # Get RGB frame
                current_frame = None
                try:
                    current_frame = camera_queue.get(timeout=0.05)
                except queue.Empty:
                    if rgb_frame is not None:
                        current_frame = rgb_frame
                    else:
                        # No frame yet — ส่ง control เก่าซ้ำเพื่อให้รถไม่หยุด
                        if self.carla and self.carla.vehicle and self._latest_control is not None:
                            try:
                                self.carla.vehicle.apply_control(self._latest_control)
                            except Exception:
                                pass
                        time.sleep(0.01)
                        continue

                # Get vehicle state
                vehicle_transform = self.carla.get_vehicle_transform()
                current_speed_ms = self.carla.get_vehicle_velocity()

                if vehicle_transform is None:
                    continue

                # Get waypoint state for fusion (WP primary)
                wp_state = None
                try:
                    wps = get_waypoints(self.carla.vehicle, self.carla.world.get_map())
                    if wps:
                        wp_state = waypoints_to_cte_heading(vehicle_transform, wps)
                except Exception:
                    pass

                # Run pipeline step: perception → fusion → MPC → safety
                steer, throttle, brake, frame_state = self.pipeline.step(
                    rgb=current_frame,
                    speed_ms=current_speed_ms,
                    wp_state=wp_state,
                    prev_steer=self._prev_steer,
                    prev_throttle=self._prev_throttle,
                    world=self.carla.world,
                    vehicle=self.carla.vehicle,
                )
                if self.frame_count < 10 or self.frame_count % 50 == 0:
                    cam_cnt = getattr(self, '_cam_frame_count', 0)
                    logger.debug("frame=%d cam=%d speed=%.2f steer=%.3f thr=%.3f brk=%.3f cte=%.3f conf=%.3f mode=%s",
                                 self.frame_count, cam_cnt, current_speed_ms, steer, throttle, brake,
                                 frame_state.cte_m if frame_state else 0,
                                 frame_state.lane_conf if frame_state else 0,
                                 frame_state.mode if frame_state else "?")

                # ── Full ADAS Suite ────────────────────────────────────────
                # AEB + ACC + LDW + LKA Pro + BSW + LCA + TSR + TJA + Stop&Go
                adas_out = None
                obstacles = []
                ego_loc = vehicle_transform.location
                ego_rot = vehicle_transform.rotation
                ego_heading = math.radians(ego_rot.yaw)
                try:
                    obstacles = get_traffic_obstacles(
                        self.carla.world,
                        exclude_actor_id=self.carla.vehicle.id,
                    )
                    timestamp = time.time()

                    adas_out = self.adas.update(
                        obstacles=obstacles,
                        ego_x=ego_loc.x,
                        ego_y=ego_loc.y,
                        ego_heading=ego_heading,
                        ego_speed=current_speed_ms,
                        cte=frame_state.cte_m if frame_state else 0.0,
                        heading_err=frame_state.heading_rad if frame_state else 0.0,
                        curvature=frame_state.curvature if frame_state else 0.0,
                        timestamp=timestamp,
                        carla_world=self.carla.world,
                        vehicle_transform=vehicle_transform,
                        nominal_target_speed_ms=self.target_speed_kmh / 3.6,
                    )

                    # Populate FrameState with ADAS status for dashboard
                    if frame_state:
                        frame_state.aeb_active = getattr(adas_out, 'aeb_active', False)
                        frame_state.aeb_ttc = getattr(adas_out, 'aeb_ttc', -1.0)
                        frame_state.aeb_warning_level = getattr(adas_out, 'aeb_warning_level', 'none')
                        frame_state.acc_active = getattr(adas_out, 'acc_active', False)
                        frame_state.acc_target_speed_ms = getattr(adas_out, 'acc_target_speed_ms', -1.0)
                        frame_state.acc_distance_m = getattr(adas_out, 'acc_distance_m', -1.0)
                        frame_state.ldw_state = getattr(adas_out, 'ldw_state', 'in_lane')
                        frame_state.ldw_warning_active = getattr(adas_out, 'ldw_warning_active', False)
                        frame_state.ldw_side = getattr(adas_out, 'ldw_side', 'none')
                        frame_state.lka_pro_assist = getattr(adas_out, 'lka_pro_assist', 0.0)
                        frame_state.bsw_left_alert = getattr(adas_out, 'bsw_left_alert', 'clear')
                        frame_state.bsw_right_alert = getattr(adas_out, 'bsw_right_alert', 'clear')
                        frame_state.bsw_safe_left = getattr(adas_out, 'bsw_safe_left', True)
                        frame_state.bsw_safe_right = getattr(adas_out, 'bsw_safe_right', True)
                        frame_state.tsr_speed_limit_kmh = getattr(adas_out, 'tsr_speed_limit_kmh', None)
                        frame_state.tsr_traffic_light = getattr(adas_out, 'tsr_traffic_light', 'unknown')
                        frame_state.tsr_traffic_light_distance = getattr(adas_out, 'tsr_traffic_light_distance', -1.0)
                        frame_state.tsr_action = getattr(adas_out, 'tsr_action', 'none')
                        frame_state.tja_state = getattr(adas_out, 'tja_state', 'inactive')
                        frame_state.tja_active = getattr(adas_out, 'tja_active', False)
                        frame_state.stop_and_go_stopped = getattr(adas_out, 'stop_and_go_stopped', False)
                        frame_state.adas_override_active = bool(adas_out.active_features)

                    if adas_out.active_features:
                        logger.debug(
                            "ADAS active: %s", ", ".join(adas_out.active_features)
                        )
                    # Detailed ADAS status every 100 frames
                    if self.frame_count % 100 == 0 and self.frame_count > 0:
                        logger.info(
                            "ADAS status: AEB=%s ttc=%.2f ACC=%s dist=%.1f "
                            "LDW=%s/%s LKA=%.3f BSW=L:%s R:%s "
                            "TJA=%s S&G=%s features=%s",
                            adas_out.aeb_active, adas_out.aeb_ttc,
                            adas_out.acc_active, adas_out.acc_distance_m,
                            adas_out.ldw_state, adas_out.ldw_warning_active,
                            adas_out.lka_pro_assist,
                            adas_out.bsw_left_alert, adas_out.bsw_right_alert,
                            adas_out.tja_state, adas_out.stop_and_go_stopped,
                            ", ".join(adas_out.active_features) or "none",
                        )
                except Exception as e:
                    logger.debug(f"ADAS update skipped: {e}")

                # ── Control arbitration (ADAS → APF → Safety → Stuck → AEB) ─
                arb_result = self.arbitrator.arbitrate(
                    mpc_steer=steer, mpc_throttle=throttle, mpc_brake=brake,
                    speed_ms=current_speed_ms, frame_state=frame_state,
                    adas_out=adas_out,
                    obstacles=obstacles,
                    ego_x=ego_loc.x,
                    ego_y=ego_loc.y,
                    ego_heading=ego_heading,
                )
                steer = arb_result.steer
                throttle = arb_result.throttle
                brake = arb_result.brake

                # Store obstacle avoidance status for visualization
                if frame_state and self.obstacle_avoidance:
                    av_status = self.obstacle_avoidance.get_status()
                    frame_state.obstacle_avoidance_active = av_status["active"]
                    frame_state.obstacle_avoidance_side = av_status["avoidance_side"]
                    frame_state.obstacle_avoidance_shift = av_status["lateral_shift_m"]
                    frame_state.obstacle_avoidance_dist = av_status["closest_obstacle_dist"]
                    frame_state.obstacle_avoidance_steer = av_status["steer_offset"]
                    frame_state.obstacle_count = av_status["num_obstacles"]
                    # Debug log when obstacles detected
                    if av_status["num_obstacles"] > 0 and self.frame_count % 10 == 0:
                        logger.info(
                            f"APF status: {av_status['num_obstacles']} obs, "
                            f"closest={av_status['closest_obstacle_dist']:.1f}m, "
                            f"active={av_status['active']}, "
                            f"steer_off={av_status['steer_offset']:.3f}"
                        )
                    # Store detected obstacles in ego frame for 3D viz
                    if obstacles and 'ego_loc' in dir() and 'ego_heading' in dir():
                        import math as _m, numpy as _np
                        cos_h = _m.cos(ego_heading)
                        sin_h = _m.sin(ego_heading)
                        det_obs = []
                        for obs in obstacles:
                            try:
                                dx = obs.position[0] - ego_loc.x
                                dy = obs.position[1] - ego_loc.y
                                fwd = dx * cos_h + dy * sin_h
                                lat = -dx * sin_h + dy * cos_h
                                dist = _np.sqrt(fwd*fwd + lat*lat)
                                if 0 < fwd < 50 and abs(lat) < 8:
                                    det_obs.append((float(fwd), float(lat), float(dist),
                                                   str(obs.type.name)))
                            except Exception:
                                pass
                        frame_state.detected_obstacles = det_obs

                # ── Telemetry export ───────────────────────────────────
                if self.telemetry and self.telemetry.is_connected():
                    try:
                        self.telemetry.export_frame(
                            vehicle={
                                "speed_ms": current_speed_ms,
                                "speed_kmh": current_speed_ms * 3.6,
                                "steering": steer,
                                "throttle": throttle,
                                "brake": brake,
                            },
                            lane={
                                "cte_m": frame_state.cte_m if frame_state else 0.0,
                                "heading_err_rad": frame_state.heading_rad if frame_state else 0.0,
                                "curvature": frame_state.curvature if frame_state else 0.0,
                                "lane_conf": frame_state.lane_conf if frame_state else 0.0,
                            },
                            adas_out=adas_out if 'adas_out' in dir() else None,
                            performance={
                                "fps": 1.0 / self._loop_times[-1] if self._loop_times else 0.0,
                                "loop_time_ms": (self._loop_times[-1] * 1000) if self._loop_times else 0.0,
                            },
                        )
                    except Exception as e:
                        logger.debug(f"Telemetry export skipped: {e}")

                # ── Metrics collection ────────────────────────────────
                if self.metrics_collector:
                    loop_time_ms = (time.time() - loop_start_time) * 1000.0
                    mpc_solve_ms = float(getattr(frame_state, 'mpc_solve_time_ms', 0.0)) if frame_state else 0.0
                    mpc_status = str(getattr(frame_state, 'solver_status', 'unknown')) if frame_state else 'unknown'
                    safety_active = bool(getattr(frame_state, 'safety_active', False)) if frame_state else False
                    self.metrics_collector.record_from_frame_state(
                        frame_idx=self.frame_count,
                        speed_ms=current_speed_ms,
                        steer=steer,
                        throttle=throttle,
                        brake=brake,
                        frame_state=frame_state,
                        mpc_solve_time_ms=mpc_solve_ms,
                        mpc_solver_status=mpc_status,
                        loop_time_ms=loop_time_ms,
                        safety_active=safety_active,
                        stuck_recovery_active=(arb_result.winning_source == "stuck"),
                        adas_out=adas_out,
                    )

                # ── Run logger + realtime stats ───────────────────────
                if self.run_logger and frame_state:
                    loop_time_ms = (time.time() - loop_start_time) * 1000.0
                    fps = 1.0 / self._loop_times[-1] if self._loop_times else 0.0
                    self.run_logger.log_frame(frame_state, {
                        "frame_idx": self.frame_count,
                        "speed_ms": current_speed_ms,
                        "loop_time_ms": loop_time_ms,
                        "fps": fps,
                        "mpc_steer_rad": getattr(frame_state, 'steer', 0.0),
                        "mpc_accel": 0.0,
                        "sim_time": time.time() - (self._start_time if hasattr(self, '_start_time') else time.time()),
                    })

                if self.realtime_stats and frame_state:
                    loop_time_ms = (time.time() - loop_start_time) * 1000.0
                    fps = 1.0 / self._loop_times[-1] if self._loop_times else 0.0
                    self.realtime_stats.update(frame_state, {
                        "frame_idx": self.frame_count,
                        "speed_ms": current_speed_ms,
                        "loop_time_ms": loop_time_ms,
                        "fps": fps,
                    })

                # Apply control to vehicle — สร้าง VehicleControl ใหม่ทุก frame
                # (CARLA 0.9.x: get_vehicle_control() คืนค่า garbage บางครั้ง)
                import carla as _carla_mod
                carla_control = _carla_mod.VehicleControl(
                    throttle=float(throttle),
                    steer=float(steer),
                    brake=float(brake),
                    hand_brake=False,
                    manual_gear_shift=False,
                    gear=1,
                    reverse=False,
                )
                self.carla.apply_control(carla_control)
                self._latest_control = carla_control  # ให้ refresher thread ใช้

                # Persist control state for next frame's smoothing
                self._prev_steer = steer
                self._prev_throttle = throttle

                # Check for stuck recovery
                stuck_active = False
                stuck_phase = "none"
                recovery = self.stuck_recovery.update(current_speed_ms, throttle)
                if recovery is not None:
                    r_steer, r_throttle, r_brake, r_reverse = recovery
                    import carla as _carla_mod
                    carla_control = _carla_mod.VehicleControl(
                        throttle=float(r_throttle),
                        steer=float(r_steer),
                        brake=float(r_brake),
                        hand_brake=False,
                        manual_gear_shift=False,
                        gear=1,
                        reverse=bool(r_reverse),
                    )
                    self.carla.apply_control(carla_control)
                    stuck_active = True
                    stuck_phase = getattr(self.stuck_recovery, '_phase', 'unknown')
                    logger.info("Applying stuck recovery (phase=%s)", stuck_phase)
                    # Override final values for dashboard
                    steer = r_steer
                    throttle = r_throttle
                    brake = r_brake
                    if self.metrics_collector and self.metrics_collector.frames:
                        self.metrics_collector.frames[-1].stuck_recovery_active = True

                # Update FrameState with final control values + vehicle pose
                if frame_state:
                    frame_state.final_steer = steer
                    frame_state.final_throttle = throttle
                    frame_state.final_brake = brake
                    frame_state.stuck_recovery_active = stuck_active
                    frame_state.stuck_recovery_phase = stuck_phase
                    frame_state.vehicle_x = ego_loc.x if 'ego_loc' in dir() else 0.0
                    frame_state.vehicle_y = ego_loc.y if 'ego_loc' in dir() else 0.0
                    frame_state.vehicle_z = ego_loc.z if 'ego_loc' in dir() else 0.0
                    frame_state.vehicle_yaw = ego_heading if 'ego_heading' in dir() else 0.0

                # ── Behavior + Lane Loss telemetry ──────────────────────
                # บันทึกพฤติกรรม + ตรวจจับ lane loss ทุก frame
                # (try/except เพื่อไม่ให้ telemetry crash การขับขี่)
                if frame_state and self.behavior_logger:
                    try:
                        loop_time_ms = (time.time() - loop_start_time) * 1000.0
                        ctx = {
                            "frame_idx": self.frame_count,
                            "speed_ms": current_speed_ms,
                            "loop_time_ms": loop_time_ms,
                        }
                        events_before = len(self.behavior_logger.events)
                        self.behavior_logger.update(frame_state, ctx)
                        if self.context_logger:
                            self.context_logger.update(frame_state, ctx)
                        if self.lane_loss_detector:
                            incident = self.lane_loss_detector.update(frame_state, ctx)
                            if incident:
                                logger.warning(
                                    "Lane loss detected: %s (frame %d, duration=%d frames)",
                                    incident.root_cause.value,
                                    incident.frame_idx_start,
                                    incident.duration_frames,
                                )
                        # ── Forward new behavior events to dashboard event timeline ──
                        # (GUI process handles its own event timeline — events are
                        #  reconstructed from FrameState in the GUI process)
                        new_events = self.behavior_logger.events[events_before:]
                    except Exception as e:
                        logger.debug(f"Behavior telemetry update skipped: {e}")

                # Update dashboard (AFTER all overrides — shows actual values sent to CARLA)
                # Send frame data to GUI process (non-blocking)
                if self.gui_bridge:
                    self.gui_bridge.send_frame(
                        current_frame, frame_state, current_speed_ms,
                        steer, throttle, brake,
                    )

                # Update frame counter
                self.frame_count += 1

                # Tick world in synchronous mode
                if getattr(self.carla, '_sync_mode', False):
                    try:
                        self.carla.world.tick()
                    except Exception:
                        pass

                # Maintain control frequency
                loop_time = time.time() - loop_start_time
                self._loop_times.append(loop_time)
                target_loop_time = 1.0 / CONTROL_HZ
                sleep_time = max(0, target_loop_time - loop_time - MAIN_LOOP_SLEEP_S)

                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Log performance periodically
                if self.frame_count % 100 == 0:
                    avg_loop = np.mean(self._loop_times[-100:]) if self._loop_times else 0.0
                    fps = 1.0 / avg_loop if avg_loop > 0 else 0.0
                    adas_status = self.adas.get_status() if self.adas else {}
                    aeb_active = adas_status.get("aeb", {}).get("aeb_active", False)
                    acc_active = adas_status.get("aeb", {}).get("acc_active", False)
                    tja_state = adas_status.get("tja", {}).get("state", "inactive")
                    logger.info(
                        "Performance: FPS=%.1f, loop=%.1fms | AEB=%s ACC=%s TJA=%s",
                        fps, avg_loop * 1000,
                        "ACTIVE" if aeb_active else "idle",
                        "ACTIVE" if acc_active else "idle",
                        tja_state,
                    )

            return 0

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return 0
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            return 1
        finally:
            self.cleanup()

    def _update_dashboard(self,
                         rgb_frame: np.ndarray,
                         frame_state,
                         current_speed_ms: float,
                         steer: float,
                         throttle: float,
                         brake: float):
        """Update dashboard display — Tesla-style layout"""
        try:
            # Clear panel
            self.dashboard.panel_surface.fill((26, 26, 46))

            # Render camera frame with lane overlay + MPC trajectory
            if frame_state and frame_state.lane_overlay is not None:
                self.dashboard.render_lane_overlay(rgb_frame, frame_state.lane_overlay, frame_state)
            else:
                self.dashboard.render_lane_overlay(rgb_frame, None, frame_state)

            # Render BEV view if available (with MPC path overlay)
            if frame_state:
                self.dashboard.render_bev_view(frame_state.bev_binary, frame_state)

            # Render top bar (speed, autopilot, target)
            self.dashboard._render_top_bar(frame_state, self.target_speed_kmh)

            # ── Factory HMI: lane health bar (P1-P5 + geometry + confidence) ──
            if frame_state:
                self.dashboard.render_health_bar(frame_state)

            # Render metrics panel
            speed_kmh = current_speed_ms * 3.6
            confidence = frame_state.lane_conf if frame_state else 0.0
            cte = frame_state.cte_m if frame_state else 0.0
            fps = 1.0 / self._loop_times[-1] if self._loop_times else 0.0

            self.dashboard.render_status_info(
                speed_kmh, steer, cte, confidence, fps, frame_state
            )

            # ── Factory HMI: industrial gauges (speedometer, steering, pedals, etc.) ──
            if frame_state:
                self.dashboard.render_hmi_gauges(frame_state, self.target_speed_kmh, fps)

            # Render ADAS panel
            if frame_state:
                self.dashboard.render_adas_panel(frame_state)

            # Render control info (legacy, now part of metrics)
            self.dashboard.render_control_info(
                throttle, brake, self.target_speed_kmh, frame_state
            )

            # ── Factory HMI: event/alarm timeline (SCADA-style) ──
            self.dashboard.render_event_timeline()

            # Render status bar
            sim_time = time.time() - loop_start_time if 'loop_start_time' in dir() else 0.0
            self.dashboard._render_status_bar(fps, self.frame_count, sim_time)

            # Update display
            self.dashboard.update()

        except Exception as e:
            logger.error(f"Dashboard update error: {e}")

    def cleanup(self):
        """Cleanup all system resources"""
        try:
            logger.info("Cleaning up system resources...")

            # หยุด control refresher thread
            self._stop_control_thread()

            # Save metrics
            if self.metrics_collector and self.metrics_collector.frames:
                paths = self.metrics_collector.save(prefix="run")
                logger.info("Metrics saved: %s", paths)
                summary = self.metrics_collector.get_summary()
                logger.info("Run summary: %d frames, %.1fs, fallback=%d (%.1f%%)",
                            summary['total_frames'], summary['duration_s'],
                            summary['fallback_count'], summary['fallback_rate'] * 100)

            # Close telemetry exporter
            if self.telemetry:
                self.telemetry.close()

            # Stop run logger and analyze
            if self.run_logger:
                self.run_logger.stop()
                logger.info("Run logger stopped, analyzing...")
                try:
                    analyzer = RunAnalyzer(str(self.run_logger.run_dir))
                    kpis = analyzer.analyze()
                    if kpis:
                        lc = kpis.get("lane_compliance", {})
                        logger.info(
                            "Run analysis: CTE RMSE=%.2fm, in-lane=%.1f%%, "
                            "AEB=%d, fallback=%.1f%%, FPS=%.1f",
                            lc.get("cte_rmse_m", 0),
                            lc.get("in_lane_pct", 0),
                            kpis.get("safety", {}).get("aeb_events", 0),
                            kpis.get("control", {}).get("fallback_rate_pct", 0),
                            kpis.get("performance", {}).get("fps_mean", 0),
                        )
                        logger.info("Report: %s", self.run_logger.run_dir / "report.md")
                except Exception as e:
                    logger.error(f"Run analysis failed: {e}")

            # ── Generate AI-readable behavior report ─────────────────
            # สร้างรายงานสรุปพฤติกรรมทั้ง run ให้ AI/LLM อ่านและวิเคราะห์
            if self.behavior_logger and self.ai_report_gen:
                try:
                    # Finalize lane loss detector (ปิด incident ที่ยัง lost อยู่)
                    if self.lane_loss_detector:
                        self.lane_loss_detector.finalize()

                    summary = self.behavior_logger.get_summary()
                    events = self.behavior_logger.events
                    incidents = (
                        [inc.to_dict() for inc in self.lane_loss_detector.get_incidents()]
                        if self.lane_loss_detector else []
                    )

                    # ดึง run metadata จาก run_logger
                    map_name = getattr(self.run_logger, 'map_name', 'unknown') if self.run_logger else 'unknown'
                    vehicle_type = getattr(self.run_logger, 'vehicle_type', 'unknown') if self.run_logger else 'unknown'
                    run_dir_path = Path(str(self.run_logger.run_dir)) if self.run_logger else Path("runs/latest")
                    # คำนวณ duration จาก loop_times (ผลรวมเวลาทั้ง run)
                    duration_s = float(np.sum(self._loop_times)) if self._loop_times else 0.0
                    run_meta = {
                        "map": map_name,
                        "vehicle": vehicle_type,
                        "target_speed": self.target_speed_kmh,
                        "target_speed_kmh": self.target_speed_kmh,
                        "total_frames": self.frame_count,
                        "duration": duration_s,
                        "duration_s": duration_s,
                    }

                    # สร้าง text report (.txt)
                    report = self.ai_report_gen.generate_report(summary, events, incidents, run_meta)
                    report_path = run_dir_path / "ai_behavior_report.txt"
                    self.ai_report_gen.save_report(report, str(report_path))

                    # สร้าง JSON report (.json)
                    json_report = self.ai_report_gen.generate_json_report(summary, events, incidents, run_meta)
                    json_path = run_dir_path / "ai_behavior_report.json"
                    json_path.write_text(json.dumps(json_report, indent=2, default=str), encoding="utf-8")

                    logger.info("AI behavior report saved → %s", report_path)
                    logger.info("AI behavior JSON report saved → %s", json_path)
                except Exception as e:
                    logger.error(f"AI behavior report generation failed: {e}")

            # Print final realtime stats
            if self.realtime_stats:
                stats = self.realtime_stats.get_current_stats()
                if stats:
                    logger.info("Final stats: %s", stats)

            # Stop camera
            if self.carla and self.carla.camera:
                self.carla.camera.stop()

            # Cleanup GUI process
            if self.gui_bridge:
                self.gui_bridge.stop()

            # Cleanup CARLA
            if self.carla:
                self.carla.cleanup()

            logger.info("System cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="CARLA MPC Lane Keeping System")
    parser.add_argument("--model", type=str, default="model/lane_unet_final.pth",
                       help="Path to model file")
    parser.add_argument("--model-type", type=str, default="unet",
                       choices=["unet", "dsunet", "ultra_fast"],
                       help="Model type: unet, dsunet, or ultra_fast")
    parser.add_argument("--town", type=str, default="Town04",
                       help="CARLA town name")
    parser.add_argument("--speed", type=float, default=TARGET_SPEED_KMH,
                       help="Target speed in km/h")
    parser.add_argument("--classical", action="store_true",
                       help="Use classical lane detection instead of UNet")
    parser.add_argument("--no-gui", action="store_true",
                       help="Disable GUI dashboard")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--record-dir", type=str, default=None,
                       help="Directory to record run data (not yet fully implemented)")
    parser.add_argument("--duration", type=float, default=None,
                       help="Maximum run duration in seconds (not yet fully implemented)")
    parser.add_argument("--metrics-dir", type=str, default="metrics_output",
                       help="Directory to save metrics CSV/JSON output")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    # Check model file
    if not args.classical:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return 1

    # Create and run system
    system = CARLAMPCSystem(
        model_path=args.model,
        target_speed_kmh=args.speed,
        use_classical=args.classical,
        no_gui=args.no_gui,
        metrics_dir=args.metrics_dir,
        model_type=args.model_type,
    )

    return system.run()


if __name__ == "__main__":
    sys.exit(main())

