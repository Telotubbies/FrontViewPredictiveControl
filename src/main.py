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

# Import system components
from pipeline import LKAPipeline
from safety.override import SafetyOverride
from safety.stuck_recovery import StuckRecovery
from safety.emergency_braking_adaptive_cruise_control import AEBACC
from adas.adas_manager import ADASManager
from bridge.obstacles import get_traffic_obstacles
from carla_input_output import get_waypoints, waypoints_to_cte_heading
from telemetry.influxdb_exporter import TelemetryExporter
from telemetry.metrics_collector import MetricsCollector
from config import (
    TARGET_SPEED_KMH, USE_TRAJECTORY_PIPELINE,
    CONTROL_HZ, MAIN_LOOP_SLEEP_S
)

logger = logging.getLogger(__name__)


class CARLAMPCSystem:
    """Main CARLA MPC system orchestrator"""

    def __init__(self,
                 model_path: str,
                 target_speed_kmh: float = TARGET_SPEED_KMH,
                 use_classical: bool = False,
                 no_gui: bool = False,
                 metrics_dir: str = "metrics_output"):
        self.model_path = model_path
        self.target_speed_kmh = target_speed_kmh
        self.use_classical = use_classical
        self.no_gui = no_gui
        self.metrics_dir = metrics_dir

        # System components
        self.carla: Optional[CarlaInterface] = None
        self.dashboard: Optional[Dashboard] = None
        self.pipeline: Optional[LKAPipeline] = None
        self.safety: Optional[SafetyOverride] = None
        self.stuck_recovery: Optional[StuckRecovery] = None
        self.aeb_acc: Optional[AEBACC] = None
        self.adas: Optional[ADASManager] = None
        self.telemetry: Optional[TelemetryExporter] = None
        self.metrics_collector: Optional[MetricsCollector] = None

        # Control state (persisted between frames for smoothing)
        self._prev_steer = 0.0
        self._prev_throttle = 0.0

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
            if not self.no_gui:
                self.dashboard = Dashboard()
                if not self.dashboard.initialize():
                    logger.warning("Failed to initialize dashboard, continuing without GUI")
                    self.dashboard = None

            # Initialize perception pipeline
            self.pipeline = LKAPipeline(
                model_path=self.model_path,
                device=self.device,
                target_speed_kmh=self.target_speed_kmh,
                use_trajectory_pipeline=USE_TRAJECTORY_PIPELINE
            )
            logger.info(f"Pipeline initialized: {'Classical' if self.use_classical else 'UNet'}")

            # Initialize safety systems
            self.safety = SafetyOverride({
                "max_speed_kmh": self.target_speed_kmh + 5.0,
                "max_steering_angle": 0.45,
                "emergency_brake_enabled": True,
            })
            self.stuck_recovery = StuckRecovery()
            self.aeb_acc = AEBACC()
            self.adas = ADASManager(aeb_acc=self.aeb_acc)
            self.telemetry = TelemetryExporter(enabled=True)
            self.metrics_collector = MetricsCollector(output_dir=self.metrics_dir)

            logger.info("System initialization completed — Full ADAS suite active")
            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False

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
                except Exception as e:
                    logger.error(f"Camera callback error: {e}")

            # Start camera
            self.carla.camera.listen(camera_callback)

            # Main control loop (no MPCRunner — use LKAPipeline.step() directly)
            while self.running:
                loop_start_time = time.time()

                # Handle GUI events
                if self.dashboard:
                    if not self.dashboard.handle_events():
                        break

                # Get RGB frame
                current_frame = None
                try:
                    current_frame = camera_queue.get(timeout=0.1)
                except queue.Empty:
                    if rgb_frame is not None:
                        current_frame = rgb_frame
                    else:
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

                # ── Full ADAS Suite ────────────────────────────────────────
                # AEB + ACC + LDW + LKA Pro + BSW + LCA + TSR + TJA + Stop&Go
                try:
                    obstacles = get_traffic_obstacles(
                        self.carla.world,
                        exclude_actor_id=self.carla.vehicle.id,
                    )
                    ego_loc = vehicle_transform.location
                    ego_rot = vehicle_transform.rotation
                    ego_heading = math.radians(ego_rot.yaw)
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

                    # Apply ADAS overrides to MPC output
                    steer, throttle, brake, _ = self.adas.apply_to_control(
                        steer, throttle, brake, adas_out
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
                except Exception as e:
                    logger.debug(f"ADAS update skipped: {e}")

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
                        stuck_recovery_active=False,
                        adas_out=adas_out if 'adas_out' in dir() else None,
                    )

                # Apply control to vehicle
                carla_control = self.carla.get_vehicle_control()
                carla_control.steering = steer
                carla_control.throttle = throttle
                carla_control.brake = brake
                self.carla.apply_control(carla_control)

                # Persist control state for next frame's smoothing
                self._prev_steer = steer
                self._prev_throttle = throttle

                # Check for stuck recovery
                stuck_active = False
                stuck_phase = "none"
                recovery = self.stuck_recovery.update(current_speed_ms, throttle)
                if recovery is not None:
                    r_steer, r_throttle, r_brake, r_reverse = recovery
                    carla_control = self.carla.get_vehicle_control()
                    carla_control.steering = r_steer
                    carla_control.throttle = r_throttle
                    carla_control.brake = r_brake
                    carla_control.reverse = r_reverse
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

                # Update dashboard (AFTER all overrides — shows actual values sent to CARLA)
                if self.dashboard:
                    self._update_dashboard(
                        current_frame, frame_state, current_speed_ms,
                        steer, throttle, brake,
                    )

                # Update frame counter
                self.frame_count += 1

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

            # Render metrics panel
            speed_kmh = current_speed_ms * 3.6
            confidence = frame_state.lane_conf if frame_state else 0.0
            cte = frame_state.cte_m if frame_state else 0.0
            fps = 1.0 / self._loop_times[-1] if self._loop_times else 0.0

            self.dashboard.render_status_info(
                speed_kmh, steer, cte, confidence, fps, frame_state
            )

            # Render ADAS panel
            if frame_state:
                self.dashboard.render_adas_panel(frame_state)

            # Render control info (legacy, now part of metrics)
            self.dashboard.render_control_info(
                throttle, brake, self.target_speed_kmh, frame_state
            )

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

            # Stop camera
            if self.carla and self.carla.camera:
                self.carla.camera.stop()

            # Cleanup dashboard
            if self.dashboard:
                self.dashboard.cleanup()

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
                       help="Path to UNet model file")
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
    )

    return system.run()


if __name__ == "__main__":
    sys.exit(main())

