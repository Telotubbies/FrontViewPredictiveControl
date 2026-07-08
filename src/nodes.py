"""
Pipeline nodes — each node processes one stage of the per-frame pipeline.

Base class: PipelineNode (abstract)
Concrete nodes:
  1. SensorNode      — ticks CARLA, gets RGB + vehicle transform + speed + waypoints
  2. PerceptionNode  — runs lane detection + fusion + MPC + safety (delegates to LKAPipeline.step)
  3. ControlNode     — reserved for future MPC extraction (currently pass-through; MPC runs
                       inside LKAPipeline.step which bundles perception+fusion+MPC+safety)
  4. ArbitrationNode — runs ControlArbitrator if available, else falls back to ADAS overrides
  5. ActuatorNode    — sends final control to CARLA + stuck recovery
  6. TelemetryNode   — records to telemetry exporter + metrics collector (non-blocking)
  7. DashboardNode   — updates dashboard display (non-blocking)

All per-frame mutable state (steer, throttle, brake, adas_out, current_frame, etc.) is
carried in a shared ``context`` dict so nodes can communicate without tight coupling.
FrameState flows through as the return value of each node.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from state import FrameState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class PipelineNode(ABC):
    """Base class for pipeline nodes. Each node processes one stage."""

    @abstractmethod
    def process(self, state: Optional[FrameState], context: dict) -> Optional[FrameState]:
        """Process frame state, return new state.

        Nodes may read/write the shared ``context`` dict for per-frame mutable
        values (steer, throttle, brake, adas_out, etc.).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Node name for logging."""
        ...


# ---------------------------------------------------------------------------
# 1. SensorNode
# ---------------------------------------------------------------------------
class SensorNode(PipelineNode):
    """Reads sensor data from CARLA: RGB frame, vehicle transform, speed, waypoints."""

    def __init__(self, carla):
        self._carla = carla

    @property
    def name(self) -> str:
        return "Sensor"

    def process(self, state: Optional[FrameState], context: dict) -> Optional[FrameState]:
        carla = self._carla

        # Get RGB frame from camera queue (populated by callback in main loop)
        current_frame = context.get("current_frame")
        if current_frame is None:
            return None  # No frame yet — caller should skip

        # Get vehicle state
        vehicle_transform = carla.get_vehicle_transform()
        current_speed_ms = carla.get_vehicle_velocity()

        if vehicle_transform is None:
            return None  # No vehicle transform — skip frame

        # Get waypoint state for fusion (WP primary)
        wp_state = None
        try:
            from carla_input_output import get_waypoints, waypoints_to_cte_heading
            wps = get_waypoints(carla.vehicle, carla.world.get_map())
            if wps:
                wp_state = waypoints_to_cte_heading(vehicle_transform, wps)
        except Exception:
            pass

        context["vehicle_transform"] = vehicle_transform
        context["current_speed_ms"] = current_speed_ms
        context["wp_state"] = wp_state
        return state


# ---------------------------------------------------------------------------
# 2. PerceptionNode
# ---------------------------------------------------------------------------
class PerceptionNode(PipelineNode):
    """Runs the LKAPipeline step (perception -> fusion -> MPC -> safety).

    NOTE: In the current architecture, ``LKAPipeline.step()`` bundles perception,
    fusion, MPC, and safety override into a single call.  This node delegates to
    that existing entry point so control logic is unchanged.  When the pipeline is
    later split, PerceptionNode will call only the perception stage.
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline

    @property
    def name(self) -> str:
        return "Perception"

    def process(self, state: Optional[FrameState], context: dict) -> Optional[FrameState]:
        current_frame = context.get("current_frame")
        if current_frame is None:
            return state

        steer, throttle, brake, frame_state = self._pipeline.step(
            rgb=current_frame,
            speed_ms=context.get("current_speed_ms", 0.0),
            wp_state=context.get("wp_state"),
            prev_steer=context.get("prev_steer", 0.0),
            prev_throttle=context.get("prev_throttle", 0.0),
            world=context.get("carla_world"),
            vehicle=context.get("carla_vehicle"),
        )

        context["steer"] = steer
        context["throttle"] = throttle
        context["brake"] = brake
        return frame_state


# ---------------------------------------------------------------------------
# 3. ControlNode
# ---------------------------------------------------------------------------
class ControlNode(PipelineNode):
    """Reserved for future MPC extraction.

    Currently a pass-through: MPC runs inside ``LKAPipeline.step()`` (called by
    PerceptionNode).  When the pipeline is split so that perception and control are
    separate stages, this node will call ``LaneMPC.solve()`` directly.
    """

    @property
    def name(self) -> str:
        return "Control"

    def process(self, state: Optional[FrameState], context: dict) -> Optional[FrameState]:
        return state


# ---------------------------------------------------------------------------
# 4. ArbitrationNode
# ---------------------------------------------------------------------------
class ArbitrationNode(PipelineNode):
    """Runs control arbitration.

    Attempts to use the new ``ControlArbitrator`` (src/control/arbitrator.py) if
    available.  If the arbitrator module is not yet created, falls back to the
    existing ADAS override logic from main.py so functionality is preserved.
    """

    def __init__(self, adas, carla=None):
        self._adas = adas
        self._carla = carla
        # Try to import the new ControlArbitrator (may not exist yet)
        self._arbitrator = None
        try:
            from control.arbitrator import ControlArbitrator
            self._arbitrator = ControlArbitrator()
            logger.info("ArbitrationNode: using ControlArbitrator")
        except Exception:
            logger.debug("ArbitrationNode: ControlArbitrator not available, using ADAS fallback")

    @property
    def name(self) -> str:
        return "Arbitration"

    def process(self, state: Optional[FrameState], context: dict) -> Optional[FrameState]:
        steer = context.get("steer", 0.0)
        throttle = context.get("throttle", 0.0)
        brake = context.get("brake", 0.0)
        current_speed_ms = context.get("current_speed_ms", 0.0)
        vehicle_transform = context.get("vehicle_transform")

        # ── Path A: new ControlArbitrator (if available) ──────────────────
        if self._arbitrator is not None:
            try:
                result = self._arbitrator.arbitrate(
                    mpc_steer=steer,
                    mpc_throttle=throttle,
                    mpc_brake=brake,
                    vehicle_state={"speed_ms": current_speed_ms},
                    frame_state=state,
                    adas_out=context.get("adas_out"),
                )
                steer = result.steer
                throttle = result.throttle
                brake = result.brake
                context["steer"] = steer
                context["throttle"] = throttle
                context["brake"] = brake
                return state
            except Exception as e:
                logger.debug(f"ArbitrationNode: arbitrator failed, ADAS fallback: {e}")

        # ── Path B: existing ADAS override fallback ───────────────────────
        if self._adas is None or self._carla is None or vehicle_transform is None:
            return state

        try:
            from bridge.obstacles import get_traffic_obstacles

            obstacles = get_traffic_obstacles(
                self._carla.world,
                exclude_actor_id=self._carla.vehicle.id,
            )
            ego_loc = vehicle_transform.location
            ego_rot = vehicle_transform.rotation
            ego_heading = math.radians(ego_rot.yaw)
            timestamp = time.time()

            adas_out = self._adas.update(
                obstacles=obstacles,
                ego_x=ego_loc.x,
                ego_y=ego_loc.y,
                ego_heading=ego_heading,
                ego_speed=current_speed_ms,
                cte=state.cte_m if state else 0.0,
                heading_err=state.heading_rad if state else 0.0,
                curvature=state.curvature if state else 0.0,
                timestamp=timestamp,
                carla_world=self._carla.world,
                vehicle_transform=vehicle_transform,
                nominal_target_speed_ms=context.get("target_speed_ms", 0.0),
            )

            # Apply ADAS overrides to MPC output
            steer, throttle, brake, _ = self._adas.apply_to_control(
                steer, throttle, brake, adas_out
            )

            # Populate FrameState with ADAS status for dashboard
            if state:
                state.aeb_active = getattr(adas_out, 'aeb_active', False)
                state.aeb_ttc = getattr(adas_out, 'aeb_ttc', -1.0)
                state.aeb_warning_level = getattr(adas_out, 'aeb_warning_level', 'none')
                state.acc_active = getattr(adas_out, 'acc_active', False)
                state.acc_target_speed_ms = getattr(adas_out, 'acc_target_speed_ms', -1.0)
                state.acc_distance_m = getattr(adas_out, 'acc_distance_m', -1.0)
                state.ldw_state = getattr(adas_out, 'ldw_state', 'in_lane')
                state.ldw_warning_active = getattr(adas_out, 'ldw_warning_active', False)
                state.ldw_side = getattr(adas_out, 'ldw_side', 'none')
                state.lka_pro_assist = getattr(adas_out, 'lka_pro_assist', 0.0)
                state.bsw_left_alert = getattr(adas_out, 'bsw_left_alert', 'clear')
                state.bsw_right_alert = getattr(adas_out, 'bsw_right_alert', 'clear')
                state.bsw_safe_left = getattr(adas_out, 'bsw_safe_left', True)
                state.bsw_safe_right = getattr(adas_out, 'bsw_safe_right', True)
                state.tsr_speed_limit_kmh = getattr(adas_out, 'tsr_speed_limit_kmh', None)
                state.tsr_traffic_light = getattr(adas_out, 'tsr_traffic_light', 'unknown')
                state.tsr_traffic_light_distance = getattr(adas_out, 'tsr_traffic_light_distance', -1.0)
                state.tsr_action = getattr(adas_out, 'tsr_action', 'none')
                state.tja_state = getattr(adas_out, 'tja_state', 'inactive')
                state.tja_active = getattr(adas_out, 'tja_active', False)
                state.stop_and_go_stopped = getattr(adas_out, 'stop_and_go_stopped', False)
                state.adas_override_active = bool(adas_out.active_features)

            if adas_out.active_features:
                logger.debug("ADAS active: %s", ", ".join(adas_out.active_features))

            context["steer"] = steer
            context["throttle"] = throttle
            context["brake"] = brake
            context["adas_out"] = adas_out
            context["ego_loc"] = ego_loc
            context["ego_heading"] = ego_heading
        except Exception as e:
            logger.debug(f"ADAS update skipped: {e}")

        return state


# ---------------------------------------------------------------------------
# 5. ActuatorNode
# ---------------------------------------------------------------------------
class ActuatorNode(PipelineNode):
    """Sends final control to CARLA and handles stuck recovery."""

    def __init__(self, carla, stuck_recovery=None):
        self._carla = carla
        self._stuck_recovery = stuck_recovery

    @property
    def name(self) -> str:
        return "Actuator"

    def process(self, state: Optional[FrameState], context: dict) -> Optional[FrameState]:
        steer = context.get("steer", 0.0)
        throttle = context.get("throttle", 0.0)
        brake = context.get("brake", 0.0)
        current_speed_ms = context.get("current_speed_ms", 0.0)

        # Apply control to vehicle
        carla_control = self._carla.get_vehicle_control()
        carla_control.steering = steer
        carla_control.throttle = throttle
        carla_control.brake = brake
        self._carla.apply_control(carla_control)

        # Persist control state for next frame's smoothing
        context["prev_steer"] = steer
        context["prev_throttle"] = throttle

        # Check for stuck recovery
        stuck_active = False
        stuck_phase = "none"
        if self._stuck_recovery is not None:
            recovery = self._stuck_recovery.update(current_speed_ms, throttle)
            if recovery is not None:
                r_steer, r_throttle, r_brake, r_reverse = recovery
                carla_control = self._carla.get_vehicle_control()
                carla_control.steering = r_steer
                carla_control.throttle = r_throttle
                carla_control.brake = r_brake
                carla_control.reverse = r_reverse
                self._carla.apply_control(carla_control)
                stuck_active = True
                stuck_phase = getattr(self._stuck_recovery, '_phase', 'unknown')
                logger.info("Applying stuck recovery (phase=%s)", stuck_phase)
                # Override final values for dashboard
                steer = r_steer
                throttle = r_throttle
                brake = r_brake
                context["steer"] = steer
                context["throttle"] = throttle
                context["brake"] = brake
                # Mark last metrics frame
                metrics_collector = context.get("metrics_collector")
                if metrics_collector and metrics_collector.frames:
                    metrics_collector.frames[-1].stuck_recovery_active = True

        # Update FrameState with final control values + vehicle pose
        if state:
            state.final_steer = steer
            state.final_throttle = throttle
            state.final_brake = brake
            state.stuck_recovery_active = stuck_active
            state.stuck_recovery_phase = stuck_phase
            ego_loc = context.get("ego_loc")
            ego_heading = context.get("ego_heading")
            state.vehicle_x = ego_loc.x if ego_loc is not None else 0.0
            state.vehicle_y = ego_loc.y if ego_loc is not None else 0.0
            state.vehicle_z = ego_loc.z if ego_loc is not None else 0.0
            state.vehicle_yaw = ego_heading if ego_heading is not None else 0.0

        return state


# ---------------------------------------------------------------------------
# 6. TelemetryNode
# ---------------------------------------------------------------------------
class TelemetryNode(PipelineNode):
    """Records frame data to telemetry exporter + metrics collector (non-blocking)."""

    def __init__(self, telemetry=None, metrics_collector=None):
        self._telemetry = telemetry
        self._metrics_collector = metrics_collector

    @property
    def name(self) -> str:
        return "Telemetry"

    def process(self, state: Optional[FrameState], context: dict) -> Optional[FrameState]:
        steer = context.get("steer", 0.0)
        throttle = context.get("throttle", 0.0)
        brake = context.get("brake", 0.0)
        current_speed_ms = context.get("current_speed_ms", 0.0)
        adas_out = context.get("adas_out")
        loop_start_time = context.get("loop_start_time", time.time())
        frame_count = context.get("frame_count", 0)
        loop_times = context.get("loop_times", [])

        # ── Telemetry export ──────────────────────────────────────────────
        if self._telemetry and self._telemetry.is_connected():
            try:
                self._telemetry.export_frame(
                    vehicle={
                        "speed_ms": current_speed_ms,
                        "speed_kmh": current_speed_ms * 3.6,
                        "steering": steer,
                        "throttle": throttle,
                        "brake": brake,
                    },
                    lane={
                        "cte_m": state.cte_m if state else 0.0,
                        "heading_err_rad": state.heading_rad if state else 0.0,
                        "curvature": state.curvature if state else 0.0,
                        "lane_conf": state.lane_conf if state else 0.0,
                    },
                    adas_out=adas_out,
                    performance={
                        "fps": 1.0 / loop_times[-1] if loop_times else 0.0,
                        "loop_time_ms": (loop_times[-1] * 1000) if loop_times else 0.0,
                    },
                )
            except Exception as e:
                logger.debug(f"Telemetry export skipped: {e}")

        # ── Metrics collection ────────────────────────────────────────────
        if self._metrics_collector:
            loop_time_ms = (time.time() - loop_start_time) * 1000.0
            mpc_solve_ms = float(getattr(state, 'mpc_solve_time_ms', 0.0)) if state else 0.0
            mpc_status = str(getattr(state, 'solver_status', 'unknown')) if state else 'unknown'
            safety_active = bool(getattr(state, 'safety_active', False)) if state else False
            self._metrics_collector.record_from_frame_state(
                frame_idx=frame_count,
                speed_ms=current_speed_ms,
                steer=steer,
                throttle=throttle,
                brake=brake,
                frame_state=state,
                mpc_solve_time_ms=mpc_solve_ms,
                mpc_solver_status=mpc_status,
                loop_time_ms=loop_time_ms,
                safety_active=safety_active,
                stuck_recovery_active=False,
                adas_out=adas_out,
            )

        return state


# ---------------------------------------------------------------------------
# 7. DashboardNode
# ---------------------------------------------------------------------------
class DashboardNode(PipelineNode):
    """Updates the GUI dashboard display (non-blocking)."""

    def __init__(self, dashboard=None, target_speed_kmh: float = 0.0):
        self._dashboard = dashboard
        self._target_speed_kmh = target_speed_kmh

    @property
    def name(self) -> str:
        return "Dashboard"

    def process(self, state: Optional[FrameState], context: dict) -> Optional[FrameState]:
        if self._dashboard is None:
            return state

        current_frame = context.get("current_frame")
        current_speed_ms = context.get("current_speed_ms", 0.0)
        steer = context.get("steer", 0.0)
        throttle = context.get("throttle", 0.0)
        brake = context.get("brake", 0.0)
        frame_count = context.get("frame_count", 0)
        loop_times = context.get("loop_times", [])
        loop_start_time = context.get("loop_start_time", time.time())

        try:
            # Clear panel
            self._dashboard.panel_surface.fill((26, 26, 46))

            # Render camera frame with lane overlay + MPC trajectory
            if state and state.lane_overlay is not None:
                self._dashboard.render_lane_overlay(current_frame, state.lane_overlay, state)
            else:
                self._dashboard.render_lane_overlay(current_frame, None, state)

            # Render BEV view if available (with MPC path overlay)
            if state:
                self._dashboard.render_bev_view(state.bev_binary, state)

            # Render top bar (speed, autopilot, target)
            self._dashboard._render_top_bar(state, self._target_speed_kmh)

            # Render metrics panel
            speed_kmh = current_speed_ms * 3.6
            confidence = state.lane_conf if state else 0.0
            cte = state.cte_m if state else 0.0
            fps = 1.0 / loop_times[-1] if loop_times else 0.0

            self._dashboard.render_status_info(
                speed_kmh, steer, cte, confidence, fps, state
            )

            # Render ADAS panel
            if state:
                self._dashboard.render_adas_panel(state)

            # Render control info (legacy, now part of metrics)
            self._dashboard.render_control_info(
                throttle, brake, self._target_speed_kmh, state
            )

            # Render status bar
            sim_time = time.time() - loop_start_time
            self._dashboard._render_status_bar(fps, frame_count, sim_time)

            # Update display
            self._dashboard.update()

        except Exception as e:
            logger.error(f"Dashboard update error: {e}")

        return state
