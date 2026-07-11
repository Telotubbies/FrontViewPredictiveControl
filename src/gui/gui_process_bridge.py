"""
GUI process bridge — runs the dashboard in a separate process.

Architecture:
  Main process (control loop, no pygame)
      │
      │  multiprocessing.Queue (frame data → GUI)
      ▼
  GUI process (pygame dashboard)
      │
      │  multiprocessing.Queue (events → main)
      ▼
  Main process receives quit/keyboard events

This decouples rendering from control, so slow GUI rendering does NOT
block the control loop.  The car drives reliably while the dashboard
updates at its own pace.

Data sent to GUI (per frame):
  - JPEG-compressed RGB frame (small, ~10-30 KB)
  - FrameState fields (serialized as dict)
  - Control values (steer, throttle, brake, speed)

Data sent from GUI to main:
  - Quit event (user closed window or pressed ESC)
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _gui_process_main(
    frame_queue: mp.Queue,
    event_queue: mp.Queue,
    target_speed_kmh: float,
) -> None:
    """
    GUI process entry point.

    Runs the pygame dashboard loop, reading frame data from frame_queue
    and sending quit events to event_queue.
    """
    import os
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

    try:
        import pygame
        import cv2
        from gui.dashboard import Dashboard
        from state import FrameState
    except Exception as e:
        logger.error(f"GUI process import failed: {e}")
        return

    try:
        pygame.init()
        dashboard = Dashboard()
        if not dashboard.initialize():
            logger.error("GUI: dashboard init failed")
            return

        logger.info("GUI process started — rendering dashboard")

        frame_count = 0
        fps = 0.0
        last_frame_time = time.time()

        while True:
            # Check for quit event from main process
            try:
                msg = event_queue.get_nowait()
                if msg == "QUIT":
                    break
            except Exception:
                pass

            # Handle pygame events (window close, ESC)
            if not dashboard.handle_events():
                try:
                    event_queue.put("QUIT")
                except Exception:
                    pass
                break

            # Get latest frame data (non-blocking, skip if not ready)
            try:
                data = frame_queue.get(timeout=0.05)
            except Exception:
                # No new frame — just keep rendering current state
                continue

            if data is None:
                break  # Shutdown signal

            # Unpack frame data
            try:
                jpg_bytes = data.get('jpg_bytes')
                fs_dict = data.get('frame_state', {})
                steer = data.get('steer', 0.0)
                throttle = data.get('throttle', 0.0)
                brake = data.get('brake', 0.0)
                speed_ms = data.get('speed_ms', 0.0)
                target_kmh = data.get('target_speed_kmh', target_speed_kmh)
                sim_time = data.get('sim_time', 0.0)
                frame_num = data.get('frame_count', 0)

                # Decode JPEG → RGB
                rgb_frame = None
                if jpg_bytes:
                    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    rgb_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                # Reconstruct FrameState from dict
                frame_state = FrameState(
                    rgb=rgb_frame,
                    speed_kmh=speed_ms * 3.6,
                    steer=steer,
                    throttle=throttle,
                    brake=brake,
                    cte_m=fs_dict.get('cte_m', 0.0),
                    heading_rad=fs_dict.get('heading_rad', 0.0),
                    curvature=fs_dict.get('curvature', 0.0),
                    mode=fs_dict.get('mode', 'WP_PRIMARY'),
                    lane_conf=fs_dict.get('lane_conf', 0.0),
                    geometry_valid=fs_dict.get('geometry_valid', True),
                    reference_path=fs_dict.get('reference_path'),
                    mpc_trajectory=fs_dict.get('mpc_trajectory'),
                    solver_status=fs_dict.get('solver_status', ''),
                    mpc_solve_time_ms=fs_dict.get('mpc_solve_time_ms', 0.0),
                    safety_override_active=fs_dict.get('safety_override_active', False),
                    adas_override_active=fs_dict.get('adas_override_active', False),
                    stuck_recovery_active=fs_dict.get('stuck_recovery_active', False),
                    stuck_recovery_phase=fs_dict.get('stuck_recovery_phase', 'none'),
                    final_steer=steer,
                    final_throttle=throttle,
                    final_brake=brake,
                    vehicle_x=fs_dict.get('vehicle_x', 0.0),
                    vehicle_y=fs_dict.get('vehicle_y', 0.0),
                    vehicle_z=fs_dict.get('vehicle_z', 0.0),
                    vehicle_yaw=fs_dict.get('vehicle_yaw', 0.0),
                    aeb_warning_level=fs_dict.get('aeb_warning_level', 'none'),
                    acc_active=fs_dict.get('acc_active', False),
                    acc_target_speed_ms=fs_dict.get('acc_target_speed_ms', -1.0),
                    acc_distance_m=fs_dict.get('acc_distance_m', -1.0),
                    obstacle_avoidance_active=fs_dict.get('obstacle_avoidance_active', False),
                    obstacle_avoidance_side=fs_dict.get('obstacle_avoidance_side', 'none'),
                    obstacle_avoidance_shift=fs_dict.get('obstacle_avoidance_shift', 0.0),
                    obstacle_avoidance_dist=fs_dict.get('obstacle_avoidance_dist', float('inf')),
                    obstacle_avoidance_steer=fs_dict.get('obstacle_avoidance_steer', 0.0),
                    obstacle_count=fs_dict.get('obstacle_count', 0),
                    detected_obstacles=fs_dict.get('detected_obstacles'),
                )

                # Set lane_overlay if available
                if 'lane_overlay' in fs_dict and fs_dict['lane_overlay'] is not None:
                    frame_state.lane_overlay = fs_dict['lane_overlay']
                if 'bev_binary' in fs_dict and fs_dict['bev_binary'] is not None:
                    frame_state.bev_binary = fs_dict['bev_binary']

                # Calculate FPS
                now = time.time()
                dt = now - last_frame_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                last_frame_time = now

                # Render dashboard
                dashboard.panel_surface.fill((26, 26, 46))

                if rgb_frame is not None:
                    if frame_state.lane_overlay is not None:
                        dashboard.render_lane_overlay(rgb_frame, frame_state.lane_overlay, frame_state)
                    else:
                        dashboard.render_lane_overlay(rgb_frame, None, frame_state)

                if frame_state:
                    dashboard.render_bev_view(frame_state.bev_binary, frame_state)
                    dashboard._render_top_bar(frame_state, target_kmh)
                    dashboard.render_health_bar(frame_state)

                speed_kmh = speed_ms * 3.6
                confidence = frame_state.lane_conf if frame_state else 0.0
                cte = frame_state.cte_m if frame_state else 0.0

                dashboard.render_status_info(
                    speed_kmh, steer, cte, confidence, fps, frame_state
                )

                if frame_state:
                    dashboard.render_hmi_gauges(frame_state, target_kmh, fps)
                    dashboard.render_adas_panel(frame_state)

                dashboard.render_control_info(throttle, brake, target_kmh, frame_state)
                dashboard.render_event_timeline()
                dashboard._render_status_bar(fps, frame_num, sim_time)
                dashboard.update()

                frame_count += 1

            except Exception as e:
                logger.debug(f"GUI render error: {e}")

        # Cleanup
        dashboard.cleanup()
        pygame.quit()
        logger.info("GUI process exited cleanly")

    except Exception as e:
        logger.error(f"GUI process crashed: {e}", exc_info=True)


class GuiProcessBridge:
    """
    Bridge between main control process and GUI rendering process.

    Usage:
        bridge = GuiProcessBridge(target_speed_kmh=20.0)
        bridge.start()
        ...
        bridge.send_frame(rgb_frame, frame_state, steer, throttle, brake, speed_ms)
        ...
        if bridge.should_quit():
            break
        ...
        bridge.stop()
    """

    def __init__(self, target_speed_kmh: float = 20.0) -> None:
        self.target_speed_kmh = target_speed_kmh
        self._frame_queue: Optional[mp.Queue] = None
        self._event_queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._running = False
        self._frame_count = 0
        self._loop_start = time.time()

    def start(self) -> bool:
        """Start the GUI process. Returns True if successful."""
        try:
            ctx = mp.get_context('spawn')  # spawn for pygame compatibility
            self._frame_queue = ctx.Queue(maxsize=3)  # small buffer, drop old
            self._event_queue = ctx.Queue(maxsize=10)
            self._process = ctx.Process(
                target=_gui_process_main,
                args=(self._frame_queue, self._event_queue, self.target_speed_kmh),
                daemon=True,
            )
            self._process.start()
            self._running = True
            self._loop_start = time.time()
            logger.info("GUI process bridge started (PID=%d)", self._process.pid)
            return True
        except Exception as e:
            logger.error(f"Failed to start GUI process: {e}")
            return False

    def send_frame(
        self,
        rgb_frame: np.ndarray,
        frame_state: Any,
        steer: float,
        throttle: float,
        brake: float,
        speed_ms: float,
    ) -> None:
        """Send a frame to the GUI process (non-blocking, drops old frames)."""
        if not self._running or self._frame_queue is None:
            return

        try:
            import cv2

            # Compress RGB frame to JPEG (smaller transfer)
            jpg_bytes = b''
            if rgb_frame is not None:
                # rgb_frame is RGB, cv2 wants BGR
                bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    jpg_bytes = buf.tobytes()

            # Serialize FrameState fields
            fs_dict = {}
            if frame_state is not None:
                for attr in ['cte_m', 'heading_rad', 'curvature', 'mode',
                             'lane_conf', 'geometry_valid', 'reference_path',
                             'mpc_trajectory', 'solver_status', 'mpc_solve_time_ms',
                             'safety_override_active', 'adas_override_active',
                             'stuck_recovery_active', 'stuck_recovery_phase',
                             'vehicle_x', 'vehicle_y', 'vehicle_z', 'vehicle_yaw',
                             'aeb_warning_level', 'acc_active', 'acc_target_speed_ms',
                             'acc_distance_m',
                             'obstacle_avoidance_active', 'obstacle_avoidance_side',
                             'obstacle_avoidance_shift', 'obstacle_avoidance_dist',
                             'obstacle_avoidance_steer', 'obstacle_count',
                             'detected_obstacles']:
                    fs_dict[attr] = getattr(frame_state, attr, None)

                # lane_overlay and bev_binary (may be large — skip if too big)
                lane_overlay = getattr(frame_state, 'lane_overlay', None)
                if lane_overlay is not None and lane_overlay.size < 500000:
                    fs_dict['lane_overlay'] = lane_overlay
                bev_binary = getattr(frame_state, 'bev_binary', None)
                if bev_binary is not None and bev_binary.size < 200000:
                    fs_dict['bev_binary'] = bev_binary

            sim_time = time.time() - self._loop_start
            self._frame_count += 1

            data = {
                'jpg_bytes': jpg_bytes,
                'frame_state': fs_dict,
                'steer': float(steer),
                'throttle': float(throttle),
                'brake': float(brake),
                'speed_ms': float(speed_ms),
                'target_speed_kmh': self.target_speed_kmh,
                'sim_time': sim_time,
                'frame_count': self._frame_count,
            }

            # Non-blocking put — drop if queue full (keeps GUI fresh)
            try:
                self._frame_queue.put_nowait(data)
            except Exception:
                # Queue full — drop this frame
                pass

        except Exception as e:
            logger.debug(f"send_frame failed: {e}")

    def should_quit(self) -> bool:
        """Check if GUI process sent a quit event."""
        if not self._running or self._event_queue is None:
            return False
        try:
            msg = self._event_queue.get_nowait()
            if msg == "QUIT":
                logger.info("GUI process requested quit")
                return True
        except Exception:
            pass
        # Check if process died
        if self._process and not self._process.is_alive():
            logger.warning("GUI process died unexpectedly")
            return True
        return False

    def stop(self) -> None:
        """Stop the GUI process."""
        if not self._running:
            return

        self._running = False
        try:
            if self._frame_queue is not None:
                self._frame_queue.put(None)  # shutdown signal
        except Exception:
            pass

        if self._process is not None:
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)

        logger.info("GUI process bridge stopped")
