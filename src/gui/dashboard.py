#!/usr/bin/env python3
"""
Tesla-style Dashboard GUI Module - จัดการการแสดงผล GUI สำหรับ CARLA MPC system

Layout (1280x800):
+----------------------------------------------------------+
|  Speed (large)    |  Autopilot Status  |  Target Speed   |  Top bar (80px)
|-------------------+--------------------+-----------------|
|  Camera View      |  BEV Map           |                 |  Center (480px)
|  (with overlay)   |  (top-down)        |                 |
|-------------------+--------------------+-----------------|
|  Metrics Panel    |  ADAS Status Panel |                 |  Bottom (240px)
|-------------------+--------------------+-----------------|
|  Route progress | FPS | Sim time                        |  Status bar (40px)
+----------------------------------------------------------+
"""
import logging
import math
from collections import deque
from typing import Optional, Tuple, List

import pygame
import numpy as np
import cv2

from config import (
    BEV_WIN_W, BEV_WIN_H,
    PANEL_W, PANEL_H, VIEW3D_W, VIEW3D_H,
    UXColors
)

logger = logging.getLogger(__name__)

# ── Layout constants ──────────────────────────────────────────────────────
DASH_W = 1280
DASH_H = 800

TOP_BAR_H = 80
CENTER_H = 380
BOTTOM_H = 280
STATUS_BAR_H = 60

CAM_W = 640
CAM_H = 360
BEV_PANEL_W = 640
BEV_PANEL_H = 380

METRICS_W = 640
METRICS_H = 280
ADAS_W = 640
ADAS_H = 280

# ── Color scheme (Tesla-inspired dark) ────────────────────────────────────
COLOR_BG = (26, 26, 46)           # #1a1a2e
COLOR_PANEL = (35, 35, 60)        # #23233c
COLOR_TEXT = (230, 230, 235)
COLOR_TEXT_DIM = (140, 140, 160)
COLOR_ACCENT = (0, 180, 255)      # cyan
COLOR_SUCCESS = (0, 220, 100)     # green
COLOR_WARNING = (255, 200, 0)     # yellow
COLOR_ERROR = (255, 80, 80)       # red
COLOR_MPC = (0, 255, 255)         # cyan for MPC trajectory
COLOR_REF = (255, 165, 0)         # orange for reference path
COLOR_LANE_LEFT = (255, 220, 0)   # yellow
COLOR_LANE_RIGHT = (60, 120, 255) # blue


class Dashboard:
    """Tesla-style Dashboard GUI สำหรับแสดงผล real-time data"""

    def __init__(self,
                 panel_w: int = DASH_W,
                 panel_h: int = DASH_H,
                 bev_win_w: int = BEV_WIN_W,
                 bev_win_h: int = BEV_WIN_H):
        self.panel_w = panel_w
        self.panel_h = panel_h
        self.bev_win_w = bev_win_w
        self.bev_win_h = bev_win_h

        # Pygame components
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        self.big_font: Optional[pygame.font.Font] = None
        self.tiny_font: Optional[pygame.font.Font] = None

        # Display surfaces
        self.panel_surface: Optional[pygame.Surface] = None
        self.bev_surface: Optional[pygame.Surface] = None
        self.view3d_surface: Optional[pygame.Surface] = None
        self.cam_surface: Optional[pygame.Surface] = None
        self.metrics_surface: Optional[pygame.Surface] = None
        self.adas_surface: Optional[pygame.Surface] = None

        # Sparkline data
        self._cte_history: deque = deque(maxlen=100)
        self._speed_history: deque = deque(maxlen=100)

        self._initialized = False
        self._paused = False

    def initialize(self) -> bool:
        """Initialize pygame and dashboard components"""
        try:
            pygame.init()
            pygame.display.set_caption("CARLA MPC — Tesla Dashboard")

            self.screen = pygame.display.set_mode((self.panel_w, self.panel_h))

            # Setup fonts
            self.big_font = pygame.font.Font(None, 48)
            self.font = pygame.font.Font(None, 28)
            self.small_font = pygame.font.Font(None, 22)
            self.tiny_font = pygame.font.Font(None, 18)

            # Create display surfaces
            self.panel_surface = pygame.Surface((self.panel_w, self.panel_h))
            self.bev_surface = pygame.Surface((BEV_PANEL_W, BEV_PANEL_H))
            self.view3d_surface = pygame.Surface((VIEW3D_W, VIEW3D_H))
            self.cam_surface = pygame.Surface((CAM_W, CAM_H))
            self.metrics_surface = pygame.Surface((METRICS_W, METRICS_H))
            self.adas_surface = pygame.Surface((ADAS_W, ADAS_H))

            self._initialized = True
            logger.info("Tesla Dashboard initialized (%dx%d)", self.panel_w, self.panel_h)
            return True

        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            return False

    # ── Text rendering helpers ─────────────────────────────────────────────

    def render_text(self, text: str, pos: Tuple[int, int],
                    color: Tuple[int, int, int] = COLOR_TEXT,
                    font: Optional[pygame.font.Font] = None,
                    surface: Optional[pygame.Surface] = None) -> None:
        """Render text on specified surface"""
        if not self._initialized:
            return
        f = font or self.font
        surf = surface or self.panel_surface
        try:
            text_surface = f.render(text, True, color)
            surf.blit(text_surface, pos)
        except Exception as e:
            logger.error(f"Failed to render text: {e}")

    def render_text_centered(self, text: str, center_pos: Tuple[int, int],
                             color: Tuple[int, int, int] = COLOR_TEXT,
                             font: Optional[pygame.font.Font] = None,
                             surface: Optional[pygame.Surface] = None) -> None:
        """Render centered text"""
        if not self._initialized:
            return
        f = font or self.font
        surf = surface or self.panel_surface
        try:
            text_surface = f.render(text, True, color)
            rect = text_surface.get_rect(center=center_pos)
            surf.blit(text_surface, rect)
        except Exception as e:
            logger.error(f"Failed to render centered text: {e}")

    # ── Image rendering ────────────────────────────────────────────────────

    def render_image(self, image: np.ndarray,
                     surface: pygame.Surface,
                     pos: Tuple[int, int] = (0, 0)) -> None:
        """Render numpy image to pygame surface"""
        if not self._initialized or image is None:
            return
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            image_surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))
            surface.blit(image_surface, pos)
        except Exception as e:
            logger.error(f"Failed to render image: {e}")

    # ── Camera view with overlay ───────────────────────────────────────────

    def render_lane_overlay(self,
                            rgb_frame: np.ndarray,
                            lane_overlay: Optional[np.ndarray] = None,
                            frame_state=None) -> None:
        """Render camera frame with lane overlay + MPC trajectory"""
        if not self._initialized or rgb_frame is None:
            return

        try:
            # Resize to camera panel
            frame = cv2.resize(rgb_frame, (CAM_W, CAM_H))

            # Apply lane overlay if available
            if lane_overlay is not None:
                overlay = cv2.resize(lane_overlay, (CAM_W, CAM_H))
                frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

            # Draw MPC trajectory if available
            if frame_state is not None and frame_state.mpc_trajectory is not None:
                frame = self._draw_mpc_trajectory_on_camera(frame, frame_state.mpc_trajectory)

            # Draw reference path if available
            if frame_state is not None and frame_state.reference_path:
                frame = self._draw_reference_path_on_camera(frame, frame_state.reference_path)

            # Convert to RGB and render
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.render_image(frame_rgb, self.cam_surface)

            # Confidence border
            if frame_state is not None:
                conf = frame_state.lane_conf
                if conf > 0.7:
                    border_color = (0, 220, 100)
                elif conf > 0.4:
                    border_color = (255, 200, 0)
                else:
                    border_color = (255, 80, 80)
                pygame.draw.rect(self.cam_surface, border_color,
                                 (0, 0, CAM_W, CAM_H), 2)

            # Speed overlay (bottom-left)
            if frame_state is not None:
                speed_text = f"{frame_state.speed_kmh:.0f} km/h"
                self.render_text(speed_text, (10, CAM_H - 35),
                                 color=COLOR_TEXT, font=self.big_font,
                                 surface=self.cam_surface)

        except Exception as e:
            logger.error(f"Failed to render lane overlay: {e}")

    def _draw_mpc_trajectory_on_camera(self, frame: np.ndarray,
                                       trajectory: np.ndarray) -> np.ndarray:
        """Project MPC trajectory (x,y in vehicle frame) onto camera image."""
        try:
            # trajectory is (4, N+1): [x, y, psi, v]
            # Simple perspective projection: assume camera at origin, looking forward (+x)
            # y is lateral, x is forward distance
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            # Simple pinhole: image_x = cx - f*y/x, image_y = cy - f*height/x
            # Approximate f from FOV
            f = w / 2.0  # rough estimate for 90deg FOV

            pts = []
            for k in range(trajectory.shape[1]):
                x = float(trajectory[0, k])  # forward
                y = float(trajectory[1, k])  # lateral
                if x < 1.0:
                    continue  # skip points too close
                px = int(cx - f * y / x)
                py = int(cy + 30)  # approximate horizon
                if 0 <= px < w and 0 <= py < h:
                    pts.append((px, py))

            # Draw connected line
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (255, 255, 0), 2)

            # Draw dots
            for pt in pts:
                cv2.circle(frame, pt, 3, (0, 255, 255), -1)

            return frame
        except Exception as e:
            logger.debug(f"MPC trajectory projection failed: {e}")
            return frame

    def _draw_reference_path_on_camera(self, frame: np.ndarray,
                                       ref_path: List[Tuple[float, float]]) -> np.ndarray:
        """Draw reference path (s_m, lat_m) onto camera image."""
        try:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            f = w / 2.0

            pts = []
            for s, lat in ref_path:
                if s < 1.0:
                    continue
                px = int(cx - f * lat / s)
                py = int(cy + 30)
                if 0 <= px < w and 0 <= py < h:
                    pts.append((px, py))

            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (0, 165, 255), 1)  # orange dashed-ish

            return frame
        except Exception:
            return frame

    # ── BEV Map ────────────────────────────────────────────────────────────

    def render_bev_view(self, bev_image: np.ndarray, frame_state=None) -> None:
        """Render BEV (Bird's Eye View) map with vehicle, lanes, MPC path"""
        if not self._initialized:
            return

        try:
            self.bev_surface.fill(COLOR_PANEL)

            if bev_image is not None:
                # Use existing BEV binary mask as background
                bev_resized = cv2.resize(bev_image, (BEV_PANEL_W, BEV_PANEL_H))
                if len(bev_resized.shape) == 2:
                    bev_rgb = cv2.cvtColor(bev_resized, cv2.COLOR_GRAY2RGB)
                else:
                    bev_rgb = bev_resized
                self.render_image(bev_rgb, self.bev_surface)

            # Draw BEV map overlay (top-down view)
            if frame_state is not None:
                self._draw_bev_overlay(frame_state)

            # Title
            self.render_text("BEV Map", (10, 5),
                             color=COLOR_TEXT_DIM, font=self.small_font,
                             surface=self.bev_surface)

        except Exception as e:
            logger.error(f"Failed to render BEV view: {e}")

    def _draw_bev_overlay(self, frame_state) -> None:
        """Draw top-down map: vehicle, lane boundaries, MPC path, waypoints."""
        try:
            px_per_m = 4.0  # scale: 4 pixels per meter
            cx = BEV_PANEL_W // 2
            cy = BEV_PANEL_H - 50  # vehicle at bottom-center

            # Draw grid lines every 10m
            for d in range(10, 100, 10):
                y = cy - int(d * px_per_m)
                if y < 0:
                    break
                pygame.draw.line(self.bev_surface, (50, 50, 70),
                                 (0, y), (BEV_PANEL_W, y), 1)
                self.render_text(f"{d}m", (5, y - 15),
                                 color=COLOR_TEXT_DIM, font=self.tiny_font,
                                 surface=self.bev_surface)

            # Draw vehicle rectangle
            veh_w = int(1.8 * px_per_m)  # 1.8m wide
            veh_h = int(4.5 * px_per_m)  # 4.5m long
            veh_rect = pygame.Rect(cx - veh_w // 2, cy - veh_h // 2, veh_w, veh_h)
            pygame.draw.rect(self.bev_surface, COLOR_ACCENT, veh_rect)
            pygame.draw.rect(self.bev_surface, COLOR_TEXT, veh_rect, 1)

            # Draw heading indicator
            heading = getattr(frame_state, 'vehicle_yaw', 0.0)
            hx = cx + int(math.sin(heading) * 30)
            hy = cy - int(math.cos(heading) * 30)
            pygame.draw.line(self.bev_surface, COLOR_SUCCESS, (cx, cy), (hx, hy), 2)

            # Draw MPC trajectory
            mpc_traj = getattr(frame_state, 'mpc_trajectory', None)
            if mpc_traj is not None and isinstance(mpc_traj, np.ndarray):
                pts = []
                for k in range(mpc_traj.shape[1]):
                    x = float(mpc_traj[0, k])  # forward
                    y = float(mpc_traj[1, k])  # lateral
                    px = cx + int(y * px_per_m)
                    py = cy - int(x * px_per_m)
                    if 0 <= px < BEV_PANEL_W and 0 <= py < BEV_PANEL_H:
                        pts.append((px, py))

                if len(pts) > 1:
                    pygame.draw.lines(self.bev_surface, COLOR_MPC, False, pts, 2)
                for pt in pts:
                    pygame.draw.circle(self.bev_surface, COLOR_MPC, pt, 3)

            # Draw reference path
            ref_path = getattr(frame_state, 'reference_path', None)
            if ref_path:
                pts = []
                for s, lat in ref_path:
                    px = cx + int(lat * px_per_m)
                    py = cy - int(s * px_per_m)
                    if 0 <= px < BEV_PANEL_W and 0 <= py < BEV_PANEL_H:
                        pts.append((px, py))
                if len(pts) > 1:
                    pygame.draw.lines(self.bev_surface, COLOR_REF, False, pts, 1)

        except Exception as e:
            logger.debug(f"BEV overlay failed: {e}")

    # ── Status / Control info (legacy API compatibility) ───────────────────

    def render_status_info(self, speed_kmh: float, steering: float,
                           cte: float, confidence: float, fps: float,
                           frame_state=None) -> None:
        """Render vehicle status — now routes to metrics panel"""
        # Update sparkline data
        self._cte_history.append(cte)
        self._speed_history.append(speed_kmh)

        if frame_state is not None:
            self._render_metrics_panel(frame_state, fps)
        else:
            # Fallback: basic metrics
            self._render_basic_metrics(speed_kmh, steering, cte, confidence, fps)

    def render_control_info(self, throttle: float, brake: float,
                            target_speed: float, frame_state=None) -> None:
        """Render control information — routes to metrics panel"""
        # Control info is now part of metrics panel
        pass

    def _render_basic_metrics(self, speed_kmh, steering, cte, confidence, fps):
        """Basic metrics when no frame_state available."""
        self.metrics_surface.fill(COLOR_PANEL)
        y = 10
        self.render_text(f"Speed: {speed_kmh:.1f} km/h", (10, y),
                         surface=self.metrics_surface)
        y += 30
        self.render_text(f"Steering: {steering:.3f}", (10, y),
                         surface=self.metrics_surface)
        y += 30
        cte_color = COLOR_SUCCESS if abs(cte) < 0.5 else COLOR_WARNING if abs(cte) < 1.0 else COLOR_ERROR
        self.render_text(f"CTE: {cte:.2f} m", (10, y), color=cte_color,
                         surface=self.metrics_surface)
        y += 30
        conf_color = COLOR_SUCCESS if confidence > 0.7 else COLOR_WARNING if confidence > 0.4 else COLOR_ERROR
        self.render_text(f"Confidence: {confidence:.2f}", (10, y), color=conf_color,
                         surface=self.metrics_surface)
        y += 30
        self.render_text(f"FPS: {fps:.1f}", (10, y),
                         surface=self.metrics_surface)

    def _render_metrics_panel(self, frame_state, fps: float):
        """Render full metrics panel with sparklines."""
        self.metrics_surface.fill(COLOR_PANEL)

        # Title
        self.render_text("Metrics", (10, 5),
                         color=COLOR_TEXT_DIM, font=self.small_font,
                         surface=self.metrics_surface)

        y = 35
        col_x = [10, 220, 440]

        # CTE
        cte = frame_state.cte_m
        cte_color = COLOR_SUCCESS if abs(cte) < 0.5 else COLOR_WARNING if abs(cte) < 1.0 else COLOR_ERROR
        self.render_text(f"CTE: {cte:+.2f} m", (col_x[0], y), color=cte_color,
                         font=self.font, surface=self.metrics_surface)

        # Heading error
        head_deg = math.degrees(frame_state.heading_rad)
        head_color = COLOR_SUCCESS if abs(head_deg) < 3.0 else COLOR_WARNING if abs(head_deg) < 7.0 else COLOR_ERROR
        self.render_text(f"Heading: {head_deg:+.1f}°", (col_x[1], y), color=head_color,
                         font=self.font, surface=self.metrics_surface)

        # Curvature
        curv = frame_state.curvature
        self.render_text(f"Curv: {curv:+.4f}", (col_x[2], y),
                         font=self.font, surface=self.metrics_surface)
        y += 35

        # Lane confidence
        conf = frame_state.lane_conf
        conf_color = COLOR_SUCCESS if conf > 0.7 else COLOR_WARNING if conf > 0.4 else COLOR_ERROR
        self.render_text(f"Confidence: {conf:.0%}", (col_x[0], y), color=conf_color,
                         font=self.font, surface=self.metrics_surface)
        # Confidence bar
        bar_w = int(120 * conf)
        pygame.draw.rect(self.metrics_surface, (60, 60, 80),
                         (col_x[0] + 140, y + 5, 120, 18))
        pygame.draw.rect(self.metrics_surface, conf_color,
                         (col_x[0] + 140, y + 5, bar_w, 18))
        y += 35

        # MPC status
        solver = frame_state.solver_status
        solver_color = COLOR_SUCCESS if "Succeed" in solver else COLOR_WARNING
        self.render_text(f"MPC: {solver}", (col_x[0], y), color=solver_color,
                         font=self.small_font, surface=self.metrics_surface)
        self.render_text(f"Solve: {frame_state.mpc_solve_time_ms:.1f} ms",
                         (col_x[1], y), font=self.small_font,
                         surface=self.metrics_surface)
        y += 30

        # Speed
        speed_kmh = frame_state.speed_kmh
        self.render_text(f"Speed: {speed_kmh:.1f} km/h", (col_x[0], y),
                         font=self.font, surface=self.metrics_surface)
        y += 35

        # Control values (final, after overrides)
        steer_deg = math.degrees(frame_state.final_steer * 0.4)  # approx
        self.render_text(f"Steer: {frame_state.final_steer:+.3f} ({steer_deg:+.1f}°)",
                         (col_x[0], y), font=self.small_font,
                         surface=self.metrics_surface)
        self.render_text(f"Throttle: {frame_state.final_throttle:.2f}",
                         (col_x[1], y),
                         color=COLOR_SUCCESS if frame_state.final_throttle > 0.1 else COLOR_TEXT_DIM,
                         font=self.small_font, surface=self.metrics_surface)
        self.render_text(f"Brake: {frame_state.final_brake:.2f}",
                         (col_x[2], y),
                         color=COLOR_ERROR if frame_state.final_brake > 0.1 else COLOR_TEXT_DIM,
                         font=self.small_font, surface=self.metrics_surface)
        y += 30

        # Override indicators
        overrides = []
        if frame_state.safety_override_active:
            overrides.append(("Safety", COLOR_WARNING))
        if frame_state.adas_override_active:
            overrides.append(("ADAS", COLOR_ACCENT))
        if frame_state.stuck_recovery_active:
            overrides.append(("Stuck", COLOR_ERROR))
        if not overrides:
            self.render_text("Overrides: none", (col_x[0], y),
                             color=COLOR_TEXT_DIM, font=self.small_font,
                             surface=self.metrics_surface)
        else:
            x = col_x[0]
            self.render_text("Overrides:", (x, y), font=self.small_font,
                             surface=self.metrics_surface)
            x += 90
            for name, color in overrides:
                self.render_text(f"[{name}]", (x, y), color=color,
                                 font=self.small_font, surface=self.metrics_surface)
                x += 70
        y += 30

        # FPS
        self.render_text(f"FPS: {fps:.1f}", (col_x[0], y),
                         color=COLOR_TEXT_DIM, font=self.small_font,
                         surface=self.metrics_surface)
        y += 25

        # Sparkline for CTE (last 100 frames)
        if len(self._cte_history) > 2:
            spark_x = 10
            spark_y = y
            spark_w = 280
            spark_h = 40
            pygame.draw.rect(self.metrics_surface, (20, 20, 35),
                             (spark_x, spark_y, spark_w, spark_h))
            cte_arr = np.array(self._cte_history)
            cte_max = max(abs(cte_arr.max()), abs(cte_arr.min()), 0.5)
            pts = []
            for i, v in enumerate(cte_arr):
                px = spark_x + int(i * spark_w / len(cte_arr))
                py = spark_y + spark_h // 2 - int(v * spark_h / 2 / cte_max)
                py = max(spark_y, min(spark_y + spark_h, py))
                pts.append((px, py))
            if len(pts) > 1:
                pygame.draw.lines(self.metrics_surface, COLOR_ACCENT, False, pts, 1)
            # Zero line
            pygame.draw.line(self.metrics_surface, (60, 60, 80),
                             (spark_x, spark_y + spark_h // 2),
                             (spark_x + spark_w, spark_y + spark_h // 2), 1)
            self.render_text("CTE", (spark_x, spark_y - 15),
                             color=COLOR_TEXT_DIM, font=self.tiny_font,
                             surface=self.metrics_surface)

        # Sparkline for speed
        if len(self._speed_history) > 2:
            spark_x = 310
            spark_y = y
            spark_w = 280
            spark_h = 40
            pygame.draw.rect(self.metrics_surface, (20, 20, 35),
                             (spark_x, spark_y, spark_w, spark_h))
            spd_arr = np.array(self._speed_history)
            spd_max = max(spd_arr.max(), 1.0)
            pts = []
            for i, v in enumerate(spd_arr):
                px = spark_x + int(i * spark_w / len(spd_arr))
                py = spark_y + spark_h - int(v * spark_h / spd_max)
                py = max(spark_y, min(spark_y + spark_h, py))
                pts.append((px, py))
            if len(pts) > 1:
                pygame.draw.lines(self.metrics_surface, COLOR_SUCCESS, False, pts, 1)
            self.render_text("Speed (km/h)", (spark_x, spark_y - 15),
                             color=COLOR_TEXT_DIM, font=self.tiny_font,
                             surface=self.metrics_surface)

    # ── ADAS Status Panel ──────────────────────────────────────────────────

    def render_adas_panel(self, frame_state) -> None:
        """Render ADAS status panel."""
        self.adas_surface.fill(COLOR_PANEL)

        # Title
        self.render_text("ADAS Status", (10, 5),
                         color=COLOR_TEXT_DIM, font=self.small_font,
                         surface=self.adas_surface)

        y = 35
        col_w = ADAS_W // 2
        col_x = [10, col_w + 10]
        row_h = 40

        # AEB
        aeb_active = frame_state.aeb_active
        aeb_color = COLOR_ERROR if aeb_active else COLOR_TEXT_DIM
        aeb_text = f"AEB: {'ACTIVE' if aeb_active else 'IDLE'}"
        if frame_state.aeb_ttc > 0:
            aeb_text += f" (TTC: {frame_state.aeb_ttc:.1f}s)"
        self._render_adas_indicator(0, col_x[0], y, "AEB", aeb_text, aeb_color)
        y += row_h

        # ACC
        acc_active = frame_state.acc_active
        acc_color = COLOR_SUCCESS if acc_active else COLOR_TEXT_DIM
        acc_text = f"ACC: {'ACTIVE' if acc_active else 'IDLE'}"
        if acc_active:
            acc_text += f" ({frame_state.acc_target_speed_ms * 3.6:.0f} km/h, {frame_state.acc_distance_m:.1f}m)"
        self._render_adas_indicator(1, col_x[0], y, "ACC", acc_text, acc_color)
        y += row_h

        # LDW
        ldw_state = frame_state.ldw_state
        if ldw_state == "in_lane":
            ldw_color = COLOR_SUCCESS
            ldw_text = "LDW: IN LANE"
        elif ldw_state == "departing":
            ldw_color = COLOR_WARNING
            ldw_text = f"LDW: DEPARTING ({frame_state.ldw_side})"
        else:
            ldw_color = COLOR_ERROR
            ldw_text = f"LDW: DEPARTED ({frame_state.ldw_side})"
        self._render_adas_indicator(0, col_x[0], y, "LDW", ldw_text, ldw_color)
        y += row_h

        # BSW
        bsw_l = frame_state.bsw_left_alert
        bsw_r = frame_state.bsw_right_alert
        bsw_color = COLOR_ERROR if "blind" in bsw_l or "blind" in bsw_r else COLOR_WARNING if "approach" in bsw_l or "approach" in bsw_r else COLOR_SUCCESS
        bsw_text = f"BSW: L={bsw_l[:4].upper()} R={bsw_r[:4].upper()}"
        self._render_adas_indicator(1, col_x[0], y, "BSW", bsw_text, bsw_color)
        y += row_h

        # TSR
        tsr_limit = frame_state.tsr_speed_limit_kmh
        tl = frame_state.tsr_traffic_light
        tsr_text = "TSR: "
        if tsr_limit is not None:
            tsr_text += f"{tsr_limit:.0f} km/h"
        else:
            tsr_text += "---"
        if tl != "unknown":
            tl_color = COLOR_ERROR if tl == "red" else COLOR_WARNING if tl == "yellow" else COLOR_SUCCESS
            tsr_text += f" | {tl.upper()}"
        else:
            tl_color = COLOR_TEXT_DIM
        self._render_adas_indicator(0, col_x[0], y, "TSR", tsr_text, tl_color)
        y += row_h

        # TJA
        tja_active = frame_state.tja_active
        tja_color = COLOR_SUCCESS if tja_active else COLOR_TEXT_DIM
        tja_text = f"TJA: {frame_state.tja_state.upper()}"
        self._render_adas_indicator(1, col_x[0], y, "TJA", tja_text, tja_color)

        # Right column: LKA Pro
        y2 = 35
        lka = frame_state.lka_pro_assist
        lka_color = COLOR_ACCENT if abs(lka) > 0.01 else COLOR_TEXT_DIM
        lka_text = f"LKA Pro: {lka:+.3f} rad"
        self._render_adas_indicator(0, col_x[1], y2, "LKA", lka_text, lka_color)
        y2 += row_h

        # Stop & Go
        sg = frame_state.stop_and_go_stopped
        sg_color = COLOR_WARNING if sg else COLOR_SUCCESS
        sg_text = f"Stop&Go: {'STOPPED' if sg else 'MOVING'}"
        self._render_adas_indicator(1, col_x[1], y2, "S&G", sg_text, sg_color)
        y2 += row_h

        # TSR action
        action = frame_state.tsr_action
        action_color = COLOR_ERROR if action == "stop" else COLOR_WARNING if action == "slow" else COLOR_SUCCESS if action == "proceed" else COLOR_TEXT_DIM
        action_text = f"Action: {action.upper()}"
        self._render_adas_indicator(0, col_x[1], y2, "ACT", action_text, action_color)
        y2 += row_h

        # Stuck recovery
        stuck = frame_state.stuck_recovery_active
        stuck_color = COLOR_ERROR if stuck else COLOR_TEXT_DIM
        stuck_text = f"Stuck: {frame_state.stuck_recovery_phase.upper() if stuck else 'NONE'}"
        self._render_adas_indicator(1, col_x[1], y2, "STK", stuck_text, stuck_color)
        y2 += row_h

        # Safety override
        safety = frame_state.safety_override_active
        safety_color = COLOR_WARNING if safety else COLOR_TEXT_DIM
        safety_text = f"Safety: {'OVERRIDE' if safety else 'OK'}"
        self._render_adas_indicator(0, col_x[1], y2, "SAF", safety_text, safety_color)
        y2 += row_h

        # Geometry valid
        geom = frame_state.geometry_valid
        geom_color = COLOR_SUCCESS if geom else COLOR_ERROR
        geom_text = f"Geometry: {'VALID' if geom else 'INVALID'}"
        self._render_adas_indicator(1, col_x[1], y2, "GEO", geom_text, geom_color)

    def _render_adas_indicator(self, idx: int, x: int, y: int,
                               label: str, text: str,
                               color: Tuple[int, int, int]) -> None:
        """Render a single ADAS indicator with status dot."""
        # Status dot
        pygame.draw.circle(self.adas_surface, color, (x + 8, y + 12), 6)
        # Label
        self.render_text(text, (x + 22, y), color=color,
                         font=self.small_font, surface=self.adas_surface)

    # ── Top bar ────────────────────────────────────────────────────────────

    def _render_top_bar(self, frame_state, target_speed_kmh: float) -> None:
        """Render top bar: speed | autopilot status | target speed."""
        y = 0
        # Background
        pygame.draw.rect(self.panel_surface, COLOR_PANEL, (0, y, self.panel_w, TOP_BAR_H))

        # Speed (large, left)
        speed_kmh = frame_state.speed_kmh if frame_state else 0.0
        speed_text = f"{speed_kmh:.0f}"
        self.render_text(speed_text, (40, y + 15),
                         color=COLOR_TEXT, font=self.big_font)
        self.render_text("km/h", (40 + 80, y + 30),
                         color=COLOR_TEXT_DIM, font=self.small_font)

        # Autopilot status (center)
        ap_active = (frame_state is not None and
                     frame_state.solver_status == "Solve_Succeeded" and
                     not frame_state.stuck_recovery_active)
        ap_color = COLOR_SUCCESS if ap_active else COLOR_WARNING
        ap_text = "AUTOPILOT" if ap_active else "MANUAL/FALLBACK"
        self.render_text_centered(ap_text, (self.panel_w // 2, y + 30),
                                  color=ap_color, font=self.font)

        # Sub-status
        sub_status = ""
        if frame_state:
            if frame_state.stuck_recovery_active:
                sub_status = f"STUCK RECOVERY ({frame_state.stuck_recovery_phase})"
            elif frame_state.aeb_active:
                sub_status = "AEB ACTIVE"
            elif frame_state.adas_override_active:
                sub_status = "ADAS OVERRIDE"
        if sub_status:
            self.render_text_centered(sub_status, (self.panel_w // 2, y + 55),
                                      color=COLOR_WARNING, font=self.tiny_font)

        # Target speed (right)
        self.render_text(f"{target_speed_kmh:.0f}", (self.panel_w - 120, y + 15),
                         color=COLOR_ACCENT, font=self.big_font)
        self.render_text("target km/h", (self.panel_w - 120, y + 55),
                         color=COLOR_TEXT_DIM, font=self.tiny_font)

    # ── Status bar (bottom) ────────────────────────────────────────────────

    def _render_status_bar(self, fps: float, frame_count: int, sim_time: float) -> None:
        """Render bottom status bar."""
        y = self.panel_h - STATUS_BAR_H
        pygame.draw.rect(self.panel_surface, COLOR_PANEL, (0, y, self.panel_w, STATUS_BAR_H))

        # FPS
        self.render_text(f"FPS: {fps:.1f}", (10, y + 20),
                         color=COLOR_TEXT_DIM, font=self.small_font)

        # Frame count
        self.render_text(f"Frame: {frame_count}", (120, y + 20),
                         color=COLOR_TEXT_DIM, font=self.small_font)

        # Sim time
        self.render_text(f"Time: {sim_time:.1f}s", (250, y + 20),
                         color=COLOR_TEXT_DIM, font=self.small_font)

        # Mode
        self.render_text("FVPC v2.0 | ESC: quit | SPACE: pause",
                         (self.panel_w - 350, y + 20),
                         color=COLOR_TEXT_DIM, font=self.tiny_font)

    # ── Main update ────────────────────────────────────────────────────────

    def update(self) -> None:
        """Update display — blit all panels to screen"""
        if not self._initialized:
            return

        try:
            self.screen.fill(COLOR_BG)

            # Top bar (rendered directly on panel_surface)
            # Camera view (left)
            self.screen.blit(self.cam_surface, (0, TOP_BAR_H))
            # BEV map (right)
            self.screen.blit(self.bev_surface, (CAM_W, TOP_BAR_H))
            # Metrics panel (bottom-left)
            self.screen.blit(self.metrics_surface, (0, TOP_BAR_H + CENTER_H))
            # ADAS panel (bottom-right)
            self.screen.blit(self.adas_surface, (METRICS_W, TOP_BAR_H + CENTER_H))

            pygame.display.flip()

        except Exception as e:
            logger.error(f"Failed to update display: {e}")

    def handle_events(self) -> bool:
        """Handle pygame events, return False if should quit"""
        if not self._initialized:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self._paused = not self._paused
                    logger.info(f"Dashboard {'paused' if self._paused else 'resumed'}")
        return True

    def cleanup(self) -> None:
        """Cleanup pygame resources"""
        try:
            if self._initialized:
                pygame.quit()
                self._initialized = False
                logger.info("Dashboard cleaned up")
        except Exception as e:
            logger.error(f"Error during dashboard cleanup: {e}")

    def __enter__(self):
        if self.initialize():
            return self
        raise RuntimeError("Failed to initialize dashboard")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class DashboardRenderer:
    """Helper class for rendering dashboard components"""

    @staticmethod
    def create_lane_overlay(rgb_frame: np.ndarray,
                            lane_mask: np.ndarray,
                            overlay_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Create lane overlay on RGB frame"""
        try:
            overlay = rgb_frame.copy()
            colored_mask = np.zeros_like(rgb_frame)
            colored_mask[lane_mask > 0] = overlay_color
            overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
            return overlay
        except Exception as e:
            logger.error(f"Failed to create lane overlay: {e}")
            return rgb_frame

    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray,
                                 target_size: Tuple[int, int],
                                 fill_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
            target = np.full((target_h, target_w) + image.shape[2:], fill_color, dtype=image.dtype)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            target[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            return target
        except Exception as e:
            logger.error(f"Failed to resize with aspect ratio: {e}")
            return cv2.resize(image, target_size)
