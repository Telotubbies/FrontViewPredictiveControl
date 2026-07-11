"""
Tesla-style 3D perspective visualization.

Renders a 3D perspective view of the road ahead, showing:
  - Lane boundaries (left/right) as 3D ribbons
  - MPC planned trajectory as a glowing path
  - Reference path as a dashed line
  - Detected obstacles as 3D boxes with risk coloring
  - Obstacle avoidance path (when active) as an alternative route
  - Vehicle icon at the bottom

The view uses a simple perspective projection:
  - Camera is behind and above the vehicle, looking forward
  - World coordinates (x=forward, y=lateral) → screen (px, py)
  - Vanishing point at the horizon

This replaces the old top-down BEV map with a more intuitive 3D view
similar to Tesla's Autopilot visualization.
"""
from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    pygame = None
    HAS_PYGAME = False

try:
    from gui.render_utils import (
        aa_line, aa_circle, aa_filled_polygon, aa_arc,
        draw_glow_line, draw_glow_circle, lerp_color, fade_color,
        SmoothValue, PulseAnimation, cached_glow,
    )
    HAS_RENDER_UTILS = True
except Exception:
    HAS_RENDER_UTILS = False
    # Fallbacks to plain pygame.draw
    def aa_line(surf, color, start, end, width=1):
        pygame.draw.line(surf, color, start, end, width)
    def aa_circle(surf, color, center, radius, filled=True):
        if filled:
            pygame.draw.circle(surf, color, center, radius)
    def aa_filled_polygon(surf, points, color):
        pygame.draw.polygon(surf, color, points)
    def aa_arc(surf, color, center, radius, sa, ea, width=2):
        pygame.draw.arc(surf, color, (center[0]-radius, center[1]-radius,
                                      radius*2, radius*2), sa, ea, width)
    def draw_glow_line(surf, color, start, end, width=3, **kw):
        pygame.draw.line(surf, color, start, end, width)
    def draw_glow_circle(surf, color, center, radius, **kw):
        pygame.draw.circle(surf, color, center, radius)
    def lerp_color(c1, c2, t):
        return c2
    def fade_color(c, a):
        return c
    class SmoothValue:
        def __init__(self, initial=0.0, **kw): self._v = initial
        def set_target(self, v): self._v = v
        def update(self, dt): return self._v
        @property
        def value(self): return self._v
    class PulseAnimation:
        def __init__(self, *a, **kw): pass
        def update(self, dt): return 1.0

logger = logging.getLogger(__name__)

# ── Colors (Tesla-inspired) ─────────────────────────────────────────────────
COLOR_BG_GRAD_TOP = (15, 20, 35)       # dark blue at top
COLOR_BG_GRAD_BOT = (25, 30, 50)       # slightly lighter at bottom
COLOR_ROAD = (40, 42, 55)              # road surface
COLOR_ROAD_EDGE = (60, 65, 80)         # road edge lines
COLOR_LANE_LEFT = (255, 220, 0)        # yellow left lane
COLOR_LANE_RIGHT = (60, 130, 255)      # blue right lane
COLOR_MPC_PATH = (0, 220, 255)         # cyan MPC trajectory
COLOR_MPC_PATH_GLOW = (0, 180, 220)    # glow effect
COLOR_REF_PATH = (255, 165, 0)         # orange reference path
COLOR_OBSTACLE_HIGH = (255, 60, 60)    # red - high risk
COLOR_OBSTACLE_MED = (255, 180, 0)     # orange - medium risk
COLOR_OBSTACLE_LOW = (0, 200, 100)     # green - low risk
COLOR_AVOIDANCE_PATH = (180, 100, 255) # purple - avoidance route
COLOR_VEHICLE = (0, 200, 230)          # cyan vehicle icon
COLOR_VEHICLE_OUTLINE = (255, 255, 255)
COLOR_HORIZON = (50, 60, 90)           # horizon line
COLOR_GRID = (35, 40, 60)              # perspective grid
COLOR_TEXT = (200, 210, 230)
COLOR_TEXT_DIM = (120, 130, 150)


class Tesla3DView:
    """
    3D perspective visualization of the road, trajectory, and obstacles.

    Renders onto a pygame surface with perspective projection.
    The camera is positioned behind and above the vehicle, looking forward.
    """

    def __init__(self, width: int = 640, height: int = 350) -> None:
        self.width = width
        self.height = height

        # Camera parameters (perspective projection)
        self.cam_height = 3.5       # camera height above ground (m)
        self.cam_back = 5.0         # camera distance behind vehicle (m)
        self.fov = 60.0             # field of view (degrees)
        self.horizon_ratio = 0.35   # horizon at 35% from top
        self.scale = 12.0           # pixels per meter at 1m distance

        # Vanishing point
        self.vp_x = width // 2
        self.vp_y = int(height * self.horizon_ratio)

        # Road parameters
        self.lane_width = 3.5       # lane width (m)
        self.road_length = 50.0     # how far ahead to render (m)

        # Smoothing for trajectory
        self._prev_traj_pts: List[Tuple[int, int]] = []
        self._alpha = 0.6           # smoothing factor

    def _project(self, x_fwd: float, y_lat: float, z: float = 0.0) -> Optional[Tuple[int, int]]:
        """
        Project a 3D world point to 2D screen coordinates.

        Args:
            x_fwd: forward distance from vehicle (m)
            y_lat: lateral distance from vehicle centerline (m)
            z: height above ground (m)

        Returns:
            (px, py) screen coordinates, or None if behind camera
        """
        # Camera is at (cam_back, 0, cam_height) looking forward
        # Transform to camera frame: relative to camera
        cam_x = x_fwd + self.cam_back
        cam_y = y_lat
        cam_z = z - self.cam_height

        if cam_x <= 0.5:
            return None  # behind camera

        # Perspective projection
        # Screen y: closer = bottom, farther = horizon
        depth = cam_x
        screen_y = self.vp_y + int((self.height - self.vp_y) * (self.cam_height - cam_z) / depth * 2.0)
        # Screen x: lateral offset scaled by 1/depth
        screen_x = self.vp_x + int(y_lat * self.scale * self.cam_height / depth)

        # Clamp to screen
        if screen_y < -50 or screen_y > self.height + 50:
            return None
        if screen_x < -100 or screen_x > self.width + 100:
            return None

        return (screen_x, screen_y)

    def _draw_gradient_bg(self, surface: pygame.Surface) -> None:
        """Draw gradient background (dark blue sky → road)."""
        for y in range(self.height):
            if y < self.vp_y:
                # Sky
                t = y / max(1, self.vp_y)
                r = int(COLOR_BG_GRAD_TOP[0] * (1 - t) + COLOR_HORIZON[0] * t)
                g = int(COLOR_BG_GRAD_TOP[1] * (1 - t) + COLOR_HORIZON[1] * t)
                b = int(COLOR_BG_GRAD_TOP[2] * (1 - t) + COLOR_HORIZON[2] * t)
            else:
                # Ground
                t = (y - self.vp_y) / max(1, self.height - self.vp_y)
                r = int(COLOR_HORIZON[0] * (1 - t) + COLOR_BG_GRAD_BOT[0] * t)
                g = int(COLOR_HORIZON[1] * (1 - t) + COLOR_BG_GRAD_BOT[1] * t)
                b = int(COLOR_HORIZON[2] * (1 - t) + COLOR_BG_GRAD_BOT[2] * t)
            pygame.draw.line(surface, (r, g, b), (0, y), (self.width, y))

    def _draw_road(self, surface: pygame.Surface) -> None:
        """Draw the road surface as a perspective trapezoid."""
        # Road edges at ±lane_width
        near_left = self._project(0, -self.lane_width)
        near_right = self._project(0, self.lane_width)
        far_left = self._project(self.road_length, -self.lane_width)
        far_right = self._project(self.road_length, self.lane_width)

        if all(p is not None for p in [near_left, near_right, far_left, far_right]):
            # Road surface (anti-aliased)
            road_pts = [near_left, far_left, far_right, near_right]
            aa_filled_polygon(surface, road_pts, COLOR_ROAD)

            # Road edges (anti-aliased)
            aa_line(surface, COLOR_ROAD_EDGE, near_left, far_left, 2)
            aa_line(surface, COLOR_ROAD_EDGE, near_right, far_right, 2)

    def _draw_lane_lines(self, surface: pygame.Surface, frame_state: Any) -> None:
        """Draw lane boundary lines in 3D perspective (anti-aliased)."""
        # Left lane line (yellow) with subtle glow
        left_pts = []
        for d in range(0, int(self.road_length), 2):
            p = self._project(float(d), -self.lane_width * 0.85)
            if p:
                left_pts.append(p)
        if len(left_pts) > 1:
            for i in range(len(left_pts) - 1):
                aa_line(surface, COLOR_LANE_LEFT, left_pts[i], left_pts[i+1], 2)

        # Right lane line (blue) with subtle glow
        right_pts = []
        for d in range(0, int(self.road_length), 2):
            p = self._project(float(d), self.lane_width * 0.85)
            if p:
                right_pts.append(p)
        if len(right_pts) > 1:
            for i in range(len(right_pts) - 1):
                aa_line(surface, COLOR_LANE_RIGHT, right_pts[i], right_pts[i+1], 2)

        # Dashed center line (white, anti-aliased)
        for d in range(0, int(self.road_length), 4):
            p1 = self._project(float(d), 0)
            p2 = self._project(float(d + 2), 0)
            if p1 and p2:
                aa_line(surface, (200, 200, 200), p1, p2, 1)

    def _draw_mpc_trajectory(self, surface: pygame.Surface, frame_state: Any) -> None:
        """Draw the MPC planned trajectory as a glowing 3D path."""
        traj = getattr(frame_state, 'mpc_trajectory', None)
        if traj is None or not isinstance(traj, np.ndarray):
            return

        pts = []
        n_pts = traj.shape[1]
        for k in range(n_pts):
            x = float(traj[0, k])  # forward
            y = float(traj[1, k])  # lateral
            p = self._project(x, y)
            if p:
                pts.append(p)

        if len(pts) < 2:
            return

        # Glow effect: draw thick semi-transparent line first
        for i in range(len(pts) - 1):
            draw_glow_line(surface, COLOR_MPC_PATH, pts[i], pts[i+1],
                          width=3, glow_radius=6, glow_alpha=60)

        # Main path line (anti-aliased)
        for i in range(len(pts) - 1):
            aa_line(surface, COLOR_MPC_PATH, pts[i], pts[i+1], 3)

        # Path dots (anti-aliased with glow)
        for p in pts:
            draw_glow_circle(surface, COLOR_MPC_PATH, p, 3,
                            glow_radius=4, glow_alpha=80)

    def _draw_reference_path(self, surface: pygame.Surface, frame_state: Any) -> None:
        """Draw the reference path as a dashed orange line."""
        ref = getattr(frame_state, 'reference_path', None)
        if not ref or len(ref) < 2:
            return

        pts = []
        for s, lat in ref:
            p = self._project(float(s), float(lat))
            if p:
                pts.append(p)

        # Dashed line
        for i in range(0, len(pts) - 1, 2):
            if i + 1 < len(pts):
                pygame.draw.line(surface, COLOR_REF_PATH, pts[i], pts[i+1], 2)

    def _draw_avoidance_path(self, surface: pygame.Surface, frame_state: Any) -> None:
        """Draw the obstacle avoidance path when active."""
        if not getattr(frame_state, 'obstacle_avoidance_active', False):
            return

        shift = getattr(frame_state, 'obstacle_avoidance_shift', 0.0)
        if abs(shift) < 0.05:
            return

        # Draw a curved path showing the avoidance maneuver
        pts = []
        for d in range(0, 30, 2):
            # Smooth lateral shift using sigmoid
            t = d / 30.0
            lat = shift * (3 * t**2 - 2 * t**3)  # smoothstep
            p = self._project(float(d), lat)
            if p:
                pts.append(p)

        if len(pts) > 1:
            # Glow + main line
            for i in range(len(pts) - 1):
                draw_glow_line(surface, COLOR_AVOIDANCE_PATH, pts[i], pts[i+1],
                              width=3, glow_radius=8, glow_alpha=100)
            for i in range(len(pts) - 1):
                aa_line(surface, COLOR_AVOIDANCE_PATH, pts[i], pts[i+1], 3)

    def _draw_obstacles(self, surface: pygame.Surface, frame_state: Any) -> None:
        """Draw detected obstacles as 3D boxes."""
        obstacles = getattr(frame_state, 'detected_obstacles', None)
        if not obstacles:
            return

        for fwd, lat, dist, type_str in obstacles:
            if fwd <= 0 or fwd > 45:
                continue

            # Determine risk color based on distance
            if dist < 8:
                color = COLOR_OBSTACLE_HIGH
            elif dist < 20:
                color = COLOR_OBSTACLE_MED
            else:
                color = COLOR_OBSTACLE_LOW

            # Draw obstacle as a 3D box
            # Box dimensions: 2m wide, 4m long, 1.5m tall
            half_w = 1.0
            half_l = 2.0
            height = 1.5

            # 8 corners of the box
            corners_3d = [
                (fwd - half_l, lat - half_w, 0),      # 0: bottom-front-left
                (fwd - half_l, lat + half_w, 0),      # 1: bottom-front-right
                (fwd + half_l, lat + half_w, 0),      # 2: bottom-back-right
                (fwd + half_l, lat - half_w, 0),      # 3: bottom-back-left
                (fwd - half_l, lat - half_w, height), # 4: top-front-left
                (fwd - half_l, lat + half_w, height), # 5: top-front-right
                (fwd + half_l, lat + half_w, height), # 6: top-back-right
                (fwd + half_l, lat - half_w, height), # 7: top-back-left
            ]

            corners_2d = []
            for cx, cy, cz in corners_3d:
                p = self._project(cx, cy, cz)
                if p:
                    corners_2d.append(p)
                else:
                    corners_2d.append(None)

            if len([c for c in corners_2d if c is not None]) < 4:
                continue

            # Draw bottom face (anti-aliased)
            bottom = [corners_2d[i] for i in range(4) if corners_2d[i] is not None]
            if len(bottom) >= 3:
                aa_filled_polygon(surface, bottom, (color[0]//3, color[1]//3, color[2]//3))

            # Draw top face (anti-aliased)
            top = [corners_2d[i] for i in range(4, 8) if corners_2d[i] is not None]
            if len(top) >= 3:
                aa_filled_polygon(surface, top, (color[0]//2, color[1]//2, color[2]//2))

            # Draw vertical edges (anti-aliased)
            for i in range(4):
                if corners_2d[i] and corners_2d[i+4]:
                    aa_line(surface, color, corners_2d[i], corners_2d[i+4], 2)

            # Draw top edges (anti-aliased)
            for i in range(4):
                j = (i + 1) % 4
                if corners_2d[i+4] and corners_2d[j+4]:
                    aa_line(surface, color, corners_2d[i+4], corners_2d[j+4], 2)

            # Draw distance label
            label_pos = self._project(fwd, lat, height + 0.5)
            if label_pos:
                try:
                    font = pygame.font.SysFont("DejaVu Sans", 12)
                    text = f"{dist:.0f}m"
                    surf = font.render(text, True, color)
                    surface.blit(surf, (label_pos[0] - surf.get_width()//2, label_pos[1]))
                except Exception:
                    pass

    def _draw_vehicle_icon(self, surface: pygame.Surface) -> None:
        """Draw the ego vehicle at the bottom of the view."""
        # Vehicle at (0, 0) in ego frame
        veh_w = 1.8
        veh_l = 4.5

        corners = [
            self._project(-veh_l/2, -veh_w/2),
            self._project(-veh_l/2, veh_w/2),
            self._project(veh_l/2, veh_w/2),
            self._project(veh_l/2, -veh_w/2),
        ]

        if all(c is not None for c in corners):
            # Vehicle body (anti-aliased)
            aa_filled_polygon(surface, corners, COLOR_VEHICLE)
            # Outline
            for i in range(len(corners)):
                j = (i + 1) % len(corners)
                aa_line(surface, COLOR_VEHICLE_OUTLINE, corners[i], corners[j], 2)

            # Headlights (with glow)
            for sign in [-1, 1]:
                hl = self._project(veh_l/2, sign * veh_w/3)
                if hl:
                    draw_glow_circle(surface, (255, 255, 200), hl, 3,
                                    glow_radius=5, glow_alpha=120)

    def _draw_grid(self, surface: pygame.Surface) -> None:
        """Draw perspective grid lines on the road (anti-aliased)."""
        # Horizontal lines (distance markers)
        for d in [5, 10, 15, 20, 30, 40]:
            p1 = self._project(float(d), -self.lane_width)
            p2 = self._project(float(d), self.lane_width)
            if p1 and p2:
                aa_line(surface, COLOR_GRID, p1, p2, 1)
                # Distance label
                try:
                    font = pygame.font.SysFont("DejaVu Sans", 10)
                    text = f"{d}m"
                    surf = font.render(text, True, COLOR_TEXT_DIM)
                    surface.blit(surf, (p1[0] + 2, p1[1] - 12))
                except Exception:
                    pass

    def _draw_info_overlay(self, surface: pygame.Surface, frame_state: Any) -> None:
        """Draw status info overlay (avoidance status, obstacle count)."""
        try:
            font = pygame.font.SysFont("DejaVu Sans", 12)
            y = 5

            # Obstacle count
            n_obs = getattr(frame_state, 'obstacle_count', 0)
            if n_obs > 0:
                color = COLOR_OBSTACLE_HIGH if n_obs > 0 else COLOR_TEXT
                text = f"Obstacles: {n_obs}"
                surf = font.render(text, True, color)
                surface.blit(surf, (5, y))
                y += 16

            # Avoidance status
            av_active = getattr(frame_state, 'obstacle_avoidance_active', False)
            if av_active:
                side = getattr(frame_state, 'obstacle_avoidance_side', 'none')
                shift = getattr(frame_state, 'obstacle_avoidance_shift', 0.0)
                dist = getattr(frame_state, 'obstacle_avoidance_dist', float('inf'))
                text = f"AVOIDANCE: {side.upper()} shift={shift:.1f}m dist={dist:.0f}m"
                surf = font.render(text, True, COLOR_AVOIDANCE_PATH)
                surface.blit(surf, (5, y))
                y += 16

            # MPC status
            solver = getattr(frame_state, 'solver_status', '')
            if solver:
                color = (0, 220, 100) if 'Succeeded' in solver else (255, 180, 0)
                text = f"MPC: {solver}"
                surf = font.render(text, True, color)
                surface.blit(surf, (5, y))
                y += 16

        except Exception as e:
            logger.debug(f"Info overlay failed: {e}")

    def render(self, surface: pygame.Surface, frame_state: Any) -> None:
        """
        Render the full 3D Tesla-style view.

        Args:
            surface: pygame Surface to render onto (width x height)
            frame_state: FrameState with trajectory, obstacles, etc.
        """
        if not HAS_PYGAME:
            return

        try:
            # 1. Background gradient
            self._draw_gradient_bg(surface)

            # 2. Perspective grid
            self._draw_grid(surface)

            # 3. Road surface
            self._draw_road(surface)

            # 4. Lane lines
            self._draw_lane_lines(surface, frame_state)

            # 5. Reference path (orange dashed)
            self._draw_reference_path(surface, frame_state)

            # 6. MPC trajectory (cyan glowing path)
            self._draw_mpc_trajectory(surface, frame_state)

            # 7. Obstacle avoidance path (purple, when active)
            self._draw_avoidance_path(surface, frame_state)

            # 8. Obstacles (3D boxes)
            self._draw_obstacles(surface, frame_state)

            # 9. Ego vehicle icon
            self._draw_vehicle_icon(surface)

            # 10. Info overlay
            self._draw_info_overlay(surface, frame_state)

        except Exception as e:
            logger.debug(f"Tesla 3D view render failed: {e}")
