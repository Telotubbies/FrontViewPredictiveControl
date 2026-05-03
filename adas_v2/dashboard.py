"""
Dashboard for ADAS v2 (Apollo-style Modern UI)

Real-time visualization:
    - Camera feed with lane overlay
    - Mode indicator (LANE/WP/FUSION)
    - Speed gauge (circular)
    - CTE bar (center-based)
    - Steering wheel indicator
    - Throttle/Brake bars
    - Performance metrics
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class DashboardState:
    """State for dashboard rendering."""
    rgb: Optional[np.ndarray] = None
    speed_kmh: float = 0.0
    target_speed_kmh: float = 0.0
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    cte: float = 0.0
    heading_deg: float = 0.0
    curvature: float = 0.0
    lane_confidence: float = 0.0
    lane_weight: float = 0.0
    mode: str = "INIT"
    wp_lookahead_curv: float = 0.0
    frame_id: int = 0
    fps: float = 0.0


class ADASv2Dashboard:
    """
    Clean dashboard for ADAS v2 - Camera view only with HUD overlay.
    
    Layout:
    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │                     Camera View (Full)                             │
    │                     + Lane Overlay                                 │
    │                                                                    │
    │  ┌──────┐                                          ┌──────────┐   │
    │  │ MODE │                                          │ SPEED    │   │
    │  │ LANE │                                          │  30 km/h │   │
    │  └──────┘                                          └──────────┘   │
    │                                                                    │
    │  ═══════════════════════╪═══════════════════════  CTE: +0.25m     │
    │  THR: ████████░░░░  BRK: ░░░░░░░░░░  STEER: +0.12                 │
    └────────────────────────────────────────────────────────────────────┘
    """
    
    # Colors (BGR) - Clean HUD style
    COLOR_BG = (0, 0, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_DIM = (180, 180, 180)
    COLOR_GREEN = (100, 255, 120)
    COLOR_YELLOW = (0, 220, 255)
    COLOR_RED = (80, 80, 255)
    COLOR_CYAN = (255, 220, 100)
    COLOR_ORANGE = (0, 165, 255)
    
    MODE_COLORS = {
        "LANE": (100, 255, 120),     # Green
        "WP": (0, 220, 255),          # Yellow
        "FUSION": (0, 165, 255),      # Orange
        "INIT": (180, 180, 180),      # Gray
    }
    
    def __init__(
        self,
        width: int = 960,
        height: int = 540,
    ):
        self.width = width
        self.height = height
        
    def render(self, state: DashboardState, bev_image: np.ndarray = None) -> np.ndarray:
        """Render clean camera-only dashboard with HUD overlay."""
        # Start with camera image as base
        if state.rgb is not None:
            h, w = state.rgb.shape[:2]
            canvas = cv2.resize(state.rgb, (self.width, self.height))
            if len(canvas.shape) == 3 and canvas.shape[2] == 3:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        else:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw HUD overlay
        self._draw_hud(canvas, state)
        
        return canvas
    
    def _draw_hud(self, canvas: np.ndarray, state: DashboardState):
        """Draw clean HUD overlay on camera view."""
        h, w = canvas.shape[:2]
        
        # === Top Left: Mode indicator ===
        mode_name = state.mode.split()[0] if state.mode else "INIT"
        mode_color = self.MODE_COLORS.get(mode_name, self.COLOR_TEXT_DIM)
        
        # Mode box with semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (15, 15), (130, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        
        cv2.rectangle(canvas, (15, 15), (130, 70), mode_color, 2)
        cv2.putText(canvas, mode_name, (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"L:{state.lane_weight*100:.0f}%", (25, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        
        # === Top Right: Speed gauge ===
        speed_x = w - 140
        overlay = canvas.copy()
        cv2.rectangle(overlay, (speed_x, 15), (w - 15, 85), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        
        cv2.rectangle(canvas, (speed_x, 15), (w - 15, 85), self.COLOR_CYAN, 2)
        cv2.putText(canvas, f"{state.speed_kmh:.0f}", (speed_x + 15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.COLOR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(canvas, "km/h", (speed_x + 70, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"/{state.target_speed_kmh:.0f}", (speed_x + 15, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        
        # === Bottom: Control bars ===
        bar_y = h - 60
        
        # Semi-transparent background for bottom bar
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, bar_y - 10), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
        
        # CTE bar (center-based)
        cte_bar_x = 150
        cte_bar_w = 300
        cte_bar_h = 16
        
        cv2.putText(canvas, "CTE", (15, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        
        # Bar background
        cv2.rectangle(canvas, (cte_bar_x, bar_y), 
                     (cte_bar_x + cte_bar_w, bar_y + cte_bar_h), (50, 50, 50), -1)
        
        # CTE indicator
        cte_center = cte_bar_x + cte_bar_w // 2
        cte_px = int(state.cte * 75)  # Scale: 1m = 75px
        cte_pos = cte_center + cte_px
        cte_pos = max(cte_bar_x + 5, min(cte_bar_x + cte_bar_w - 5, cte_pos))
        
        cte_color = self.COLOR_GREEN if abs(state.cte) < 0.5 else \
                    self.COLOR_YELLOW if abs(state.cte) < 1.0 else self.COLOR_RED
        
        # Draw CTE bar fill
        if state.cte >= 0:
            cv2.rectangle(canvas, (cte_center, bar_y + 2), 
                         (cte_pos, bar_y + cte_bar_h - 2), cte_color, -1)
        else:
            cv2.rectangle(canvas, (cte_pos, bar_y + 2), 
                         (cte_center, bar_y + cte_bar_h - 2), cte_color, -1)
        
        # Center line
        cv2.line(canvas, (cte_center, bar_y - 3), (cte_center, bar_y + cte_bar_h + 3), 
                 self.COLOR_TEXT, 2)
        
        # CTE value
        cv2.putText(canvas, f"{state.cte:+.2f}m", (cte_bar_x + cte_bar_w + 10, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cte_color, 1, cv2.LINE_AA)
        
        # === Second row: THR/BRK/STEER ===
        row2_y = bar_y + 28
        
        # Throttle
        cv2.putText(canvas, "THR", (15, row2_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        self._draw_bar(canvas, 55, row2_y, 80, 14, state.throttle, self.COLOR_GREEN)
        
        # Brake
        cv2.putText(canvas, "BRK", (150, row2_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        self._draw_bar(canvas, 190, row2_y, 80, 14, state.brake, self.COLOR_RED)
        
        # Steering
        steer_color = self.COLOR_GREEN if abs(state.steer) < 0.3 else \
                      self.COLOR_YELLOW if abs(state.steer) < 0.6 else self.COLOR_RED
        cv2.putText(canvas, f"STEER: {state.steer:+.2f}", (290, row2_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, steer_color, 1, cv2.LINE_AA)
        
        # Curvature
        cv2.putText(canvas, f"CURV: {state.curvature:.3f}", (420, row2_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        
        # FPS & Frame
        cv2.putText(canvas, f"FPS:{state.fps:.0f} F:{state.frame_id}", (w - 130, row2_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    
    def _draw_bar(self, canvas: np.ndarray, x: int, y: int, w: int, h: int, 
                  value: float, color: tuple):
        """Draw simple horizontal bar."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 50), -1)
        bar_w = int(max(0, min(1, value)) * (w - 4))
        if bar_w > 0:
            cv2.rectangle(canvas, (x + 2, y + 2), (x + 2 + bar_w, y + h - 2), color, -1)
        


def create_dashboard_state(
    rgb: Optional[np.ndarray],
    output,  # ControlOutput
    speed_ms: float,
    wp_lookahead_curv: float,
    frame_id: int,
    fps: float,
) -> DashboardState:
    """Create dashboard state from ADAS output."""
    import math
    
    return DashboardState(
        rgb=rgb,
        speed_kmh=speed_ms * 3.6,
        target_speed_kmh=output.target_speed_ms * 3.6,
        steer=output.steer,
        throttle=output.throttle,
        brake=output.brake,
        cte=output.fused_cte,
        heading_deg=math.degrees(output.fused_heading),
        curvature=output.fused_curvature,
        lane_confidence=0.0,  # TODO: get from output
        lane_weight=output.lane_weight,
        mode=output.mode,
        wp_lookahead_curv=wp_lookahead_curv,
        frame_id=frame_id,
        fps=fps,
    )
