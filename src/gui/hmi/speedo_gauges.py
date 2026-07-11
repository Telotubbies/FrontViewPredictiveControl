#!/usr/bin/env python3
"""
Factory HMI gauge renderer — industrial gauges rendered with pygame primitives.

แต่ละ gauge วาดด้วย pygame primitives เท่านั้น (ไม่ใช้ไฟล์รูปภายนอก)
เพื่อให้ทำงานได้ทันทีไม่ต้องโหลด asset.  สไตล์ตาม ISA 101 industrial HMI:
พื้นหลังเข้ม คอนทราสต์สูง ใช้สีตามความหมาย (semantic colors).

Gauges provided:
    1. Speedometer          — circular analog + digital readout + target marker
    2. Steering indicator   — horizontal bar (-1 .. +1) with center mark
    3. Pedal indicators     — throttle (green) + brake (red) vertical bars
    4. MPC status panel     — solver status + solve time + trajectory indicator
    5. CTE indicator        — horizontal bar (-2m .. +2m), color-coded
    6. Heading error        — circular gauge showing heading error in degrees
    7. Curvature indicator  — small gauge showing road curvature
    8. Lane confidence      — semicircular gauge 0-100%, color-coded
    9. Lane health strip    — P1-P5 phase LEDs + geometry_valid indicator

Main entry point: ``render(surface, frame_state, target_speed_kmh, fps)``
"""
from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Type aliases.
RGB = Tuple[int, int, int]
Pos = Tuple[int, int]
Rect = Tuple[int, int, int, int]

# ── Graceful pygame import ──────────────────────────────────────────────────
try:
    import pygame
    HAS_PYGAME = True
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore
    HAS_PYGAME = False

# ── Theme import with fallback ──────────────────────────────────────────────
try:
    from gui.hmi.theme import HmiColors, HmiFonts, HmiTheme
    _HAS_THEME = True
except Exception:  # pragma: no cover - theme may not exist yet
    _HAS_THEME = False

    class _FallbackColors:
        BG_DARK = (18, 20, 28)
        BG_PANEL = (28, 32, 44)
        BG_PANEL_ALT = (34, 38, 52)
        BG_HEADER = (22, 26, 38)
        BG_FOOTER = (16, 18, 26)
        TEXT_PRIMARY = (235, 238, 245)
        TEXT_SECONDARY = (170, 178, 195)
        TEXT_DIM = (110, 120, 140)
        TEXT_INVERSE = (20, 22, 30)
        STATUS_OK = (0, 210, 90)
        STATUS_WARNING = (255, 190, 0)
        STATUS_ERROR = (240, 70, 70)
        STATUS_CRITICAL = (255, 40, 40)
        STATUS_INFO = (0, 170, 255)
        STATUS_INACTIVE = (120, 125, 140)
        ACCENT_CYAN = (0, 200, 230)
        ACCENT_ORANGE = (255, 150, 40)
        ACCENT_PURPLE = (180, 100, 240)
        LANE_LEFT = (255, 220, 0)
        LANE_RIGHT = (60, 130, 255)
        LANE_CENTER = (0, 220, 100)
        LANE_LOST = (255, 80, 80)
        GAUGE_BG = (40, 46, 62)
        GAUGE_FILL = (0, 200, 230)
        GAUGE_NEEDLE = (255, 240, 240)
        GAUGE_REDLINE = (240, 70, 70)
        GRID_LINE = (50, 56, 74)
        GRID_LINE_DIM = (36, 42, 56)

        @classmethod
        def severity(cls, name: str) -> RGB:
            mapping = {
                "ok": cls.STATUS_OK,
                "warning": cls.STATUS_WARNING,
                "error": cls.STATUS_ERROR,
                "critical": cls.STATUS_CRITICAL,
                "info": cls.STATUS_INFO,
                "inactive": cls.STATUS_INACTIVE,
            }
            return mapping.get(str(name).lower(), cls.STATUS_INFO)

    class _FallbackFonts:
        TITLE = 32
        HEADER = 24
        LABEL = 18
        VALUE = 28
        SMALL = 14
        TINY = 10
        BIG_VALUE = 48

        @classmethod
        def get(cls, size: int, bold: bool = False):
            if HAS_PYGAME:
                try:
                    f = pygame.font.Font(None, size)
                    if bold:
                        f.set_bold(True)
                    return f
                except Exception:
                    return None
            return None

    class _FallbackTheme:
        colors = _FallbackColors
        fonts = _FallbackFonts

    HmiColors = _FallbackColors  # type: ignore
    HmiFonts = _FallbackFonts    # type: ignore
    HmiTheme = _FallbackTheme()  # type: ignore


# ── Drawing helpers (local copies matching assets.py style) ─────────────────
def _pygame_ok() -> bool:
    """Return True if pygame is imported and initialized."""
    try:
        import pygame
        return bool(pygame.get_init())
    except Exception:  # pragma: no cover
        return False


def _aacircle(surface, center: Pos, radius: int, color: RGB) -> None:
    """Draw a filled anti-aliased circle, falling back to pygame.draw."""
    import pygame
    try:
        import pygame.gfxdraw
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
    except Exception:
        pygame.draw.circle(surface, color, center, radius)


def _aaline(surface, start: Pos, end: Pos, color: RGB, width: int = 1) -> None:
    """Draw a line, using anti-aliasing for width 1 and thick line otherwise."""
    import pygame
    if width <= 1:
        try:
            pygame.draw.aaline(surface, color, start, end)
        except Exception:
            pygame.draw.line(surface, color, start, end, 1)
    else:
        pygame.draw.line(surface, color, start, end, width)


def _gauge_arc(surface, center: Pos, radius: int,
               start_angle: float, end_angle: float,
               color: RGB, thickness: int = 4) -> None:
    """Draw an arc segment (annulus section) for gauges.

    Angles in radians from positive x-axis, counter-clockwise.
    """
    if not _pygame_ok():
        return
    import pygame

    cx, cy = center
    span = end_angle - start_angle
    steps = max(8, int(abs(span) * radius / 3))
    if steps < 2:
        return

    pts_outer: List[Pos] = []
    pts_inner: List[Pos] = []
    for i in range(steps + 1):
        t = start_angle + span * i / steps
        ox = cx + math.cos(t) * radius
        oy = cy - math.sin(t) * radius
        ix = cx + math.cos(t) * (radius - thickness)
        iy = cy - math.sin(t) * (radius - thickness)
        pts_outer.append((ox, oy))
        pts_inner.append((ix, iy))

    polygon = pts_outer + list(reversed(pts_inner))
    try:
        import pygame.gfxdraw
        pygame.gfxdraw.filled_polygon(surface, polygon, color)
        pygame.gfxdraw.aapolygon(surface, polygon, color)
    except Exception:
        pygame.draw.polygon(surface, color, polygon)


def _rounded_panel(surface, rect: Rect, color: RGB,
                   border_color: Optional[RGB] = None,
                   border_width: int = 1, radius: int = 6) -> None:
    """Draw a rounded-rectangle panel with optional border."""
    if not _pygame_ok():
        return
    import pygame
    x, y, w, h = rect
    r = min(radius, w // 2, h // 2)
    if r < 1:
        r = 1
    pygame.draw.rect(surface, color, (x, y, w, h), border_radius=r)
    if border_color is not None and border_width > 0:
        pygame.draw.rect(surface, border_color, (x, y, w, h), border_width,
                         border_radius=r)


def _font(size: int, bold: bool = False):
    """Get a pygame font of the given size (lazy, cached via theme if available)."""
    if not _pygame_ok():
        return None
    try:
        get = getattr(HmiFonts, "get", None)
        if callable(get):
            return get(size, bold)
        import pygame
        f = pygame.font.Font(None, size)
        if bold:
            f.set_bold(True)
        return f
    except Exception:
        return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ═══════════════════════════════════════════════════════════════════════════════
# HmiGaugeRenderer
# ═══════════════════════════════════════════════════════════════════════════════
class HmiGaugeRenderer:
    """Factory-style industrial HMI gauge renderer (pygame primitives only).

    แต่ละ gauge เป็น method แยก เพื่อให้เรียกใช้แบบ standalone หรือผ่าน
    ``render()`` ที่วาดทุก gauge พร้อมกัน.  ทุก method จะ degrade gracefully
    ถ้า pygame ยังไม่ init (เพียงแค่ return ไม่วาด).
    """

    # Speedometer range (km/h).
    SPEED_MAX = 160.0
    SPEED_REDLINE = 130.0

    # CTE range (meters).
    CTE_MAX = 2.0

    # Heading error range (degrees).
    HEADING_MAX_DEG = 30.0

    # Curvature range (1/m).
    CURV_MAX = 0.05

    def __init__(self) -> None:
        """Initialize the gauge renderer (no pygame dependency at construction)."""
        self._C = HmiColors  # color palette shortcut
        self._F = HmiFonts   # font constants shortcut
        # Cached fonts.
        self._f_big: Any = None
        self._f_val: Any = None
        self._f_lbl: Any = None
        self._f_sm: Any = None
        self._f_tiny: Any = None
        self._fonts_ready = False
        # Smooth animated values for gauge needles (using pytweening easing)
        try:
            from gui.render_utils import SmoothValue, PulseAnimation
            self._smooth_speed = SmoothValue(initial=0.0, duration=0.25)
            self._smooth_steer = SmoothValue(initial=0.0, duration=0.15)
            self._smooth_cte = SmoothValue(initial=0.0, duration=0.2)
            self._smooth_conf = SmoothValue(initial=0.0, duration=0.3)
            self._warn_pulse = PulseAnimation(freq_hz=1.5)
            self._has_smooth = True
        except Exception:
            self._has_smooth = False

    # ── Font setup (lazy) ────────────────────────────────────────────────────
    def _ensure_fonts(self) -> None:
        if self._fonts_ready or not _pygame_ok():
            return
        try:
            self._f_big = _font(self._F.BIG_VALUE, bold=True)
            self._f_val = _font(self._F.VALUE, bold=True)
            self._f_lbl = _font(self._F.LABEL, bold=False)
            self._f_sm = _font(self._F.SMALL, bold=False)
            self._f_tiny = _font(self._F.TINY, bold=False)
            self._fonts_ready = True
        except Exception as e:
            logger.warning("HmiGaugeRenderer: font init failed: %s", e)
            self._fonts_ready = False

    # ── Text helper ──────────────────────────────────────────────────────────
    def _text(self, surface, text: str, pos: Pos,
              color: RGB, font_obj, centered: bool = False) -> None:
        """Render text on surface with the given font (safe wrapper)."""
        if font_obj is None:
            return
        try:
            ts = font_obj.render(str(text), True, color)
            if centered:
                surface.blit(ts, ts.get_rect(center=pos))
            else:
                surface.blit(ts, pos)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Speedometer
    # ─────────────────────────────────────────────────────────────────────────
    def draw_speedometer(self, surface, center: Pos, radius: int,
                         speed_kmh: float, target_kmh: float = 0.0) -> None:
        """Circular analog speedometer with needle, digital readout, target marker.

        วงกลม 270° sweep จาก 7 นาฬิกา (lower-left) ไป 5 นาฬิกา (lower-right)
        ผ่าน 12 นาฬิกา (top).  มี color zones: green (normal), amber (caution),
        red (redline).
        """
        if not _pygame_ok() or surface is None:
            return
        self._ensure_fonts()
        import pygame

        cx, cy = center
        # Sweep: 225° start (lower-left) to -45° end (lower-right) = 270° total.
        # In radians: start = 225° = 5π/4, end = -45° = -π/4 (going clockwise over top).
        start_a = math.radians(225)   # lower-left
        end_a = math.radians(-45)     # lower-right
        total_sweep = start_a - end_a  # 270° = 3π/2 (clockwise)

        # ── Background ring (track) ──
        _gauge_arc(surface, (cx, cy), radius, end_a, start_a,
                   self._C.GAUGE_BG, thickness=10)

        # ── Color zones ──
        # Green: 0 .. SPEED_REDLINE-30, Amber: REDLINE-30 .. REDLINE, Red: REDLINE .. MAX
        green_end = start_a - total_sweep * _clamp(
            (self.SPEED_REDLINE - 30) / self.SPEED_MAX, 0, 1)
        amber_end = start_a - total_sweep * _clamp(
            self.SPEED_REDLINE / self.SPEED_MAX, 0, 1)
        # Green zone
        _gauge_arc(surface, (cx, cy), radius, green_end, start_a,
                   self._C.STATUS_OK, thickness=10)
        # Amber zone
        _gauge_arc(surface, (cx, cy), radius, amber_end, green_end,
                   self._C.STATUS_WARNING, thickness=10)
        # Red zone
        _gauge_arc(surface, (cx, cy), radius, end_a, amber_end,
                   self._C.GAUGE_REDLINE, thickness=10)

        # ── Tick marks ──
        for i in range(9):  # 0, 20, 40, ... 160
            t = start_a - total_sweep * i / 8.0
            x1 = cx + math.cos(t) * (radius - 14)
            y1 = cy - math.sin(t) * (radius - 14)
            x2 = cx + math.cos(t) * (radius - 4)
            y2 = cy - math.sin(t) * (radius - 4)
            _aaline(surface, (x1, y1), (x2, y2), self._C.TEXT_DIM, 1)
            # Tick label
            if self._f_tiny is not None:
                label_val = int(self.SPEED_MAX * i / 8)
                lx = cx + math.cos(t) * (radius - 24)
                ly = cy - math.sin(t) * (radius - 24)
                self._text(surface, str(label_val), (int(lx), int(ly)),
                           self._C.TEXT_DIM, self._f_tiny, centered=True)

        # ── Needle ──
        ratio = _clamp(speed_kmh / self.SPEED_MAX, 0, 1)
        needle_a = start_a - total_sweep * ratio
        nx = cx + math.cos(needle_a) * (radius - 8)
        ny = cy - math.sin(needle_a) * (radius - 8)
        _aaline(surface, (cx, cy), (int(nx), int(ny)), self._C.GAUGE_NEEDLE, 3)
        _aacircle(surface, (cx, cy), 5, self._C.GAUGE_NEEDLE)
        _aacircle(surface, (cx, cy), 3, self._C.BG_PANEL)

        # ── Target speed marker (small triangle on the ring) ──
        if target_kmh > 0:
            tgt_ratio = _clamp(target_kmh / self.SPEED_MAX, 0, 1)
            tgt_a = start_a - total_sweep * tgt_ratio
            tx = cx + math.cos(tgt_a) * radius
            ty = cy - math.sin(tgt_a) * radius
            # Small marker dot
            _aacircle(surface, (int(tx), int(ty)), 4, self._C.ACCENT_CYAN)
            _aacircle(surface, (int(tx), int(ty)), 2, self._C.BG_DARK)

        # ── Digital readout (center-bottom) ──
        self._text(surface, f"{speed_kmh:.0f}", (cx, cy + radius // 3),
                   self._C.TEXT_PRIMARY, self._f_big, centered=True)
        self._text(surface, "km/h", (cx, cy + radius // 3 + 28),
                   self._C.TEXT_DIM, self._f_tiny, centered=True)

        # ── Label (top) ──
        self._text(surface, "SPEED", (cx, cy - radius - 8),
                   self._C.TEXT_SECONDARY, self._f_sm, centered=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Steering angle indicator
    # ─────────────────────────────────────────────────────────────────────────
    def draw_steering(self, surface, rect: Rect, steer: float) -> None:
        """Horizontal bar showing steering angle (-1 .. +1) with center mark.

        ค่า steer จาก FrameState.steer หรือ final_steer, range -1.0 .. +1.0.
        แท่งแนวนอน จุดศูนย์กลาง = 0, ซ้าย = -1 (สีฟ้า), ขวา = +1 (สีส้ม).
        """
        if not _pygame_ok() or surface is None:
            return
        self._ensure_fonts()
        import pygame

        x, y, w, h = rect
        _rounded_panel(surface, rect, self._C.BG_PANEL_ALT,
                       border_color=self._C.GRID_LINE, border_width=1, radius=4)

        # Label
        self._text(surface, "STEER", (x + 6, y + 2),
                   self._C.TEXT_SECONDARY, self._f_tiny)

        # Bar geometry
        bar_y = y + h // 2 + 2
        bar_h = max(8, h // 4)
        bar_x = x + 8
        bar_w = w - 16
        center_x = bar_x + bar_w // 2

        # Background track
        pygame.draw.rect(surface, self._C.GAUGE_BG,
                         (bar_x, bar_y, bar_w, bar_h), border_radius=3)

        # Center mark
        pygame.draw.line(surface, self._C.TEXT_DIM,
                         (center_x, bar_y - 3), (center_x, bar_y + bar_h + 3), 1)

        # Fill from center
        steer_clamped = _clamp(steer, -1.0, 1.0)
        fill_w = int(abs(steer_clamped) * bar_w / 2)
        fill_color = self._C.ACCENT_ORANGE if steer_clamped >= 0 else self._C.ACCENT_CYAN
        if steer_clamped >= 0:
            pygame.draw.rect(surface, fill_color,
                             (center_x, bar_y, fill_w, bar_h), border_radius=3)
        else:
            pygame.draw.rect(surface, fill_color,
                             (center_x - fill_w, bar_y, fill_w, bar_h),
                             border_radius=3)

        # Value text
        steer_deg = math.degrees(steer_clamped * 0.4)  # approx steering wheel deg
        val_text = f"{steer_clamped:+.2f} ({steer_deg:+.1f}°)"
        self._text(surface, val_text, (x + w - 120, y + 2),
                   self._C.TEXT_PRIMARY, self._f_tiny)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Pedal indicators (throttle + brake)
    # ─────────────────────────────────────────────────────────────────────────
    def draw_pedals(self, surface, rect: Rect,
                    throttle: float, brake: float) -> None:
        """Vertical bars for throttle (green, 0-100%) and brake (red, 0-100%).

        วาด 2 แท่งแนวตั้ง: throttle (ซ้าย, เติมจากล่างขึ้น, สีเขียว)
        และ brake (ขวา, เติมจากล่างขึ้น, สีแดง).
        """
        if not _pygame_ok() or surface is None:
            return
        self._ensure_fonts()
        import pygame

        x, y, w, h = rect
        _rounded_panel(surface, rect, self._C.BG_PANEL_ALT,
                       border_color=self._C.GRID_LINE, border_width=1, radius=4)

        # Layout: two vertical bars side by side
        bar_w = 28
        bar_gap = 12
        label_h = 14
        val_h = 14
        bar_area_h = h - label_h - val_h - 8
        bar_y = y + label_h + 4
        bar_h = max(10, bar_area_h)

        # Throttle bar (left)
        th_x = x + 20
        self._text(surface, "THR", (th_x + bar_w // 2, y + 2),
                   self._C.TEXT_SECONDARY, self._f_tiny, centered=True)
        pygame.draw.rect(surface, self._C.GAUGE_BG,
                         (th_x, bar_y, bar_w, bar_h), border_radius=3)
        th_ratio = _clamp(throttle, 0.0, 1.0)
        th_fill_h = int(bar_h * th_ratio)
        if th_fill_h > 0:
            pygame.draw.rect(surface, self._C.STATUS_OK,
                             (th_x, bar_y + bar_h - th_fill_h, bar_w, th_fill_h),
                             border_radius=3)
        self._text(surface, f"{throttle * 100:.0f}%",
                   (th_x + bar_w // 2, bar_y + bar_h + 2),
                   self._C.TEXT_PRIMARY, self._f_tiny, centered=True)

        # Brake bar (right)
        br_x = th_x + bar_w + bar_gap
        self._text(surface, "BRK", (br_x + bar_w // 2, y + 2),
                   self._C.TEXT_SECONDARY, self._f_tiny, centered=True)
        pygame.draw.rect(surface, self._C.GAUGE_BG,
                         (br_x, bar_y, bar_w, bar_h), border_radius=3)
        br_ratio = _clamp(brake, 0.0, 1.0)
        br_fill_h = int(bar_h * br_ratio)
        if br_fill_h > 0:
            pygame.draw.rect(surface, self._C.STATUS_ERROR,
                             (br_x, bar_y + bar_h - br_fill_h, bar_w, br_fill_h),
                             border_radius=3)
        self._text(surface, f"{brake * 100:.0f}%",
                   (br_x + bar_w // 2, bar_y + bar_h + 2),
                   self._C.TEXT_PRIMARY, self._f_tiny, centered=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. MPC status panel
    # ─────────────────────────────────────────────────────────────────────────
    def draw_mpc_status(self, surface, rect: Rect,
                        solver_status: str, solve_time_ms: float,
                        has_trajectory: bool = False) -> None:
        """MPC solver status panel: status LED + solve time + trajectory indicator.

        Solve_Succeeded = เขียว, Fallback/อื่นๆ = amber.
        แสดง solve time ในหน่วย ms พร้อม color coding (< 20ms เขียว, < 50ms amber, > 50ms แดง).
        """
        if not _pygame_ok() or surface is None:
            return
        self._ensure_fonts()
        import pygame

        x, y, w, h = rect
        _rounded_panel(surface, rect, self._C.BG_PANEL_ALT,
                       border_color=self._C.GRID_LINE, border_width=1, radius=4)

        # Title
        self._text(surface, "MPC", (x + 6, y + 2),
                   self._C.TEXT_SECONDARY, self._f_tiny)

        # Solver status LED + text
        succeeded = "Succeed" in str(solver_status) or "succeed" in str(solver_status).lower()
        led_color = self._C.STATUS_OK if succeeded else self._C.STATUS_WARNING
        led_x = x + 10
        led_y = y + h // 2
        _aacircle(surface, (led_x, led_y), 5, led_color)
        _aacircle(surface, (led_x, led_y), 3, self._C.BG_PANEL_ALT)

        status_short = "OK" if succeeded else "FALLBACK"
        self._text(surface, status_short, (led_x + 14, y + h // 2 - 7),
                   led_color, self._f_sm)

        # Solve time
        time_color = (self._C.STATUS_OK if solve_time_ms < 20
                      else self._C.STATUS_WARNING if solve_time_ms < 50
                      else self._C.STATUS_ERROR)
        self._text(surface, f"{solve_time_ms:.1f}ms",
                   (x + w // 2, y + h // 2 - 7),
                   time_color, self._f_sm, centered=True)

        # Trajectory indicator (right side)
        traj_color = self._C.STATUS_OK if has_trajectory else self._C.STATUS_INACTIVE
        traj_x = x + w - 30
        _aacircle(surface, (traj_x, led_y), 5, traj_color)
        self._text(surface, "TRAJ", (traj_x + 12, y + h // 2 - 7),
                   traj_color, self._f_tiny)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. CTE indicator
    # ─────────────────────────────────────────────────────────────────────────
    def draw_cte(self, surface, rect: Rect, cte_m: float) -> None:
        """Horizontal bar showing cross-track error (-2m .. +2m), center=0.

        Color-coded: |CTE| < 0.3m เขียว, < 0.8m amber, > 0.8m แดง.
        แท่งแนวนอน จุดศูนย์กลาง = 0m, บวก = ขวา, ลบ = ซ้าย.
        """
        if not _pygame_ok() or surface is None:
            return
        self._ensure_fonts()
        import pygame

        x, y, w, h = rect
        _rounded_panel(surface, rect, self._C.BG_PANEL_ALT,
                       border_color=self._C.GRID_LINE, border_width=1, radius=4)

        # Label
        self._text(surface, "CTE", (x + 6, y + 2),
                   self._C.TEXT_SECONDARY, self._f_tiny)

        # Bar geometry
        bar_y = y + h // 2 + 2
        bar_h = max(8, h // 4)
        bar_x = x + 8
        bar_w = w - 16
        center_x = bar_x + bar_w // 2

        # Background track
        pygame.draw.rect(surface, self._C.GAUGE_BG,
                         (bar_x, bar_y, bar_w, bar_h), border_radius=3)

        # Center mark (0m)
        pygame.draw.line(surface, self._C.TEXT_DIM,
                         (center_x, bar_y - 3), (center_x, bar_y + bar_h + 3), 1)

        # Color based on magnitude
        abs_cte = abs(cte_m)
        if abs_cte < 0.3:
            fill_color = self._C.STATUS_OK
        elif abs_cte < 0.8:
            fill_color = self._C.STATUS_WARNING
        else:
            fill_color = self._C.STATUS_ERROR

        # Fill from center
        cte_clamped = _clamp(cte_m, -self.CTE_MAX, self.CTE_MAX)
        fill_w = int(abs(cte_clamped) / self.CTE_MAX * bar_w / 2)
        if cte_clamped >= 0:
            pygame.draw.rect(surface, fill_color,
                             (center_x, bar_y, fill_w, bar_h), border_radius=3)
        else:
            pygame.draw.rect(surface, fill_color,
                             (center_x - fill_w, bar_y, fill_w, bar_h),
                             border_radius=3)

        # Value text
        self._text(surface, f"{cte_m:+.2f}m", (x + w - 70, y + 2),
                   fill_color, self._f_tiny)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Heading error indicator
    # ─────────────────────────────────────────────────────────────────────────
    def draw_heading_error(self, surface, center: Pos, radius: int,
                           heading_rad: float) -> None:
        """Circular gauge showing heading error in degrees.

        วงกลมเล็ก แสดง heading error จาก heading_rad (radians).
        เข็มชี้บน/ล่าง = 0°, ซ้าย/ขวา = ±max.  Color: < 3° เขียว, < 7° amber, > 7° แดง.
        """
        if not _pygame_ok() or surface is None:
            return
        self._ensure_fonts()
        import pygame

        cx, cy = center
        head_deg = math.degrees(heading_rad)

        # Background ring
        _aacircle(surface, (cx, cy), radius, self._C.GAUGE_BG)
        _aacircle(surface, (cx, cy), radius - 1, self._C.BG_PANEL)

        # Color
        abs_deg = abs(head_deg)
        if abs_deg < 3.0:
            color = self._C.STATUS_OK
        elif abs_deg < 7.0:
            color = self._C.STATUS_WARNING
        else:
            color = self._C.STATUS_ERROR

        # Arc fill (semicircle top half, from center toward error direction)
        ratio = _clamp(head_deg / self.HEADING_MAX_DEG, -1.0, 1.0)
        # 0° = straight up (π/2), positive = right (toward 0), negative = left (toward π)
        start_a = math.pi / 2  # straight up
        fill_end = start_a - ratio * math.pi / 2  # ±90° max
        if ratio >= 0:
            _gauge_arc(surface, (cx, cy), radius, fill_end, start_a,
                       color, thickness=6)
        else:
            _gauge_arc(surface, (cx, cy), radius, start_a, fill_end,
                       color, thickness=6)

        # Tick marks at 0°, ±max
        for deg_mark, label_txt in [(0, "0"), (-self.HEADING_MAX_DEG, "-"),
                                     (self.HEADING_MAX_DEG, "+")]:
            t = math.pi / 2 - (deg_mark / self.HEADING_MAX_DEG) * math.pi / 2
            x1 = cx + math.cos(t) * (radius - 6)
            y1 = cy - math.sin(t) * (radius - 6)
            x2 = cx + math.cos(t) * radius
            y2 = cy - math.sin(t) * radius
            _aaline(surface, (x1, y1), (x2, y2), self._C.TEXT_DIM, 1)

        # Needle
        needle_a = math.pi / 2 - ratio * math.pi / 2
        nx = cx + math.cos(needle_a) * (radius - 4)
        ny = cy - math.sin(needle_a) * (radius - 4)
        _aaline(surface, (cx, cy), (int(nx), int(ny)), self._C.GAUGE_NEEDLE, 2)
        _aacircle(surface, (cx, cy), 3, self._C.GAUGE_NEEDLE)

        # Digital readout
        self._text(surface, f"{head_deg:+.1f}°", (cx, cy + radius + 2),
                   color, self._f_tiny, centered=True)
        self._text(surface, "HDG", (cx, cy - radius - 6),
                   self._C.TEXT_SECONDARY, self._f_tiny, centered=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Curvature indicator
    # ─────────────────────────────────────────────────────────────────────────
    def draw_curvature(self, surface, rect: Rect, curvature: float) -> None:
        """Small gauge showing road curvature (1/m).

        แท่งแนวนอนเล็ก  center=0, บวก=โค้งขวา (สีฟ้า), ลบ=โค้งซ้าย (สีส้ม).
        Range: -CURV_MAX .. +CURV_MAX (1/m).
        """
        if not _pygame_ok() or surface is None:
            return
        self._ensure_fonts()
        import pygame

        x, y, w, h = rect
        _rounded_panel(surface, rect, self._C.BG_PANEL_ALT,
                       border_color=self._C.GRID_LINE, border_width=1, radius=4)

        # Label
        self._text(surface, "CURV", (x + 6, y + 2),
                   self._C.TEXT_SECONDARY, self._f_tiny)

        # Bar
        bar_y = y + h // 2 + 2
        bar_h = max(6, h // 5)
        bar_x = x + 8
        bar_w = w - 16
        center_x = bar_x + bar_w // 2

        pygame.draw.rect(surface, self._C.GAUGE_BG,
                         (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        pygame.draw.line(surface, self._C.TEXT_DIM,
                         (center_x, bar_y - 2), (center_x, bar_y + bar_h + 2), 1)

        curv_clamped = _clamp(curvature, -self.CURV_MAX, self.CURV_MAX)
        fill_w = int(abs(curv_clamped) / self.CURV_MAX * bar_w / 2)
        fill_color = self._C.ACCENT_CYAN if curv_clamped >= 0 else self._C.ACCENT_ORANGE
        if curv_clamped >= 0:
            pygame.draw.rect(surface, fill_color,
                             (center_x, bar_y, fill_w, bar_h), border_radius=3)
        else:
            pygame.draw.rect(surface, fill_color,
                             (center_x - fill_w, bar_y, fill_w, bar_h),
                             border_radius=3)

        # Value
        self._text(surface, f"{curvature:+.5f}", (x + w - 90, y + 2),
                   self._C.TEXT_PRIMARY, self._f_tiny)

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Lane confidence gauge
    # ─────────────────────────────────────────────────────────────────────────
    def draw_lane_confidence(self, surface, center: Pos, radius: int,
                             confidence: float) -> None:
        """Semicircular gauge 0-100%, color-coded.

        ครึ่งวงกลม (top half) วัดจาก 0% (ซ้าย) ถึง 100% (ขวา).
        Color: > 70% เขียว, > 40% amber, < 40% แดง.
        """
        if not _pygame_ok() or surface is None:
            return
        self._ensure_fonts()
        import pygame

        cx, cy = center
        start_a = math.pi   # left (180°)
        end_a = 0.0          # right (0°)
        total_sweep = math.pi  # 180°

        # Background track
        _gauge_arc(surface, (cx, cy), radius, end_a, start_a,
                   self._C.GAUGE_BG, thickness=8)

        # Color
        if confidence > 0.7:
            color = self._C.STATUS_OK
        elif confidence > 0.4:
            color = self._C.STATUS_WARNING
        else:
            color = self._C.STATUS_ERROR

        # Fill arc
        ratio = _clamp(confidence, 0.0, 1.0)
        fill_end = start_a - ratio * total_sweep
        _gauge_arc(surface, (cx, cy), radius, fill_end, start_a,
                   color, thickness=8)

        # Tick marks
        for i in range(5):  # 0, 25, 50, 75, 100%
            t = start_a - total_sweep * i / 4.0
            x1 = cx + math.cos(t) * (radius - 10)
            y1 = cy - math.sin(t) * (radius - 10)
            x2 = cx + math.cos(t) * (radius - 2)
            y2 = cy - math.sin(t) * (radius - 2)
            _aaline(surface, (x1, y1), (x2, y2), self._C.TEXT_DIM, 1)

        # Needle
        needle_a = start_a - ratio * total_sweep
        nx = cx + math.cos(needle_a) * (radius - 6)
        ny = cy - math.sin(needle_a) * (radius - 6)
        _aaline(surface, (cx, cy), (int(nx), int(ny)), self._C.GAUGE_NEEDLE, 2)
        _aacircle(surface, (cx, cy), 3, self._C.GAUGE_NEEDLE)

        # Digital readout
        self._text(surface, f"{confidence * 100:.0f}%",
                   (cx, cy + radius // 3), color, self._f_val, centered=True)
        self._text(surface, "LANE CONF", (cx, cy - radius - 6),
                   self._C.TEXT_SECONDARY, self._f_tiny, centered=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Lane health strip (P1-P5 phases + geometry_valid)
    # ─────────────────────────────────────────────────────────────────────────
    def draw_lane_health(self, surface, rect: Rect,
                         phases: Sequence[bool],
                         geometry_valid: bool,
                         confidence: float = 0.0) -> None:
        """Horizontal bar showing P1-P5 phase status + geometry_valid + confidence.

        5 LEDs สำหรับ phase_p1_ok .. phase_p5_ok (เขียว=ok, แดง=failed)
        + 1 LED สำหรับ geometry_valid + confidence bar ด้านขวา.
        """
        if not _pygame_ok() or surface is None:
            return
        self._ensure_fonts()
        import pygame

        x, y, w, h = rect
        _rounded_panel(surface, rect, self._C.BG_PANEL,
                       border_color=self._C.GRID_LINE, border_width=1, radius=4)

        # Label
        self._text(surface, "LANE HEALTH", (x + 6, y + 2),
                   self._C.TEXT_SECONDARY, self._f_tiny)

        # Phase LEDs
        led_r = 6
        led_gap = 26
        led_start_x = x + 80
        led_y = y + h // 2
        phase_labels = ["P1", "P2", "P3", "P4", "P5"]

        for i in range(5):
            led_x = led_start_x + i * led_gap
            ok = bool(phases[i]) if i < len(phases) else False
            color = self._C.STATUS_OK if ok else self._C.STATUS_ERROR
            _aacircle(surface, (led_x, led_y), led_r, color)
            # Glow effect for OK
            if ok:
                _aacircle(surface, (led_x, led_y), led_r - 2,
                          tuple(min(255, c + 40) for c in color))
            # Label
            self._text(surface, phase_labels[i], (led_x, led_y + led_r + 2),
                       self._C.TEXT_DIM, self._f_tiny, centered=True)

        # Geometry valid LED
        geom_x = led_start_x + 5 * led_gap + 10
        geom_color = self._C.STATUS_OK if geometry_valid else self._C.STATUS_ERROR
        _aacircle(surface, (geom_x, led_y), led_r, geom_color)
        if geometry_valid:
            _aacircle(surface, (geom_x, led_y), led_r - 2,
                      tuple(min(255, c + 40) for c in geom_color))
        self._text(surface, "GEOM", (geom_x, led_y + led_r + 2),
                   self._C.TEXT_DIM, self._f_tiny, centered=True)

        # Confidence bar (right side)
        conf_x = geom_x + 40
        conf_w = x + w - conf_x - 10
        conf_h = max(8, h // 4)
        conf_y = led_y - conf_h // 2
        if conf_w > 20:
            pygame.draw.rect(surface, self._C.GAUGE_BG,
                             (conf_x, conf_y, conf_w, conf_h), border_radius=3)
            conf_ratio = _clamp(confidence, 0.0, 1.0)
            conf_fill_w = int(conf_w * conf_ratio)
            if conf_fill_w > 0:
                conf_color = (self._C.STATUS_OK if confidence > 0.7
                              else self._C.STATUS_WARNING if confidence > 0.4
                              else self._C.STATUS_ERROR)
                pygame.draw.rect(surface, conf_color,
                                 (conf_x, conf_y, conf_fill_w, conf_h),
                                 border_radius=3)
            self._text(surface, f"{confidence * 100:.0f}%",
                       (conf_x + conf_w + 2, conf_y - 2),
                       self._C.TEXT_PRIMARY, self._f_tiny)

    # ─────────────────────────────────────────────────────────────────────────
    # Main render entry point
    # ─────────────────────────────────────────────────────────────────────────
    def render(self, surface, frame_state, target_speed_kmh: float,
               fps: float) -> None:
        """Render all factory HMI gauges onto ``surface``.

        วาดทุก gauge ลงบน surface ที่กำหนด.  ใช้ FrameState เป็นแหล่งข้อมูล.
        ถ้า frame_state เป็น None จะใช้ค่า default (0/False).

        Args:
            surface: pygame.Surface ปลายทาง
            frame_state: FrameState dataclass (จาก src/state.py)
            target_speed_kmh: ความเร็วเป้าหมาย (km/h)
            fps: FPS ปัจจุบัน (สำหรับแสดงผล)
        """
        if not _pygame_ok() or surface is None:
            return

        try:
            self._ensure_fonts()
            import pygame

            # Extract values from frame_state with safe defaults
            fs = frame_state
            speed = getattr(fs, "speed_kmh", 0.0) if fs else 0.0
            steer = getattr(fs, "final_steer", 0.0) if fs else 0.0
            throttle = getattr(fs, "final_throttle", 0.0) if fs else 0.0
            brake = getattr(fs, "final_brake", 0.0) if fs else 0.0
            cte = getattr(fs, "cte_m", 0.0) if fs else 0.0
            heading = getattr(fs, "heading_rad", 0.0) if fs else 0.0
            curvature = getattr(fs, "curvature", 0.0) if fs else 0.0
            lane_conf = getattr(fs, "lane_conf", 0.0) if fs else 0.0
            solver = getattr(fs, "solver_status", "Solve_Succeeded") if fs else "Solve_Succeeded"
            solve_time = getattr(fs, "mpc_solve_time_ms", 0.0) if fs else 0.0
            mpc_traj = getattr(fs, "mpc_trajectory", None) if fs else None
            has_traj = mpc_traj is not None

            phases = [
                getattr(fs, "phase_p1_ok", False) if fs else False,
                getattr(fs, "phase_p2_ok", False) if fs else False,
                getattr(fs, "phase_p3_ok", False) if fs else False,
                getattr(fs, "phase_p4_ok", False) if fs else False,
                getattr(fs, "phase_p5_ok", False) if fs else False,
            ]
            geom_valid = getattr(fs, "geometry_valid", False) if fs else False

            sw = surface.get_width()
            sh = surface.get_height()

            # ── Layout: gauges arranged in the bottom panel area ──
            # We draw into a surface that is expected to be ~1280 wide.
            # Left section: speedometer (circular, ~180px)
            # Center-left: pedals + steering
            # Center-right: CTE + heading + curvature
            # Right: MPC status + lane confidence

            # Speedometer (left)
            speed_cx = 100
            speed_cy = 110
            speed_r = 80
            if sw >= 200 and sh >= 200:
                self.draw_speedometer(surface, (speed_cx, speed_cy), speed_r,
                                      speed, target_speed_kmh)

            # Pedals (left, below speedometer)
            if sh >= 240:
                self.draw_pedals(surface, (10, 200, 180, 70),
                                 throttle, brake)

            # Steering (left, below pedals)
            if sh >= 300:
                self.draw_steering(surface, (10, 275, 180, 30), steer)

            # CTE (center-left)
            if sw >= 500:
                self.draw_cte(surface, (210, 10, 250, 40), cte)

            # Curvature (center-left, below CTE)
            if sw >= 500:
                self.draw_curvature(surface, (210, 55, 250, 30), curvature)

            # Heading error (center, circular)
            if sw >= 600:
                self.draw_heading_error(surface, (340, 130), 45, heading)

            # MPC status (center-right)
            if sw >= 700:
                self.draw_mpc_status(surface, (480, 10, 200, 40),
                                     solver, solve_time, has_traj)

            # Lane confidence (right, semicircular)
            if sw >= 800:
                self.draw_lane_confidence(surface, (620, 110), 60, lane_conf)

            # Lane health strip (bottom, full width)
            if sh >= 350:
                self.draw_lane_health(surface, (10, 310, sw - 20, 30),
                                      phases, geom_valid, lane_conf)

        except Exception as e:
            logger.warning("HmiGaugeRenderer.render failed: %s", e)
