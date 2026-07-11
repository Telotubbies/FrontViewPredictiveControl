#!/usr/bin/env python3
"""
Factory HMI visual asset renderers.

Every function in this module draws directly with pygame primitives so that no
external image files are required.  Each renderer accepts a target
``pygame.Surface`` and degrades gracefully when pygame is not initialized
(the call simply returns without drawing).

Angles are expressed in radians unless noted otherwise.  Coordinates use the
standard pygame convention (origin at the top-left, y increasing downward).
"""
import logging
import math
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Type aliases.
RGB = Tuple[int, int, int]
Pos = Tuple[int, int]
Rect = Tuple[int, int, int, int]

# Default colors reused when callers omit a color argument.
_DEFAULT_BG = (28, 32, 44)
_DEFAULT_GRID = (50, 56, 74)


# ── Helpers ──────────────────────────────────────────────────────────────
def _pygame_ok() -> bool:
    """Return True if pygame is imported and initialized."""
    try:
        import pygame
        return bool(pygame.get_init())
    except Exception:  # pragma: no cover - pygame missing
        return False


def _ensure_color(color: Optional[RGB], default: RGB) -> RGB:
    """Return ``color`` or ``default`` if color is None."""
    return default if color is None else color


def _aacircle(surface, center: Pos, radius: int, color: RGB) -> None:
    """Draw a filled anti-aliased circle, falling back to gfxdraw/pygame.draw."""
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


# ── 1. Status LED ────────────────────────────────────────────────────────
def draw_status_led(surface,
                    pos: Pos,
                    color: RGB,
                    size: int = 12,
                    blinking: bool = False,
                    time_s: float = 0.0) -> None:
    """Draw an industrial status LED indicator (circle with glow).

    Parameters
    ----------
    surface : pygame.Surface
        Target surface to draw on.
    pos : (x, y)
        Center of the LED.
    color : (r, g, b)
        LED fill color.
    size : int
        Radius of the LED in pixels.
    blinking : bool
        If True the LED toggles on/off at ~1 Hz using ``time_s``.
    time_s : float
        Elapsed time in seconds used to drive the blink phase.
    """
    if not _pygame_ok():
        return

    # Blink at 1 Hz (50% duty cycle): visible for the first half of each second.
    if blinking and int(time_s * 2) % 2 == 1:
        # Off state — draw a dim outline only.
        dim = tuple(int(c * 0.25) for c in color)
        _aaline_outline_circle(surface, pos, size, dim)
        return

    import pygame

    cx, cy = pos

    # Outer glow: a few translucent concentric circles.
    glow_color = tuple(min(255, int(c)) for c in color)
    for i in range(3, 0, -1):
        glow_r = size + i * 3
        glow_surf = pygame.Surface((glow_r * 2 + 2, glow_r * 2 + 2), pygame.SRCALPHA)
        alpha = 40 // i
        pygame.draw.circle(
            glow_surf,
            (glow_color[0], glow_color[1], glow_color[2], alpha),
            (glow_r + 1, glow_r + 1),
            glow_r,
        )
        surface.blit(glow_surf, (cx - glow_r - 1, cy - glow_r - 1))

    # LED body.
    _aacircle(surface, (cx, cy), size, color)

    # Specular highlight (top-left) for a 3D look.
    hl_r = max(1, size // 3)
    hl_color = tuple(min(255, c + 80) for c in color)
    _aacircle(surface, (cx - size // 3, cy - size // 3), hl_r, hl_color)


def _aaline_outline_circle(surface, center: Pos, radius: int, color: RGB) -> None:
    """Draw a thin outlined circle (used for the LED off-state)."""
    import pygame
    try:
        import pygame.gfxdraw
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
    except Exception:
        pygame.draw.circle(surface, color, center, radius, 1)


# ── 2. Warning icon ──────────────────────────────────────────────────────
def draw_warning_icon(surface,
                      pos: Pos,
                      size: int = 24,
                      color: Optional[RGB] = None) -> None:
    """Draw a triangular warning icon centered at ``pos``.

    ``size`` is the half-height of the triangle.  An exclamation mark is drawn
    in the center using the inverse of ``color`` for contrast.
    """
    if not _pygame_ok():
        return

    import pygame

    color = _ensure_color(color, (255, 190, 0))
    cx, cy = pos
    s = size

    # Triangle vertices (pointing up).
    top = (cx, cy - s)
    bl = (cx - int(s * 0.9), cy + int(s * 0.7))
    br = (cx + int(s * 0.9), cy + int(s * 0.7))

    try:
        import pygame.gfxdraw
        pygame.gfxdraw.filled_polygon(surface, [top, br, bl], color)
        pygame.gfxdraw.aapolygon(surface, [top, br, bl], color)
    except Exception:
        pygame.draw.polygon(surface, color, [top, br, bl])

    # Exclamation mark in inverse color.
    inv = _inverse_color(color)
    bar_w = max(2, s // 8)
    bar_h = int(s * 0.7)
    pygame.draw.rect(surface, inv,
                     (cx - bar_w // 2, cy - int(s * 0.25), bar_w, bar_h))
    dot_r = max(2, s // 7)
    _aacircle(surface, (cx, cy + int(s * 0.35)), dot_r, inv)


def _inverse_color(color: RGB) -> RGB:
    """Return a high-contrast inverse of ``color``."""
    lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return (20, 20, 20) if lum > 128 else (240, 240, 240)


# ── 3. Gauge arc segment ─────────────────────────────────────────────────
def draw_gauge_arc(surface,
                   center: Pos,
                   radius: int,
                   start_angle: float,
                   end_angle: float,
                   color: RGB,
                   thickness: int = 4) -> None:
    """Draw an arc segment (annulus section) for gauges.

    Angles are in radians measured from the positive x-axis, counter-clockwise.
    The segment is rendered as a series of short thick line segments approximating
    the arc, which gives a clean anti-aliased look without requiring gfxdraw.
    """
    if not _pygame_ok():
        return

    import pygame

    cx, cy = center
    # Step size keeps segments short enough to look smooth.
    span = end_angle - start_angle
    steps = max(8, int(abs(span) * radius / 3))
    if steps < 2:
        return

    pts_outer: List[Pos] = []
    pts_inner: List[Pos] = []
    for i in range(steps + 1):
        t = start_angle + span * i / steps
        ox = cx + math.cos(t) * radius
        oy = cy - math.sin(t) * radius  # negate: pygame y is down
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


# ── 4. Progress bar ──────────────────────────────────────────────────────
def draw_progress_bar(surface,
                      rect: Rect,
                      value: float,
                      max_val: float,
                      color: RGB,
                      bg_color: Optional[RGB] = None,
                      label: Optional[str] = None) -> None:
    """Draw a horizontal progress bar with optional label.

    The bar fills left-to-right proportional to ``value / max_val``.  A thin
    border is drawn around the track and the label (if provided) is rendered
    to the left of the bar.
    """
    if not _pygame_ok():
        return

    import pygame

    bg_color = _ensure_color(bg_color, (40, 46, 62))
    x, y, w, h = rect

    label_w = 0
    if label:
        font = pygame.font.Font(None, max(12, h - 4))
        lbl_surf = font.render(str(label), True, (170, 178, 195))
        label_w = lbl_surf.get_width() + 6
        surface.blit(lbl_surf, (x, y + (h - lbl_surf.get_height()) // 2))

    track_x = x + label_w
    track_w = w - label_w
    if track_w <= 0:
        return

    # Background track.
    pygame.draw.rect(surface, bg_color, (track_x, y, track_w, h), border_radius=4)

    # Fill.
    ratio = 0.0
    if max_val > 0:
        ratio = max(0.0, min(1.0, float(value) / float(max_val)))
    fill_w = int(track_w * ratio)
    if fill_w > 0:
        pygame.draw.rect(surface, color, (track_x, y, fill_w, h), border_radius=4)

    # Border.
    pygame.draw.rect(surface, (70, 78, 100), (track_x, y, track_w, h), 1,
                     border_radius=4)


# ── 5. Rounded panel ─────────────────────────────────────────────────────
def draw_rounded_panel(surface,
                       rect: Rect,
                       color: RGB,
                       border_color: Optional[RGB] = None,
                       border_width: int = 2,
                       radius: int = 8) -> None:
    """Draw a rounded-rectangle panel with optional border."""
    if not _pygame_ok():
        return

    import pygame

    x, y, w, h = rect
    r = min(radius, w // 2, h // 2)
    if r < 1:
        r = 1

    # Filled body.
    pygame.draw.rect(surface, color, (x, y, w, h), border_radius=r)

    # Border.
    if border_color is not None and border_width > 0:
        pygame.draw.rect(surface, border_color, (x, y, w, h), border_width,
                         border_radius=r)


# ── 6. Seven-segment display ─────────────────────────────────────────────
# Segment encoding for digits 0-9 and a blank.
# Segments are indexed: 0=top, 1=upper-right, 2=lower-right, 3=bottom,
#                       4=lower-left, 5=upper-left, 6=middle.
_SEG_TABLE = {
    "0": (1, 1, 1, 1, 1, 1, 0),
    "1": (0, 1, 1, 0, 0, 0, 0),
    "2": (1, 1, 0, 1, 1, 0, 1),
    "3": (1, 1, 1, 1, 0, 0, 1),
    "4": (0, 1, 1, 0, 0, 1, 1),
    "5": (1, 0, 1, 1, 0, 1, 1),
    "6": (1, 0, 1, 1, 1, 1, 1),
    "7": (1, 1, 1, 0, 0, 0, 0),
    "8": (1, 1, 1, 1, 1, 1, 1),
    "9": (1, 1, 1, 1, 0, 1, 1),
    "-": (0, 0, 0, 0, 0, 0, 1),
    " ": (0, 0, 0, 0, 0, 0, 0),
}


def _draw_segment(surface, seg_idx: int, ox: int, oy: int,
                  w: int, h: int, t: int, color: RGB) -> None:
    """Draw a single 7-segment element at origin (ox, oy) for a digit box w x h."""
    import pygame
    # Horizontal segments are drawn as thick rounded lines.
    half = t // 2
    if seg_idx == 0:  # top
        pygame.draw.rect(surface, color, (ox + t, oy, w - 2 * t, t), border_radius=half)
    elif seg_idx == 1:  # upper-right
        pygame.draw.rect(surface, color, (ox + w - t, oy + t, t, h // 2 - t),
                         border_radius=half)
    elif seg_idx == 2:  # lower-right
        pygame.draw.rect(surface, color, (ox + w - t, oy + h // 2 + 1, t, h // 2 - t),
                         border_radius=half)
    elif seg_idx == 3:  # bottom
        pygame.draw.rect(surface, color, (ox + t, oy + h - t, w - 2 * t, t),
                         border_radius=half)
    elif seg_idx == 4:  # lower-left
        pygame.draw.rect(surface, color, (ox, oy + h // 2 + 1, t, h // 2 - t),
                         border_radius=half)
    elif seg_idx == 5:  # upper-left
        pygame.draw.rect(surface, color, (ox, oy + t, t, h // 2 - t),
                         border_radius=half)
    elif seg_idx == 6:  # middle
        pygame.draw.rect(surface, color, (ox + t, oy + h // 2 - half, w - 2 * t, t),
                         border_radius=half)


def draw_seven_segment(surface,
                       pos: Pos,
                       value,
                       num_digits: int,
                       color: RGB,
                       digit_size: int = 28) -> None:
    """Draw a 7-segment display style number (factory HMI aesthetic).

    Parameters
    ----------
    pos : (x, y)
        Top-left position of the display.
    value : int or float or str
        Value to display.  Non-integer values are truncated to ``num_digits``.
    num_digits : int
        Number of digit cells to render (value is right-justified, zero-padded
        for non-negative integers).
    color : (r, g, b)
        Lit-segment color.  Unlit segments are drawn very dim.
    digit_size : int
        Height of each digit cell in pixels; width is ~0.6 * height.
    """
    if not _pygame_ok():
        return

    import pygame

    # Normalize value to a string of digits.
    if isinstance(value, float):
        if value.is_integer():
            text = str(int(value))
        else:
            text = str(value)
    else:
        text = str(value)

    # Right-justify / zero-pad non-negative integers.
    if text.lstrip("-").isdigit() and not text.startswith("-"):
        text = text.zfill(num_digits)
    if len(text) > num_digits:
        text = text[-num_digits:]

    # Digit cell geometry.
    dw = int(digit_size * 0.6)
    dh = digit_size
    t = max(2, digit_size // 8)  # segment thickness
    gap = max(2, dw // 5)

    dim = tuple(int(c * 0.12) for c in color)
    ox, oy = pos

    for i in range(num_digits):
        dx = ox + i * (dw + gap)
        ch = text[i] if i < len(text) else " "
        segs = _SEG_TABLE.get(ch, _SEG_TABLE[" "])
        for s_idx in range(7):
            seg_color = color if segs[s_idx] else dim
            _draw_segment(surface, s_idx, dx, oy, dw, dh, t, seg_color)


# ── 7. Alarm banner ──────────────────────────────────────────────────────
def draw_alarm_banner(surface,
                      rect: Rect,
                      text: str,
                      severity: str,
                      time_s: float) -> None:
    """Draw a flashing alarm banner with severity color.

    ``severity`` is one of: ``"warning"``, ``"error"``, ``"critical"``.
    For ``"critical"`` the banner flashes (background/text swap) at ~2 Hz.
    """
    if not _pygame_ok():
        return

    import pygame

    from .theme import HmiColors

    sev = str(severity).lower()
    color = HmiColors.severity(severity)

    x, y, w, h = rect
    flash_on = True
    if sev == "critical":
        # 2 Hz flash.
        flash_on = int(time_s * 4) % 2 == 0

    bg = color if flash_on else (30, 30, 30)
    fg = (20, 20, 20) if flash_on else color

    pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=4)
    pygame.draw.rect(surface, color, (x, y, w, h), 2, border_radius=4)

    font = pygame.font.Font(None, max(16, h - 8))
    txt_surf = font.render(text, True, fg)
    tx = x + (w - txt_surf.get_width()) // 2
    ty = y + (h - txt_surf.get_height()) // 2
    surface.blit(txt_surf, (tx, ty))


# ── 8. Grid overlay ──────────────────────────────────────────────────────
def draw_grid_overlay(surface,
                      rect: Rect,
                      spacing: int = 40,
                      color: Optional[RGB] = None) -> None:
    """Draw a technical grid background within ``rect``."""
    if not _pygame_ok():
        return

    import pygame

    color = _ensure_color(color, _DEFAULT_GRID)
    x, y, w, h = rect

    # Clip drawing to the rect to avoid bleeding outside.
    prev_clip = surface.get_clip()
    surface.set_clip(pygame.Rect(x, y, w, h))

    gx = x + spacing
    while gx < x + w:
        pygame.draw.line(surface, color, (gx, y), (gx, y + h), 1)
        gx += spacing
    gy = y + spacing
    while gy < y + h:
        pygame.draw.line(surface, color, (x, gy), (x + w, gy), 1)
        gy += spacing

    surface.set_clip(prev_clip)


# ── 9. Compass rose ──────────────────────────────────────────────────────
def draw_compass_rose(surface,
                      center: Pos,
                      radius: int,
                      heading_deg: float,
                      color: Optional[RGB] = None) -> None:
    """Draw a compass / heading indicator.

    The rose shows N/E/S/W cardinal labels and a needle pointing in the
    direction of ``heading_deg`` (0 = north, clockwise positive).
    """
    if not _pygame_ok():
        return

    import pygame

    color = _ensure_color(color, (0, 200, 230))
    cx, cy = center

    # Outer ring.
    try:
        import pygame.gfxdraw
        pygame.gfxdraw.aacircle(surface, cx, cy, radius, color)
        pygame.gfxdraw.aacircle(surface, cx, cy, radius - 1, color)
    except Exception:
        pygame.draw.circle(surface, color, (cx, cy), radius, 2)

    # Cardinal ticks and labels.
    font = pygame.font.Font(None, max(12, radius // 3))
    cardinals = [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
    for label, deg in cardinals:
        rad = math.radians(90 - deg)  # 0 deg => up
        tx = cx + math.cos(rad) * (radius - 12)
        ty = cy - math.sin(rad) * (radius - 12)
        lbl_color = (255, 80, 80) if label == "N" else color
        txt = font.render(label, True, lbl_color)
        surface.blit(txt, txt.get_rect(center=(int(tx), int(ty))))

    # Minor ticks every 45 degrees.
    for deg in range(0, 360, 45):
        if deg in (0, 90, 180, 270):
            continue
        rad = math.radians(90 - deg)
        x1 = cx + math.cos(rad) * (radius - 6)
        y1 = cy - math.sin(rad) * (radius - 6)
        x2 = cx + math.cos(rad) * radius
        y2 = cy - math.sin(rad) * radius
        _aaline(surface, (x1, y1), (x2, y2), color, 1)

    # Heading needle.
    rad = math.radians(90 - heading_deg)
    tip = (cx + math.cos(rad) * (radius - 14),
           cy - math.sin(rad) * (radius - 14))
    tail = (cx - math.cos(rad) * (radius - 14),
            cy + math.sin(rad) * (radius - 14))
    _aaline(surface, (cx, cy), tip, (255, 80, 80), 2)
    _aaline(surface, (cx, cy), tail, (120, 120, 130), 1)
    _aacircle(surface, (cx, cy), 3, color)


# ── 10. Bar graph ────────────────────────────────────────────────────────
def draw_bar_graph(surface,
                   rect: Rect,
                   values: Sequence[float],
                   color: RGB,
                   bg_color: Optional[RGB] = None) -> None:
    """Draw a vertical bar graph (sparkline alternative).

    ``values`` is a sequence of non-negative samples plotted left-to-right.
    The height of each bar is proportional to its value relative to the max.
    """
    if not _pygame_ok():
        return

    import pygame

    bg_color = _ensure_color(bg_color, (40, 46, 62))
    x, y, w, h = rect

    # Background.
    pygame.draw.rect(surface, bg_color, (x, y, w, h), border_radius=4)

    n = len(values)
    if n == 0:
        return

    max_val = max(values) if max(values) > 0 else 1.0
    bar_w = max(2, w // n)
    gap = max(1, bar_w // 6)
    bw = bar_w - gap

    for i, v in enumerate(values):
        ratio = max(0.0, min(1.0, float(v) / float(max_val)))
        bh = int((h - 4) * ratio)
        bx = x + i * bar_w + gap // 2
        by = y + h - bh - 2
        if bh > 0:
            pygame.draw.rect(surface, color, (bx, by, bw, bh), border_radius=2)


# ── 11. Arc gauge ────────────────────────────────────────────────────────
def draw_arc_gauge(surface,
                   center: Pos,
                   radius: int,
                   value: float,
                   min_val: float,
                   max_val: float,
                   label: str,
                   unit: str,
                   color_scheme: Optional[str] = None) -> None:
    """Draw a semicircular gauge with needle and digital readout.

    The gauge sweeps 180 degrees from the left (180 deg) to the right (0 deg),
    i.e. the top half of a circle.  ``color_scheme`` selects the fill palette:

    - ``"speed"``  : cyan fill, red redline near the top of the range.
    - ``"steer"``  : cyan fill, centered at the midpoint.
    - ``"thermal"``: green -> amber -> red gradient based on value ratio.
    - default / ``None``: solid cyan fill.
    """
    if not _pygame_ok():
        return

    import pygame

    from .theme import HmiColors

    color_scheme = (color_scheme or "default").lower()
    cx, cy = center

    # Sweep: left (pi) to right (0), drawn clockwise over the top.
    start_a = math.pi
    end_a = 0.0

    # Background arc (track).
    draw_gauge_arc(surface, (cx, cy), radius, start_a, end_a,
                   HmiColors.GAUGE_BG, thickness=8)

    # Normalized value ratio [0, 1].
    span = max_val - min_val
    ratio = 0.0
    if span > 0:
        ratio = max(0.0, min(1.0, (float(value) - min_val) / float(span)))

    # Filled arc.
    fill_end = start_a - ratio * math.pi  # clockwise from pi toward 0
    fill_color = HmiColors.GAUGE_FILL
    if color_scheme == "thermal":
        fill_color = _thermal_color(ratio)
    draw_gauge_arc(surface, (cx, cy), radius, start_a, fill_end,
                   fill_color, thickness=8)

    # Redline zone for speed scheme (top 20% of range).
    if color_scheme == "speed":
        redline_start = start_a - 0.8 * math.pi
        draw_gauge_arc(surface, (cx, cy), radius, redline_start, end_a,
                       HmiColors.GAUGE_REDLINE, thickness=8)

    # Tick marks every 10% of the range.
    for i in range(11):
        t = start_a - i / 10.0 * math.pi
        x1 = cx + math.cos(t) * (radius - 12)
        y1 = cy - math.sin(t) * (radius - 12)
        x2 = cx + math.cos(t) * (radius - 4)
        y2 = cy - math.sin(t) * (radius - 4)
        _aaline(surface, (x1, y1), (x2, y2), HmiColors.TEXT_DIM, 1)

    # Needle.
    needle_a = start_a - ratio * math.pi
    nx = cx + math.cos(needle_a) * (radius - 6)
    ny = cy - math.sin(needle_a) * (radius - 6)
    _aaline(surface, (cx, cy), (nx, ny), HmiColors.GAUGE_NEEDLE, 3)
    _aacircle(surface, (cx, cy), 4, HmiColors.GAUGE_NEEDLE)

    # Digital readout (value + unit) below the gauge center.
    val_font = pygame.font.Font(None, max(20, radius // 3))
    val_text = f"{value:.0f}"
    val_surf = val_font.render(val_text, True, HmiColors.TEXT_PRIMARY)
    surface.blit(val_surf, val_surf.get_rect(center=(cx, cy + radius // 3)))

    unit_font = pygame.font.Font(None, max(12, radius // 5))
    unit_surf = unit_font.render(unit, True, HmiColors.TEXT_DIM)
    surface.blit(unit_surf, unit_surf.get_rect(
        center=(cx, cy + radius // 3 + val_surf.get_height() // 2 + 6)))

    # Label above the gauge.
    lbl_font = pygame.font.Font(None, max(12, radius // 4))
    lbl_surf = lbl_font.render(label, True, HmiColors.TEXT_SECONDARY)
    surface.blit(lbl_surf, lbl_surf.get_rect(center=(cx, cy - radius - 6)))


def _thermal_color(ratio: float) -> RGB:
    """Return a green->amber->red color for a 0..1 ratio."""
    ratio = max(0.0, min(1.0, ratio))
    if ratio < 0.5:
        # green -> amber
        t = ratio / 0.5
        r = int(0 + t * 255)
        g = int(210 - t * 20)
        b = int(90 - t * 90)
    else:
        # amber -> red
        t = (ratio - 0.5) / 0.5
        r = int(255)
        g = int(190 - t * 120)
        b = int(0 + t * 70)
    return (r, g, b)
