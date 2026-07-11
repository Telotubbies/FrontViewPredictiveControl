"""
Render utilities — anti-aliased drawing, glow effects, smooth animations.

Free libraries used:
  - pygame.gfxdraw (built-in) — anti-aliased primitives
  - numpy (already installed) — fast array ops for blur/glow
  - Pillow (already installed) — Gaussian blur for glow effects
  - pytweening (MIT, free) — easing functions for smooth animations

All functions are self-contained and can be dropped into any pygame renderer.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import pygame
import pygame.gfxdraw

try:
    from PIL import Image, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytweening
    HAS_TWEEN = True
except ImportError:
    HAS_TWEEN = False

logger = logging.getLogger(__name__)


# ── Anti-aliased drawing (using pygame.gfxdraw) ─────────────────────────────

def aa_line(surface: pygame.Surface, color: Tuple[int, int, int],
            start: Tuple[float, float], end: Tuple[float, float],
            width: int = 1) -> None:
    """Draw anti-aliased line with optional thickness."""
    if width <= 1:
        pygame.gfxdraw.aaline(surface, int(start[0]), int(start[1]),
                              int(end[0]), int(end[1]), color)
    else:
        # Thick AA line: draw filled polygon
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.5:
            return
        # Perpendicular offset
        px = -dy / length * width / 2
        py = dx / length * width / 2
        points = [
            (start[0] + px, start[1] + py),
            (end[0] + px, end[1] + py),
            (end[0] - px, end[1] - py),
            (start[0] - px, start[1] - py),
        ]
        aa_filled_polygon(surface, points, color)


def aa_circle(surface: pygame.Surface, color: Tuple[int, int, int],
              center: Tuple[float, float], radius: float,
              filled: bool = True) -> None:
    """Draw anti-aliased circle."""
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    if r <= 0:
        return
    if filled:
        pygame.gfxdraw.filled_circle(surface, cx, cy, r, color)
    pygame.gfxdraw.aacircle(surface, cx, cy, r, color)


def aa_polygon(surface: pygame.Surface, color: Tuple[int, int, int],
               points: List[Tuple[float, float]], filled: bool = True) -> None:
    """Draw anti-aliased polygon."""
    int_points = [(int(p[0]), int(p[1])) for p in points]
    if filled:
        pygame.gfxdraw.filled_polygon(surface, int_points, color)
    pygame.gfxdraw.aapolygon(surface, int_points, color)


def aa_filled_polygon(surface: pygame.Surface,
                      points: List[Tuple[float, float]],
                      color: Tuple[int, int, int]) -> None:
    """Draw filled anti-aliased polygon."""
    int_points = [(int(p[0]), int(p[1])) for p in points]
    pygame.gfxdraw.filled_polygon(surface, int_points, color)
    pygame.gfxdraw.aapolygon(surface, int_points, color)


def aa_arc(surface: pygame.Surface, color: Tuple[int, int, int],
           center: Tuple[float, float], radius: float,
           start_angle: float, stop_angle: float, width: int = 2) -> None:
    """Draw anti-aliased arc using line segments."""
    cx, cy = center
    r = radius
    # Number of segments based on radius
    n = max(8, int(r * (stop_angle - start_angle) / 0.1))
    n = min(n, 64)
    prev = None
    for i in range(n + 1):
        t = i / n
        angle = start_angle + t * (stop_angle - start_angle)
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        if prev is not None:
            aa_line(surface, color, prev, (x, y), width)
        prev = (x, y)


# ── Glow effects (using Pillow Gaussian blur) ───────────────────────────────

def make_glow_surface(width: int, height: int,
                      color: Tuple[int, int, int],
                      intensity: float = 1.0,
                      blur_radius: int = 10) -> pygame.Surface:
    """
    Create a glowing surface of the given color.
    Uses Pillow Gaussian blur for smooth glow.
    """
    if not HAS_PIL:
        # Fallback: simple radial gradient with numpy
        return _numpy_glow(width, height, color, intensity)

    # Create solid color image
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :, 0] = color[0]
    arr[:, :, 1] = color[1]
    arr[:, :, 2] = color[2]
    arr[:, :, 3] = int(255 * intensity)

    # Convert to PIL, blur, convert back
    pil_img = Image.fromarray(arr, mode='RGBA')
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Convert back to pygame surface
    result = pygame.image.frombuffer(
        pil_img.tobytes(), (width, height), 'RGBA'
    )
    return result.convert_alpha()


def _numpy_glow(width: int, height: int,
                color: Tuple[int, int, int],
                intensity: float = 1.0) -> pygame.Surface:
    """Fallback glow using numpy radial gradient."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    cx, cy = width // 2, height // 2
    max_r = math.sqrt(cx * cx + cy * cy)

    # Create gradient using numpy
    y, x = np.ogrid[:height, :width]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha = np.clip(1.0 - dist / max_r, 0, 1) * intensity * 255
    alpha = alpha.astype(np.uint8)

    # Fill surface
    arr = pygame.surfarray.pixels_alpha(surf)
    arr[:] = alpha.T
    del arr

    # Set color
    color_arr = pygame.surfarray.pixels3d(surf)
    color_arr[:, :, 0] = color[0]
    color_arr[:, :, 1] = color[1]
    color_arr[:, :, 2] = color[2]
    del color_arr

    return surf


def draw_glow_line(surface: pygame.Surface,
                   color: Tuple[int, int, int],
                   start: Tuple[float, float],
                   end: Tuple[float, float],
                   width: int = 3,
                   glow_radius: int = 8,
                   glow_alpha: int = 80) -> None:
    """Draw a line with glow effect."""
    # Draw glow (thick, semi-transparent)
    glow_color = (*color, glow_alpha)
    glow_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    aa_line(glow_surf, glow_color, start, end, width + glow_radius)
    if HAS_PIL:
        # Blur the glow surface
        arr = pygame.surfarray.array3d(glow_surf)
        alpha = pygame.surfarray.array_alpha(glow_surf)
        rgba = np.dstack([arr, alpha])
        pil_img = Image.fromarray(rgba, mode='RGBA')
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=glow_radius // 2))
        glow_surf = pygame.image.frombuffer(
            pil_img.tobytes(), surface.get_size(), 'RGBA'
        ).convert_alpha()
    surface.blit(glow_surf, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)

    # Draw main line on top
    aa_line(surface, color, start, end, width)


def draw_glow_circle(surface: pygame.Surface,
                     color: Tuple[int, int, int],
                     center: Tuple[float, float],
                     radius: float,
                     glow_radius: int = 6,
                     glow_alpha: int = 60) -> None:
    """Draw a circle with glow effect."""
    # Glow
    glow_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    glow_color = (*color, glow_alpha)
    aa_circle(glow_surf, glow_color, center, radius + glow_radius)
    if HAS_PIL:
        arr = pygame.surfarray.array3d(glow_surf)
        alpha = pygame.surfarray.array_alpha(glow_surf)
        rgba = np.dstack([arr, alpha])
        pil_img = Image.fromarray(rgba, mode='RGBA')
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=glow_radius))
        glow_surf = pygame.image.frombuffer(
            pil_img.tobytes(), surface.get_size(), 'RGBA'
        ).convert_alpha()
    surface.blit(glow_surf, (0, 0))

    # Main circle
    aa_circle(surface, color, center, radius)


# ── Smooth animations (using pytweening) ────────────────────────────────────

class SmoothValue:
    """
    Smoothly animated value using easing functions.

    Usage:
        speed_display = SmoothValue(initial=0.0, duration=0.3)
        speed_display.set_target(20.0)
        current = speed_display.update(dt)  # call every frame
    """

    def __init__(self, initial: float = 0.0, duration: float = 0.3,
                 easing: str = "easeOutCubic") -> None:
        self._start = initial
        self._target = initial
        self._current = initial
        self._duration = duration
        self._elapsed = 0.0
        self._easing_name = easing
        self._easing = self._get_easing(easing)

    @staticmethod
    def _get_easing(name: str):
        if HAS_TWEEN:
            return getattr(pytweening, name, pytweening.linear)
        # Fallback: simple linear
        return lambda t: t

    def set_target(self, value: float) -> None:
        """Set new target value — will animate towards it."""
        if abs(value - self._target) > 0.01:
            self._start = self._current
            self._target = value
            self._elapsed = 0.0

    def update(self, dt: float) -> float:
        """Update animation. Returns current value."""
        if self._elapsed < self._duration:
            self._elapsed += dt
            t = min(self._elapsed / self._duration, 1.0)
            eased = self._easing(t)
            self._current = self._start + (self._target - self._start) * eased
        else:
            self._current = self._target
        return self._current

    @property
    def value(self) -> float:
        return self._current

    @property
    def target(self) -> float:
        return self._target

    @property
    def is_animating(self) -> bool:
        return self._elapsed < self._duration


class PulseAnimation:
    """
    Pulsing animation (e.g., for warning indicators).

    Usage:
        pulse = PulseAnimation(freq_hz=2.0)
        alpha = pulse.update(dt)  # returns 0.0-1.0
    """

    def __init__(self, freq_hz: float = 2.0) -> None:
        self.freq = freq_hz
        self._t = 0.0

    def update(self, dt: float) -> float:
        """Returns pulse value 0.0-1.0."""
        self._t += dt
        # Sine wave 0-1
        return 0.5 + 0.5 * math.sin(2 * math.pi * self.freq * self._t)

    def reset(self) -> None:
        self._t = 0.0


# ── Color utilities ─────────────────────────────────────────────────────────

def lerp_color(c1: Tuple[int, int, int],
               c2: Tuple[int, int, int],
               t: float) -> Tuple[int, int, int]:
    """Linear interpolation between two colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def fade_color(color: Tuple[int, int, int],
               alpha: float) -> Tuple[int, int, int]:
    """Fade color towards black by alpha (0=black, 1=full color)."""
    alpha = max(0.0, min(1.0, alpha))
    return (
        int(color[0] * alpha),
        int(color[1] * alpha),
        int(color[2] * alpha),
    )


def with_alpha(color: Tuple[int, int, int],
               alpha: int) -> Tuple[int, int, int, int]:
    """Add alpha channel to RGB color."""
    return (color[0], color[1], color[2], max(0, min(255, alpha)))


# ── Performance: cached surfaces ────────────────────────────────────────────

class SurfaceCache:
    """
    Cache for frequently-used surfaces (e.g., glow effects).
    Prevents re-creating expensive surfaces every frame.
    """

    def __init__(self, max_size: int = 50) -> None:
        self._cache: dict = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[pygame.Surface]:
        return self._cache.get(key)

    def set(self, key: str, surface: pygame.Surface) -> None:
        if len(self._cache) >= self._max_size:
            # Remove oldest (simple FIFO — could use LRU)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = surface

    def clear(self) -> None:
        self._cache.clear()


# Global cache instance
_surface_cache = SurfaceCache()


def cached_glow(width: int, height: int,
                color: Tuple[int, int, int],
                blur_radius: int = 10) -> pygame.Surface:
    """Get cached glow surface (reuses if same params)."""
    key = f"glow_{width}_{height}_{color}_{blur_radius}"
    surf = _surface_cache.get(key)
    if surf is None:
        surf = make_glow_surface(width, height, color, 1.0, blur_radius)
        _surface_cache.set(key, surf)
    return surf
