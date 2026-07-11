#!/usr/bin/env python3
"""
Factory HMI theme — industrial color scheme, typography and layout constants.

The palette follows ISA 101 "High Visibility" style conventions for industrial
HMI displays: dark backgrounds, high-contrast text and semantic status colors
that are distinguishable for operators under varying lighting conditions.

All colors are RGB tuples compatible with pygame.  Typography and layout
constants are plain integers so they can be used before pygame is initialized.
"""
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Type alias for an RGB color triple used throughout the module.
RGB = Tuple[int, int, int]


class HmiColors:
    """Industrial HMI color palette (ISA 101 style).

    Colors are organized by functional category so that consumers can pick a
    semantic color rather than a raw hex value, keeping the visual language
    consistent across the dashboard.
    """

    # ── Backgrounds ──────────────────────────────────────────────────────
    # Deep dark base used for the overall dashboard backdrop.
    BG_DARK: RGB = (18, 20, 28)
    # Standard panel background (slightly lighter than the base).
    BG_PANEL: RGB = (28, 32, 44)
    # Alternate panel background for zebra-striped / secondary panels.
    BG_PANEL_ALT: RGB = (34, 38, 52)
    # Header bar background (top of the dashboard).
    BG_HEADER: RGB = (22, 26, 38)
    # Footer / status bar background (bottom of the dashboard).
    BG_FOOTER: RGB = (16, 18, 26)

    # ── Text ─────────────────────────────────────────────────────────────
    # Primary high-contrast text for values and titles.
    TEXT_PRIMARY: RGB = (235, 238, 245)
    # Secondary text for labels and supporting information.
    TEXT_SECONDARY: RGB = (170, 178, 195)
    # Dimmed text for hints, units and disabled information.
    TEXT_DIM: RGB = (110, 120, 140)
    # Inverse text used on light / status-colored backgrounds.
    TEXT_INVERSE: RGB = (20, 22, 30)

    # ── Status ───────────────────────────────────────────────────────────
    # Normal / healthy status (green).
    STATUS_OK: RGB = (0, 210, 90)
    # Cautionary status (amber).
    STATUS_WARNING: RGB = (255, 190, 0)
    # Fault status (red).
    STATUS_ERROR: RGB = (240, 70, 70)
    # Critical / flashing alarm status (bright red).
    STATUS_CRITICAL: RGB = (255, 40, 40)
    # Informational status (blue).
    STATUS_INFO: RGB = (0, 170, 255)
    # Inactive / offline status (gray).
    STATUS_INACTIVE: RGB = (120, 125, 140)

    # ── Accents ──────────────────────────────────────────────────────────
    ACCENT_CYAN: RGB = (0, 200, 230)
    ACCENT_ORANGE: RGB = (255, 150, 40)
    ACCENT_PURPLE: RGB = (180, 100, 240)

    # ── Lane detection ───────────────────────────────────────────────────
    LANE_LEFT: RGB = (255, 220, 0)     # yellow
    LANE_RIGHT: RGB = (60, 130, 255)   # blue
    LANE_CENTER: RGB = (0, 220, 100)   # green
    LANE_LOST: RGB = (255, 80, 80)     # red

    # ── Gauges ───────────────────────────────────────────────────────────
    GAUGE_BG: RGB = (40, 46, 62)
    GAUGE_FILL: RGB = (0, 200, 230)
    GAUGE_NEEDLE: RGB = (255, 240, 240)
    GAUGE_REDLINE: RGB = (240, 70, 70)

    # ── Grid ─────────────────────────────────────────────────────────────
    GRID_LINE: RGB = (50, 56, 74)
    GRID_LINE_DIM: RGB = (36, 42, 56)

    # Convenience mapping of severity name -> status color.
    SEVERITY_COLORS: Dict[str, RGB] = {
        "ok": STATUS_OK,
        "warning": STATUS_WARNING,
        "error": STATUS_ERROR,
        "critical": STATUS_CRITICAL,
        "info": STATUS_INFO,
        "inactive": STATUS_INACTIVE,
    }

    @classmethod
    def severity(cls, name: str) -> RGB:
        """Return the status color for a severity name (defaults to INFO)."""
        return cls.SEVERITY_COLORS.get(str(name).lower(), cls.STATUS_INFO)


class HmiFonts:
    """Font size constants (in pixels) for the factory HMI dashboard."""

    # Large dashboard title.
    TITLE: int = 32
    # Section / panel header.
    HEADER: int = 24
    # Field labels and captions.
    LABEL: int = 18
    # Standard numeric value readout.
    VALUE: int = 28
    # Small body text.
    SMALL: int = 14
    # Tiny annotations / units.
    TINY: int = 10
    # Large prominent value (e.g. primary speed readout).
    BIG_VALUE: int = 48


class HmiLayout:
    """Layout constants for the 1280x800 factory HMI dashboard.

    Coordinates are in pixels from the top-left origin.  The dashboard is
    divided into a header, a main content area and a footer, with standard
    margins and padding applied throughout.
    """

    # ── Canvas ───────────────────────────────────────────────────────────
    WIDTH: int = 1280
    HEIGHT: int = 800

    # ── Vertical regions ─────────────────────────────────────────────────
    HEADER_H: int = 70
    FOOTER_H: int = 50
    # Main content height is derived: HEIGHT - HEADER_H - FOOTER_H
    MAIN_H: int = HEIGHT - HEADER_H - FOOTER_H  # 680

    # ── Margins / padding ────────────────────────────────────────────────
    MARGIN: int = 12
    PADDING: int = 10
    GAP: int = 10
    PANEL_RADIUS: int = 8

    # ── Header region ────────────────────────────────────────────────────
    HEADER_RECT = (0, 0, WIDTH, HEADER_H)
    HEADER_TITLE_POS = (MARGIN + 4, 10)
    HEADER_CLOCK_POS = (WIDTH - 160, 10)

    # ── Footer region ────────────────────────────────────────────────────
    FOOTER_RECT = (0, HEIGHT - FOOTER_H, WIDTH, FOOTER_H)
    FOOTER_TEXT_POS = (MARGIN + 4, HEIGHT - FOOTER_H + 14)

    # ── Main content columns ─────────────────────────────────────────────
    # Three-column layout: left gauges, center camera/BEV, right status.
    COL_LEFT_X: int = MARGIN
    COL_LEFT_W: int = 300
    COL_CENTER_X: int = COL_LEFT_X + COL_LEFT_W + GAP
    COL_CENTER_W: int = WIDTH - COL_LEFT_W - 320 - 3 * MARGIN
    COL_RIGHT_X: int = COL_CENTER_X + COL_CENTER_W + GAP
    COL_RIGHT_W: int = 300

    MAIN_Y: int = HEADER_H + MARGIN

    # ── Left column: arc gauges ──────────────────────────────────────────
    GAUGE_SPEED_CENTER = (COL_LEFT_X + COL_LEFT_W // 2, MAIN_Y + 110)
    GAUGE_SPEED_RADIUS: int = 90

    GAUGE_STEER_CENTER = (COL_LEFT_X + COL_LEFT_W // 2, MAIN_Y + 320)
    GAUGE_STEER_RADIUS: int = 70

    # ── Right column: status panels ──────────────────────────────────────
    PANEL_STATUS_RECT = (COL_RIGHT_X, MAIN_Y, COL_RIGHT_W, 220)
    PANEL_LANE_RECT = (COL_RIGHT_X, MAIN_Y + 230, COL_RIGHT_W, 200)
    PANEL_ALARM_RECT = (COL_RIGHT_X, MAIN_Y + 440, COL_RIGHT_W, 120)

    # ── Center column: camera + BEV ──────────────────────────────────────
    PANEL_CAMERA_RECT = (COL_CENTER_X, MAIN_Y, COL_CENTER_W, 360)
    PANEL_BEV_RECT = (COL_CENTER_X, MAIN_Y + 370, COL_CENTER_W, 200)

    # ── LED strip (lane health) ──────────────────────────────────────────
    LED_STRIP_Y: int = HEADER_H + 2
    LED_SIZE: int = 12
    LED_SPACING: int = 28


class HmiTheme:
    """Combined theme providing colors, fonts and lazy-initialized pygame fonts.

    Calling ``get_font`` requires pygame to be initialized; the method handles
    the case where pygame is not yet available by returning ``None``.
    """

    colors = HmiColors
    fonts = HmiFonts
    layout = HmiLayout

    def __init__(self) -> None:
        # Cache of already-created pygame fonts keyed by (size, bold).
        self._font_cache: Dict[Tuple[int, bool], "pygame.font.Font"] = None  # type: ignore[assignment]
        self._pygame_ok: bool = False

    # ── Pygame readiness ─────────────────────────────────────────────────
    def _check_pygame(self) -> bool:
        """Return True if pygame and its font module are initialized."""
        try:
            import pygame  # noqa: WPS433 - intentional local import
            ok = pygame.get_init() and pygame.font.get_init()
            self._pygame_ok = bool(ok)
            return self._pygame_ok
        except Exception:  # pragma: no cover - pygame missing
            self._pygame_ok = False
            return False

    # ── Font access ──────────────────────────────────────────────────────
    def get_font(self, size: int, bold: bool = False) -> Optional["pygame.font.Font"]:
        """Return a cached pygame font of the given size.

        Uses the pygame default font (``Font(None, size)``) so that no external
        font files are required.  Returns ``None`` if pygame is not initialized.
        """
        if not self._check_pygame():
            return None

        if self._font_cache is None:
            self._font_cache = {}

        import pygame  # noqa: WPS433

        key = (size, bold)
        font = self._font_cache.get(key)
        if font is None:
            try:
                font = pygame.font.Font(None, size)
                font.set_bold(bold)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to create HMI font (size=%d): %s", size, exc)
                return None
            self._font_cache[key] = font
        return font

    # ── Convenience accessors ────────────────────────────────────────────
    def color(self, name: str) -> RGB:
        """Return a color by attribute name from :class:`HmiColors`."""
        return getattr(self.colors, name)

    def font(self, name: str, bold: bool = False) -> Optional["pygame.font.Font"]:
        """Return a pygame font by size-name from :class:`HmiFonts`."""
        size = getattr(self.fonts, name)
        return self.get_font(size, bold=bold)
