"""
EventTimelinePanel — Factory HMI event/alarm timeline strip

แสดง behavior events และ warnings แบบ chronological เหมือน SCADA/HMI alarm list
ใน factory control room. ประกอบด้วย 4 ส่วน:

  A. ALARM BANNER — flashing banner สำหรับ critical/error ที่ยังไม่ acknowledge
  B. EVENT LIST  — scrolling list ของ recent events (newest at top, factory style)
  C. EVENT STATISTICS — count by severity + total + time since last event
  D. MINI TIMELINE GRAPH — horizontal bar แสดง event density ใน 60 วินาทีล่าสุด

ใช้คู่กับ BehaviorLogger (src/telemetry/behavior_logger.py).
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Graceful pygame import ──────────────────────────────────────────────────
try:
    import pygame
    HAS_PYGAME = True
except ImportError:  # pragma: no cover
    pygame = None
    HAS_PYGAME = False

# ── Theme import with fallback ──────────────────────────────────────────────
try:
    from gui.hmi.theme import HmiColors, HmiFonts, HmiTheme
    _HAS_THEME = True
except Exception:  # pragma: no cover - theme may not exist yet
    _HAS_THEME = False

    class _FallbackColors:
        BG = (18, 18, 24)
        PANEL = (28, 28, 38)
        PANEL_BORDER = (60, 60, 80)
        TEXT = (230, 230, 235)
        TEXT_DIM = (140, 140, 160)
        TEXT_BRIGHT = (255, 255, 255)
        ACCENT = (0, 180, 255)
        STATUS_OK = (0, 220, 100)
        STATUS_WARNING = (255, 200, 0)
        STATUS_ERROR = (255, 80, 80)
        STATUS_CRITICAL = (255, 40, 40)
        ROW_TINT_INFO = (20, 40, 30)
        ROW_TINT_WARNING = (50, 42, 18)
        ROW_TINT_ERROR = (50, 22, 22)
        ROW_TINT_CRITICAL = (60, 16, 16)
        ALARM_FLASH_ON = (180, 20, 20)
        ALARM_FLASH_OFF = (60, 10, 10)
        GRID = (50, 50, 70)

    class _FallbackFonts:
        def get(self, size: int, bold: bool = False):
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
        colors = _FallbackColors()
        fonts = _FallbackFonts()

    HmiColors = _FallbackColors()  # type: ignore
    HmiFonts = _FallbackFonts()    # type: ignore
    HmiTheme = _FallbackTheme()    # type: ignore


# ── Severity → color/icon mapping ───────────────────────────────────────────
SEVERITY_ORDER = ("info", "warning", "error", "critical")

_SEVERITY_COLOR = {
    "info": lambda: HmiColors.STATUS_OK,
    "warning": lambda: HmiColors.STATUS_WARNING,
    "error": lambda: HmiColors.STATUS_ERROR,
    "critical": lambda: HmiColors.STATUS_CRITICAL,
}

_SEVERITY_TINT = {
    "info": lambda: getattr(HmiColors, "ROW_TINT_INFO", (20, 40, 30)),
    "warning": lambda: getattr(HmiColors, "ROW_TINT_WARNING", (50, 42, 18)),
    "error": lambda: getattr(HmiColors, "ROW_TINT_ERROR", (50, 22, 22)),
    "critical": lambda: getattr(HmiColors, "ROW_TINT_CRITICAL", (60, 16, 16)),
}

_SEVERITY_LABEL = {
    "info": "INFO",
    "warning": "WARN",
    "error": "ERR",
    "critical": "CRIT",
}

# Compact event type labels (factory HMI style)
_EVENT_TYPE_LABEL = {
    "lane_conf_drop": "LANE-CONF",
    "lane_lost": "LANE-LOST",
    "lane_recovered": "LANE-RECOV",
    "phase_failure": "PHASE-FAIL",
    "geometry_invalid": "GEOM-INV",
    "drifting_left": "DRIFT-L",
    "drifting_right": "DRIFT-R",
    "off_track": "OFF-TRACK",
    "returned_to_lane": "RTN-LANE",
    "oscillation": "OSC",
    "sudden_brake": "SUD-BRAKE",
    "excessive_steering": "EXC-STEER",
    "mpc_unstable": "MPC-UNST",
    "perception_mode_change": "MODE-CHG",
}


@dataclass
class TimelineEvent:
    """Internal representation of an event in the timeline."""
    event_type: str
    severity: str
    message: str
    frame_idx: int
    timestamp: float          # seconds since session start
    wall_time: float          # time.time() at insertion
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    @property
    def is_alarm(self) -> bool:
        return self.severity in ("error", "critical")

    @property
    def color(self) -> Tuple[int, int, int]:
        return _SEVERITY_COLOR.get(self.severity, lambda: HmiColors.TEXT)()

    @property
    def tint(self) -> Tuple[int, int, int]:
        return _SEVERITY_TINT.get(self.severity, lambda: HmiColors.PANEL)()

    @property
    def type_label(self) -> str:
        return _EVENT_TYPE_LABEL.get(self.event_type, self.event_type[:10].upper())

    @property
    def severity_label(self) -> str:
        return _SEVERITY_LABEL.get(self.severity, self.severity.upper()[:4])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "acknowledged": self.acknowledged,
            "context": self.context,
        }


def _format_timestamp(ts: float) -> str:
    """แปลง seconds-since-start → HH:MM:SS.mmm"""
    try:
        dt = datetime(1970, 1, 1) + timedelta(seconds=float(ts))
        return dt.strftime("%H:%M:%S.") + f"{int(dt.microsecond / 1000):03d}"
    except Exception:
        return "--:--:--.---"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "\u2026"


class EventTimelinePanel:
    """
    Factory HMI event/alarm timeline strip.

    แสดง behavior events แบบ chronological เหมือน SCADA alarm list.

    Layout (rect = x, y, w, h):
        +----------------------------------------------------------+
        | ALARM BANNER (flashing, if active critical/error)        |  ~36px
        +----------------------------------------------------------+
        | EVENT LIST (scrolling, newest at top)                    |  flexible
        |  HH:MM:SS.mmm  F#  ● TYPE      message...               |
        |  ...                                                     |
        +----------------------------------------------------------+
        | STATS: INFO:N WARN:M ERR:K CRIT:J | TOTAL | LAST: 12.3s |  ~22px
        +----------------------------------------------------------+
        | MINI TIMELINE GRAPH (last 60s, color-coded markers)      |  ~40px
        +----------------------------------------------------------+
    """

    AUTO_ACK_SECONDS = 5.0
    MINI_TIMELINE_WINDOW = 60.0  # seconds shown in mini graph
    BANNER_H = 36
    STATS_H = 22
    MINI_GRAPH_H = 40
    ROW_H = 20
    HEADER_H = 18

    def __init__(self, rect: Tuple[int, int, int, int], max_events: int = 50):
        """
        Args:
            rect: (x, y, w, h) ของ panel ใน screen coordinates
            max_events: จำนวน events สูงสุดที่เก็บใน timeline (rolling)
        """
        self.rect = pygame.Rect(rect) if HAS_PYGAME else None
        self.x, self.y, self.w, self.h = rect
        self.max_events = max_events

        # Rolling event store (newest appended at right/end; displayed newest-at-top)
        self._events: deque = deque(maxlen=max_events)

        # Active alarm (latest unacknowledged error/critical)
        self._active_alarm: Optional[TimelineEvent] = None
        self._alarm_triggered_at: float = 0.0  # wall time

        # Time tracking
        self._last_frame_time: float = time.time()
        self._last_event_time: Optional[float] = None  # session time of last event
        self._current_time_s: float = 0.0

        # Scroll offset (rows scrolled from top); 0 = show newest at top
        self._scroll_offset: int = 0

        # Cached fonts
        self._font_banner: Any = None
        self._font_row: Any = None
        self._font_small: Any = None
        self._font_tiny: Any = None
        self._fonts_ready = False

        # Cached surfaces for performance
        self._surface: Any = None

    # ── Font setup (lazy, requires pygame display) ──────────────────────────
    def _ensure_fonts(self) -> None:
        if self._fonts_ready or not HAS_PYGAME:
            return
        try:
            # Prefer theme fonts if available
            get = getattr(HmiFonts, "get", None)
            if callable(get):
                self._font_banner = get(20, bold=True)
                self._font_row = get(16, bold=False)
                self._font_small = get(14, bold=False)
                self._font_tiny = get(12, bold=False)
            else:
                self._font_banner = pygame.font.Font(None, 22)
                self._font_row = pygame.font.Font(None, 18)
                self._font_small = pygame.font.Font(None, 16)
                self._font_tiny = pygame.font.Font(None, 14)
            # Fallback if any None
            if self._font_banner is None:
                self._font_banner = pygame.font.Font(None, 22)
            if self._font_row is None:
                self._font_row = pygame.font.Font(None, 18)
            if self._font_small is None:
                self._font_small = pygame.font.Font(None, 16)
            if self._font_tiny is None:
                self._font_tiny = pygame.font.Font(None, 14)
            self._fonts_ready = True
        except Exception as e:
            logger.warning("EventTimelinePanel: font init failed: %s", e)
            self._fonts_ready = False

    # ── Public API ──────────────────────────────────────────────────────────

    def add_event(self,
                  event_type: str,
                  severity: str,
                  message: str,
                  frame_idx: int,
                  timestamp: float,
                  context: Optional[Dict[str, Any]] = None) -> None:
        """
        เพิ่ม event เข้า timeline.

        Args:
            event_type: BehaviorType value (e.g. "lane_lost") หรือ custom string
            severity: Severity value ("info" | "warning" | "error" | "critical")
            message: ข้อความมนุษย์อ่านได้
            frame_idx: frame number ตอนเกิด event
            timestamp: session time (seconds since start)
            context: optional metadata dict
        """
        sev = str(severity).lower()
        if sev not in SEVERITY_ORDER:
            sev = "info"

        evt = TimelineEvent(
            event_type=str(event_type),
            severity=sev,
            message=str(message),
            frame_idx=int(frame_idx),
            timestamp=float(timestamp),
            wall_time=time.time(),
            context=context or {},
        )
        self._events.append(evt)
        self._last_event_time = float(timestamp)

        # New error/critical → raise alarm (auto-ack previous)
        if evt.is_alarm:
            self._active_alarm = evt
            self._alarm_triggered_at = time.time()

        logger.debug("EventTimeline: +%s [%s] %s", evt.type_label, sev, message)

    def update(self, frame_state: Any, frame_idx: int, time_s: float) -> None:
        """
        เรียกทุก frame เพื่อ track time และ auto-acknowledge alarms.

        Args:
            frame_state: FrameState (unused for now, reserved for future hooks)
            frame_idx: current frame index
            time_s: current session time in seconds
        """
        self._current_time_s = float(time_s)
        self._last_frame_time = time.time()

        # Auto-acknowledge alarm after AUTO_ACK_SECONDS
        if self._active_alarm is not None:
            elapsed = time.time() - self._alarm_triggered_at
            if elapsed >= self.AUTO_ACK_SECONDS:
                self._active_alarm.acknowledged = True
                self._active_alarm = None

    def acknowledge_all(self) -> None:
        """Acknowledge ทุก alarm ปัจจุบัน (manual clear)."""
        for evt in self._events:
            evt.acknowledged = True
        self._active_alarm = None

    def get_active_alarms(self) -> List[Dict[str, Any]]:
        """คืน list ของ unacknowledged ERROR/CRITICAL events (newest first)."""
        out: List[Dict[str, Any]] = []
        for evt in reversed(self._events):
            if evt.is_alarm and not evt.acknowledged:
                out.append(evt.to_dict())
        return out

    def get_event_stats(self) -> Dict[str, Any]:
        """คืนสถิติ events: {total, by_severity, last_event_time}."""
        by_sev = {"info": 0, "warning": 0, "error": 0, "critical": 0}
        for evt in self._events:
            by_sev[evt.severity] = by_sev.get(evt.severity, 0) + 1
        return {
            "total": len(self._events),
            "by_severity": by_sev,
            "last_event_time": self._last_event_time,
        }

    def clear(self) -> None:
        """ล้าง events ทั้งหมด."""
        self._events.clear()
        self._active_alarm = None
        self._last_event_time = None
        self._scroll_offset = 0

    # ── Drawing ─────────────────────────────────────────────────────────────

    def draw(self, surface: Any, time_s: float) -> None:
        """
        วาด timeline panel ลงบน surface.

        Args:
            surface: pygame.Surface ปลายทาง
            time_s: current session time (seconds)
        """
        if not HAS_PYGAME or surface is None:
            return
        self._ensure_fonts()
        if not self._fonts_ready:
            return

        self._current_time_s = float(time_s)

        try:
            # Panel background
            pygame.draw.rect(surface, HmiColors.PANEL, self.rect)
            pygame.draw.rect(surface, HmiColors.PANEL_BORDER, self.rect, 1)

            # Layout sub-rects
            x = self.x
            y = self.y
            w = self.w

            # A. Alarm banner (only if active alarm)
            banner_h = self.BANNER_H if self._active_alarm is not None else 0
            if banner_h:
                self._draw_alarm_banner(surface, pygame.Rect(x, y, w, banner_h))
            y += banner_h

            # B. Event list (flexible height)
            stats_h = self.STATS_H
            graph_h = self.MINI_GRAPH_H
            list_h = self.h - banner_h - stats_h - graph_h
            if list_h > 0:
                self._draw_event_list(surface, pygame.Rect(x, y, w, list_h))
            y += list_h

            # C. Statistics strip
            self._draw_stats(surface, pygame.Rect(x, y, w, stats_h))
            y += stats_h

            # D. Mini timeline graph
            self._draw_mini_timeline(surface, pygame.Rect(x, y, w, graph_h))

        except Exception as e:
            logger.warning("EventTimelinePanel.draw failed: %s", e)

    # ── A. Alarm banner ─────────────────────────────────────────────────────
    def _draw_alarm_banner(self, surface: Any, rect: Any) -> None:
        """Flashing red banner สำหรับ latest unacknowledged alarm."""
        alarm = self._active_alarm
        if alarm is None:
            return

        # Flash at ~2 Hz
        flash_on = (int(time.time() * 2) % 2) == 0
        bg = getattr(HmiColors, "ALARM_FLASH_ON", (180, 20, 20)) if flash_on \
            else getattr(HmiColors, "ALARM_FLASH_OFF", (60, 10, 10))
        pygame.draw.rect(surface, bg, rect)

        # Time since alarm
        elapsed = time.time() - self._alarm_triggered_at
        ts_str = _format_timestamp(alarm.timestamp)
        line = f"\u26a0 ALARM: [{alarm.severity_label}] {alarm.message}  " \
               f"({ts_str}  +{elapsed:4.1f}s)"
        line = _truncate(line, max_chars=int(rect.w / 7))  # rough char width

        if self._font_banner is not None:
            txt = self._font_banner.render(line, True, HmiColors.TEXT_BRIGHT)
            surface.blit(txt, (rect.x + 8, rect.y + (rect.h - txt.get_height()) // 2))

        # Right-side ACK hint
        hint = "AUTO-ACK 5s"
        if self._font_tiny is not None:
            ht = self._font_tiny.render(hint, True, HmiColors.TEXT_BRIGHT)
            surface.blit(ht, (rect.right - ht.get_width() - 8,
                              rect.y + (rect.h - ht.get_height()) // 2))

    # ── B. Event list ───────────────────────────────────────────────────────
    def _draw_event_list(self, surface: Any, rect: Any) -> None:
        """Scrolling list ของ events (newest at top, factory HMI style)."""
        # Header
        header = "TIME            F#     SEV  TYPE        MESSAGE"
        if self._font_tiny is not None:
            ht = self._font_tiny.render(header, True, HmiColors.TEXT_DIM)
            surface.blit(ht, (rect.x + 6, rect.y + 2))
        pygame.draw.line(surface, HmiColors.PANEL_BORDER,
                         (rect.x, rect.y + self.HEADER_H),
                         (rect.right, rect.y + self.HEADER_H), 1)

        # Visible rows
        list_top = rect.y + self.HEADER_H + 2
        avail_h = rect.bottom - list_top
        max_rows = max(0, avail_h // self.ROW_H)

        # Newest first
        events_desc = list(reversed(self._events))
        # Apply scroll offset (0 = newest at top)
        start = max(0, self._scroll_offset)
        visible = events_desc[start: start + max_rows]

        flash_on = (int(time.time() * 2) % 2) == 0

        for i, evt in enumerate(visible):
            row_y = list_top + i * self.ROW_H
            row_rect = pygame.Rect(rect.x, row_y, rect.w, self.ROW_H)

            # Row background tint by severity
            pygame.draw.rect(surface, evt.tint, row_rect)

            # Unacknowledged alarm → flashing left border
            if evt.is_alarm and not evt.acknowledged and flash_on:
                pygame.draw.rect(surface, evt.color,
                                 pygame.Rect(rect.x, row_y, 3, self.ROW_H))

            # Severity LED (colored circle)
            led_x = rect.x + 8
            led_y = row_y + self.ROW_H // 2
            led_color = evt.color
            if evt.severity == "critical" and flash_on and not evt.acknowledged:
                # Flashing red for unack critical
                led_color = HmiColors.TEXT_BRIGHT
            pygame.draw.circle(surface, led_color, (led_x, led_y), 5)
            pygame.draw.circle(surface, HmiColors.PANEL_BORDER, (led_x, led_y), 5, 1)

            # Text columns
            if self._font_row is None:
                continue
            col_x = rect.x + 22

            # Timestamp
            ts_surf = self._font_row.render(_format_timestamp(evt.timestamp),
                                            True, HmiColors.TEXT_DIM)
            surface.blit(ts_surf, (col_x, row_y + 1))
            col_x += 96

            # Frame number
            fno = self._font_row.render(f"F{evt.frame_idx}", True, HmiColors.TEXT_DIM)
            surface.blit(fno, (col_x, row_y + 1))
            col_x += 52

            # Severity label
            sev_surf = self._font_row.render(evt.severity_label, True, evt.color)
            surface.blit(sev_surf, (col_x, row_y + 1))
            col_x += 44

            # Event type
            type_surf = self._font_row.render(evt.type_label, True, HmiColors.TEXT)
            surface.blit(type_surf, (col_x, row_y + 1))
            col_x += 84

            # Message (truncated to remaining width)
            remaining = rect.right - col_x - 6
            if remaining > 10 and self._font_row is not None:
                # Estimate char width
                avg_w = max(6, self._font_row.size("x")[0])
                max_chars = remaining // avg_w
                msg = _truncate(evt.message, max_chars)
                msg_surf = self._font_row.render(msg, True, HmiColors.TEXT)
                surface.blit(msg_surf, (col_x, row_y + 1))

        # Scroll indicator if more events than visible
        if len(events_desc) > max_rows:
            ind = f"\u2191{len(events_desc) - max_rows - start} more  " \
                  f"(scroll:{start}/{len(events_desc)})"
            if self._font_tiny is not None:
                it = self._font_tiny.render(ind, True, HmiColors.TEXT_DIM)
                surface.blit(it, (rect.right - it.get_width() - 4, rect.bottom - 14))

    # ── C. Statistics strip ─────────────────────────────────────────────────
    def _draw_stats(self, surface: Any, rect: Any) -> None:
        """Count by severity + total + time since last event."""
        stats = self.get_event_stats()
        by = stats["by_severity"]

        # Time since last event
        if self._last_event_time is not None:
            dt = max(0.0, self._current_time_s - self._last_event_time)
            last_str = f"LAST: {dt:6.1f}s"
        else:
            last_str = "LAST: --"

        parts = [
            (f"INFO:{by['info']}", HmiColors.STATUS_OK),
            (f"WARN:{by['warning']}", HmiColors.STATUS_WARNING),
            (f"ERR:{by['error']}", HmiColors.STATUS_ERROR),
            (f"CRIT:{by['critical']}", HmiColors.STATUS_CRITICAL),
            (f"TOTAL:{stats['total']}", HmiColors.TEXT),
            (last_str, HmiColors.TEXT_DIM),
        ]

        # Background
        pygame.draw.rect(surface, HmiColors.BG, rect)
        pygame.draw.line(surface, HmiColors.PANEL_BORDER,
                         (rect.x, rect.y), (rect.right, rect.y), 1)

        if self._font_small is None:
            return

        cx = rect.x + 6
        cy = rect.y + (rect.h - self._font_small.get_height()) // 2 + 1
        for text, color in parts:
            ts = self._font_small.render(text, True, color)
            surface.blit(ts, (cx, cy))
            cx += ts.get_width() + 14

    # ── D. Mini timeline graph ──────────────────────────────────────────────
    def _draw_mini_timeline(self, surface: Any, rect: Any) -> None:
        """Horizontal bar แสดง event density ใน 60 วินาทีล่าสุด."""
        # Background
        pygame.draw.rect(surface, HmiColors.BG, rect)
        pygame.draw.line(surface, HmiColors.PANEL_BORDER,
                         (rect.x, rect.y), (rect.right, rect.y), 1)

        window = self.MINI_TIMELINE_WINDOW
        t_now = self._current_time_s
        t_start = t_now - window

        # Axis baseline
        base_y = rect.y + rect.h // 2
        pygame.draw.line(surface, HmiColors.GRID,
                         (rect.x + 4, base_y), (rect.right - 4, base_y), 1)

        # Time grid lines (every 15s)
        if self._font_tiny is not None:
            for tick in range(int(t_start // 15) * 15, int(t_now) + 1, 15):
                if tick < t_start:
                    continue
                frac = (tick - t_start) / window
                gx = int(rect.x + 4 + frac * (rect.w - 8))
                pygame.draw.line(surface, HmiColors.GRID,
                                 (gx, rect.y + 4), (gx, rect.bottom - 4), 1)
                lbl = self._font_tiny.render(f"{tick}s", True, HmiColors.TEXT_DIM)
                surface.blit(lbl, (gx + 2, rect.bottom - lbl.get_height() - 1))

        # Event markers
        for evt in self._events:
            if evt.timestamp < t_start:
                continue
            frac = (evt.timestamp - t_start) / window
            mx = int(rect.x + 4 + frac * (rect.w - 8))
            # Stagger vertically by severity to reduce overlap
            sev_idx = SEVERITY_ORDER.index(evt.severity)
            my = base_y - 6 + sev_idx * 3
            if my > rect.bottom - 8:
                my = rect.bottom - 8
            color = evt.color
            r = 3 if evt.is_alarm else 2
            pygame.draw.circle(surface, color, (mx, my), r)
            if evt.is_alarm and not evt.acknowledged:
                pygame.draw.circle(surface, HmiColors.TEXT_BRIGHT, (mx, my), r + 2, 1)

        # Current time marker (vertical line)
        cur_x = rect.right - 4
        pygame.draw.line(surface, HmiColors.ACCENT,
                         (cur_x, rect.y + 2), (cur_x, rect.bottom - 2), 2)
        if self._font_tiny is not None:
            now_lbl = self._font_tiny.render("NOW", True, HmiColors.ACCENT)
            surface.blit(now_lbl, (cur_x - now_lbl.get_width() - 2, rect.y + 2))

    # ── Scroll control (optional, for future mouse wheel support) ───────────
    def scroll_up(self, rows: int = 1) -> None:
        """Scroll ขึ้น (ดู events เก่ากว่า)."""
        self._scroll_offset = min(max(0, len(self._events) - 1),
                                  self._scroll_offset + rows)

    def scroll_down(self, rows: int = 1) -> None:
        """Scroll ลง (กลับไปดู events ใหม่สุด)."""
        self._scroll_offset = max(0, self._scroll_offset - rows)

    def scroll_reset(self) -> None:
        """รีเซ็ต scroll ไปที่ newest."""
        self._scroll_offset = 0
