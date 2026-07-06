"""
apollo_dashboard.py
═══════════════════════════════════════════════════════════════════════════════
Apollo-Style Lane Detection + MPC Visualization Dashboard
─────────────────────────────────────────────────────────
PURPOSE  : Standalone Python visualization ที่ใช้ประกอบโปรเจกต์
           CARLA–Apollo Bridge + UNet Lane Detection + MPC Controller
           เลียนแบบ Apollo DreamView HUD ด้วย matplotlib เท่านั้น

LAYOUT   :
  ┌──────────────────────────┬───────────────────────┬──────────────────────┐
  │  [A] PIPELINE STATUS     │  [B] CAMERA / LANE    │  [C] BEV MAP         │
  │      workflow steps      │      UNet overlay      │      bird-eye view   │
  ├──────────────────────────┼───────────────────────┼──────────────────────┤
  │  [D] MPC STATE           │  [E] CONTROL SIGNALS   │  [F] SYSTEM LOG      │
  │      speed / CTE / head  │      steer/thr/brake   │      live text feed  │
  └──────────────────────────┴───────────────────────┴──────────────────────┘

DEPENDENCIES : matplotlib, numpy, scipy  (ไม่ต้องการ pygame หรือ ROS)
USAGE        : python3 apollo_dashboard.py
               กด Ctrl+C หรือปิดหน้าต่างเพื่อหยุด

INTEGRATION  : เพื่อใช้กับ CARLA + run_unet_mpc pipeline จริง
               → สร้าง ApolloDashboardState ที่รับ (rgb, cte_m, heading_rad,
                  curvature, steer, throttle, brake, speed_kmh, lane_conf,
                  reference_path, mode) จากแต่ละ frame แล้วอัปเดต SimState
               → เรียก render_frame() ใน thread แยกจาก control loop (10Hz)
"""
# This file is the full Apollo dashboard; implementation below.

import logging
import math
import os
import random
import time
import collections

import numpy as np

logger = logging.getLogger(__name__)
import matplotlib
# Backend: caller sets TkAgg/Qt5Agg for live; we use Agg only when run as __main__
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════
#  THEME — Apollo Dark HUD
# ══════════════════════════════════════════════════════════════════
BG_DEEP = "#010409"
BG_PANEL = "#0a0e1a"
BG_CARD = "#0d1220"
CYAN = "#00d4ff"
CYAN_DIM = "#0099cc"
GREEN = "#00ff88"
AMBER = "#ffaa00"
RED = "#ff4444"
PURPLE = "#8b5cf6"
TEXT_PRI = "#e2eaf7"
TEXT_SEC = "#8fa3c0"
TEXT_DIM = "#4a607a"
BORDER = "#1e2d45"

plt.rcParams.update({
    "figure.facecolor": BG_DEEP,
    "axes.facecolor": BG_PANEL,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT_SEC,
    "xtick.color": TEXT_DIM,
    "ytick.color": TEXT_DIM,
    "text.color": TEXT_PRI,
    "grid.color": BORDER,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "font.family": "monospace",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titlecolor": CYAN,
    "axes.titleweight": "bold",
})


# ══════════════════════════════════════════════════════════════════
#  SIMULATION STATE (replace with real CARLA/pipeline data)
# ══════════════════════════════════════════════════════════════════
class SimState:
    """Mock state for standalone demo. Replace with data from run_unet_mpc pipeline."""

    def __init__(self):
        self.t = 0.0
        self.speed = 0.0
        self.target_v = 10.0
        self.cte = 0.0
        self.heading_err = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.unet_conf = 0.0
        self.unet_ms = 0.0
        self.mpc_ms = 0.0
        self.mpc_converged = True
        self.mask_px = 0
        self.module_status = {
            "SENSOR CAPTURE": "DONE",
            "PREPROCESS": "DONE",
            "UNET INFERENCE": "RUN",
            "IPM TRANSFORM": "RUN",
            "CENTERLINE FIT": "RUN",
            "MPC SOLVE": "RUN",
            "CONTROL APPLY": "RUN",
        }
        N = 200
        self.hist_speed = collections.deque([0.0] * N, maxlen=N)
        self.hist_cte = collections.deque([0.0] * N, maxlen=N)
        self.hist_steer = collections.deque([0.0] * N, maxlen=N)
        self.hist_thr = collections.deque([0.0] * N, maxlen=N)
        self.hist_brk = collections.deque([0.0] * N, maxlen=N)
        self.log_lines = collections.deque(maxlen=18)
        self._log_pool = [
            (GREEN, "[camera] frame {0:05d}"),
            (CYAN, "[preprocess] ROI-crop done"),
            (GREEN, "[unet] {0:.0f}ms · mask {1:d}px · conf {2:.2f}"),
            (GREEN, "[ipm] homography ok"),
            (GREEN, "[centerline] waypoints · poly-deg2"),
            (GREEN, "[mpc] converged {0:.0f}ms"),
            (CYAN, "[control] apply_control"),
            (GREEN, "[chassis] speed={0:.1f}m/s steer={1:.1f}%"),
            (AMBER, "[unet] low-px → fallback waypoints"),
            (GREEN, "[mpc] δ={0:.1f}° thr={1:.0f}% brk={2:.0f}%"),
            (GREEN, "[centerline] confidence {0:.2f}"),
        ]
        self._frame = 0
        self._log_t = 0.0
        self.rgb = None  # optional: live camera (H,W,3) RGB from pipeline

    def update_from_pipeline(
        self,
        t: float,
        speed_kmh: float,
        target_v_kmh: float,
        cte_m: float,
        heading_err_rad: float,
        steer: float,
        throttle: float,
        brake: float,
        unet_conf: float = 0.0,
        unet_ms: float = 0.0,
        mpc_ms: float = 0.0,
        mpc_converged: bool = True,
        mask_px: int = 0,
        rgb=None,
        mode: str = "FUSION",
        road_symbol: str = "",
    ):
        """อัปเดต state จากข้อมูล pipeline จริง (ใช้เมื่อรันกับ run_unet_mpc)."""
        self.t = t
        self.speed = speed_kmh / 3.6
        self.target_v = target_v_kmh / 3.6
        self.cte = float(cte_m)
        self.heading_err = float(heading_err_rad)
        self.steer = steer
        self.throttle = throttle
        self.brake = brake
        self.unet_conf = unet_conf
        self.unet_ms = unet_ms
        self.mpc_ms = mpc_ms
        self.mpc_converged = mpc_converged
        self.mask_px = mask_px
        if rgb is not None:
            self.rgb = np.asarray(rgb)
        self._frame += 1
        self.hist_speed.append(speed_kmh)
        self.hist_cte.append(cte_m * 100)
        self.hist_steer.append(steer)
        self.hist_thr.append(throttle * 100)
        self.hist_brk.append(brake * 100)

    def tick(self, dt: float):
        self.t += dt
        self._frame += 1
        self.speed += (self.target_v - self.speed) * 0.06 * dt * 60
        self.speed = float(np.clip(self.speed, 0, 20))
        self.cte = 0.18 * math.sin(self.t * 0.35) + 0.04 * random.gauss(0, 1)
        self.cte = float(np.clip(abs(self.cte), 0, 0.8))
        self.heading_err = 2.5 * math.sin(self.t * 0.35 - 0.3) + random.gauss(0, 0.2)
        self.steer = 12.0 * math.sin(self.t * 0.28) + 3 * math.sin(self.t * 0.91)
        self.steer += random.gauss(0, 0.3)
        self.throttle = float(np.clip(0.35 + self.speed * 0.02, 0, 1))
        self.brake = float(np.clip(-min(0, self.speed - self.target_v) * 0.15, 0, 1))
        self.heading += (self.steer / 29.4) * self.speed / 100.0 * dt * 60
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
        self.unet_conf = 0.89 + 0.07 * math.sin(self.t * 0.4) + random.gauss(0, 0.01)
        self.unet_conf = float(np.clip(self.unet_conf, 0, 1))
        self.unet_ms = 22 + 6 * random.random()
        self.mpc_ms = 38 + 12 * random.random()
        self.mask_px = int(3400 + 400 * math.sin(self.t * 0.6) + random.gauss(0, 80))
        self.mpc_converged = self.unet_conf > 0.55
        self.hist_speed.append(self.speed * 3.6)
        self.hist_cte.append(self.cte * 100)
        self.hist_steer.append(self.steer)
        self.hist_thr.append(self.throttle * 100)
        self.hist_brk.append(self.brake * 100)
        self._log_t += dt
        if self._log_t > 0.4 + 0.2 * random.random():
            self._log_t = 0.0
            color, tmpl = random.choice(self._log_pool)
            try:
                if "unet" in tmpl and "ms" in tmpl:
                    msg = tmpl.format(self.unet_ms, self.mask_px, self.unet_conf)
                elif "speed" in tmpl:
                    msg = tmpl.format(self.speed, self.steer)
                elif "δ=" in tmpl:
                    msg = tmpl.format(self.steer, self.throttle * 100, self.brake * 100)
                elif "confidence" in tmpl:
                    msg = tmpl.format(self.unet_conf)
                elif "converged" in tmpl:
                    msg = tmpl.format(self.mpc_ms)
                elif "frame" in tmpl:
                    msg = tmpl.format(self._frame)
                else:
                    msg = tmpl
            except Exception:
                msg = tmpl.split("{")[0] + "..."
            ts = f"[{self.t:08.3f}]"
            self.log_lines.append((color, ts, msg))


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def section_header(ax, title: str, color=CYAN):
    ax.text(0.0, 1.015, f"  {title}", transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=color,
            fontfamily="monospace", va="bottom",
            bbox=dict(boxstyle="square,pad=0.18", facecolor=BG_CARD,
                      edgecolor=color, linewidth=0.8, alpha=0.9))


def corner_marks(ax, color=CYAN, size=0.04):
    corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for cx, cy in corners:
        dx = size if cx == 0 else -size
        dy = size if cy == 0 else -size
        ax.plot([cx, cx + dx, None, cx, cx],
                [cy + dy, cy, None, cy, cy + dy],
                color=color, lw=0.8, alpha=0.5, transform=ax.transAxes,
                clip_on=False, solid_capstyle="round")


def draw_gauge_arc(ax, cx, cy, r, value, vmin, vmax, color, label, unit,
                   start_angle=200, end_angle=-20, linewidth=5):
    theta_start = math.radians(start_angle)
    theta_end = math.radians(end_angle)
    total_span = theta_start - theta_end
    frac = np.clip((value - vmin) / (vmax - vmin), 0, 1)
    thetas_bg = np.linspace(theta_end, theta_start, 120)
    ax.plot(cx + r * np.cos(thetas_bg), cy + r * np.sin(thetas_bg),
            color=BORDER, lw=linewidth, solid_capstyle="round", zorder=2)
    theta_val = theta_start - frac * total_span
    thetas_fg = np.linspace(theta_val, theta_start, 120)
    ax.plot(cx + r * np.cos(thetas_fg), cy + r * np.sin(thetas_fg),
            color=color, lw=linewidth, solid_capstyle="round", zorder=3,
            path_effects=[pe.Stroke(linewidth=linewidth + 4, foreground=color, alpha=0.15), pe.Normal()])
    ax.text(cx, cy + 0.02, f"{value:.1f}", ha="center", va="center",
            fontsize=14, fontweight="bold", color=color, fontfamily="monospace",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG_PANEL)])
    ax.text(cx, cy - 0.14, unit, ha="center", va="center",
            fontsize=7, color=TEXT_DIM, fontfamily="monospace")
    ax.text(cx, cy - 0.26, label, ha="center", va="center",
            fontsize=7, color=TEXT_SEC, fontfamily="monospace")


# ══════════════════════════════════════════════════════════════════
#  SUBPLOT A — PIPELINE WORKFLOW
# ══════════════════════════════════════════════════════════════════
def draw_pipeline(ax, state: SimState):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    section_header(ax, "PIPELINE  WORKFLOW")
    corner_marks(ax)
    modules = [
        ("01", "SENSOR CAPTURE", "camera · 30Hz", "DONE", GREEN),
        ("02", "PREPROCESS", "resize · ROI", "DONE", GREEN),
        ("03", "UNET INFERENCE", f"conf={state.unet_conf:.2f} · {state.unet_ms:.0f}ms", "RUN", CYAN),
        ("04", "IPM TRANSFORM", "vehicle frame", "RUN", CYAN),
        ("05", "CENTERLINE FIT", "poly-deg2 · waypoints", "RUN", CYAN),
        ("06", "MPC SOLVE", f"{state.mpc_ms:.0f}ms · {'OK' if state.mpc_converged else 'FAIL'}", "RUN", CYAN if state.mpc_converged else RED),
        ("07", "CONTROL APPLY", f"δ={state.steer:.1f}° thr={state.throttle*100:.0f}%", "RUN", AMBER),
    ]
    step_h = 0.115
    y_start = 0.93
    for i, (num, name, desc, status, color) in enumerate(modules):
        y = y_start - i * (step_h + 0.005)
        bg = FancyBboxPatch((0.03, y - step_h + 0.01), 0.94, step_h - 0.01,
                            boxstyle="square,pad=0", linewidth=0.6,
                            edgecolor=color if status == "RUN" else BORDER,
                            facecolor=BG_CARD if status != "RUN" else plt.matplotlib.colors.to_rgba(color, 0.04))
        ax.add_patch(bg)
        bar = Rectangle((0.03, y - step_h + 0.01), 0.006, step_h - 0.01,
                         facecolor=color, linewidth=0)
        ax.add_patch(bar)
        ax.text(0.07, y - step_h * 0.45, f"MOD·{num}", color=TEXT_DIM, fontsize=6, fontfamily="monospace", va="center")
        ax.text(0.13, y - step_h * 0.28, name, color=color if status == "RUN" else TEXT_PRI, fontsize=8.5, fontweight="bold", fontfamily="monospace", va="center")
        ax.text(0.13, y - step_h * 0.68, desc, color=TEXT_DIM, fontsize=6.5, fontfamily="monospace", va="center")
        badge_color = GREEN if status == "DONE" else (RED if not state.mpc_converged and i == 5 else color)
        badge_bg = plt.matplotlib.colors.to_rgba(badge_color, 0.12)
        ax.text(0.88, y - step_h * 0.45, status, color=badge_color, fontsize=6.5, fontweight="bold",
                fontfamily="monospace", va="center", ha="center",
                bbox=dict(boxstyle="square,pad=0.25", facecolor=badge_bg, edgecolor=badge_color, linewidth=0.6, alpha=0.9))
        if i < len(modules) - 1:
            ay = y - step_h + 0.01
            ax.annotate("", xy=(0.5, ay - 0.004), xytext=(0.5, ay + 0.0),
                        arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=0.8, mutation_scale=6))


# ══════════════════════════════════════════════════════════════════
#  SUBPLOT B — CAMERA / LANE
# ══════════════════════════════════════════════════════════════════
def draw_camera(ax, state: SimState):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_facecolor("#040810")
    section_header(ax, f"CAMERA·FRONT  ·  CONF:{state.unet_conf:.2f}  ·  {state.unet_ms:.0f}ms")
    corner_marks(ax)
    # Live RGB from pipeline (เมื่อใช้กับ run_unet_mpc)
    rgb = getattr(state, "rgb", None)
    if rgb is not None and rgb.size > 0:
        h, w = rgb.shape[:2]
        ax.imshow(rgb, extent=[0, 1, 1, 0], aspect="auto", interpolation="bilinear")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.text(0.02, 0.98, f"FRAME {state._frame:06d}", transform=ax.transAxes,
                fontsize=6.5, color=TEXT_SEC, fontfamily="monospace",
                bbox=dict(facecolor=BG_DEEP, edgecolor="none", alpha=0.7, pad=1.5))
        ax.text(0.02, 0.94, f"MASK {state.mask_px:5d}px", transform=ax.transAxes,
                fontsize=6.5, color=TEXT_SEC, fontfamily="monospace",
                bbox=dict(facecolor=BG_DEEP, edgecolor="none", alpha=0.7, pad=1.5))
        return
    t = state.t
    VPX, VPY = 0.5 + 0.015 * math.sin(t * 0.18), 0.47
    for i, alpha in enumerate(np.linspace(0.0, 0.4, 12)):
        yb = 0.47 - i * 0.005
        ax.axhspan(yb - 0.005, yb, alpha=alpha * 0.3, color="#0a1828", zorder=0)
    road_poly = plt.Polygon(
        [(0.05, 0), (0.95, 0), (0.75, VPY), (0.25, VPY)],
        facecolor="#0e141f", edgecolor="none", zorder=1, alpha=0.95)
    ax.add_patch(road_poly)

    def persp_x(x_base, y_norm):
        return VPX + (x_base - VPX) * (1.0 - y_norm * 0.72)

    lane_xs = [-0.38, -0.13, 0.13, 0.38]
    dash_off = (t * 0.18) % 0.14
    for i, lx in enumerate(lane_xs):
        center = 0.5
        is_solid = (i == 0 or i == 3)
        y_vals = np.linspace(0, VPY, 80)
        x_vals = [persp_x(center + lx, y / VPY) for y in y_vals]
        if is_solid:
            ax.plot(x_vals, y_vals, color="white", lw=1.5, alpha=0.7, zorder=3, solid_capstyle="round")
        else:
            seg_len, gap_len, total, pos = 0.13, 0.09, VPY, dash_off
            while pos < total:
                seg_end = min(pos + seg_len, total)
                ys = np.linspace(pos, seg_end, 10)
                xs = [persp_x(center + lx, y / VPY) for y in ys]
                ax.plot(xs, ys, color="#ffdd00", lw=1.2, alpha=0.65, zorder=3, solid_capstyle="round")
                pos += seg_len + gap_len

    ego_l = [persp_x(0.5 - 0.13, y / VPY) for y in np.linspace(0, VPY, 60)]
    ego_r = [persp_x(0.5 + 0.13, y / VPY) for y in np.linspace(0, VPY, 60)]
    ys60 = np.linspace(0, VPY, 60)
    mask_alpha = 0.22 * state.unet_conf
    ax.fill_betweenx(ys60, ego_l, ego_r, color=GREEN, alpha=mask_alpha, zorder=2)
    ax.plot(ego_l, ys60, color=GREEN, lw=0.8, alpha=0.35, zorder=4)
    ax.plot(ego_r, ys60, color=GREEN, lw=0.8, alpha=0.35, zorder=4)

    n_wpts = 10
    wpt_xs, wpt_ys = [], []
    for k in range(n_wpts):
        frac = k / (n_wpts - 1)
        wy = frac * VPY * 0.95
        lateral_offset = state.cte * 0.3 * (1 - frac)
        wx = persp_x(0.5 + lateral_offset + 0.012 * math.sin(t * 0.3 + k * 0.4), frac)
        wpt_xs.append(wx)
        wpt_ys.append(wy)
        r = 0.007 * (1 - frac * 0.5)
        ax.add_patch(Circle((wx, wy), r, color=CYAN, zorder=6, alpha=0.85))
        ax.add_patch(Circle((wx, wy), r * 2.2, color=CYAN, zorder=5, alpha=0.12, linewidth=0))
    ax.plot(wpt_xs, wpt_ys, color=CYAN, lw=1.0, ls="--", alpha=0.5, zorder=5, dashes=(4, 4))

    arc_pts_x = [persp_x(0.5 + math.sin(math.radians(state.steer * 0.4) * frac * 3.5) * 0.18 * frac, frac) for frac in np.linspace(0, 0.88, 25)]
    arc_pts_y = np.linspace(0, VPY * 0.88, 25)
    ax.plot(arc_pts_x, arc_pts_y, color=AMBER, lw=1.8, ls=(0, (4, 3)), alpha=0.7, zorder=7)

    cte_px = state.cte * 0.35
    cte_color = RED if state.cte > 0.4 else (AMBER if state.cte > 0.2 else GREEN)
    ax.annotate("", xy=(0.5 + cte_px, 0.05), xytext=(0.5, 0.05),
                arrowprops=dict(arrowstyle="<->", color=cte_color, lw=1.2, mutation_scale=8))
    ax.text(0.5 + cte_px / 2, 0.08, f"CTE {state.cte*100:.1f}cm", ha="center", fontsize=6.5, color=cte_color, fontfamily="monospace")
    for tx, ty, txt in [(0.02, 0.97, f"FRAME {state._frame:06d}"), (0.02, 0.93, f"MASK {state.mask_px:5d}px"), (0.65, 0.97, f"T+{state.t:07.2f}s"), (0.65, 0.93, f"GPU {state.unet_ms:.0f}ms")]:
        ax.text(tx, ty, txt, transform=ax.transAxes, fontsize=6.5, color=TEXT_SEC, fontfamily="monospace",
                bbox=dict(facecolor=BG_DEEP, edgecolor="none", alpha=0.7, pad=1.5))


# ══════════════════════════════════════════════════════════════════
#  SUBPLOT C — BEV MAP
# ══════════════════════════════════════════════════════════════════
def draw_bev(ax, state: SimState):
    ax.set_aspect("equal")
    ax.set_facecolor("#040810")
    section_header(ax, "BEV  VEHICLE  FRAME", color=GREEN)
    corner_marks(ax, color=GREEN)
    ax.set_xlim(-6, 45)
    ax.set_ylim(-8, 8)
    ax.set_xlabel("x  [m forward]", fontsize=6.5, color=TEXT_DIM)
    ax.set_ylabel("y  [m lateral]", fontsize=6.5, color=TEXT_DIM)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.15, color=BORDER)
    ax.axhline(0, color=BORDER, lw=0.5, alpha=0.4)
    t = state.t
    lane_half = 1.8
    road = FancyBboxPatch((-2, -lane_half * 2), 47, lane_half * 4,
                          boxstyle="square,pad=0", facecolor="#0a0f1a",
                          edgecolor=BORDER, linewidth=0.5, alpha=0.8)
    ax.add_patch(road)
    for ly in [-lane_half * 2, -lane_half, 0, lane_half, lane_half * 2]:
        style = "--" if ly == 0 else "-"
        col = "#ffdd00" if ly == 0 else "white"
        alpha = 0.5 if ly == 0 else 0.35
        lw = 1.0 if ly == 0 else 0.8
        if ly == 0:
            for dx in np.arange(-2, 44, 4):
                ax.plot([dx, dx + 2.2], [ly, ly], color=col, lw=lw, alpha=alpha, solid_capstyle="round")
        else:
            ax.axhline(ly, xmin=0.04, color=col, lw=lw, alpha=alpha)
    road_curve_y = np.array([state.cte + 0.3 * math.sin(t * 0.35 + x * 0.08) for x in range(45)])
    ax.plot(range(45), road_curve_y, color=GREEN, lw=1.5, alpha=0.35, ls="--", zorder=3)
    wpts_x = np.linspace(2, 42, 20)
    wpts_y = np.array([state.cte + 0.25 * math.sin(t * 0.35 + x * 0.08) for x in wpts_x])
    ax.scatter(wpts_x, wpts_y, s=12, color=CYAN, zorder=5, alpha=0.85, edgecolors=CYAN, linewidths=0)
    ax.plot(wpts_x, wpts_y, color=CYAN, lw=1.0, alpha=0.4, ls="--", zorder=4, dashes=(3, 3))
    pred_x = np.linspace(0, 25, 11)
    steer_r = math.radians(state.steer * 0.5)
    pred_y = np.array([state.cte + math.sin(steer_r * 0.5) * x * 0.15 for x in pred_x])
    ax.plot(pred_x, pred_y, color=AMBER, lw=2.0, alpha=0.75, zorder=6, solid_capstyle="round")
    for px, py in zip(pred_x[1:], pred_y[1:]):
        ax.add_patch(Circle((px, py), 0.25, color=AMBER, alpha=0.3, zorder=6))
    veh_l, veh_w = 4.93, 1.93
    veh = FancyBboxPatch((-veh_l / 2, -veh_w / 2), veh_l, veh_w,
                         boxstyle="round,pad=0.1",
                         facecolor=plt.matplotlib.colors.to_rgba(CYAN, 0.12),
                         edgecolor=CYAN, linewidth=1.5, zorder=8)
    ax.add_patch(veh)
    ax.annotate("", xy=(veh_l / 2 + 3.5, 0), xytext=(veh_l / 2, 0),
                arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=1.5, mutation_scale=10))
    cte_color = RED if state.cte > 0.4 else (AMBER if state.cte > 0.2 else GREEN)
    ax.annotate("", xy=(0, state.cte), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color=cte_color, lw=1.5, mutation_scale=8))
    ax.text(0.5, state.cte / 2, f"CTE\n{state.cte*100:.1f}cm", fontsize=6, color=cte_color, fontfamily="monospace", va="center", ha="left")
    legend_elements = [
        Line2D([0], [0], color=GREEN, lw=1.5, ls="--", label="Centerline ref"),
        Line2D([0], [0], color=CYAN, lw=1, ls="--", label="Lane waypoints"),
        Line2D([0], [0], color=AMBER, lw=2, label="MPC prediction"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=6, facecolor=BG_CARD,
              edgecolor=BORDER, labelcolor=TEXT_SEC, framealpha=0.9)


# ══════════════════════════════════════════════════════════════════
#  SUBPLOT D — MPC STATE GAUGES
# ══════════════════════════════════════════════════════════════════
def draw_mpc_state(ax, state: SimState):
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    section_header(ax, "MPC  STATE  METRICS", color=PURPLE)
    corner_marks(ax, color=PURPLE)
    speed_kmh = state.speed * 3.6
    cte_cm = state.cte * 100
    head_deg = abs(state.heading_err)
    speed_color = GREEN if abs(speed_kmh - state.target_v * 3.6) < 3 else AMBER
    cte_color = RED if cte_cm > 80 else (AMBER if cte_cm > 50 else GREEN)
    head_color = RED if head_deg > 8 else (AMBER if head_deg > 5 else CYAN)
    draw_gauge_arc(ax, 0.5, 0.45, 0.3, speed_kmh, 0, 72, speed_color, "SPEED", "km/h")
    draw_gauge_arc(ax, 1.5, 0.45, 0.3, cte_cm, 0, 100, cte_color, "CTE", "cm")
    draw_gauge_arc(ax, 2.5, 0.45, 0.3, head_deg, 0, 15, head_color, "HEADING ERR", "deg")
    ax.text(0.5, 0.08, f"TGT {state.target_v*3.6:.0f} km/h", ha="center", fontsize=6.5, color=TEXT_DIM, fontfamily="monospace")
    total_ms = state.unet_ms + state.mpc_ms
    timing_color = RED if total_ms > 95 else (AMBER if total_ms > 80 else GREEN)
    ax.text(1.5, 0.08, f"PIPELINE {total_ms:.0f}ms / 100ms", ha="center", fontsize=6.5, color=timing_color, fontfamily="monospace")
    ax.text(2.5, 0.08, "CONVERGED" if state.mpc_converged else "FALLBACK", ha="center", fontsize=6.5,
            color=GREEN if state.mpc_converged else RED, fontfamily="monospace")


# ══════════════════════════════════════════════════════════════════
#  SUBPLOT E — CONTROL SIGNALS
# ══════════════════════════════════════════════════════════════════
def draw_control_signals(ax, state: SimState):
    ax.set_facecolor(BG_PANEL)
    section_header(ax, "CONTROL  SIGNALS  (ROLLING  10s)")
    corner_marks(ax)
    N = len(state.hist_steer)
    xs = np.linspace(0, 10, N)
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    steer_arr = np.array(state.hist_steer)
    ax.plot(xs, steer_arr, color=AMBER, lw=1.2, alpha=0.85, label="Steer (°)")
    ax.fill_between(xs, steer_arr, 0, alpha=0.08, color=AMBER)
    ax.axhline(29.4, color=RED, lw=0.6, ls="--", alpha=0.4)
    ax.axhline(-29.4, color=RED, lw=0.6, ls="--", alpha=0.4)
    ax.set_ylim(-35, 35)
    ax.set_ylabel("steer  (°)", fontsize=6.5, color=AMBER)
    ax.tick_params(axis="y", colors=AMBER, labelsize=6)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_xlabel("time  [s]", fontsize=6.5)
    thr_arr = np.array(state.hist_thr)
    brk_arr = np.array(state.hist_brk)
    ax2.plot(xs, thr_arr, color=GREEN, lw=1.0, alpha=0.7, ls="-", label="Thr (%)")
    ax2.fill_between(xs, thr_arr, 0, alpha=0.06, color=GREEN)
    ax2.plot(xs, -brk_arr, color=RED, lw=1.0, alpha=0.7, ls="-", label="Brk (%)")
    ax2.fill_between(xs, -brk_arr, 0, alpha=0.06, color=RED)
    ax2.set_ylim(-120, 120)
    ax2.set_ylabel("thr / brk  (%)", fontsize=6.5, color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN, labelsize=6)
    for val, col, label, yp in [(state.steer, AMBER, f"δ={state.steer:.1f}°", 0.88), (state.throttle * 100, GREEN, f"THR {state.throttle*100:.0f}%", 0.74), (state.brake * 100, RED, f"BRK {state.brake*100:.0f}%", 0.60)]:
        ax.text(1.01, yp, label, transform=ax.transAxes, fontsize=7, color=col, fontfamily="monospace", va="center")
    lines = [Line2D([0], [0], color=AMBER, lw=1.2, label="Steer (°)"), Line2D([0], [0], color=GREEN, lw=1.0, label="Throttle (%)"), Line2D([0], [0], color=RED, lw=1.0, label="Brake (%)")]
    ax.legend(handles=lines, loc="upper left", fontsize=6, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_SEC, framealpha=0.85)
    ax.grid(True, alpha=0.12)


# ══════════════════════════════════════════════════════════════════
#  SUBPLOT F — SYSTEM LOG
# ══════════════════════════════════════════════════════════════════
def draw_log(ax, state: SimState):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    section_header(ax, "SYSTEM  LOG", color=TEXT_SEC)
    corner_marks(ax, color=TEXT_DIM)
    lines = list(state.log_lines)
    max_lines = 17
    y_start = 0.94
    line_h = 0.05
    ax.text(0.0, y_start + 0.03, "  TIMESTAMP      MODULE", fontsize=5.5, color=TEXT_DIM, fontfamily="monospace", transform=ax.transAxes)
    ax.axhline(y_start, color=BORDER, lw=0.4, xmin=0, xmax=1)
    for i, (color, ts, msg) in enumerate(reversed(lines[-max_lines:])):
        y = y_start - (i + 1) * line_h
        if y < 0.02:
            break
        ax.text(0.0, y, ts, fontsize=6, color=TEXT_DIM, fontfamily="monospace", va="center")
        short_msg = msg[:58] + ("…" if len(msg) > 58 else "")
        ax.text(0.30, y, short_msg, fontsize=6, color=color, fontfamily="monospace", va="center")
    ok_count = sum(1 for c, _, _ in state.log_lines if c == GREEN)
    warn_count = sum(1 for c, _, _ in state.log_lines if c == AMBER)
    err_count = sum(1 for c, _, _ in state.log_lines if c == RED)
    ax.text(0.0, 0.03,
            f"  OK:{ok_count}  WARN:{warn_count}  ERR:{err_count}  "
            f"| UNet:{state.unet_ms:.0f}ms  MPC:{state.mpc_ms:.0f}ms  "
            f"| TOTAL:{state.unet_ms+state.mpc_ms:.0f}/100ms",
            fontsize=6, color=TEXT_DIM, fontfamily="monospace",
            bbox=dict(facecolor=BG_CARD, edgecolor=BORDER, pad=2))


# ══════════════════════════════════════════════════════════════════
#  MAIN RENDER LOOP
# ══════════════════════════════════════════════════════════════════
def build_figure():
    fig = plt.figure(figsize=(22, 12), facecolor=BG_DEEP)
    fig.patch.set_facecolor(BG_DEEP)
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32, left=0.03, right=0.97, top=0.91, bottom=0.05)
    ax_pipe = fig.add_subplot(gs[0, 0])
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_bev = fig.add_subplot(gs[0, 2])
    ax_gauge = fig.add_subplot(gs[1, 0])
    ax_ctrl = fig.add_subplot(gs[1, 1])
    ax_log = fig.add_subplot(gs[1, 2])
    return fig, ax_pipe, ax_cam, ax_bev, ax_gauge, ax_ctrl, ax_log


def render_frame(fig, axes, state: SimState):
    ax_pipe, ax_cam, ax_bev, ax_gauge, ax_ctrl, ax_log = axes
    for ax in axes:
        ax.cla()
    draw_pipeline(ax_pipe, state)
    draw_camera(ax_cam, state)
    draw_bev(ax_bev, state)
    draw_mpc_state(ax_gauge, state)
    draw_control_signals(ax_ctrl, state)
    draw_log(ax_log, state)
    fig.text(0.5, 0.965, "APOLLO  AV  ·  LANE DETECTION + MPC CONTROL SYSTEM  ·  v8.0",
             ha="center", fontsize=11, fontweight="bold", color=CYAN, fontfamily="monospace",
             path_effects=[pe.withStroke(linewidth=4, foreground=BG_DEEP)])
    time_str = time.strftime("%H:%M:%S")
    fig.text(0.97, 0.965, f"SIM  T+{state.t:07.2f}s  ·  {time_str}", ha="right", fontsize=8, color=TEXT_DIM, fontfamily="monospace")
    fig.text(0.03, 0.965, "CARLA·0.9.16  →  UNet + MPC  →  Apollo-style", ha="left", fontsize=8, color=TEXT_DIM, fontfamily="monospace")
    line = plt.matplotlib.lines.Line2D([0.01, 0.99], [0.955, 0.955], transform=fig.transFigure, color=BORDER, linewidth=0.8)
    fig.add_artist(line)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    state = SimState()
    fig, *axes = build_figure()
    total_frames = 300
    output_dir = "frames"
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Apollo Dashboard — rendering frames...")
    logger.info("Output → ./frames/frame_NNNN.png")
    for frame_idx in range(total_frames):
        state.tick(dt=0.033)
        render_frame(fig, axes, state)
        fname = f"{output_dir}/frame_{frame_idx:04d}.png"
        fig.savefig(fname, dpi=110, facecolor=BG_DEEP, bbox_inches="tight", pad_inches=0.05)
        plt.close("all")
        if frame_idx < total_frames - 1:
            fig, *axes = build_figure()
        if frame_idx % 30 == 0:
            logger.info(
                "  frame %04d/%d  T=%.1fs  speed=%.1fkm/h  CTE=%.1fcm  steer=%.1f°",
                frame_idx, total_frames, state.t, state.speed * 3.6,
                state.cte * 100, state.steer,
            )
    logger.info("Done. %d frames saved to ./%s/", total_frames, output_dir)
    logger.info("สร้าง video: ffmpeg -r 30 -i frames/frame_%%04d.png -c:v libx264 apollo_demo.mp4")


if __name__ == "__main__":
    main()
