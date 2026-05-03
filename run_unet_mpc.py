#!/usr/bin/env python3
"""
UNet + Waypoints + LSTM + MPC: Camera -> Waypoints (ทิศทาง) + UNet (เลน) -> MPC -> Drive

Pipeline:
  1. CARLA waypoints = รถรู้ว่าจะไปทางไหน (ตามถนนใน map)
  2. Camera (320x240 RGB) -> UNet -> ego-lane mask (fallback เมื่อไม่มี waypoint)
  3. Reference: ใช้ waypoint path สำหรับ CTE + heading + curvature เมื่อมี; ไม่มีถึงใช้ lane
  4. LSTM temporal smoother -> ลด jitter
  5. MPC -> steering + acceleration
  6. CARLA vehicle control

Changes v2:
  - ใช้ LaneTrajectoryPipeline ใหม่ที่มี bev_binary / bev_window_vis / lane_overlay
  - Dashboard แสดง lane_overlay (green fill) แทน raw mask
  - BEV window vis แสดงใน panel แยก (ช่องที่ 3 เล็กๆ ด้านล่าง)
  - fix: TrajectoryOutput.cte อยู่ใน vehicle-frame meters แล้ว (ไม่ต้อง * CTE_TO_METERS)
  - fix: geometry_valid ถูกส่งไป Dashboard ถูกต้อง
  - fix: waypoint_only fallback logic ชัดเจน
"""

import math
import os
import threading
import time
import queue
import logging
import argparse
import sys
import csv
import json
from collections import deque, Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

# CARLA Python API (แบบเดียวกับ carla_lstm_mpc_project)
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
try:
    import carla
except ImportError:
    for _p in (_root.parent / "PythonAPI", _root / ".carla_py"):
        if _p.exists() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import carla
import numpy as np
import cv2
import torch
import torch.nn as nn
import pygame

from safety.safety_override import SafetyOverride
from safety.stuck_recovery import StuckRecovery
import subprocess
import signal


def cleanup_gpu_processes():
    """Kill old GPU processes to free CUDA memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
        current_pid = str(os.getpid())
        for pid in pids:
            if pid and pid != current_pid:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        if pids:
            time.sleep(2)
            print(f"✅ Cleaned up {len(pids)} GPU processes")
    except Exception as e:
        print(f"⚠️ GPU cleanup failed: {e}")


def start_carla_server(carla_root: str = None, port: int = 2000):
    """Start CARLA server if not already running."""
    # Check if CARLA is already running
    try:
        client = carla.Client("localhost", port)
        client.set_timeout(2.0)
        client.get_server_version()
        print(f"✅ CARLA server already running on port {port}")
        return None
    except:
        pass
    
    # Find CARLA root
    if carla_root is None:
        carla_root = os.environ.get("CARLA_ROOT", "/home/supawich/Desktop/CARLA_0.9.16")
    
    carla_sh = os.path.join(carla_root, "CarlaUE4.sh")
    if not os.path.exists(carla_sh):
        print(f"⚠️ CARLA not found at {carla_sh}")
        return None
    
    # Start CARLA server
    print(f"🚗 Starting CARLA server from {carla_root}...")
    proc = subprocess.Popen(
        [carla_sh, "-RenderOffScreen", "-quality-level=Low"],
        cwd=carla_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp
    )
    
    # Wait for server to be ready
    for i in range(30):
        time.sleep(2)
        try:
            client = carla.Client("localhost", port)
            client.set_timeout(2.0)
            client.get_server_version()
            print(f"✅ CARLA server started (pid={proc.pid})")
            return proc
        except:
            if i % 5 == 4:
                print(f"   Waiting for CARLA... ({i+1}/30)")
    
    print("⚠️ CARLA server failed to start")
    return None
from visualization.peter_moran_viz import compose_peter_moran_overlay
from bridge import get_traffic_obstacles
from alg import LKAStep, make_fallback_trajectory_out, get_reference_path, dynamic_lookahead_m, resample_path
from carla_io import cleanup, get_waypoints, get_waypoints_all_lanes, waypoints_to_cte_heading, waypoints_to_path, waypoint_lookahead_curvature
from config import (
    BEV_H,
    BEV_PX_PER_M,
    BEV_W,
    BEV_WIN_H,
    BEV_WIN_W,
    CAM_FOV_DEG,
    CAM_H,
    CAM_W,
    CENTER_LANE_WIDTH_RATIO,
    CONTROL_HZ,
    CTE_TO_METERS,
    DASHED_LANE_MIN_PIXELS,
    LANE_CONF_THRESHOLD,
    LANE_EXTEND_ROWS_UP,
    LANE_FILL_STEP,
    LANE_HALF_WIDTH_DRAW_M,
    LANE_MEMORY_MARGIN_PX,
    MAX_LANE_WIDTH_PX,
    MIN_LANE_WIDTH_PX,
    PANEL_H,
    PANEL_W,
    PLANNING_VIEW_RADIUS_M,
    REF_PATH_DISPLAY_EMA_ALPHA,
    REF_PATH_DISPLAY_LOOKAHEAD_M,
    REF_PATH_DISPLAY_NUM_PTS,
    REF_PATH_LOOKAHEAD_M,
    REF_PATH_NUM_PTS,
    ROAD_MARGIN_PX,
    ROAD_TOP_RATIO,
    ROI_BOTTOM_RATIO,
    ROI_TOP_PENALTY_RATIO,
    SAFETY_MAX_STEER_RAD,
    STALE_PERCEPTION_S,
    TARGET_SPEED_KMH,
    USE_PERCEPTION_THREAD,
    USE_TRAJECTORY_PIPELINE,
    PERCEPTION_SKIP_FRAME,
    MAIN_LOOP_SLEEP_S,
    DASHBOARD_DRAW_EVERY_N,
    DISPLAY_PATH_CACHE_INTERVAL,
    UXColors,
    VIEW3D_H,
    VIEW3D_W,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("unet_mpc")

# ═══════════════════════════════════════════════════════════════════════════
#  Imports (delayed so CARLA not needed for unit tests)
# ═══════════════════════════════════════════════════════════════════════════

from perception.lane_detector import LaneDetector

if USE_TRAJECTORY_PIPELINE:
    from perception.lane_trajectory import (
        LaneTrajectoryPipeline,
        TrajectoryOutput,
        visualize_pipeline,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  BEV Road Perception (robust BEV-based lane detection)
# ═══════════════════════════════════════════════════════════════════════════

class BEVRoadPerception:
    """BEV-based lane detection with polynomial fitting."""

    EMA = 0.25

    def __init__(self, model_path, device):
        self.dev = device
        self.detector = LaneDetector(model_path=model_path, use_carla=False, model_type="unet")
        logger.info("UNet loaded with BEV pipeline (BEVRoadPerception)")
        self.reset()

    def reset(self):
        self._prev_cte = 0.0
        self._prev_head = 0.0
        self._prev_curv = 0.0
        self._prev_left_coeffs = None
        self._prev_right_coeffs = None

    def process(self, rgb):
        """Process RGB image and return lane tracking outputs."""
        h0, w0 = rgb.shape[:2]
        img = cv2.resize(rgb, (CAM_W, CAM_H))
        
        # Use BEV pipeline
        left_c, right_c, raw_cte, raw_head, raw_curv, conf = self.detector.detect_lanes_bev(img)
        
        if left_c is None and right_c is None:
            # Fallback to previous
            return self._prev_cte, self._prev_head, self._prev_curv, \
                   np.zeros((h0, w0), dtype=np.uint8), max(0.0, conf - 0.3), []
        
        # EMA smoothing
        a = self.EMA
        cte = a * self._prev_cte + (1 - a) * float(np.clip(raw_cte, -1, 1))
        head = a * self._prev_head + (1 - a) * float(np.clip(raw_head, -0.5, 0.5))
        curv = a * self._prev_curv + (1 - a) * float(np.clip(raw_curv, -0.04, 0.04))
        
        self._prev_cte = cte
        self._prev_head = head
        self._prev_curv = curv
        self._prev_left_coeffs = left_c
        self._prev_right_coeffs = right_c
        
        # Generate visualization mask
        mask_vis = np.zeros((h0, w0), dtype=np.uint8)
        
        # Generate centerline for visualization
        centerline = []
        if left_c is not None and right_c is not None:
            center_c = (left_c + right_c) / 2
            pts = self.detector.bev_pipeline.bev_to_perspective_points(center_c, n_points=20)
            centerline = [(int(y), int(x)) for x, y in pts if 0 <= x < w0 and 0 <= y < h0]
        
        return cte, head, curv, mask_vis, conf, centerline


# ═══════════════════════════════════════════════════════════════════════════
#  Road Perception (legacy — U-Net row-by-row, ใช้เมื่อ --legacy-perception)
# ═══════════════════════════════════════════════════════════════════════════

class RoadPerception:
    """Legacy UNet lane detection (row-wise poly fit, no BEV)."""

    EMA = 0.25

    def __init__(self, model_path, device):
        self.dev = device
        self.detector = LaneDetector(model_path=model_path, use_carla=False, model_type="unet")
        logger.info("UNet loaded (legacy RoadPerception)")
        self.reset()

    def reset(self):
        self._prev_cte = 0.0
        self._prev_head = 0.0
        self._prev_curv = 0.0
        self._prev_rows = np.array([], dtype=np.float64)
        self._prev_centers = np.array([], dtype=np.float64)
        self._prev_poly = None
        self._prev_center_bottom = None

    def process(self, rgb):
        h0, w0 = rgb.shape[:2]
        img = cv2.resize(rgb, (CAM_W, CAM_H))
        mask_uint8, _, _ = self.detector.detect_lanes(img, world=None, vehicle=None)
        mask = (mask_uint8 > 0).astype(np.uint8)
        mask[:int(CAM_H * ROAD_TOP_RATIO), :] = 0

        road_roi = mask[int(CAM_H * (1 - ROI_BOTTOM_RATIO)):, :]
        sky_roi  = mask[:int(CAM_H * ROI_TOP_PENALTY_RATIO), :]
        cov_road = min(1.0, np.sum(road_roi > 0) / max(road_roi.size * 0.05, 1))
        cov_sky  = min(1.0, np.sum(sky_roi > 0)  / max(sky_roi.size  * 0.15, 1))
        raw_conf = cov_road * (1.0 - 0.85 * cov_sky)

        cx = CAM_W / 2.0
        row_start = int(CAM_H * ROAD_TOP_RATIO)
        center_lo = int(CAM_W * (0.5 - CENTER_LANE_WIDTH_RATIO / 2))
        center_hi = int(CAM_W * (0.5 + CENTER_LANE_WIDTH_RATIO / 2))

        rows_obs, centers_obs = [], []
        for r in range(row_start, CAM_H, 2):
            cols = np.where(mask[r] > 0)[0]
            if len(cols) < DASHED_LANE_MIN_PIXELS:
                continue
            left, right = cols[0], cols[-1]
            lane_w = right - left
            if lane_w < MIN_LANE_WIDTH_PX or lane_w > MAX_LANE_WIDTH_PX:
                continue
            if left < ROAD_MARGIN_PX or right > (CAM_W - ROAD_MARGIN_PX):
                continue
            mid = (left + right) / 2.0
            if center_lo <= mid <= center_hi:
                if self._prev_center_bottom is not None and abs(mid - self._prev_center_bottom) > LANE_MEMORY_MARGIN_PX:
                    continue
                centers_obs.append(mid)
                rows_obs.append(r)

        poly = None
        if len(rows_obs) >= 3:
            rows_a = np.array(rows_obs, dtype=np.float64)
            ctrs_a = np.array(centers_obs, dtype=np.float64)
            r_bot, r_top = rows_a[-1], rows_a[0]
            rn_obs = (rows_a - r_bot) / max(1.0, r_top - r_bot)
            cn_obs = (ctrs_a - cx) / (CAM_W / 2.0)
            try:
                poly = np.polyfit(rn_obs, cn_obs, 2)
                self._prev_poly = tuple(poly)
            except Exception:
                poly = self._prev_poly
        else:
            poly = self._prev_poly

        if poly is None or len(rows_obs) < 2:
            conf = max(0.0, raw_conf - 0.2)
            mask_vis = cv2.resize(mask * 255, (w0, h0), interpolation=cv2.INTER_NEAREST)
            return self._prev_cte, self._prev_head, self._prev_curv, mask_vis, conf, []

        row_extend  = max(0, row_start - LANE_EXTEND_ROWS_UP)
        all_rows    = np.arange(row_extend, CAM_H, LANE_FILL_STEP, dtype=np.float64)
        poly_arr    = np.array(poly)
        r_bot = float(CAM_H - 1)
        r_top = float(row_start)
        rn_all = (all_rows - r_bot) / max(1.0, r_top - r_bot)
        cn_all = np.polyval(poly_arr, rn_all)
        centers_filled = np.clip(cx + cn_all * (CAM_W / 2.0), center_lo, center_hi)
        rows_a, ctrs_a = all_rows, centers_filled

        raw_cte  = (ctrs_a[-1] - cx) / (CAM_W / 2.0)
        la  = max(0, len(rows_a) // 3)
        dx  = ctrs_a[la] - ctrs_a[-1]
        dy  = rows_a[-1] - rows_a[la]
        raw_head = math.atan2(dx, max(dy, 5.0))
        rn  = (rows_a - rows_a[-1]) / max(1.0, rows_a[-1] - rows_a[0])
        cn  = (ctrs_a - cx) / (CAM_W / 2.0)
        raw_curv = 0.0
        if len(rn) >= 4:
            try:
                p = np.polyfit(rn, cn, 2)
                raw_curv = p[0] * 0.03
            except Exception:
                pass

        a    = self.EMA
        cte  = a * self._prev_cte  + (1-a) * float(np.clip(raw_cte,  -1,    1))
        head = a * self._prev_head + (1-a) * float(np.clip(raw_head, -0.5,  0.5))
        curv = a * self._prev_curv + (1-a) * float(np.clip(raw_curv, -0.04, 0.04))
        self._prev_cte = cte; self._prev_head = head; self._prev_curv = curv
        self._prev_rows = rows_a; self._prev_centers = ctrs_a
        self._prev_center_bottom = float(ctrs_a[-1])

        conf = min(1.0, raw_conf + 0.15 if len(rows_obs) >= 10 else raw_conf)
        mask_vis   = cv2.resize(mask * 255, (w0, h0), interpolation=cv2.INTER_NEAREST)
        centerline = [(int(rows_a[i]), int(np.clip(ctrs_a[i], 0, CAM_W-1))) for i in range(len(rows_a))]
        return cte, head, curv, mask_vis, conf, centerline


def reference_path_to_image(path_s_lat, cam_w, cam_h, lookahead_m,
                             cx=None, ch=None, px_per_m=None):
    if cx is None:       cx = cam_w / 2.0
    if ch is None:       ch = float(cam_h)
    if px_per_m is None: px_per_m = (cam_w / 2.0) / CTE_TO_METERS
    pts = []
    for s, lat in path_s_lat:
        col = cx + lat * px_per_m
        row = ch * (1.0 - s / lookahead_m)
        if 0 <= row < ch and 0 <= col < cam_w:
            pts.append((int(row), int(col)))
    return pts


# ═══════════════════════════════════════════════════════════════════════════
#  Bird's Eye View — Map Vector (Apollo-style)
# ═══════════════════════════════════════════════════════════════════════════

def draw_bird_eye_view(reference_path, heading_rad=0.0, speed_kmh=0.0,
                       lookahead_m=REF_PATH_LOOKAHEAD_M,
                       bev_w=BEV_W, bev_h=BEV_H, px_per_m=BEV_PX_PER_M,
                       waypoint_path=None, all_lanes_paths=None):
    """
    Map vector แบบ Apollo: กริดพื้นหลัง, vector ถนนทั้งหมดรวมทุกเลน (ซ้าย/กลาง/ขวา),
    planned path สีส้มหนา, รถ ego พร้อมป้าย (ระยะ/ความเร็ว).
    all_lanes_paths = (ego_path, left_lane_path, right_lane_path) แต่ละอัน list of (s, lat).
    """
    # พื้นหลังโทนน้ำเงินเข้มแบบแผนที่ (Apollo-style grid)
    img = np.full((bev_h, bev_w, 3), (42, 48, 62), dtype=np.uint8)  # BGR dark blue-gray
    cx, cy = bev_w // 2, bev_h - 28

    # กริดระยะ — step ใหญ่เมื่อ lookahead ยาว (หลายร้อยเมตร) เพื่อไม่ให้เส้นทับกัน
    grid_step = 10 if lookahead_m > 50 else 2
    for d in range(grid_step, int(lookahead_m) + 1, grid_step):
        y = cy - d * px_per_m
        if y < 0:
            break
        color = (70, 75, 90) if d % (5 * max(1, grid_step)) != 0 else (85, 92, 110)
        lw = 1 if d % (5 * max(1, grid_step)) != 0 else 2
        cv2.line(img, (0, int(y)), (bev_w, int(y)), color, lw)
        cv2.putText(img, "%dm" % d, (cx - 16, int(y) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 125, 140))
    for lat in range(-4, 5):
        if lat == 0:
            continue
        x = cx + lat * px_per_m
        if 0 <= x < bev_w:
            color = (65, 70, 85) if abs(lat) % 2 != 0 else (75, 80, 95)
            cv2.line(img, (int(x), 0), (int(x), bev_h), color, 1)

    # ── Vector ถนนทั้งหมด รวมทุกเลน (ซ้าย/ขวา) — เส้นบางสีเทา/ฟ้าอ่อน ──
    if all_lanes_paths is not None:
        ego_path, left_path, right_path = all_lanes_paths[0], all_lanes_paths[1], all_lanes_paths[2]
        for path, color_bgr in [(left_path, (200, 180, 120)), (right_path, (200, 180, 120))]:  # BGR tan/gray
            if path and len(path) >= 2:
                pts = []
                for s, lat in path:
                    if s > lookahead_m:
                        break
                    x = cx + lat * px_per_m
                    y = cy - s * px_per_m
                    if 0 <= y < bev_h and 0 <= x < bev_w:
                        pts.append((int(x), int(y)))
                if len(pts) >= 2:
                    cv2.polylines(img, [np.array(pts)], False, color_bgr, 1)

    # ── ถนนเป็น vector: polygon เลน + ขอบซ้าย/ขวา (เส้นขาว-เหลือง) ──
    if reference_path and len(reference_path) >= 2:
        pts_l, pts_r, pts_c = [], [], []
        for s, lat in reference_path:
            x = cx + lat * px_per_m
            y = cy - s * px_per_m
            if 0 <= y < bev_h:
                pts_l.append((int(cx + (lat - LANE_HALF_WIDTH_DRAW_M) * px_per_m), int(y)))
                pts_r.append((int(cx + (lat + LANE_HALF_WIDTH_DRAW_M) * px_per_m), int(y)))
            if 0 <= y < bev_h and 0 <= x < bev_w:
                pts_c.append((int(x), int(y)))
        if len(pts_l) >= 2 and len(pts_r) >= 2:
            cv2.fillPoly(img, [np.array(pts_l + pts_r[::-1])], (55, 60, 72))
            cv2.polylines(img, [np.array(pts_l)], False, (0, 255, 255), 2)   # yellow left
            cv2.polylines(img, [np.array(pts_r)], False, (0, 255, 255), 2)   # yellow right
        if len(pts_c) >= 2:
            cv2.polylines(img, [np.array(pts_c)], False, (200, 200, 200), 1)  # center thin
        # Planned trajectory (MPC reference) — เส้นส้มหนาแบบ Apollo
        if len(pts_c) >= 2:
            cv2.polylines(img, [np.array(pts_c)], False, (0, 100, 255), 4)   # BGR orange

    # ── (Optional) Waypoint path จาก map — เส้นเทาแสดง lane ระยะไกลจาก map
    if waypoint_path and len(waypoint_path) >= 2:
        pts_wp = []
        for s, lat in waypoint_path:
            x = cx + lat * px_per_m
            y = cy - s * px_per_m
            if 0 <= y < bev_h and 0 <= x < bev_w:
                pts_wp.append((int(x), int(y)))
        if len(pts_wp) >= 2:
            cv2.polylines(img, [np.array(pts_wp)], False, (140, 140, 140), 1)  # BGR gray

    # ── Ego vehicle: รูปสี่เหลี่ยมโทนฟ้าพร้อมป้าย (Apollo-style) ──
    car_len, car_wid = 20, 12
    angle_deg = math.degrees(-heading_rad)
    pts_car = np.array([
        [0, -car_len // 2], [car_wid // 2, car_len // 2],
        [0, car_len // 4], [-car_wid // 2, car_len // 2]
    ], dtype=np.float32)
    R = cv2.getRotationMatrix2D((0, 0), angle_deg, 1.0)
    pts_car = cv2.transform(pts_car.reshape(1, -1, 2), R).reshape(-1, 2) + np.array([cx, cy])
    pts_car = pts_car.astype(np.int32)
    cv2.fillPoly(img, [pts_car], (230, 216, 173))   # BGR light blue
    cv2.polylines(img, [pts_car], True, (255, 255, 255), 1)
    # ป้ายใต้รถ: ระยะ 0.0m | ความเร็ว km/h
    label = "ego | 0.0m | %.1f km/h" % speed_kmh
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    tx, ty = cx - tw // 2, cy + 22
    cv2.rectangle(img, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (42, 48, 62), -1)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cv2.putText(img, "N", (cx - 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 155, 170))
    return img


# ═══════════════════════════════════════════════════════════════════════════
#  Dashboard  (3 panels: Camera, Planning BEV, Perception BEV)
# ═══════════════════════════════════════════════════════════════════════════

class Dashboard:
    """
    Layout: Header | Camera | Planning BEV | Perception BEV | Control | Footer.
    Simplified: 3 main panels without 3D vector view or road symbol.
    """
    HEADER_H = 26
    FOOTER_H = 22
    GAP = 10
    # Theme (RGB for pygame)
    BG = (22, 24, 28)
    HEADER_BG = (32, 36, 42)
    CARD_BORDER = (55, 60, 70)
    ACCENT = (90, 140, 200)
    STATUS_OK = (100, 220, 140)
    STATUS_WARN = (220, 180, 100)

    def __init__(self, screenshot_dir: str = None):
        # บังคับ X11 ถ้ามี DISPLAY (แก้ปัญหา pygame ไม่อัปเดตบนบาง Wayland)
        if os.environ.get("DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "x11")
        pygame.init()
        self.scr = pygame.display.set_mode((1320, 680), pygame.RESIZABLE)
        pygame.display.set_caption("LKA — UNet + MPC | CARLA")
        self.ft = pygame.font.SysFont("monospace", 14)
        self.ft_small = pygame.font.SysFont("monospace", 11)
        self.ft_header = pygame.font.SysFont("monospace", 13)
        self.ft_symbol = pygame.font.SysFont("monospace", 18)
        self.clk = pygame.time.Clock()
        self._spd = deque(maxlen=200)
        self._st = deque(maxlen=200)
        self._cte = deque(maxlen=200)
        self._prev_ref_path = None  # EMA สำหรับ reference_path ที่วาด BEV/3D (ลดเหวี่ยง)
        
        # Screenshot capture setup
        self._screenshot_dir = screenshot_dir
        self._screenshot_interval = 100  # Capture every N frames
        self._frame_count = 0
        if screenshot_dir:
            Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"📸 Dashboard screenshots will be saved to: {screenshot_dir}")
        
        self.scr.fill(self.BG)
        try:
            msg = self.ft.render("Starting CARLA + UNet...", True, (150, 200, 255))
            self.scr.blit(msg, (40, 320))
        except Exception as e:
            logger.debug("Dashboard init render: %s", e)
        pygame.display.flip()

    # ── internal blit helper ──────────────────────────────────────────────
    @staticmethod
    def _blit_rgb(scr, rgb_np, x, y, w, h):
        """Resize numpy RGB to (w, h) and blit at (x, y)."""
        img = cv2.resize(rgb_np, (w, h))
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        scr.blit(surf, (x, y))

    @staticmethod
    def _blit_bgr(scr, bgr_np, x, y, w, h):
        Dashboard._blit_rgb(scr, bgr_np[:, :, ::-1], x, y, w, h)

    # ── main draw ─────────────────────────────────────────────────────────
    def draw(self, rgb, spd, st, th, br, cte, hd, curv, mode,
             lane_conf=0.0,
             reference_path=None,
             waypoint_path=None,  # Optional: (s, lat) จาก map สำหรับวาดเส้นเสริมใน BEV
             all_lanes_paths=None,  # (ego_path, left_path, right_path) vector ถนนทั้งหมดรวมทุกเลน รัศมี 500m
             geometry_valid=False,
             lane_overlay=None,
             mask=None,
             bev_window_vis=None,
             left_px_img=None,   # (N, 2) row,col image-space for segment display
             right_px_img=None,
             tracked=False,
             used_completion=False,
             road_symbol_result=None,  # RoadSymbolResult: เลี้ยวขวา/ซ้าย/ตรง
             obstacles=None,  # bridge.get_traffic_obstacles (Apollo-style)
             phase_p2_case="none",  # P2: both | mirror_left | mirror_right | completion | none
             phase_p3_case="none",  # P3: from_both | completion | none
             # Peter Moran visualization data
             bev_binary=None,
             raw_windows_left=None,
             raw_windows_right=None,
             filt_windows_left=None,
             filt_windows_right=None,
             left_px_bev=None,
             right_px_bev=None,
             left_curvature_m=9999.0,
             right_curvature_m=9999.0,
             ):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False

        scr_w, scr_h = self.scr.get_size()
        TOP = self.HEADER_H
        BOT = scr_h - self.FOOTER_H

        self.scr.fill(self.BG)
        # Header bar
        pygame.draw.rect(self.scr, self.HEADER_BG, (0, 0, scr_w, TOP))
        pygame.draw.line(self.scr, self.CARD_BORDER, (0, TOP), (scr_w, TOP), 1)
        title = self.ft_header.render("LKA  |  UNet + MPC", True, (220, 225, 230))
        self.scr.blit(title, (self.GAP, (TOP - title.get_height()) // 2))
        status_text = "IN LANE" if (abs(cte) < 0.15 and geometry_valid) else ("LANE INVALID" if not geometry_valid else "OFF CENTER")
        status_col = self.STATUS_OK if (abs(cte) < 0.15 and geometry_valid) else (self.STATUS_WARN if geometry_valid else UXColors.RED_INVALID)
        geom_text = "geom ok" if geometry_valid else "geom invalid"
        right_x = scr_w - self.GAP
        for part, col in [
            (geom_text, UXColors.GREEN_VALID if geometry_valid else UXColors.RED_INVALID),
            (status_text, status_col),
            ("conf %.2f" % lane_conf, UXColors.GREEN_VALID if lane_conf >= 0.55 else UXColors.TEXT_NORMAL),
            ("CTE %+.2fm" % cte, UXColors.GREEN_VALID if abs(cte) < 0.25 else self.STATUS_WARN),
            (mode or "—", self.ACCENT),
            ("%.1f km/h" % spd, (180, 220, 200)),
        ]:
            r = self.ft_small.render(part, True, col)
            right_x -= r.get_width() + 14
            self.scr.blit(r, (right_x, (TOP - r.get_height()) // 2))

        pw, ph = PANEL_W, PANEL_H

        # Lookahead สำหรับ scale: รัศมีรอบตัวรถได้ถึง PLANNING_VIEW_RADIUS_M (500m), ใช้ path จริงหรือ ref
        lookahead_used = (
            max((s for s, _ in reference_path), default=0.0)
            if reference_path and len(reference_path) >= 2
            else REF_PATH_DISPLAY_LOOKAHEAD_M
        )
        if lookahead_used < 1.0:
            lookahead_used = REF_PATH_DISPLAY_LOOKAHEAD_M
        if all_lanes_paths is not None:
            lookahead_used = PLANNING_VIEW_RADIUS_M  # แสดง vector ถนนทั้งหมดในรัศมี 500m

        # Reference path สำหรับวาด: EMA ลดเหวี่ยงใน BEV/3D; resample ก่อน blend เพื่อให้จำนวนจุดเท่ากัน
        display_path = reference_path or []
        if reference_path and len(reference_path) >= 2:
            lookahead_m = max((s for s, _ in reference_path), default=REF_PATH_DISPLAY_LOOKAHEAD_M)
            if lookahead_m < 1.0:
                lookahead_m = REF_PATH_DISPLAY_LOOKAHEAD_M
            curr_resampled = resample_path(
                reference_path, lookahead_m, REF_PATH_DISPLAY_NUM_PTS
            )
            if self._prev_ref_path is not None and len(self._prev_ref_path) >= 2:
                prev_resampled = resample_path(
                    self._prev_ref_path, lookahead_m, REF_PATH_DISPLAY_NUM_PTS
                )
                if len(prev_resampled) == len(curr_resampled):
                    a = REF_PATH_DISPLAY_EMA_ALPHA
                    display_path = [
                        (s, a * prev_lat + (1.0 - a) * curr_lat)
                        for (s, curr_lat), (_, prev_lat) in zip(curr_resampled, prev_resampled)
                    ]
                else:
                    display_path = curr_resampled
            else:
                display_path = curr_resampled
            self._prev_ref_path = display_path
        else:
            self._prev_ref_path = None

        # ── Zone: Perception — Peter Moran Style ─────────────────────────────
        if lane_overlay is not None:
            cam_show = lane_overlay
        elif mask is not None:
            h0, w0 = rgb.shape[:2]
            m = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
            if m.ndim == 3:
                m_bin = (m.max(axis=2) > 0).astype(np.uint8)
            else:
                m_bin = (m > 0).astype(np.uint8)
            overlay = np.zeros((h0, w0, 3), dtype=np.uint8)
            overlay[m_bin > 0] = [100, 200, 0]
            cam_show = np.clip(rgb.astype(np.float32) + overlay.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
        else:
            cam_show = rgb

        # Convert to BGR for OpenCV drawing
        cam_bgr = cv2.cvtColor(cam_show, cv2.COLOR_RGB2BGR)

        # Apply Peter Moran diagnostic overlay if BEV data available
        if bev_binary is not None:
            raw_camera_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cam_bgr = compose_peter_moran_overlay(
                lane_overlay_bgr=cam_bgr,
                raw_camera_bgr=raw_camera_bgr,
                bev_binary=bev_binary,
                raw_windows_left=raw_windows_left or [],
                raw_windows_right=raw_windows_right or [],
                filt_windows_left=filt_windows_left or [],
                filt_windows_right=filt_windows_right or [],
                left_px_bev=left_px_bev,
                right_px_bev=right_px_bev,
                left_curvature_m=left_curvature_m,
                right_curvature_m=right_curvature_m,
                cte_m=cte,
            )

        # Card border + blit
        left_h = ph + 32
        pygame.draw.rect(self.scr, self.CARD_BORDER, (2, TOP, pw + 6, left_h), 1)
        self._blit_bgr(self.scr, cam_bgr, 5, TOP + 22, pw, ph)
        # Label + TRACKED/COMPLETION badge
        self.scr.blit(self.ft_small.render("Perception — Peter Moran", True, UXColors.SECTION_LABEL), (5, TOP + 2))
        bx = 215
        if used_completion:
            self.scr.blit(self.ft_small.render("COMPLETION", True, (255, 200, 0)), (bx, TOP + 2))
            bx += 88
        if tracked and not used_completion:
            self.scr.blit(self.ft_small.render("TRACKED", True, (100, 255, 100)), (bx, TOP + 2))

        y_info_base = TOP + ph + 32

        # ── Zone: Planning (ทางที่ไป + แผนที่) — ใช้ display_path (EMA); scale ตาม path จริง (PART 5)
        mid_x = 5 + pw + 10
        pygame.draw.rect(self.scr, self.CARD_BORDER, (mid_x - 2, TOP, pw + 4, ph + 24), 1)
        px_per_m_display = ph / lookahead_used
        bev_img = draw_bird_eye_view(
            display_path, heading_rad=hd, speed_kmh=spd,
            lookahead_m=lookahead_used, bev_w=pw, bev_h=ph, px_per_m=px_per_m_display,
            waypoint_path=waypoint_path, all_lanes_paths=all_lanes_paths)
        self._blit_rgb(self.scr, bev_img, 5 + pw + 10, TOP + 22, pw, ph)
        self.scr.blit(self.ft_small.render("Planning — Bird's Eye", True, UXColors.SECTION_LABEL), (5 + pw + 10, TOP + 2))

        # ── Panel 2: BEV Sliding Window ───────────────────────────────────
        if bev_window_vis is not None:
            self._blit_bgr(self.scr, bev_window_vis,
                           5 + pw * 2 + 20, TOP + 22, BEV_WIN_W, BEV_WIN_H)
            self.scr.blit(self.ft_small.render("Perception — BEV", True, UXColors.SECTION_LABEL),
                          (5 + pw * 2 + 20, TOP + 2))

        # ── Zone: Control (telemetry + graphs) ─────────────────────────────
        x = 5
        y_info = y_info_base
        self._spd.append(spd); self._st.append(st); self._cte.append(cte)
        conf_col = UXColors.GREEN_VALID if lane_conf >= LANE_CONF_THRESHOLD else UXColors.TEXT_NORMAL
        cte_col = UXColors.RED_INVALID if abs(cte) > 0.35 else UXColors.TEXT_NORMAL
        n_obs = len(obstacles) if obstacles else 0
        info = [
            ("MODE:  %s"            % mode,       (100, 255, 200)),
            ("Conf:  %.2f"          % lane_conf,  conf_col),
            ("Speed: %5.1f km/h"   % spd,        (200, 255, 200)),
            ("Steer: %+.3f"        % st,          (200, 200, 200)),
            ("Thr: %.2f  Brk: %.2f" % (th, br),  (200, 200, 200)),
            ("CTE:  %+.2f m"       % cte,         cte_col),
            ("Head: %+.1f deg"     % math.degrees(hd), (200, 200, 200)),
            ("Curv: %+.5f"         % curv,        (200, 200, 200)),
            ("Obstacles: %d"       % n_obs,       (180, 180, 200)),
        ]
        for i, (t, col) in enumerate(info):
            self.scr.blit(self.ft.render(t, True, col), (x, y_info + i * 16))

        # ── Graphs ────────────────────────────────────────────────────────
        gh = 48
        gx, gy = 5, y_info + 18
        self._graph(gx, gy,              280, gh, self._spd, (0, 200, 255), mx=60)
        self._graph(gx, gy + gh + 4,     280, gh, self._st,  (255, 200, 0),  sym=True)
        self._graph(gx, gy + 2*(gh+4),   280, gh, self._cte, (255, 100, 100), sym=True)

        # Footer strip
        pygame.draw.rect(self.scr, self.HEADER_BG, (0, BOT, scr_w, self.FOOTER_H))
        pygame.draw.line(self.scr, self.CARD_BORDER, (0, BOT), (scr_w, BOT), 1)
        footer_parts = [
            "FPS %.0f" % self.clk.get_fps(),
            "|",
            mode or "—",
            "Conf %.2f" % lane_conf,
            "P2:%s" % (phase_p2_case or "—"),
            "P3:%s" % (phase_p3_case or "—"),
        ]
        fx = self.GAP
        for part in footer_parts:
            if part == "|":
                fx += 8
                continue
            r = self.ft_small.render(part, True, UXColors.TEXT_DIM)
            self.scr.blit(r, (fx, BOT + (self.FOOTER_H - r.get_height()) // 2))
            fx += r.get_width() + 12
        r = self.ft_small.render("LKA UNet+MPC", True, self.ACCENT)
        self.scr.blit(r, (scr_w - self.GAP - r.get_width(), BOT + (self.FOOTER_H - r.get_height()) // 2))

        pygame.display.flip()
        
        # Screenshot capture
        self._frame_count += 1
        if self._screenshot_dir and self._frame_count % self._screenshot_interval == 0:
            screenshot_path = Path(self._screenshot_dir) / f"dashboard_{self._frame_count:06d}.png"
            pygame.image.save(self.scr, str(screenshot_path))
        
        self.clk.tick(20)
        return True

    def _graph(self, gx, gy, gw, gh, data, col, mx=None, sym=False):
        pygame.draw.rect(self.scr, (30, 32, 36), (gx, gy, gw, gh))
        pygame.draw.rect(self.scr, self.CARD_BORDER, (gx, gy, gw, gh), 1)
        if len(data) < 2: return
        if sym:
            mv  = max(max(abs(d) for d in data), 0.01)
            pts = [(gx + j * gw // 200, gy + gh//2 - int(d/mv * gh//2)) for j, d in enumerate(data)]
            pygame.draw.line(self.scr, (50, 50, 50), (gx, gy+gh//2), (gx+gw, gy+gh//2))
        else:
            mv  = mx or max(max(data), 0.01)
            pts = [(gx + j * gw // 200, gy + gh - int(d/mv * gh)) for j, d in enumerate(data)]
        if len(pts) > 1:
            pygame.draw.lines(self.scr, col, False, pts, 2)

    def close(self):
        pygame.quit()


def _write_run_summary(record_path, n_frames, t0, safety_status):
    meta_file = record_path / "meta.csv"
    if not meta_file.exists(): return
    try:
        import csv as _csv
        rows = list(_csv.DictReader(open(meta_file, newline="")))
    except Exception as e:
        logger.warning("meta.csv read error: %s", e); return
    if not rows: return
    cte_vals   = [float(r["cte_m"])     for r in rows]
    steer_vals = [float(r["steer"])     for r in rows]
    speed_vals = [float(r["speed_kmh"]) for r in rows]
    conf_vals  = [float(r["lane_conf"]) for r in rows]
    mode_counts = dict(Counter(r["mode"] for r in rows))
    duration = time.time() - t0
    summary = {
        "record_dir": str(record_path),
        "timestamp":  datetime.now().isoformat(),
        "duration_sec": round(duration, 1),
        "frame_count":  n_frames,
        "fps": round(n_frames / duration, 1) if duration > 0 else 0,
        "cte_m":     {"mean": round(float(np.mean(np.abs(cte_vals))), 4),
                      "max":  round(float(np.max(np.abs(cte_vals))),  4)},
        "steer":     {"mean_abs": round(float(np.mean(np.abs(steer_vals))), 4)},
        "speed_kmh": {"mean": round(float(np.mean(speed_vals)), 1)},
        "lane_conf": {"min": round(min(conf_vals), 3),
                      "mean": round(float(np.mean(conf_vals)), 3)},
        "mode_counts": mode_counts,
        "safety": {"override_count": safety_status.get("override_count", 0),
                   "last_override":  safety_status.get("last_override")},
    }
    out = record_path / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary written: %s", out)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main(model_path, town="Town04", speed=TARGET_SPEED_KMH,
         record_dir=None, use_trajectory_pipeline=None, dashboard="apollo", duration_sec=0,
         auto_start_carla=True):
    
    # Cleanup old GPU processes and start CARLA if needed
    if auto_start_carla:
        cleanup_gpu_processes()
        carla_proc = start_carla_server()
    else:
        carla_proc = None

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    record_path = record_csv = record_queue = None
    record_writer_thread = None
    if record_dir:
        record_path = Path(record_dir)
        record_path.mkdir(parents=True, exist_ok=True)
        (record_path / "rgb").mkdir(exist_ok=True)
        (record_path / "mask").mkdir(exist_ok=True)
        record_csv = open(record_path / "meta.csv", "w", newline="")
        record_csv.write("frame_id,cte_m,heading_rad,curvature,steer,throttle,"
                         "brake,lane_conf,mode,speed_kmh,solver_status\n")
        record_queue = queue.Queue(maxsize=30)

        def _record_writer():
            write_count = 0
            while True:
                item = record_queue.get()
                if item is None:
                    break
                n, rgb_bgr, mask_bgr, csv_line = item
                cv2.imwrite(str(record_path / "rgb" / ("%06d.png" % n)), rgb_bgr)
                if mask_bgr is not None:
                    cv2.imwrite(str(record_path / "mask" / ("%06d.png" % n)), mask_bgr)
                record_csv.write(csv_line)
                write_count += 1
                if write_count % 20 == 0:
                    record_csv.flush()

        record_writer_thread = threading.Thread(target=_record_writer, daemon=True)
        record_writer_thread.start()
        logger.info("Recording to %s (async writer)", record_path)

    safety = SafetyOverride({
        "max_speed_kmh": float(speed) + 5.0,
        "max_steering_angle": SAFETY_MAX_STEER_RAD,
        "emergency_brake_enabled": True,
    })

    use_traj = use_trajectory_pipeline if use_trajectory_pipeline is not None else USE_TRAJECTORY_PIPELINE
    lka = LKAStep(model_path, dev, float(speed), use_trajectory_pipeline=use_traj)
    trajectory_pipeline = lka._trajectory  # for perception thread
    logger.info("Using alg.LKAStep (Perception→Fusion→Reference→MPC→Safety)")
    recovery = StuckRecovery()
    use_apollo = (dashboard == "apollo")
    dash = None
    apollo_fig = apollo_axes = apollo_state = None
    plt = None
    if use_apollo:
        try:
            import matplotlib
            for backend in ("TkAgg", "Qt5Agg", "GTK4Agg", "GTK3Agg", "WXAgg"):
                try:
                    matplotlib.use(backend)
                    import matplotlib.pyplot as _plt
                    plt = _plt
                    break
                except Exception:
                    continue
            if plt is None:
                raise RuntimeError("No GUI backend available (install python3-tk or PyQt5)")
            from gui.apollo_dashboard import build_figure, render_frame, SimState
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                logger.warning("DISPLAY not set — Apollo dashboard may not appear. Run with: export DISPLAY=:0")
            apollo_fig, *apollo_axes = build_figure()
            apollo_state = SimState()
            plt.ion()
            plt.show(block=False)
            logger.info("Using Apollo-style dashboard (wp70)")
        except Exception as e:
            logger.warning("Apollo dashboard failed (%s), falling back to Pygame", e)
            use_apollo = False
            apollo_fig = apollo_axes = apollo_state = None
    if not use_apollo:
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            logger.warning("DISPLAY not set — dashboard may not appear. Run with: export DISPLAY=:0")
        # Create screenshot directory for this run
        screenshot_dir = f"logs/screenshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dash = Dashboard(screenshot_dir=screenshot_dir)

    # ── CARLA setup ──────────────────────────────────────────────────────
    cli = carla.Client("localhost", 2000)
    cli.set_timeout(60)
    w = cli.get_world()
    if w.get_map().name.split("/")[-1] != town:
        logger.info("Loading %s...", town)
        w = cli.load_world(town)
    cleanup(w)
    st_w = w.get_settings()
    st_w.synchronous_mode = False
    st_w.fixed_delta_seconds = None
    w.apply_settings(st_w)

    bp  = w.get_blueprint_library()
    veh = None
    for sp in w.get_map().get_spawn_points()[:30]:
        veh = w.try_spawn_actor(bp.find("vehicle.tesla.model3"), sp)
        if veh: break
    if not veh:
        raise RuntimeError("No spawn point available")
    logger.info("Vehicle id=%d", veh.id)

    cq  = queue.Queue(maxsize=2)
    cbp = bp.find("sensor.camera.rgb")
    cbp.set_attribute("image_size_x", str(CAM_W))
    cbp.set_attribute("image_size_y", str(CAM_H))
    cbp.set_attribute("fov", str(int(CAM_FOV_DEG)))
    cam = w.spawn_actor(cbp,
        carla.Transform(carla.Location(x=1.5, z=2.0), carla.Rotation(pitch=-8)),
        attach_to=veh)
    cam.listen(cq.put)
    cmap = w.get_map()

    logger.info("Warmup 3s...")
    time.sleep(3)
    while not cq.empty():
        try: cq.get_nowait()
        except queue.Empty: break

    logger.info("Driving started! (UNet+WP+LSTM+MPC)")
    n = 0
    t0 = time.time()
    stale_count = 0  # frames where perception was stale → used fallback
    last_img = None
    # Cache for dashboard: refresh waypoint_path / all_lanes_paths every N frames to reduce cost
    _display_wp_path_cache: Optional[list] = None
    _display_all_lanes_cache: Optional[tuple] = None
    target_v = speed / 3.6
    prev_th  = 0.0; prev_steer_carla = 0.0
    last_state = None
    last_lane_overlay = None
    last_bev = None
    out = None
    age = 999.0

    # Fix 3: optional perception thread (30Hz) so control can run at 10Hz
    # Atomic snapshot to avoid torn read: writer assigns one reference, reader copies reference under lock
    @dataclass(frozen=False)
    class PerceptionSnapshot:
        ts: float
        out: Any
        rgb: Optional[np.ndarray]
        perception_ms: float = 0.0  # time for trajectory_pipeline.process()

    _shared_snapshot: Optional[PerceptionSnapshot] = None
    _perception_lock = threading.Lock()
    _stop_perception = threading.Event()
    _perception_frame_count = [0]  # mutable so worker can increment

    def _perception_worker():
        nonlocal _shared_snapshot
        while not _stop_perception.is_set():
            try:
                img = cq.get(timeout=0.25)
            except queue.Empty:
                continue
            raw = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))
            rgb = raw[:, :, 2::-1].copy()
            if trajectory_pipeline is None:
                continue
            _perception_frame_count[0] += 1
            if PERCEPTION_SKIP_FRAME > 1 and _perception_frame_count[0] % PERCEPTION_SKIP_FRAME != 0:
                continue  # keep previous snapshot, reduce load
            try:
                t0 = time.perf_counter()
                out = trajectory_pipeline.process(rgb, world=w, vehicle=veh)
                perception_ms = (time.perf_counter() - t0) * 1000.0
                snap = PerceptionSnapshot(ts=time.time(), out=out, rgb=rgb, perception_ms=perception_ms)
                with _perception_lock:
                    _shared_snapshot = snap
            except Exception as e:
                logger.warning("Perception worker failed: %s", e, exc_info=True)

    if USE_PERCEPTION_THREAD and trajectory_pipeline is not None:
        _perception_thread = threading.Thread(target=_perception_worker, daemon=True)
        _perception_thread.start()
        logger.info("Perception thread started (control at %d Hz)", CONTROL_HZ)

    try:
        while True:
            if USE_PERCEPTION_THREAD and trajectory_pipeline is not None:
                time.sleep(1.0 / CONTROL_HZ)
                vel = veh.get_velocity()
                spd_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                wps = get_waypoints(veh, cmap)
                wp_state = waypoints_to_cte_heading(veh.get_transform(), wps) if wps else None
                waypoint_path = waypoints_to_path(veh.get_transform(), wps) if wps else None
                wp_lookahead_curv = waypoint_lookahead_curvature(wps, lookahead_m=50.0) if wps else 0.0

                with _perception_lock:
                    snap = _shared_snapshot
                if snap is not None:
                    ts, out, rgb = snap.ts, snap.out, snap.rgb
                    perception_ms = getattr(snap, "perception_ms", 0.0)
                else:
                    ts, out, rgb = 0.0, None, None
                    perception_ms = 0.0
                age = time.time() - ts if out is not None else 999.0
                use_fallback = out is None or age > STALE_PERCEPTION_S
                if rgb is None:
                    rgb = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            else:
                # ── Sequential: ดึงภาพล่าสุด ──────────────────────────────────────
                perception_ms = 0.0
                use_fallback = False
                img = None
                try:
                    while True:
                        img = cq.get_nowait()
                except queue.Empty:
                    pass
                if img is None:
                    if last_img is None:
                        time.sleep(0.01); continue
                    img = last_img
                last_img = img

                raw = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))
                rgb = raw[:, :, 2::-1].copy()   # BGRA→RGB

                vel = veh.get_velocity()
                spd_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                wps = get_waypoints(veh, cmap)
                wp_state = waypoints_to_cte_heading(veh.get_transform(), wps) if wps else None
                waypoint_path = waypoints_to_path(veh.get_transform(), wps) if wps else None
                wp_lookahead_curv = waypoint_lookahead_curvature(wps, lookahead_m=50.0) if wps else 0.0

            # ── 3. Algorithm step (alg layer) or recovery ──────────────────
            rec = recovery.update(spd_ms, prev_th)
            state = None
            mpc_ms = 0.0
            if rec is not None:
                s_val, th, br, rev = rec
                s_val, th, br = safety.apply_safety_override({"velocity": spd_ms}, s_val, th, br)
                prev_steer_carla = s_val
                prev_th = th
                veh.apply_control(carla.VehicleControl(
                    steer=float(s_val), throttle=float(th), brake=float(br), reverse=rev))
                state = last_state
            else:
                if recovery.just_recovered:
                    lka.reset()
                if USE_PERCEPTION_THREAD and trajectory_pipeline is not None:
                    trajectory_out = out if (out is not None and age <= STALE_PERCEPTION_S) else make_fallback_trajectory_out(target_v, last_lane_overlay, last_bev, CAM_H, CAM_W)
                else:
                    trajectory_out = None
                if use_fallback:
                    stale_count += 1
                t_mpc0 = time.perf_counter()
                s_val, th, br, state = lka.step(rgb, spd_ms, wp_state, prev_steer_carla, prev_th, trajectory_out=trajectory_out, waypoint_path=waypoint_path, wp_lookahead_curv=wp_lookahead_curv)
                mpc_ms = (time.perf_counter() - t_mpc0) * 1000.0
                prev_steer_carla, prev_th = s_val, th
                veh.apply_control(carla.VehicleControl(steer=float(s_val), throttle=float(th), brake=float(br)))
                last_lane_overlay = state.lane_overlay
                last_bev = state.bev_window_vis
                last_state = state

            if state is None:
                state = last_state
            if state is None:
                time.sleep(MAIN_LOOP_SLEEP_S)
                continue

            # ── Traffic obstacles + Dashboard ─────────────────────────────
            t_draw0 = time.perf_counter()
            do_draw = (n % DASHBOARD_DRAW_EVERY_N == 0)
            obstacles = get_traffic_obstacles(w, exclude_actor_id=veh.id) if do_draw else []
            if do_draw and use_apollo and apollo_state is not None:
                # Simple curvature-based direction
                if state.curvature > 0.012:
                    road_sym = "TURN_RIGHT"
                elif state.curvature < -0.012:
                    road_sym = "TURN_LEFT"
                else:
                    road_sym = "STRAIGHT"
                apollo_state.update_from_pipeline(
                    t=time.time() - t0,
                    speed_kmh=state.speed_kmh,
                    target_v_kmh=speed,
                    cte_m=state.cte_m,
                    heading_err_rad=state.heading_rad,
                    steer=state.steer,
                    throttle=state.throttle,
                    brake=state.brake,
                    unet_conf=state.lane_conf,
                    unet_ms=perception_ms if USE_PERCEPTION_THREAD else 0.0,
                    mpc_ms=mpc_ms,
                    mpc_converged=True,
                    mask_px=int(state.lane_conf * 5000) if state.lane_conf else 0,
                    rgb=state.lane_overlay if state.lane_overlay is not None else state.rgb,
                    mode=state.mode,
                    road_symbol=road_sym,
                )
                render_frame(apollo_fig, apollo_axes, apollo_state)
                apollo_fig.canvas.draw()
                apollo_fig.canvas.flush_events()
                if not plt.fignum_exists(apollo_fig.number):
                    break
            elif do_draw and dash is not None:
                # PART 5: ใช้ path ชุดเดียวกับที่ step ใช้ (control = visualization)
                ref_path_display = (
                    state.reference_path
                    if (getattr(state, "reference_path", None) and len(state.reference_path) >= 2)
                    else get_reference_path(
                        state.cte_m or 0.0, state.heading_rad or 0.0, state.curvature or 0.0,
                        lookahead_m=REF_PATH_DISPLAY_LOOKAHEAD_M, num_pts=REF_PATH_DISPLAY_NUM_PTS,
                    )
                )
                # Refresh display path cache every N frames to reduce get_waypoints_all_lanes cost
                if _display_all_lanes_cache is None or n % DISPLAY_PATH_CACHE_INTERVAL == 0:
                    wps_display = get_waypoints(veh, cmap)
                    _display_wp_path_cache = waypoints_to_path(veh.get_transform(), wps_display) if wps_display else None
                    _display_all_lanes_cache = get_waypoints_all_lanes(veh, cmap, max_s_m=PLANNING_VIEW_RADIUS_M)
                waypoint_path = _display_wp_path_cache
                all_lanes_paths = _display_all_lanes_cache if _display_all_lanes_cache else None
                if not dash.draw(state.rgb, state.speed_kmh, state.steer, state.throttle, state.brake,
                                state.cte_m, state.heading_rad, state.curvature, state.mode,
                                lane_conf=state.lane_conf,
                                reference_path=ref_path_display,
                                waypoint_path=waypoint_path,
                                all_lanes_paths=all_lanes_paths if (all_lanes_paths and any(all_lanes_paths)) else None,
                                geometry_valid=state.geometry_valid,
                                lane_overlay=state.lane_overlay,
                                mask=state.mask,
                                bev_window_vis=state.bev_window_vis,
                                left_px_img=state.left_px_img,
                                right_px_img=state.right_px_img,
                                tracked=state.tracked,
                                used_completion=state.used_completion,
                                road_symbol_result=None,
                                obstacles=obstacles,
                                phase_p2_case=getattr(state, "phase_p2_case", "none"),
                                phase_p3_case=getattr(state, "phase_p3_case", "none"),
                                # Peter Moran visualization
                                bev_binary=getattr(state, "bev_binary", None),
                                raw_windows_left=getattr(state, "raw_windows_left", None),
                                raw_windows_right=getattr(state, "raw_windows_right", None),
                                filt_windows_left=getattr(state, "filt_windows_left", None),
                                filt_windows_right=getattr(state, "filt_windows_right", None),
                                left_px_bev=getattr(state, "left_px_bev", None),
                                right_px_bev=getattr(state, "right_px_bev", None),
                                left_curvature_m=getattr(state, "left_curvature_m", 9999.0),
                                right_curvature_m=getattr(state, "right_curvature_m", 9999.0)):
                    break
            draw_ms = (time.perf_counter() - t_draw0) * 1000.0

            if record_path and record_csv is not None:
                csv_line = "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%s,%.2f,%s\n" % (
                    n, state.cte_m, state.heading_rad, state.curvature,
                    state.steer, state.throttle, state.brake,
                    state.lane_conf, state.mode, state.speed_kmh,
                    getattr(state, "solver_status", "Solve_Succeeded"))
                if record_queue is not None:
                    mask_bgr = cv2.cvtColor(state.lane_overlay, cv2.COLOR_RGB2BGR) if state.lane_overlay is not None else None
                    try:
                        record_queue.put_nowait((n, state.rgb[:, :, ::-1].copy(), mask_bgr, csv_line))
                    except queue.Full:
                        pass
                else:
                    cv2.imwrite(str(record_path / "rgb" / ("%06d.png" % n)), state.rgb[:, :, ::-1])
                    if state.lane_overlay is not None:
                        cv2.imwrite(str(record_path / "mask" / ("%06d.png" % n)),
                                    cv2.cvtColor(state.lane_overlay, cv2.COLOR_RGB2BGR))
                    record_csv.write(csv_line)
                    if n % 20 == 0:
                        record_csv.flush()

            n += 1
            # Stop after duration_sec if set (for automated test)
            if duration_sec > 0 and (time.time() - t0) >= duration_sec:
                logger.info("Duration %.0fs reached, stopping.", duration_sec)
                break
            if n % 200 == 0:
                elapsed = time.time() - t0
                # Phase report P1..P5 (back-to-basic: จรครบ P5 = จับเลนและตีเส้นแบ่ง phase ครบ)
                p1 = getattr(state, "phase_p1_ok", False)
                p2 = getattr(state, "phase_p2_ok", False)
                p3 = getattr(state, "phase_p3_ok", False)
                p4 = getattr(state, "phase_p4_ok", False)
                p5 = getattr(state, "phase_p5_ok", False)
                phase_str = " ".join(
                    "P%d%s" % (i, "✓" if p else "✗")
                    for i, p in enumerate([p1, p2, p3, p4, p5], 1)
                )
                all_p5 = p1 and p2 and p3 and p4 and p5
                logger.info(
                    "f=%d spd=%.1f st=%+.3f cte=%+.3fm hd=%+.1fdeg crv=%+.5f mode=%s fps=%.0f",
                    n, state.speed_kmh, state.steer, state.cte_m, math.degrees(state.heading_rad),
                    state.curvature, state.mode, n / max(elapsed, 1))
                logger.info("  phases: %s  %s", phase_str, "P1–P5 ครบ" if all_p5 else "รอจรครบ P5")
                # Diagnosis: timing and stale (perception not in time → fallback)
                if USE_PERCEPTION_THREAD:
                    logger.info(
                        "  timing: perception=%.0fms mpc=%.0fms draw=%.0fms | stale_frames=%d (%.0f%%)",
                        perception_ms, mpc_ms, draw_ms, stale_count, 100.0 * stale_count / max(n, 1))

            time.sleep(MAIN_LOOP_SLEEP_S)

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        _stop_perception.set()
        # Drain async record queue and stop writer thread before closing CSV
        if record_queue is not None and record_writer_thread is not None:
            try:
                record_queue.put(None, timeout=2.0)
                record_writer_thread.join(timeout=5.0)
            except Exception as e:
                logger.warning("Record writer shutdown: %s", e)
        if record_csv:
            record_csv.flush()
            record_csv.close()
        if record_path:
            _write_run_summary(record_path, n, t0, safety.get_safety_status())
        cam.destroy()
        veh.destroy()
        if dash is not None:
            dash.close()
        elif apollo_fig is not None:
            import matplotlib.pyplot as plt
            plt.close(apollo_fig)
        st_s = safety.get_safety_status()
        logger.info("Done: %d frames %.0fs | ADAS overrides: %d (%s)",
                    n, time.time() - t0,
                    st_s["override_count"], st_s["last_override"] or "none")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=str(Path(__file__).resolve().parent / "model" / "lane_unet_final.pth"),
                    help="U-Net .pth model path")
    ap.add_argument("--town",  default="Town04")
    ap.add_argument("--speed", type=float, default=25.0)
    ap.add_argument("--record-dir", default=None)
    ap.add_argument("--record", action="store_true",
                    help="Auto-save to logs/run_YYYYMMDD_HHMMSS/")
    ap.add_argument("--legacy-perception", action="store_true",
                    help="Use legacy row-wise perception (no BEV+sliding window)")
    ap.add_argument("--dashboard", choices=("apollo", "pygame"), default="apollo",
                    help="Dashboard UI: apollo (default) or pygame")
    ap.add_argument("--duration", type=float, default=0,
                    help="Stop after N seconds (0 = run until Ctrl+C). Use with --record for eval.")
    ap.add_argument("--no-auto-carla", action="store_true",
                    help="Skip automatic GPU cleanup and CARLA server start")
    a = ap.parse_args()

    record_dir = a.record_dir
    if a.record and record_dir is None:
        record_dir = "logs/run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(record_dir).parent.mkdir(parents=True, exist_ok=True)

    main(a.model, a.town, a.speed,
         record_dir=record_dir,
         use_trajectory_pipeline=not a.legacy_perception,
         dashboard=a.dashboard,
         duration_sec=float(a.duration),
         auto_start_carla=not a.no_auto_carla)