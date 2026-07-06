#!/usr/bin/env python3
"""
รันเทส Phase P1–P5 (Lane detection pipeline).

เก็บสถิติ phase_p1_ok .. phase_p5_ok ต่อ frame แล้วรายงาน pass rate.
ใช้เมื่อ: ตรวจว่า pipeline จับเลนและตีเส้นแบ่ง phase ครบก่อนไป tune MPC.

Usage:
  # รันกับ CARLA (ต้องเปิด CARLA server ก่อน) — เก็บสถิติ 200 frames หรือ 30 วินาที
  python scripts/run_test_p1_p5.py --town Town04 --frames 200
  python scripts/run_test_p1_p5.py --town Town04 --duration 30

  # ไม่ส่งคำสั่งขับ (รถนิ่ง) — แค่รัน perception + step แล้วเก็บ P1–P5
  python scripts/run_test_p1_p5.py --town Town04 --frames 100 --no-drive

  # เปิด GUI (Pygame dashboard) ขณะรันเทส — เห็นภาพกล้อง BEV และ P1–P5 แบบ real-time
  python scripts/run_test_p1_p5.py --town Town04 --duration 30 --dashboard

  # รันจากโฟลเดอร์ภาพ (ไม่ใช้ CARLA)
  python scripts/run_test_p1_p5.py --image-dir path/to/images --frames 50

  # เกณฑ์ผ่าน: แต่ละ phase ต้องผ่าน >= 80% ของ frames (ปรับได้ด้วย --threshold)
  python scripts/run_test_p1_p5.py --town Town04 --frames 150 --threshold 0.75
"""
import argparse
import logging
import math
import queue
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CARLA
try:
    import carla
except ImportError:
    for _p in (ROOT.parent / "PythonAPI", ROOT / ".carla_py"):
        if _p.exists() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import carla

import torch

from carla_input_output import get_waypoints, waypoints_to_cte_heading
from config import CAM_W, CAM_H, USE_TRAJECTORY_PIPELINE
from algorithms import LKAStep


def run_carla(
    client: carla.Client,
    town: str,
    model_path: str,
    device: torch.device,
    speed_kmh: float,
    max_frames: int,
    max_duration_s: float,
    no_drive: bool,
    threshold: float,
    show_dashboard: bool = False,
) -> Tuple[List[bool], List[bool], List[bool], List[bool], List[bool], List[str], List[str], int, float]:
    """Run with CARLA; return (p1..p5_list, p2_case_list, p3_case_list, n_frames, duration_s)."""
    world = client.load_world(town)
    m = world.get_map()
    spawns = m.get_spawn_points()
    if not spawns:
        raise RuntimeError("No spawn points")
    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    v = world.spawn_actor(bp, spawns[0])
    try:
        # Camera
        cq = queue.Queue()
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM_W))
        cam_bp.set_attribute("image_size_y", str(CAM_H))
        cam = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15)),
            attach_to=v,
        )
        cam.listen(cq.put)
        try:
            time.sleep(1.0)
            while not cq.empty():
                try:
                    cq.get_nowait()
                except queue.Empty:
                    break
        except Exception:
            pass

        lka = LKAStep(model_path, device, speed_kmh, use_trajectory_pipeline=USE_TRAJECTORY_PIPELINE)
        prev_steer, prev_th = 0.0, 0.0
        p1_list, p2_list, p3_list, p4_list, p5_list = [], [], [], [], []
        p2_case_list, p3_case_list = [], []
        n = 0
        t0 = time.time()
        last_img = None

        dash = None
        if show_dashboard:
            try:
                import os
                if "DISPLAY" not in os.environ:
                    os.environ.setdefault("DISPLAY", ":0")
                from gui.dashboard import Dashboard
                dash = Dashboard()
                logger.info("Dashboard opened (Pygame) — ปิดหน้าต่างหรือกด ESC เพื่อจบเทสก่อนครบ frames")
            except Exception as e:
                logger.warning("Dashboard init failed (%s), continuing without GUI", e)
                dash = None

        while n < max_frames and (time.time() - t0) < max_duration_s:
            img = None
            try:
                img = cq.get(timeout=0.5)
            except queue.Empty:
                pass
            if img is None:
                if last_img is None:
                    time.sleep(0.02)
                    continue
                img = last_img
            last_img = img
            raw = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))
            rgb = raw[:, :, 2::-1].copy()

            if no_drive:
                spd_ms = 0.0
                wp_state = None
            else:
                vel = v.get_velocity()
                spd_ms = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                wps = get_waypoints(v, m)
                wp_state = waypoints_to_cte_heading(v.get_transform(), wps) if wps else None

            steer, throttle, brake, state = lka.step(rgb, spd_ms, wp_state, prev_steer, prev_th, trajectory_out=None)
            if not no_drive:
                prev_steer, prev_th = steer, throttle
                v.apply_control(carla.VehicleControl(
                    steer=float(steer), throttle=float(throttle), brake=float(brake)
                ))

            if dash is not None:
                if not dash.draw(
                    state.rgb, state.speed_kmh, state.steer, state.throttle, state.brake,
                    state.cte_m, state.heading_rad, state.curvature, state.mode,
                    lane_conf=state.lane_conf,
                    reference_path=state.reference_path,
                    geometry_valid=state.geometry_valid,
                    lane_overlay=state.lane_overlay,
                    mask=state.mask,
                    bev_window_vis=state.bev_window_vis,
                    left_px_img=getattr(state, "left_px_img", None),
                    right_px_img=getattr(state, "right_px_img", None),
                    tracked=state.tracked,
                    used_completion=state.used_completion,
                    road_symbol_result=state.road_symbol_result,
                    obstacles=None,
                    phase_p2_case=getattr(state, "phase_p2_case", "none"),
                    phase_p3_case=getattr(state, "phase_p3_case", "none"),
                ):
                    break
                dash.clk.tick(15)

            p1_list.append(getattr(state, "phase_p1_ok", False))
            p2_list.append(getattr(state, "phase_p2_ok", False))
            p3_list.append(getattr(state, "phase_p3_ok", False))
            p4_list.append(getattr(state, "phase_p4_ok", False))
            p5_list.append(getattr(state, "phase_p5_ok", False))
            p2_case_list.append(getattr(state, "phase_p2_case", "none"))
            p3_case_list.append(getattr(state, "phase_p3_case", "none"))
            n += 1
            if n % 50 == 0:
                s1, s2, s3, s4, s5 = (
                    sum(p1_list[-50:]),
                    sum(p2_list[-50:]),
                    sum(p3_list[-50:]),
                    sum(p4_list[-50:]),
                    sum(p5_list[-50:]),
                )
                logger.info("  frames: %d  P1–P5: %d/50, %d/50, %d/50, %d/50, %d/50", n, s1, s2, s3, s4, s5)

        duration = time.time() - t0
        return p1_list, p2_list, p3_list, p4_list, p5_list, p2_case_list, p3_case_list, n, duration
    finally:
        cam.destroy()
        v.destroy()


def run_image_dir(
    image_dir: Path,
    model_path: str,
    device: torch.device,
    speed_kmh: float,
    max_frames: int,
    threshold: float,
) -> Tuple[List[bool], List[bool], List[bool], List[bool], List[bool], List[str], List[str], int, float]:
    """Run on images from a folder; return same as run_carla."""
    exts = {".png", ".jpg", ".jpeg"}
    images = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in exts])[:max_frames]
    if not images:
        raise FileNotFoundError(f"No images in {image_dir}")

    import cv2

    lka = LKAStep(model_path, device, speed_kmh, use_trajectory_pipeline=USE_TRAJECTORY_PIPELINE)
    prev_steer, prev_th = 0.0, 0.0
    p1_list, p2_list, p3_list, p4_list, p5_list = [], [], [], [], []
    p2_case_list, p3_case_list = [], []
    t0 = time.time()

    for i, path in enumerate(images):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (CAM_H, CAM_W):
            rgb = cv2.resize(rgb, (CAM_W, CAM_H))
        spd_ms = 0.0
        wp_state = None
        _, _, _, state = lka.step(rgb, spd_ms, wp_state, prev_steer, prev_th, trajectory_out=None)
        prev_steer, prev_th = state.steer, state.throttle
        p1_list.append(getattr(state, "phase_p1_ok", False))
        p2_list.append(getattr(state, "phase_p2_ok", False))
        p3_list.append(getattr(state, "phase_p3_ok", False))
        p4_list.append(getattr(state, "phase_p4_ok", False))
        p5_list.append(getattr(state, "phase_p5_ok", False))
        p2_case_list.append(getattr(state, "phase_p2_case", "none"))
        p3_case_list.append(getattr(state, "phase_p3_case", "none"))
        if (i + 1) % 20 == 0:
            logger.info("  images: %d/%d", i + 1, len(images))

    n = len(p1_list)
    duration = time.time() - t0
    return p1_list, p2_list, p3_list, p4_list, p5_list, p2_case_list, p3_case_list, n, duration


def report(
    p1_list: List[bool],
    p2_list: List[bool],
    p3_list: List[bool],
    p4_list: List[bool],
    p5_list: List[bool],
    n: int,
    duration: float,
    threshold: float,
    p2_case_list: Optional[List[str]] = None,
    p3_case_list: Optional[List[str]] = None,
) -> bool:
    """Print P1–P5 pass rates and overall PASS/FAIL. If p2_case_list/p3_case_list given, print by-case summary."""
    if n == 0:
        logger.warning("No frames collected.")
        return False
    r1 = sum(p1_list) / n
    r2 = sum(p2_list) / n
    r3 = sum(p3_list) / n
    r4 = sum(p4_list) / n
    r5 = sum(p5_list) / n
    logger.info("")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  P1–P5 TEST REPORT")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  Frames: %d  |  Duration: %.1fs  |  Threshold: %.0f%%", n, duration, threshold * 100)
    logger.info("──────────────────────────────────────────────────")
    fmt = "  {label:<32} : {pct:5.1f}%  {icon}"
    for label, r in [
        ("P1 (Lane mask + confidence)", r1),
        ("P2 (Left/right boundary)", r2),
        ("P3 (Centerline)", r3),
        ("P4 (State finite: cte/h/curv)", r4),
        ("P5 (Mode set)", r5),
    ]:
        icon = "✅" if r >= threshold else "❌"
        logger.info(fmt.format(label=label, pct=r * 100, icon=icon))
    if p2_case_list is not None and len(p2_case_list) == n:
        logger.info("──────────────────────────────────────────────────")
        logger.info("  P2 by case (count | pass rate within case)")
        for case in sorted(set(p2_case_list)):
            idx = [i for i, c in enumerate(p2_case_list) if c == case]
            cnt = len(idx)
            passed = sum(1 for i in idx if p2_list[i])
            pct = (passed / cnt * 100) if cnt else 0.0
            logger.info("    %-14s : %4d frames  pass %4d (%.1f%%)", case, cnt, passed, pct)
    if p3_case_list is not None and len(p3_case_list) == n:
        logger.info("  P3 by case (count | pass rate within case)")
        for case in sorted(set(p3_case_list)):
            idx = [i for i, c in enumerate(p3_case_list) if c == case]
            cnt = len(idx)
            passed = sum(1 for i in idx if p3_list[i])
            pct = (passed / cnt * 100) if cnt else 0.0
            logger.info("    %-14s : %4d frames  pass %4d (%.1f%%)", case, cnt, passed, pct)
    logger.info("──────────────────────────────────────────────────")
    all_ok = all(r >= threshold for r in (r1, r2, r3, r4, r5))
    logger.info("  OVERALL: %s", "PASS" if all_ok else "FAIL")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="Run P1–P5 phase test (lane detection pipeline)")
    ap.add_argument("--town", default="Town04", help="CARLA town (ignored if --image-dir)")
    ap.add_argument("--model", type=Path, default=ROOT / "model" / "lane_unet_final.pth", help="UNet .pth")
    ap.add_argument("--frames", type=int, default=200, help="Max frames to collect")
    ap.add_argument("--duration", type=float, default=0, help="Max duration (s); 0 = use --frames only")
    ap.add_argument("--no-drive", action="store_true", help="Do not apply control (vehicle static)")
    ap.add_argument("--image-dir", type=Path, default=None, help="Run on images in folder (no CARLA)")
    ap.add_argument("--threshold", type=float, default=0.80, help="Pass rate threshold per phase (0–1)")
    ap.add_argument("--speed", type=float, default=25.0, help="Target speed km/h (for LKA step)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--dashboard", action="store_true", help="Open Pygame dashboard (GUI) while testing")
    a = ap.parse_args()

    model_path = a.model if a.model.is_absolute() else (ROOT / a.model)
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        return 1
    model_path = str(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_duration = a.duration if a.duration > 0 else 1e9

    if a.image_dir is not None:
        a.image_dir = Path(a.image_dir)
        if not a.image_dir.is_dir():
            logger.error("Not a directory: %s", a.image_dir)
            return 1
        logger.info("Running P1–P5 test on image dir: %s", a.image_dir)
        p1, p2, p3, p4, p5, p2_case, p3_case, n, dur = run_image_dir(
            a.image_dir, model_path, device, a.speed, a.frames, a.threshold
        )
    else:
        logger.info("Connecting to CARLA %s %s ...", a.host, a.port)
        client = carla.Client(a.host, a.port)
        client.set_timeout(10.0)
        logger.info("Running P1–P5 test in CARLA %s (no_drive=%s, dashboard=%s)", a.town, a.no_drive, a.dashboard)
        p1, p2, p3, p4, p5, p2_case, p3_case, n, dur = run_carla(
            client, a.town, model_path, device, a.speed,
            a.frames, max_duration, a.no_drive, a.threshold,
            show_dashboard=a.dashboard,
        )

    passed = report(p1, p2, p3, p4, p5, n, dur, a.threshold, p2_case_list=p2_case, p3_case_list=p3_case)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
