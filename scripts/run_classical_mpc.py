#!/usr/bin/env python3
"""
รัน LKA ด้วย classical lane detection (sliding window + poly) + LaneMPC.

ไม่ใช้ UNet: ใช้ perception.classical_lane (edge → warp → sliding window → poly)
Output: cte_m, heading_rad, curvature → control.lane_mpc
"""

import argparse
import logging
import math
import queue
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# CARLA Python API (ถ้ายังไม่มี carla ใน path)
try:
    import carla
except ImportError:
    _carla_api = ROOT.parent / "PythonAPI"
    _carla_py = ROOT / ".carla_py"
    for _p in (_carla_api, _carla_py):
        if _p.exists() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import carla
import numpy as np
import cv2

from perception.classical import ClassicalLane
from control.lane_mpc import LaneMPC, MPCConfig

CAM_W, CAM_H = 320, 240
TARGET_SPEED_MS = 25.0 / 3.6


def main():
    ap = argparse.ArgumentParser(description="Classical lane + MPC in CARLA")
    ap.add_argument("--town", default="Town04")
    ap.add_argument("--speed", type=float, default=25.0, help="Target speed km/h")
    args = ap.parse_args()
    target_ms = args.speed / 3.6

    lane = ClassicalLane(image_width=CAM_W, image_height=CAM_H)
    mpc_cfg = MPCConfig(N=10, dt=0.1, w_cte=50.0, w_heading=30.0, w_vel=5.0, w_steer_rate=200.0)
    mpc = LaneMPC(mpc_cfg)

    client = carla.Client("localhost", 2000)
    client.set_timeout(60)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != args.town:
        world = client.load_world(args.town)

    # Spawn vehicle
    bp = world.get_blueprint_library()
    vehicle = None
    for sp in world.get_map().get_spawn_points()[:30]:
        vehicle = world.try_spawn_actor(bp.find("vehicle.tesla.model3"), sp)
        if vehicle:
            break
    if not vehicle:
        sys.exit("Could not spawn vehicle")
    logger.info("Vehicle spawned.")

    # Camera
    cq = queue.Queue(maxsize=2)
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_W))
    cam_bp.set_attribute("image_size_y", str(CAM_H))
    cam_bp.set_attribute("fov", "110")
    cam = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15)),
        attach_to=vehicle,
    )
    cam.listen(cq.put)
    time.sleep(2)
    while not cq.empty():
        try:
            cq.get_nowait()
        except queue.Empty:
            break

    logger.info("Classical lane + MPC driving. Ctrl+C to stop.")
    n = 0
    last_img = None
    try:
        while True:
            img = None
            try:
                img = cq.get_nowait()
            except queue.Empty:
                pass
            if img is None and last_img is not None:
                img = last_img
            if img is None:
                time.sleep(0.01)
                continue
            last_img = img
            raw = np.frombuffer(img.raw_data, np.uint8)
            raw = raw.reshape((img.height, img.width, 4))[:, :, :3]
            bgr = raw[:, :, ::-1].copy()

            cte_m, heading_rad, curvature, valid = lane.process(bgr)
            if not valid:
                cte_m, heading_rad, curvature = 0.0, 0.0, 0.0

            vel = vehicle.get_velocity()
            spd_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            steer_rad, accel, status = mpc.solve(
                x0=0, y0=cte_m, psi0=heading_rad, v0=spd_ms,
                v_ref=target_ms, cte=cte_m, heading_err=heading_rad, curvature=curvature,
            )
            steer_carla = mpc.steer_to_carla(steer_rad)
            throttle, brake = mpc.accel_to_carla(accel)

            vehicle.apply_control(carla.VehicleControl(steer=steer_carla, throttle=throttle, brake=brake))

            vis = lane.overlay_lane_lines(bgr)
            cv2.putText(vis, "CTE: %+.2fm" % cte_m, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, "Classical+MPC", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("classical_mpc", vis)
            if cv2.waitKey(1) == 27:
                break
            n += 1
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if cam:
            cam.destroy()
        if vehicle:
            vehicle.destroy()
    logger.info("Done. Frames: %d", n)


if __name__ == "__main__":
    main()
