#!/usr/bin/env python3
"""
ADAS (Advanced Driver Assistance) — ทดสอบทั้งระบบ LKA

คิดว่าเป็นระบบ ADAS แล้วทำทั้งหมด:
  - Pre-flight: CARLA, model, perception/control/safety โหลดได้
  - Run: Camera → U-Net lane → Waypoint fusion → MPC → SafetyOverride → Control
  - Log: frames, ADAS overrides, mode
"""

import sys
import socket
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("adas_full")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "model" / "lane_unet_final.pth"
CARLA_PORT = 2000


def check_carla(port: int = CARLA_PORT) -> bool:
    """ตรวจว่า CARLA server พร้อมรับ connection หรือไม่"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        r = s.connect_ex(("127.0.0.1", port))
        s.close()
        return r == 0
    except Exception:
        return False


def check_model(path: Path) -> bool:
    """ตรวจว่าไฟล์ model มีอยู่"""
    return path.is_file()


def check_imports() -> bool:
    """ตรวจว่า perception, control, safety โหลดได้"""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from perception.lane_detector import LaneDetector
        from control.lane_mpc import LaneMPC, MPCConfig
        from safety.safety_override import SafetyOverride
        return True
    except Exception as e:
        logger.error("Import check failed: %s", e)
        return False


def preflight(model_path: Path, carla: bool = True) -> bool:
    """Pre-flight checks สำหรับ ADAS run. คืน True ถ้าพร้อมรัน."""
    ok = True
    if not check_model(model_path):
        logger.error("Model not found: %s", model_path)
        ok = False
    else:
        logger.info("OK model: %s", model_path)

    if not check_imports():
        logger.error("Import check failed")
        ok = False
    else:
        logger.info("OK imports (perception, control, safety)")

    if carla and not check_carla():
        logger.error("CARLA not reachable at 127.0.0.1:%s — start server first", CARLA_PORT)
        ok = False
    elif carla:
        logger.info("OK CARLA 127.0.0.1:%s", CARLA_PORT)

    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser(description="ADAS full run: pre-flight + LKA pipeline")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="U-Net .pth path")
    ap.add_argument("--town", default="Town04")
    ap.add_argument("--speed", type=float, default=25.0)
    ap.add_argument("--record-dir", default=None, help="Record rgb/mask + meta.csv ไปโฟลเดอร์นี้")
    ap.add_argument("--record", action="store_true",
                    help="บันทึกผลที่ logs/run_YYYYMMDD_HHMMSS + summary.json (ส่ง path นี้ให้แชทเพื่อดูสถานการณ์)")
    ap.add_argument("--check-only", action="store_true", help="Only run pre-flight, do not start sim")
    ap.add_argument("--no-carla-check", action="store_true", help="Skip CARLA port check (e.g. run sim elsewhere)")
    ap.add_argument("--dashboard", default="pygame", choices=("pygame", "apollo"),
                    help="Dashboard แสดงภาพกล้อง: pygame (เห็นภาพชัด) หรือ apollo")
    a = ap.parse_args()

    model_path = a.model if isinstance(a.model, Path) else Path(a.model)
    if not preflight(model_path, carla=not a.no_carla_check):
        logger.error("Pre-flight failed. Fix and retry.")
        sys.exit(1)

    if a.check_only:
        logger.info("Pre-flight passed. Exiting (--check-only).")
        sys.exit(0)

    record_dir = a.record_dir
    if a.record and record_dir is None:
        record_dir = "logs/run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(record_dir).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Recording to %s (summary.json will be written on exit)", record_dir)

    logger.info("Starting ADAS full run (LKA pipeline + SafetyOverride)...")
    from run_unet_mpc import main as run_main
    run_main(str(model_path), a.town, a.speed, record_dir=record_dir, dashboard=a.dashboard)


if __name__ == "__main__":
    main()
