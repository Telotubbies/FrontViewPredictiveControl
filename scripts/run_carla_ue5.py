#!/usr/bin/env python3
"""
Entry point สำหรับรัน FVPC กับ CARLA UE5 (0.10.0).

ความแตกต่างจาก main.py หลัก:
- โหลด CARLA map จาก config (CARLA_MAP, default Town10) ก่อนเริ่ม pipeline
- รับ --host / --port สำหรับ CARLA server (default localhost:2000)
- ใช้ pip-installed carla>=0.10.0 (ไม่ต้องใช้ local egg)

Usage:
    python scripts/run_carla_ue5.py
    python scripts/run_carla_ue5.py --host 192.168.1.10 --port 2000
    python scripts/run_carla_ue5.py --classical --no-gui
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_src = ROOT / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# CARLA Python API — pip-installed (carla>=0.10.0)
# Optional at import time; guard at entry point (main / load_town).
try:
    import carla
except ImportError:
    carla = None  # type: ignore[assignment]

from config import CARLA_MAP, TARGET_SPEED_KMH

logger = logging.getLogger(__name__)


def load_town(host: str, port: int, town: str) -> bool:
    """เชื่อมต่อ CARLA และโหลด map ที่กำหนด (เช่น Town10)."""
    if carla is None:
        raise ImportError(
            "CARLA Python API not installed. "
            "Install with: pip install carla-0.10.0-cp311-cp311-win_amd64.whl "
            "from https://github.com/carla-simulator/carla/releases"
        )
    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.get_world()
    current = world.get_map().name.split("/")[-1]
    if current != town:
        logger.info(f"Loading world: {town} (current: {current})")
        client.load_world(town)
    else:
        logger.info(f"World already loaded: {town}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run FVPC with CARLA UE5")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--town", type=str, default=CARLA_MAP,
                       help=f"CARLA map name (default: {CARLA_MAP})")
    parser.add_argument("--model", type=str, default="model/lane_unet_final.pth",
                       help="Path to UNet model file")
    parser.add_argument("--speed", type=float, default=TARGET_SPEED_KMH,
                       help="Target speed in km/h")
    parser.add_argument("--classical", action="store_true",
                       help="Use classical lane detection instead of UNet")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI dashboard")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--metrics-dir", type=str, default="metrics_output",
                       help="Directory to save metrics CSV/JSON output")
    args = parser.parse_args()

    if not (1 <= args.port <= 65535):
        parser.error(f"Port must be between 1 and 65535, got {args.port}")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # 1. โหลด CARLA map ก่อน (UE5: Town10)
    logger.info(f"Connecting to CARLA UE5 at {args.host}:{args.port}")
    if not load_town(args.host, args.port, args.town):
        return 1

    # 2. รัน FVPC pipeline (CARLAMPCSystem เชื่อมต่อ localhost:2000 ตามค่าเดิม)
    from main import CARLAMPCSystem

    system = CARLAMPCSystem(
        model_path=args.model,
        target_speed_kmh=args.speed,
        use_classical=args.classical,
        no_gui=args.no_gui,
        metrics_dir=args.metrics_dir,
    )
    return system.run()


if __name__ == "__main__":
    sys.exit(main())
