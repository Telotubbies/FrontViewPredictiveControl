#!/usr/bin/env python3
"""
จุดเข้า ADAS — รัน LKA pipeline (แบบเดียวกับ carla_lstm_mpc_project).

ใช้จาก CARLA_0.9.16:
  python carla_mpc_classical/adas/run.py --town Town04 --speed 25

หรือจากโฟลเดอร์ carla_mpc_classical:
  ./scripts/run.sh unet --town Town04
  python adas/run.py --town Town04
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# CARLA Python API (เหมือน run_ufld_mpc ใน lstm)
try:
    import carla
except ImportError:
    for _p in (ROOT.parent / "PythonAPI", ROOT / ".carla_py"):
        if _p.exists() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import carla

from run_adas_full import main

if __name__ == "__main__":
    main()
