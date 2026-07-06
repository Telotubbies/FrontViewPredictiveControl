#!/usr/bin/env python3
"""
CARLA MPC Lane Keeping System — entry point wrapper.

The actual implementation lives in ``src/main.py``. This thin wrapper
keeps the launch command ``python main.py`` working for CARLA scripts
and shell wrappers that expect the entry point at the repository root.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
