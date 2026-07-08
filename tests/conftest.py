"""
Pytest configuration — ensures src/ and project root are on sys.path
so that imports like ``from control.lane_mpc import LaneMPC`` work
without requiring ``pip install -e .`` first.
"""
import os
import sys
from pathlib import Path

# Use dummy SDL video driver so pygame works headless in CI.
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
