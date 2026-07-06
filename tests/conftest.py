"""
Pytest configuration — ensures src/ and project root are on sys.path
so that imports like ``from control.lane_mpc import LaneMPC`` work
without requiring ``pip install -e .`` first.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
