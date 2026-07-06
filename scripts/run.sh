#!/bin/bash
# รัน LKA pipeline (แบบเดียวกับ carla_lstm_mpc_project)
# ต้องเปิด CARLA server ก่อน (port 2000)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# ให้ Dashboard (Pygame) แสดงได้ — ใช้เมื่อรันจาก terminal บนเครื่องที่มีจอ
[ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ] && export DISPLAY=:0

# CARLA Python API — โครงเดียวกับ carla_lstm_mpc_project
export PYTHONPATH="${ROOT}:${ROOT}/.carla_py:${ROOT}/../PythonAPI:${PYTHONPATH}"

# Python: ใช้ venv ในโปรเจกต์นี้ก่อน แล้ว fallback ไป venv ของ carla_lstm_mpc_project แล้วค่อย python3 (เหมือน lstm)
PYTHON=""
for v in "$ROOT/venv/bin/python" "$ROOT/.venv/bin/python" "$ROOT/../carla_lstm_mpc_project/venv/bin/python"; do
  if [ -x "$v" ] && "$v" -c "import numpy, cv2" 2>/dev/null; then
    PYTHON="$v"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  if python3 -c "import numpy, cv2" 2>/dev/null; then
    PYTHON="python3"
  fi
fi
if [ -z "$PYTHON" ]; then
  echo "❌ ขาด numpy หรือ opencv-python"
  echo "   pip3 install numpy opencv-python pygame torch"
  echo "   หรือใช้ venv: python3 -m venv venv && ./venv/bin/pip install numpy opencv-python pygame torch"
  exit 1
fi

MODE="${1:-classical}"
shift || true

case "$MODE" in
  classical)
    exec "$PYTHON" "$ROOT/main.py" --classical "$@"
    ;;
  unet)
    exec "$PYTHON" "$ROOT/main.py" "$@"
    ;;
  *)
    echo "Usage: $0 [classical|unet] [--town Town04] [--speed 25] ..."
    echo "  classical — edge + sliding window (ไม่ใช้ model)"
    echo "  unet      — UNet + LaneTrajectoryPipeline (ต้องมี model/lane_unet_final.pth)"
    exit 1
    ;;
esac
