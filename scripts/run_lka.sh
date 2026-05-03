#!/usr/bin/env bash
# รัน LKA — จุดเข้าเดียว แยก classical / unet (แบบเดียวกับ carla_lstm_mpc_project)
# Usage: ./scripts/run_lka.sh [classical|unet] [--town Town04] ...

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/.carla_py:${ROOT}/../PythonAPI:${PYTHONPATH}"

PYTHON=""
for v in "$ROOT/venv/bin/python" "$ROOT/.venv/bin/python" "$ROOT/../carla_lstm_mpc_project/venv/bin/python"; do
  if [ -x "$v" ] && "$v" -c "import numpy, cv2" 2>/dev/null; then
    PYTHON="$v"
    break
  fi
done
[ -z "$PYTHON" ] && command -v python3 >/dev/null && PYTHON="python3"
[ -z "$PYTHON" ] && { echo "❌ ไม่พบ Python (numpy, cv2)"; exit 1; }

MODE="${1:-classical}"
shift || true

case "$MODE" in
  classical)
    exec "$PYTHON" "$SCRIPT_DIR/run_classical_mpc.py" "$@"
    ;;
  unet)
    exec "$PYTHON" "$ROOT/run_unet_mpc.py" "$@"
    ;;
  *)
    echo "Usage: $0 [classical|unet] [--town Town04] [--speed 25] ..."
    echo "  classical — edge + sliding window (ไม่ใช้ model)"
    echo "  unet      — UNet + LaneTrajectoryPipeline (ต้องมี model/lane_unet_final.pth)"
    exit 1
    ;;
esac
