#!/bin/bash
# เปิดใช้ venv ของ carla_lstm_mpc_project (มี numpy, torch, carla ฯลฯ)
# Usage: source activate_venv.sh   หรือ  . activate_venv.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$(cd "$SCRIPT_DIR/../carla_lstm_mpc_project/venv" && pwd)"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
  echo "venv activated: $VENV_DIR"
else
  echo "venv not found: $VENV_DIR" >&2
  return 1 2>/dev/null || exit 1
fi
