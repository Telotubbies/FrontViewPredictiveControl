#!/bin/bash

# Run ADAS Test with GPU (DSUNet training stopped)

echo "=========================================="
echo "🚀 ADAS Test with UNet (GPU mode)"
echo "=========================================="
echo ""
echo "Fixes applied:"
echo "  ✅ MPC_W_CTE: 20.0 (ลดจาก 30)"
echo "  ✅ MPC_W_HEADING: 50.0 (เพิ่มจาก 40)"
echo "  ✅ MPC_W_STEER: 200.0 (เพิ่มจาก 150)"
echo "  ✅ MPC_W_STEER_RATE: 400.0 (เพิ่มจาก 300)"
echo "  ✅ MPC_W_STEER_JERK: 200.0 (เพิ่มจาก 150)"
echo "  ✅ STEER_SMOOTH_ALPHA: 0.80 (เพิ่มจาก 0.70)"
echo "  ✅ Rate limits: 0.03/0.02"
echo ""
echo "Expected: Steering ±20-25° (was ±32.6°)"
echo ""
echo "=========================================="
echo ""

cd /home/supawich/Desktop/CARLA_0.9.16/carla_mpc_classical

# Use venv with torch
source ../carla_lstm_mpc_project/venv/bin/activate

# Use GPU (no FORCE_CPU)
unset FORCE_CPU

# Run ADAS
python adas/run.py
