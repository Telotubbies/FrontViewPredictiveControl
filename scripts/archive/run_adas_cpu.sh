#!/bin/bash

# Run ADAS with CPU (avoid CUDA OOM when DSUNet training)

echo "🚀 Running ADAS with UNet on CPU"
echo ""
echo "Note: Using CPU because DSUNet is training on GPU"
echo ""
echo "Fixes applied:"
echo "  ✅ MPC_W_CTE: 20.0 (ลดจาก 30)"
echo "  ✅ MPC_W_HEADING: 50.0 (เพิ่มจาก 40)"
echo "  ✅ MPC_W_STEER: 200.0 (เพิ่มจาก 150)"
echo "  ✅ MPC_W_STEER_RATE: 400.0 (เพิ่มจาก 300)"
echo "  ✅ MPC_W_STEER_JERK: 200.0 (เพิ่มจาก 150)"
echo "  ✅ STEER_SMOOTH_ALPHA: 0.80 (เพิ่มจาก 0.70)"
echo "  ✅ Rate limits: 0.03/0.02 (ลดจาก 0.05/0.03)"
echo ""
echo "Expected: Steering ±20-25° (was ±32.6°)"
echo ""
echo "=========================================="
echo ""

cd /home/supawich/Desktop/CARLA_0.9.16/carla_mpc_classical

# Force CPU usage
export FORCE_CPU=1

# Run ADAS
python3 adas/run.py
