#!/bin/bash

# Quick Test ADAS with New Fixes

echo "🚀 Testing ADAS with UNet + New Fixes"
echo ""
echo "Fixes:"
echo "  - MPC weights adjusted (CTE↓, Heading↑, Steer↑)"
echo "  - Steering smoothing increased (0.70 → 0.80)"
echo "  - Rate limits tightened (0.05 → 0.03)"
echo ""
echo "Expected: Steering ±20-25° (was ±32.6°)"
echo ""
echo "=========================================="
echo ""

cd /home/supawich/Desktop/CARLA_0.9.16/carla_mpc_classical
source venv/bin/activate
python adas/run.py
