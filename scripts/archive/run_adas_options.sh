#!/bin/bash
# ADAS Run Options Script
# ใช้สำหรับรัน CARLA ADAS พร้อม dashboard และ options ต่างๆ

echo "=== CARLA ADAS Run Options ==="
echo ""

# Activate virtual environment
source ../carla_lstm_mpc_project/venv/bin/activate

echo "1. Pre-flight Check:"
echo "   python adas/run.py --check-only"
echo ""

echo "2. Basic Run (No Dashboard):"
echo "   python adas/run.py --town Town04 --speed 25"
echo ""

echo "3. Run with Pygame Dashboard:"
echo "   python adas/run.py --town Town04 --speed 25 --dashboard pygame"
echo ""

echo "4. Run with Apollo Dashboard:"
echo "   python adas/run.py --town Town04 --speed 25 --dashboard apollo"
echo ""

echo "5. Run with Recording:"
echo "   python adas/run.py --town Town04 --speed 25 --record"
echo ""

echo "6. Run with Custom Model:"
echo "   python adas/run.py --town Town04 --speed 25 --model model/lane_unet_final.pth"
echo ""

echo "7. Run Different Towns:"
echo "   python adas/run.py --town Town01 --speed 20"
echo "   python adas/run.py --town Town02 --speed 30"
echo "   python adas/run.py --town Town03 --speed 25"
echo "   python adas/run.py --town Town04 --speed 25"
echo "   python adas/run.py --town Town05 --speed 20"
echo ""

echo "8. Run with Speed Variations:"
echo "   python adas/run.py --town Town04 --speed 15  # Slow"
echo "   python adas/run.py --town Town04 --speed 25  # Normal"
echo "   python adas/run.py --town Town04 --speed 35  # Fast"
echo "   python adas/run.py --town Town04 --speed 50  # Very Fast"
echo ""

echo "9. Debug Options:"
echo "   python adas/run.py --town Town04 --speed 25 --record"
echo "   # Check logs/run_YYYYMMDD_HHMMSS/ for recorded data"
echo ""

echo "10. BEV Transform Test (Standalone):"
echo "    python test_bev_transform_gui.py"
echo "    # Interactive BEV transform visualization"
echo ""

echo "=== Current System Status ==="
echo "Model: model/lane_unet_final.pth"
echo "CARLA: 127.0.0.1:2000"
echo "GPU: CUDA Available"
echo "Config: USE_CLASSICAL_DETECTOR=false (UNet mode)"
echo ""

echo "=== Quick Start Commands ==="
echo "# Check system:"
echo "python adas/run.py --check-only"
echo ""
echo "# Run with dashboard:"
echo "python adas/run.py --town Town04 --speed 25 --dashboard pygame"
echo ""
echo "# Test BEV transform:"
echo "python test_bev_transform_gui.py"
echo ""

read -p "Enter option number or command to run: " choice

case $choice in
    1)
        echo "Running pre-flight check..."
        python adas/run.py --check-only
        ;;
    2)
        echo "Running basic ADAS..."
        python adas/run.py --town Town04 --speed 25
        ;;
    3)
        echo "Running ADAS with Pygame dashboard..."
        python adas/run.py --town Town04 --speed 25 --dashboard pygame
        ;;
    4)
        echo "Running ADAS with Apollo dashboard..."
        python adas/run.py --town Town04 --speed 25 --dashboard apollo
        ;;
    5)
        echo "Running ADAS with recording..."
        python adas/run.py --town Town04 --speed 25 --record
        ;;
    10)
        echo "Running BEV Transform GUI..."
        python test_bev_transform_gui.py
        ;;
    *)
        echo "Running custom command: $choice"
        eval $choice
        ;;
esac
