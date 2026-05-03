#!/bin/bash

# Test ADAS with UNet and New Fixes
# ===================================

echo "=========================================="
echo "Testing ADAS with UNet + New Fixes"
echo "=========================================="
echo ""
echo "Fixes applied:"
echo "  ✅ MPC_W_CTE: 30 → 20"
echo "  ✅ MPC_W_HEADING: 40 → 50"
echo "  ✅ MPC_W_STEER: 150 → 200"
echo "  ✅ MPC_W_STEER_RATE: 300 → 400"
echo "  ✅ MPC_W_STEER_JERK: 150 → 200"
echo "  ✅ STEER_SMOOTH_ALPHA: 0.70 → 0.80"
echo "  ✅ STEER_MAX_DELTA_PER_FRAME: 0.05 → 0.03"
echo "  ✅ STEER_MAX_DELTA_HIGH_SPEED: 0.03 → 0.02"
echo ""
echo "Expected improvements:"
echo "  - Max steering: ±32.6° → ±20-25°"
echo "  - Oscillation: -50 to -70%"
echo "  - Smoothness: +150%"
echo ""
echo "=========================================="
echo ""

# Activate venv
source venv/bin/activate

# Run ADAS
echo "Starting ADAS..."
echo ""
python adas/run.py 2>&1 | tee test_unet_fixes.log &

ADAS_PID=$!
echo "ADAS PID: $ADAS_PID"
echo ""

# Wait a bit for startup
sleep 5

echo "=========================================="
echo "Monitoring (Ctrl+C to stop)"
echo "=========================================="
echo ""

# Monitor in real-time
tail -f test_unet_fixes.log &
TAIL_PID=$!

# Wait for user to stop
wait $ADAS_PID

# Cleanup
kill $TAIL_PID 2>/dev/null

echo ""
echo "=========================================="
echo "Test completed! Analyzing results..."
echo "=========================================="
echo ""

# Analyze results
echo "📊 Steering Analysis:"
echo "-------------------"
grep "steer=" test_unet_fixes.log | tail -20 | awk -F'steer=' '{print $2}' | awk '{print $1}' | sort -n | tail -5 | while read val; do
    echo "  Max steering: $val"
done

echo ""
echo "⚠️  Steering Limit Warnings:"
echo "-------------------"
WARNINGS=$(grep -c "STEERING LIMIT" test_unet_fixes.log)
echo "  Total warnings: $WARNINGS"

echo ""
echo "🎯 Mode Usage:"
echo "-------------------"
grep "mode=" test_unet_fixes.log | tail -20 | awk -F'mode=' '{print $2}' | awk '{print $1}' | sort | uniq -c

echo ""
echo "📈 Phase Status:"
echo "-------------------"
grep "phases:" test_unet_fixes.log | tail -10

echo ""
echo "Full log saved to: test_unet_fixes.log"
