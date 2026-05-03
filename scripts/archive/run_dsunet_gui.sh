#!/bin/bash
# DSUNet Dashboard Runner Script

echo "🚀 Starting DSUNet Dashboard with CARLA..."

# Check if model exists
MODEL_PATH="checkpoints/dsunet_best_iou.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "Please ensure DSUNet model is trained and saved."
    exit 1
fi

echo "✅ Using DSUNet model: $MODEL_PATH"

# Activate virtual environment
if [ -d "../carla_lstm_mpc_project/venv" ]; then
    echo "🔧 Activating virtual environment..."
    source ../carla_lstm_mpc_project/venv/bin/activate
else
    echo "⚠️  No virtual environment found, using system Python"
fi

# Set display for GUI
export DISPLAY=${DISPLAY:-:0}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-x11}

# Check if CARLA is running
if ! pgrep -f "CarlaUE4" > /dev/null; then
    echo "🎮 Starting CARLA server..."
    cd ..
    ./run_carla_visible.sh &
    CARLA_PID=$!
    echo "CARLA PID: $CARLA_PID"
    
    # Wait for CARLA to start
    echo "⏳ Waiting for CARLA to initialize..."
    sleep 10
    
    cd carla_mpc_classical
fi

# Run DSUNet Dashboard
echo "🎯 Starting DSUNet Dashboard..."
python run_dsunet_dashboard.py \
    --model "$MODEL_PATH" \
    --town "Town04" \
    --speed 25.0

# Cleanup on exit
echo "🧹 Cleaning up..."
if [ ! -z "$CARLA_PID" ]; then
    kill $CARLA_PID 2>/dev/null
fi

echo "✅ DSUNet Dashboard stopped"
