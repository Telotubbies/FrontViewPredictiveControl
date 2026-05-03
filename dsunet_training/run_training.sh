#!/bin/bash

# DSUNet Training Script for CARLA Lane Detection

# Dataset path
DATA_ROOT="/home/supawich/Desktop/CARLA_0.9.16/carla_lstm_mpc_project/data/lane_train_diverse_20260227_121538"

# Output directory
OUTPUT_DIR="./outputs/dsunet_carla_$(date +%Y%m%d_%H%M%S)"

# Training parameters
EPOCHS=100
BATCH_SIZE=4  # Reduced from 16 to avoid CUDA OOM
LR=0.001
BASE_CHANNELS=64

# Image size
IMAGE_HEIGHT=480
IMAGE_WIDTH=640

# Number of workers
NUM_WORKERS=8

echo "=========================================="
echo "DSUNet Training on CARLA Lane Detection"
echo "=========================================="
echo ""
echo "Dataset: $DATA_ROOT"
echo "Output:  $OUTPUT_DIR"
echo "Epochs:  $EPOCHS"
echo "Batch:   $BATCH_SIZE"
echo "LR:      $LR"
echo ""

# Activate virtual environment
source ../venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -q -r requirements.txt

# Run training
python train.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --base-channels $BASE_CHANNELS \
    --image-height $IMAGE_HEIGHT \
    --image-width $IMAGE_WIDTH \
    --num-workers $NUM_WORKERS \
    --save-freq 10

echo ""
echo "✅ Training completed!"
echo "Results saved to: $OUTPUT_DIR"
