#!/bin/bash

# Resume DSUNet Training from Checkpoint

echo "=========================================="
echo "Resume DSUNet Training"
echo "=========================================="
echo ""

# Find latest output directory
LATEST_DIR=$(ls -td outputs/dsunet_carla_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ No training directory found!"
    echo "Please run ./run_training.sh first"
    exit 1
fi

echo "Found training directory: $LATEST_DIR"
echo ""

# Find latest checkpoint
CHECKPOINT=$(ls -t "$LATEST_DIR"/checkpoint_epoch_*.pth 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "❌ No checkpoint found!"
    echo "Starting new training instead..."
    ./run_training.sh
    exit 0
fi

echo "Latest checkpoint: $CHECKPOINT"
echo ""

# Extract epoch number
EPOCH=$(echo "$CHECKPOINT" | grep -oP 'epoch_\K[0-9]+')
echo "Resuming from epoch: $EPOCH"
echo ""

# Dataset path
DATA_ROOT="/home/supawich/Desktop/CARLA_0.9.16/carla_lstm_mpc_project/data/lane_train_diverse_20260227_121538"

# Training parameters
TOTAL_EPOCHS=100
REMAINING_EPOCHS=$((TOTAL_EPOCHS - EPOCH))
BATCH_SIZE=4
LR=0.001
BASE_CHANNELS=64
IMAGE_HEIGHT=480
IMAGE_WIDTH=640
NUM_WORKERS=8

echo "Training parameters:"
echo "  - Resume from epoch: $EPOCH"
echo "  - Remaining epochs: $REMAINING_EPOCHS"
echo "  - Total epochs: $TOTAL_EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo ""

# Activate venv
source ../venv/bin/activate

# Resume training
python train.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$LATEST_DIR" \
    --epochs $TOTAL_EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --base-channels $BASE_CHANNELS \
    --image-height $IMAGE_HEIGHT \
    --image-width $IMAGE_WIDTH \
    --num-workers $NUM_WORKERS \
    --resume "$CHECKPOINT" \
    --save-freq 10

echo ""
echo "✅ Training completed!"
