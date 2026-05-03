#!/bin/bash

# DSUNet Evaluation Script

# Dataset path
DATA_ROOT="/home/supawich/Desktop/CARLA_0.9.16/carla_lstm_mpc_project/data/lane_train_diverse_20260227_121538"

# Checkpoint path (update this after training)
CHECKPOINT="./outputs/dsunet_carla_XXXXXX/best_model_iou.pth"

# Output directory
OUTPUT_DIR="./eval_results/$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "DSUNet Evaluation on CARLA Test Set"
echo "=========================================="
echo ""
echo "Dataset:    $DATA_ROOT"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_DIR"
echo ""

# Activate virtual environment
source ../../venv/bin/activate

# Run evaluation
python evaluate.py \
    --data-root "$DATA_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 16 \
    --num-workers 8 \
    --save-vis

echo ""
echo "✅ Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
