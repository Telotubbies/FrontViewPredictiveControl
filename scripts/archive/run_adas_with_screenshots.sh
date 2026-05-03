#!/bin/bash
#
# รัน ADAS พร้อมบันทึกภาพหน้าจอ GUI อัตโนมัติ
#
# Usage:
#   ./run_adas_with_screenshots.sh --town Town04 --speed 25 --frames 100
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRIPT_DIR}/gui_screenshots/${TIMESTAMP}"

# Parse arguments
TOWN="Town04"
SPEED=25
FRAMES=100
INTERVAL=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --town)
            TOWN="$2"
            shift 2
            ;;
        --speed)
            SPEED="$2"
            shift 2
            ;;
        --frames)
            FRAMES="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "📸 ADAS with GUI Screenshot Capture"
echo "===================================="
echo "Town: $TOWN"
echo "Speed: $SPEED km/h"
echo "Frames: $FRAMES"
echo "Screenshot interval: every $INTERVAL frames"
echo "Output: $OUTPUT_DIR"
echo ""

# Set environment variable for screenshot capture
export ADAS_SCREENSHOT_DIR="$OUTPUT_DIR"
export ADAS_SCREENSHOT_INTERVAL="$INTERVAL"
export ADAS_MAX_FRAMES="$FRAMES"

# Run ADAS
cd "$SCRIPT_DIR"
source activate_venv.sh

echo "🚗 Starting ADAS..."
python adas/run.py \
    --town "$TOWN" \
    --speed "$SPEED" \
    --dashboard pygame \
    --screenshot-dir "$OUTPUT_DIR" \
    --screenshot-interval "$INTERVAL" \
    --max-frames "$FRAMES"

echo ""
echo "✅ Complete!"
echo "📁 Screenshots saved to: $OUTPUT_DIR"
echo ""
echo "ดูภาพ:"
echo "  cd $OUTPUT_DIR"
echo "  eog *.png"
