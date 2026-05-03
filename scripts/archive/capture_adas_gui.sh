#!/bin/bash
#
# Capture ADAS GUI screenshots automatically
# 
# Usage:
#   Terminal 1: ./run_adas.sh --town Town04 --speed 25
#   Terminal 2: ./capture_adas_gui.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRIPT_DIR}/gui_screenshots/${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "📸 ADAS GUI Screenshot Capture"
echo "================================"
echo "Output: $OUTPUT_DIR"
echo ""
echo "วิธีใช้:"
echo "1. รัน ADAS ในเทอร์มินัลอื่น: ./run_adas.sh --town Town04 --speed 25"
echo "2. รอให้ GUI window ขึ้น"
echo "3. สคริปต์นี้จะ capture ทุก 2 วินาที"
echo "4. กด Ctrl+C เพื่อหยุด"
echo ""
echo "เริ่มใน 5 วินาที..."
sleep 5

FRAME=0
INTERVAL=2

echo ""
echo "🎬 เริ่ม capture (interval: ${INTERVAL}s)"
echo "กด Ctrl+C เพื่อหยุด"
echo ""

# Function to capture screenshot
capture_screenshot() {
    local frame_num=$1
    local output_file="${OUTPUT_DIR}/frame_$(printf "%04d" $frame_num).png"
    
    # Try different screenshot methods
    if command -v gnome-screenshot &> /dev/null; then
        # GNOME Screenshot (works with Wayland)
        gnome-screenshot -w -f "$output_file" 2>/dev/null && \
            echo "✅ Frame $frame_num: $(basename $output_file)" && \
            return 0
    fi
    
    if command -v scrot &> /dev/null; then
        # scrot (X11)
        scrot -u "$output_file" 2>/dev/null && \
            echo "✅ Frame $frame_num: $(basename $output_file)" && \
            return 0
    fi
    
    if command -v import &> /dev/null; then
        # ImageMagick import
        import -window root "$output_file" 2>/dev/null && \
            echo "✅ Frame $frame_num: $(basename $output_file)" && \
            return 0
    fi
    
    # Fallback: use xwd + convert
    if command -v xwd &> /dev/null && command -v convert &> /dev/null; then
        WINDOW_ID=$(xdotool search --name "LKA" 2>/dev/null | head -1)
        if [ -n "$WINDOW_ID" ]; then
            xwd -id "$WINDOW_ID" | convert xwd:- "$output_file" 2>/dev/null && \
                echo "✅ Frame $frame_num: $(basename $output_file)" && \
                return 0
        fi
    fi
    
    echo "⚠️  Screenshot failed for frame $frame_num"
    return 1
}

# Main capture loop
trap 'echo -e "\n\n🛑 Stopped"; echo "📊 Total frames: $FRAME"; echo "📁 Output: $OUTPUT_DIR"; exit 0' INT

while true; do
    FRAME=$((FRAME + 1))
    capture_screenshot $FRAME
    sleep $INTERVAL
done
