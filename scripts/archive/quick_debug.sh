#!/bin/bash
#
# Quick Debug: รัน ADAS พร้อมเก็บ log และคำแนะนำการ screenshot
#

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="debug_output_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "🔍 ADAS Quick Debug Mode"
echo "========================"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo ""
echo "📸 วิธีเก็บภาพหน้าจอ GUI:"
echo "   1. ขณะที่ ADAS กำลังรัน ให้กด PrintScreen หรือ"
echo "   2. เปิด terminal ใหม่แล้วรัน:"
echo "      scrot -u $OUTPUT_DIR/screenshot_\$(date +%H%M%S).png"
echo "   3. หรือใช้ gnome-screenshot:"
echo "      gnome-screenshot -w -f $OUTPUT_DIR/screenshot_\$(date +%H%M%S).png"
echo ""
echo "📊 Log จะถูกบันทึกอัตโนมัติที่: $OUTPUT_DIR/adas.log"
echo ""
echo "กด Enter เพื่อเริ่ม ADAS..."
read

echo ""
echo "🚗 Starting ADAS..."
echo ""

cd "$(dirname "$0")"

# Run ADAS with log capture
./run_adas.sh --town Town04 --speed 25 2>&1 | tee "$OUTPUT_DIR/adas.log"

echo ""
echo "✅ ADAS stopped"
echo ""
echo "📊 วิเคราะห์ผล:"
echo "==============="
echo ""

# Analyze log
LOG_FILE="$OUTPUT_DIR/adas.log"

if [ -f "$LOG_FILE" ]; then
    echo "1️⃣ Lane Detection Phases:"
    grep -E "phases: P1|P2|P3|P4|P5" "$LOG_FILE" | tail -5
    echo ""
    
    echo "2️⃣ Detection Mode:"
    grep "mode=" "$LOG_FILE" | tail -10
    echo ""
    
    echo "3️⃣ Steering Issues:"
    STEERING_COUNT=$(grep -c "STEERING LIMIT EXCEEDED" "$LOG_FILE" || echo "0")
    echo "   Steering limit exceeded: $STEERING_COUNT times"
    if [ "$STEERING_COUNT" -gt 0 ]; then
        echo "   ⚠️  มีปัญหา steering เหวี่ยง!"
        grep "STEERING LIMIT EXCEEDED" "$LOG_FILE" | head -5
    fi
    echo ""
    
    echo "4️⃣ Lane Confidence:"
    grep "lane_conf=" "$LOG_FILE" | tail -5
    echo ""
    
    echo "📁 Full log: $OUTPUT_DIR/adas.log"
    echo "📸 Screenshots: $OUTPUT_DIR/*.png (ถ้ามี)"
fi

echo ""
echo "💡 Tips:"
echo "   - ถ้า P1✗ = UNet ไม่จับเลน"
echo "   - ถ้า mode=WP_PRIMARY ตลอด = ไม่ใช้ UNet"
echo "   - ถ้า STEERING LIMIT เยอะ = control เหวี่ยง"
echo ""
