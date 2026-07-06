#!/usr/bin/env bash
# ดู GitHub Actions status ของ FrontViewPredictiveControl
# เรียก GitHub REST API โดยตรง (ไม่ต้องติดตั้ง gh CLI)
#
# Usage:
#   ./scripts/ci_status.sh
#   ./scripts/ci_status.sh project-ADAS 5
#   BRANCH=project-ADAS LIMIT=5 ./scripts/ci_status.sh

set -euo pipefail

REPO="Telotubbies/FrontViewPredictiveControl"
BRANCH="${1:-${BRANCH:-}}"
LIMIT="${2:-${LIMIT:-10}}"
API_BASE="https://api.github.com/repos/$REPO/actions"

# สี
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
WHITE='\033[1;37m'
NC='\033[0m'

# Icon ตามสถานะ
icon_for() {
    case "$1" in
        success)          echo "✓" ;;
        failure)          echo "✗" ;;
        cancelled)        echo "⊘" ;;
        in_progress|queued|waiting) echo "●" ;;
        timed_out|startup_failure) echo "✗" ;;
        *)                echo "?" ;;
    esac
}

color_for() {
    case "$1" in
        success)          echo "$GREEN" ;;
        failure|timed_out|startup_failure) echo "$RED" ;;
        in_progress)      echo "$YELLOW" ;;
        queued|waiting)   echo "$CYAN" ;;
        cancelled|stale|neutral) echo "$GRAY" ;;
        *)                echo "$NC" ;;
    esac
}

echo ""
echo -e "  ${WHITE}GitHub Actions — $REPO${NC}"
echo -e "  ${GRAY}────────────────────────────────────────────────────────────────${NC}"
echo ""

# สร้าง query
QUERY="per_page=$LIMIT"
if [ -n "$BRANCH" ]; then
    QUERY="${QUERY}&branch=${BRANCH}"
fi

# เรียก API
API_RESPONSE=$(curl -sf -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "$API_BASE/runs?$QUERY" 2>&1) || {
    echo -e "  ${RED}ERROR: ไม่สามารถดึงข้อมูลได้${NC}"
    echo ""
    echo -e "  ${YELLOW}ตรวจสอบ:${NC}"
    echo "    1. อินเทอร์เน็ตเชื่อมต่อ"
    echo "    2. repo เป็น public (ไม่ต้องใช้ token)"
    echo "    3. ถ้า private ต้อง set GITHUB_TOKEN ก่อน"
    echo ""
    exit 1
}

# ตรวจว่ามี runs
RUN_COUNT=$(echo "$API_RESPONSE" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('workflow_runs',[])))" 2>/dev/null || echo "0")

if [ "$RUN_COUNT" = "0" ]; then
    echo -e "  ${YELLOW}ไม่มี workflow runs${NC}"
    echo ""
    exit 0
fi

# แสดงแต่ละ run (ใช้ python  parse JSON เพราะมี jq ไม่ทุกเครื่อง)
echo "$API_RESPONSE" | python3 -c "
import sys, json
from datetime import datetime

data = json.load(sys.stdin)
runs = data.get('workflow_runs', [])

for run in runs:
    status = run.get('conclusion') or run.get('status', 'unknown')
    icon_map = {
        'success': '✓', 'failure': '✗', 'cancelled': '⊘',
        'in_progress': '●', 'queued': '●', 'waiting': '●',
        'timed_out': '✗', 'startup_failure': '✗',
    }
    color_map = {
        'success': '\033[0;32m', 'failure': '\033[0;31m',
        'in_progress': '\033[1;33m', 'queued': '\033[0;36m',
        'cancelled': '\033[0;90m', 'timed_out': '\033[0;31m',
    }
    icon = icon_map.get(status, '?')
    color = color_map.get(status, '\033[0m')
    nc = '\033[0m'
    gray = '\033[0;90m'
    white = '\033[1;37m'
    cyan = '\033[0;36m'

    created = run.get('created_at', '')[:16].replace('T', ' ')
    branch = run.get('head_branch', '—')
    event = run.get('event', '—')
    commit_msg = (run.get('head_commit', {}).get('message', '') or '').split('\n')[0]
    if len(commit_msg) > 45:
        commit_msg = commit_msg[:42] + '...'

    # duration
    duration_str = '—'
    started = run.get('run_started_at')
    updated = run.get('updated_at')
    if started and updated:
        try:
            s = datetime.fromisoformat(started.replace('Z', '+00:00'))
            u = datetime.fromisoformat(updated.replace('Z', '+00:00'))
            dur = (u - s).total_seconds()
            if dur < 60:
                duration_str = f'{int(dur)}s'
            elif dur < 3600:
                duration_str = f'{dur/60:.1f}m'
            else:
                duration_str = f'{dur/3600:.1f}h'
        except Exception:
            pass

    print(f'  {color}{icon}{nc} #{run[\"run_number\"]:>6}  {color}{status:<16}{nc} {gray}{created}{nc}  {duration_str:>8}  {cyan}{branch:<18}{nc}  {commit_msg}')

    # show jobs if failed/in-progress
    if status in ('failure', 'in_progress', 'queued', 'timed_out', 'startup_failure'):
        # would need another API call — skip for bash version, suggest URL
        pass
"

echo ""
echo -e "  ${GRAY}────────────────────────────────────────────────────────────────${NC}"
echo -e "  ${GRAY}ดูรายละเอียด: https://github.com/$REPO/actions${NC}"
echo ""
