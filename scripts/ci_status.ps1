<#
.SYNOPSIS
  ดู GitHub Actions status ของ FrontViewPredictiveControl

.DESCRIPTION
  เรียก GitHub REST API โดยตรง (ไม่ต้องติดตั้ง gh CLI)
  แสดง workflow runs ล่าสุด + สถานะ + ลิงก์

.PARAMETER Branch
  กรองตาม branch (default: ทุก branch)

.PARAMETER Limit
  จำนวน runs ที่แสดง (default: 10)

.EXAMPLE
  .\scripts\ci_status.ps1
  .\scripts\ci_status.ps1 -Branch project-ADAS -Limit 5
#>
param(
    [string]$Branch = "",
    [int]$Limit = 10
)

$Repo = "Telotubbies/FrontViewPredictiveControl"
$ApiBase = "https://api.github.com/repos/$Repo/actions"

# Icon ตามสถานะ (ASCII เพราะ PowerShell hashtable มีปัญหา unicode)
function Get-Icon([string]$Status) {
    switch ($Status) {
        "success"         { "[OK]" }
        "failure"         { "[FAIL]" }
        "cancelled"       { "[CXL]" }
        "in_progress"     { "[RUN]" }
        "queued"          { "[WAIT]" }
        "waiting"         { "[WAIT]" }
        "timed_out"       { "[TMO]" }
        "startup_failure" { "[ERR]" }
        "neutral"         { "[--]" }
        "stale"           { "[--]" }
        "action_required" { "[!]" }
        default           { "[?]" }
    }
}

function Get-Color([string]$Status) {
    switch ($Status) {
        "success"         { "Green" }
        "failure"         { "Red" }
        "timed_out"       { "Red" }
        "startup_failure" { "Red" }
        "in_progress"     { "Yellow" }
        "queued"          { "Cyan" }
        "waiting"         { "Cyan" }
        "cancelled"       { "DarkGray" }
        "stale"           { "DarkGray" }
        "neutral"         { "Gray" }
        "action_required" { "Magenta" }
        default           { "Gray" }
    }
}

Write-Host ""
Write-Host "  GitHub Actions - $Repo" -ForegroundColor White
Write-Host "  ------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""

# ดึง workflow runs
$Query = "per_page=$Limit"
if ($Branch) { $Query += "&branch=$Branch" }

try {
    $Headers = @{
        "Accept" = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
    }
    if ($env:GITHUB_TOKEN) {
        $Headers["Authorization"] = "Bearer $env:GITHUB_TOKEN"
    }
    $RunsResponse = Invoke-RestMethod -Uri "$ApiBase/runs?$Query" -Headers $Headers -ErrorAction Stop
} catch {
    Write-Host "  ERROR: ไม่สามารถดึงข้อมูลได้" -ForegroundColor Red
    Write-Host "  $_" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  ตรวจสอบ:" -ForegroundColor Yellow
    Write-Host "    1. อินเทอร์เน็ตเชื่อมต่อ"
    Write-Host "    2. repo เป็น public (ไม่ต้องใช้ token)"
    Write-Host "    3. ถ้า private ต้อง set `$env:GITHUB_TOKEN ก่อน"
    Write-Host ""
    exit 1
}

$Runs = $RunsResponse.workflow_runs
if (-not $Runs -or $Runs.Count -eq 0) {
    Write-Host "  ไม่มี workflow runs" -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

# แสดงแต่ละ run
foreach ($Run in $Runs) {
    $Status = if ($Run.conclusion) { $Run.conclusion } else { $Run.status }
    $Icon = Get-Icon $Status
    $Color = Get-Color $Status

    $Time = ([datetime]$Run.created_at).ToString("yyyy-MM-dd HH:mm")
    $BranchName = $Run.head_branch
    $CommitMsg = ($Run.head_commit.message -split "`n")[0]
    if ($CommitMsg.Length -gt 45) { $CommitMsg = $CommitMsg.Substring(0, 42) + "..." }

    Write-Host "  " -NoNewline
    Write-Host $Icon.PadRight(7) -ForegroundColor $Color -NoNewline
    Write-Host ("#{0,6}" -f $Run.run_number) -ForegroundColor White -NoNewline
    Write-Host "  $Time" -ForegroundColor DarkGray -NoNewline
    Write-Host "  $BranchName".PadRight(20) -ForegroundColor Cyan -NoNewline
    Write-Host "  $CommitMsg" -ForegroundColor Gray

    # แสดง jobs ใน run นี้ (เฉพาะที่ fail/in_progress)
    if ($Status -in @("failure", "in_progress", "queued", "timed_out", "startup_failure")) {
        try {
            $Jobs = Invoke-RestMethod -Uri "$ApiBase/runs/$($Run.id)/jobs" -Headers $Headers
            foreach ($Job in $Jobs.jobs) {
                $JobStatus = if ($Job.conclusion) { $Job.conclusion } else { $Job.status }
                $JobIcon = Get-Icon $JobStatus
                $JobColor = Get-Color $JobStatus
                Write-Host "        " -NoNewline
                Write-Host $JobIcon.PadRight(7) -ForegroundColor $JobColor -NoNewline
                Write-Host $Job.name -ForegroundColor $JobColor
            }
        } catch {}
    }
}

Write-Host ""
Write-Host "  ------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  ดูรายละเอียด: https://github.com/$Repo/actions" -ForegroundColor DarkGray
Write-Host ""
