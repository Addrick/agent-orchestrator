# Polls the in-flight S v2a ingest. Once its retain queue drains,
# sanity-checks the bank's retain_mission via API, then fires the M v2a
# ingest in this shell. Designed to run in a separate PowerShell window
# while the S ingest finishes in its existing background job.
#
# Usage:
#   .\scripts\wait_and_fire_m_v2a.ps1
#
# Run from repo root. Will tee M ingest output to
# .eval_cache\lme_m_91b15a6e_v2a_v2.log.

$ErrorActionPreference = 'Stop'

$base         = 'http://10.0.0.70:8888'
$sBank        = 'lme_s_1c0ddc50_v2a'
$mBank        = 'lme_m_91b15a6e_v2a'
$missionFile  = '.eval_cache\retain_mission_v2a.txt'
$mLog         = '.eval_cache\lme_m_91b15a6e_v2a_v2.log'
$mStateFile   = '.eval_cache\lme_ingest_queue_v2a_m.state.json'
$pollSec      = 60

if (-not (Test-Path $missionFile)) {
    Write-Error "Mission file not found at $missionFile"
    exit 1
}

Write-Host "Waiting for $sBank retain queue to drain (poll every ${pollSec}s)..."

while ($true) {
    try {
        $stats = Invoke-RestMethod "$base/v1/default/banks/$sBank/stats" -TimeoutSec 15
        # API shape: top-level pending_operations / pending_consolidation; per-status
        # counts under operations_by_status (no "processing" key when zero).
        # Missing props -> $null, [int]$null is 0 in PS5.1.
        $pending     = [int]$stats.pending_operations
        $processing  = [int]$stats.operations_by_status.processing
        $completed   = [int]$stats.operations_by_status.completed
        $pendingCons = [int]$stats.pending_consolidation
        $busy  = $pending + $processing + $pendingCons
        $facts = [int]$stats.total_nodes
        $ts = Get-Date -Format 'HH:mm:ss'
        Write-Host ("  [{0}] done={1} busy={2}p+{3}r+{4}cons facts={5}" -f $ts, $completed, $pending, $processing, $pendingCons, $facts)
        if ($busy -eq 0 -and $completed -gt 0) { break }
    } catch {
        Write-Warning ("poll error: {0}" -f $_.Exception.Message)
    }
    Start-Sleep -Seconds $pollSec
}

Write-Host ""
Write-Host "S drained. Sanity-checking retain_mission on $sBank..."

$cfg = Invoke-RestMethod "$base/v1/default/banks/$sBank/config" -TimeoutSec 15
$actual   = [string]$cfg.config.retain_mission
$expected = (Get-Content -Raw $missionFile).TrimEnd()

if ([string]::IsNullOrEmpty($actual)) {
    Write-Error ("retain_mission is NULL on {0}. Aborting M ingest." -f $sBank)
    exit 1
}
if ($actual.Trim() -ne $expected.Trim()) {
    Write-Host ("Expected: {0}..." -f $expected.Substring(0, [Math]::Min(120, $expected.Length)))
    Write-Host ("Actual:   {0}..." -f $actual.Substring(0, [Math]::Min(120, $actual.Length)))
    Write-Error ("retain_mission mismatch on {0}. Aborting M ingest." -f $sBank)
    exit 1
}

Write-Host ("Mission verified on {0} ({1} chars). Firing M v2a ingest..." -f $sBank, $actual.Length)
Write-Host ""

# Delete the existing (mission-less) M v2a bank so we get a fresh ingest
# rather than appending to old facts under different config.
try {
    $null = Invoke-RestMethod "$base/v1/default/banks/$mBank" -Method Delete -TimeoutSec 30
    Write-Host ("Deleted stale {0}." -f $mBank)
} catch {
    Write-Host ("No existing {0} to delete (or already gone): {1}" -f $mBank, $_.Exception.Message)
}

if (Test-Path $mStateFile) {
    Remove-Item $mStateFile
    Write-Host ("Cleared {0}." -f $mStateFile)
}

# Foreground so output streams to this window AND the log file.
python -m eval_harnesses.suites.memory_recall.lme_ingest_queue `
    --tier m `
    --qids 91b15a6e `
    --bank-prefix lme_m `
    --bank-suffix _v2a `
    --state $mStateFile `
    --retain-mission-file $missionFile 2>&1 | Tee-Object -FilePath $mLog
