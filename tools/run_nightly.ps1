# Nightly batch for v0.2 (2026-10-22).
#
# Order is short -> short -> long, so a failure late in the night still leaves
# usable results.  Every script resumes from its own CSV: re-running this file
# skips whatever already finished, so a crash is never a lost night.
#
#   [1] WP-C control  tACS with A=0 (no plasticity), 3 seeds      ~15 min
#   [2] WP-E          T2.6d v0.1 regression, 60 runs              ~13 min
#   [3] WP-A2         (g x amp) mode map, 150 runs                ~75 min
#
# Usage:  powershell -ExecutionPolicy Bypass -File tools\run_nightly.ps1
#     or: .\tools\run_nightly.ps1

$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

$pyb   = "C:\ANACONDA\envs\mleDSar\python.exe"
$root  = "E:\gamma_sar"
$logs  = Join-Path $root "stage2_network_gamma"
Set-Location $root

# keep the box awake for the duration
powercfg /change standby-timeout-ac 0 2>$null | Out-Null

function Invoke-Stage {
    param(
        [string]   $Name,
        [string]   $Log,
        [string[]] $ArgList
    )
    $t0 = Get-Date
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host ("[{0}] start {1:HH:mm:ss}" -f $Name, $t0)
    Write-Host ("=" * 72)
    & $pyb @ArgList 2>&1 | Tee-Object -FilePath $Log
    $dt = (Get-Date) - $t0
    Write-Host ("[{0}] finished in {1:N1} min (exit {2})" -f $Name, $dt.TotalMinutes, $LASTEXITCODE)
}

$batchStart = Get-Date

Invoke-Stage -Name "C-control"  -Log (Join-Path $logs "c_ctrl.log") `
    -ArgList @("stage2_network_gamma\run_c_stdp.py", "--arms", "ctrl",
               "--amag", "0", "--seeds", "0-2", "--save-spikes")

Invoke-Stage -Name "E-t26d"     -Log (Join-Path $logs "t26d_regression.log") `
    -ArgList @("stage2_network_gamma\run_t26d_stats.py")

Invoke-Stage -Name "A2-mode-map" -Log (Join-Path $logs "a2_mode_map.log") `
    -ArgList @("stage2_network_gamma\run_a2_mode_map.py", "--save-spikes")

$total = (Get-Date) - $batchStart
Write-Host ""
Write-Host ("=" * 72)
Write-Host ("BATCH DONE in {0:N0} min" -f $total.TotalMinutes)
Write-Host "=" * 72
Write-Host ""
Write-Host "Morning checklist:"
Write-Host "  1. stage2_network_gamma\c_ctrl.log      -- does A=0 also raise post power?"
Write-Host "  2. stage2_network_gamma\t26d_regression.log -- p=0.00195 and the 6 tolerance rows"
Write-Host "  3. stage2_network_gamma\a2_mode_map.log -- the two 5x5 mode tables (+ '*' = seed split)"
Write-Host ""
Write-Host "If A2 stopped early, just run this file again: it resumes."
