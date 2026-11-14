# Launch the full 50-step OPSD run detached, so it survives closing the terminal.
#
#   powershell -ExecutionPolicy Bypass -File scripts\launch_50steps.ps1
#
# Writes the log to C:\Users\Khali\projects\OPSD\runs\opsd_50steps.log and the
# checkpoints to C:\Users\Khali\projects\OPSD\runs\qwen31b_3070_gen1024_fixteacher\.

param(
    [string]$RunConfig = "qwen31b_3070_gen1024_fixteacher",
    [int]$MaxSteps    = 50,
    [int]$SaveSteps   = 10,
    [int]$MaxCompletion = 1024,
    [int]$BatchSize   = 4,
    [int]$EffBatch    = 32,
    [int]$ChunkSize   = 32,
    [string]$ReportTo  = "none"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RunRoot  = "C:\Users\Khali\projects\OPSD\runs"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Get-FreeLogPath([string]$base) {
    $p = Join-Path $RunRoot $base
    if (-not (Test-Path $p)) { return $p }
    try {
        Remove-Item $p -Force -ErrorAction Stop
        return $p
    } catch {
        # still locked by a previous run -> fall back to a timestamped name
        return (Join-Path $RunRoot ($base.Replace(".log", "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")))
    }
}

$log = Get-FreeLogPath "opsd_${MaxSteps}steps.log"
$err = Get-FreeLogPath "opsd_${MaxSteps}steps.err.log"

$inner = "-ExecutionPolicy Bypass -File `"$RepoRoot\scripts\run_opsd_1b_3070.ps1`" " +
         "-RunConfig $RunConfig -MaxSteps $MaxSteps -SaveSteps $SaveSteps " +
         "-MaxCompletion $MaxCompletion -BatchSize $BatchSize -EffBatch $EffBatch " +
         "-ChunkSize $ChunkSize -ReportTo $ReportTo"

$p = Start-Process -FilePath "powershell.exe" `
        -ArgumentList $inner `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $log `
        -RedirectStandardError $err `
        -WindowStyle Hidden `
        -PassThru

Write-Host "launched PID $($p.Id)"
Write-Host "log     : $log"
Write-Host "stderr  : $err"
Write-Host "checkpoints: $RunRoot\$RunConfig\checkpoint-<step>"
Write-Host "stop    : Stop-Process -Id $($p.Id)"
