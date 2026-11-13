# Silo daily paper run (research only; paper and Trading 212 Practice only).
# Fetches new UK prices, advances every paper book, writes the leaderboard,
# and optionally mirrors one book to the Trading 212 Practice account.
#
# Usage (from the Silo folder):
#   powershell -ExecutionPolicy Bypass -File scripts\windows\silo-daily.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\windows\silo-daily.ps1 -Mirror league-silo_agent
param(
    [string]$Mirror = "",
    [string]$Model = ""
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$silo = Join-Path $root ".venv\Scripts\silo.exe"
if (-not (Test-Path $silo)) { $silo = "silo" }
if ($Model) { $env:SILO_LLM = $Model }
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null
$log = Join-Path $root ("logs\daily-" + (Get-Date -Format "yyyy-MM-dd") + ".log")
$cmd = @("daily")
if ($Mirror) { $cmd += @("--mirror", $Mirror) }
& $silo @cmd *>&1 | Tee-Object -FilePath $log -Append
exit $LASTEXITCODE
