# Register a Windows scheduled task that runs the Silo daily paper run on
# weekdays at 18:30 local time (after the London close at 16:35 and after
# Yahoo has published the day's bar). Research only: paper and Practice.
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\install-daily-task.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\windows\install-daily-task.ps1 -Mirror league-silo_agent -Model ollama:qwen2.5:7b
#
# Remove it with:  Unregister-ScheduledTask -TaskName "Silo daily paper run"
param(
    [string]$Mirror = "",
    [string]$Model = "",
    [string]$At = "18:30"
)
$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "silo-daily.ps1"
$argline = "-NoProfile -ExecutionPolicy Bypass -File `"$script`""
if ($Mirror) { $argline += " -Mirror $Mirror" }
if ($Model) { $argline += " -Model $Model" }
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argline -WorkingDirectory (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 6)
Register-ScheduledTask -TaskName "Silo daily paper run" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Scheduled 'Silo daily paper run' on weekdays at $At. Logs go to the logs folder."
