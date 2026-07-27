param(
  [string]$Repo,
  [string]$Python,
  [string]$ExecLog,
  [string]$LogDir
)
$ErrorActionPreference = 'Continue'
Set-Location $Repo

function Write-ExecLog {
  param([string]$Message)
  $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  Add-Content -Path $ExecLog -Value "- [$ts] $Message"
}

function Run-Step {
  param(
    [string]$Name,
    [string[]]$Args,
    [string]$BaseLog
  )
  $stdout = "$BaseLog.stdout"
  $stderr = "$BaseLog.stderr"
  if (Test-Path $stdout) { Remove-Item -LiteralPath $stdout -Force }
  if (Test-Path $stderr) { Remove-Item -LiteralPath $stderr -Force }
  Set-Content -Path $BaseLog -Value ("[{0}] COMMAND: {1} {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Python, ($Args -join ' '))
  Write-ExecLog "$Name started."
  & $Python @Args 1>> $stdout 2>> $stderr
  $exitCode = $LASTEXITCODE
  if (Test-Path $stdout) {
    Get-Content $stdout -Raw -ErrorAction SilentlyContinue | Add-Content -Path $BaseLog -Encoding utf8
  }
  if ((Test-Path $stderr) -and ((Get-Item $stderr).Length -gt 0)) {
    Add-Content -Path $BaseLog -Encoding utf8 -Value "`n[stderr]`n"
    Get-Content $stderr -Raw -ErrorAction SilentlyContinue | Add-Content -Path $BaseLog -Encoding utf8
  }
  if ($exitCode -ne 0) {
    Write-ExecLog "$Name failed with exit code $exitCode."
    throw "$Name failed with exit code $exitCode"
  }
  Write-ExecLog "$Name completed successfully."
}

Run-Step -Name 'Step 4 single-model evaluation (stratified 500, timed)' -BaseLog (Join-Path $LogDir '04_eval_stratified_500_timed.log') -Args @('-X','utf8','-u','scripts/eval.py','--config','configs/full_500_restart.yaml','--checkpoint','artifacts/full_500_restart/lora_full_500_restart.safetensors','--out','artifacts/full_500_restart','--stratified-max-records','500','--sample-seed','42')
Run-Step -Name 'Step 5 smoke-vs-full comparison' -BaseLog (Join-Path $LogDir '05_compare_smoke_vs_full.log') -Args @('-X','utf8','-u','scripts/eval.py','--config','configs/full_500_restart.yaml','--compare-smoke-vs-full','--smoke-checkpoint','artifacts/baseline/lora_baseline.safetensors','--full-checkpoint','artifacts/full_500_restart/lora_full_500_restart.safetensors','--out','artifacts/full_500_restart','--stratified-max-records','500','--sample-seed','42')
Run-Step -Name 'Step 6 offline package rebuild' -BaseLog (Join-Path $LogDir '06_package_offline_v2.log') -Args @('-X','utf8','-u','scripts/package_offline.py','--ckpt','artifacts/full_500_restart/lora_full_500_restart.safetensors','--out','mpid_offline_v2')
Run-Step -Name 'Step 7 offline smoke validation' -BaseLog (Join-Path $LogDir '07_smoke_offline_v2.log') -Args @('-X','utf8','-u','scripts/smoke_offline.py','--pkg','mpid_offline_v2')
Write-ExecLog 'Manual continuation from Step 4 finished successfully.'
