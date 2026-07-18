$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $RepoRoot "logs\phase2_2_full_800"
$ExecLog = Join-Path $RepoRoot "logs\phase2_2_full_800_execution_log.md"
$PidFile = Join-Path $LogDir "runner.pid"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-ExecLog {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $ExecLog -Value "- [$ts] $Message"
}

function Test-RequiredPath {
    param([string]$PathValue)
    if (-not (Test-Path $PathValue)) {
        throw "Required path missing: $PathValue"
    }
}

function Run-ShortStep {
    param(
        [string]$Name,
        [string]$Command,
        [string]$StepLog,
        [int]$MaxAttempts = 2
    )

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Write-ExecLog "$Name started (attempt $attempt/$MaxAttempts)."
        "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] COMMAND: $Command" | Set-Content -Path $StepLog
        & powershell -NoProfile -Command $Command *>> $StepLog
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            Write-ExecLog "$Name completed successfully."
            return
        }
        Write-ExecLog "$Name failed with exit code $exitCode."
    }
    throw "$Name failed after $MaxAttempts attempts."
}

function Run-LongStep {
    param(
        [string]$Name,
        [string]$Command,
        [string]$StepLog,
        [int]$StatusIntervalMinutes = 20
    )

    Write-ExecLog "$Name started."
    "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] COMMAND: $Command" | Set-Content -Path $StepLog

    $proc = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-Command", $Command) `
        -RedirectStandardOutput $StepLog `
        -RedirectStandardError $StepLog `
        -PassThru `
        -WindowStyle Hidden

    while (-not $proc.HasExited) {
        Start-Sleep -Seconds ($StatusIntervalMinutes * 60)
        if (-not $proc.HasExited) {
            $sizeKb = if (Test-Path $StepLog) { [math]::Round((Get-Item $StepLog).Length / 1KB, 1) } else { 0 }
            $tail = ""
            if (Test-Path $StepLog) {
                $tail = (Get-Content $StepLog -Tail 3 -ErrorAction SilentlyContinue) -join " || "
            }
            Write-ExecLog "$Name status update: still running, pid=$($proc.Id), log_size_kb=$sizeKb, tail=$tail"
            $proc.Refresh()
        }
    }

    if ($proc.ExitCode -ne 0) {
        Write-ExecLog "$Name failed with exit code $($proc.ExitCode)."
        throw "$Name failed."
    }
    Write-ExecLog "$Name completed successfully."
}

Set-Content -Path $PidFile -Value $PID
Write-ExecLog "Runner started. PID=$PID"

try {
    Write-ExecLog "Step 1: preflight checks started."
    Test-RequiredPath $Python
    Test-RequiredPath (Join-Path $RepoRoot "configs\full_800.yaml")
    Test-RequiredPath (Join-Path $RepoRoot "configs\benchmark_100.yaml")
    Test-RequiredPath (Join-Path $RepoRoot "data\mpid-v1\train.jsonl")
    Test-RequiredPath (Join-Path $RepoRoot "data\mpid-v1\val.jsonl")
    Test-RequiredPath (Join-Path $RepoRoot "models\smolvlm-500m\config.json")
    Test-RequiredPath (Join-Path $RepoRoot "artifacts\baseline\lora_baseline.safetensors")
    Write-ExecLog "Step 1: preflight checks passed."

    $smokeLog = Join-Path $LogDir "02_smoke_train.log"
    $smokeCmd = "& `"$Python`" scripts/train.py --config configs/benchmark_100.yaml --preload-dataset --max-train-steps 20 --checkpoint-name lora_benchmark_100.safetensors --partial-name lora_benchmark_100.safetensors -u"
    Run-ShortStep -Name "Step 2 smoke training check" -Command $smokeCmd -StepLog $smokeLog -MaxAttempts 2

    $trainLog = Join-Path $LogDir "03_full_800_train.log"
    $trainCmd = "& `"$Python`" scripts/train.py --config configs/full_800.yaml --preload-dataset --save-every 100 --max-train-seconds 172800 --checkpoint-name lora_full_800.safetensors --partial-name lora_full_800.safetensors -u"
    Run-LongStep -Name "Step 3 full 800-sample training" -Command $trainCmd -StepLog $trainLog -StatusIntervalMinutes 20

    $evalLog = Join-Path $LogDir "04_eval_full_800.log"
    $evalCmd = "& `"$Python`" scripts/eval.py --config configs/full_800.yaml --checkpoint artifacts/full_800/lora_full_800.safetensors --out artifacts/full_800"
    Run-ShortStep -Name "Step 4 single-model evaluation" -Command $evalCmd -StepLog $evalLog -MaxAttempts 2

    $cmpLog = Join-Path $LogDir "05_compare_smoke_vs_full.log"
    $cmpCmd = "& `"$Python`" scripts/eval.py --config configs/full_800.yaml --compare-smoke-vs-full --smoke-checkpoint artifacts/baseline/lora_baseline.safetensors --full-checkpoint artifacts/full_800/lora_full_800.safetensors --out artifacts/full_800"
    Run-ShortStep -Name "Step 5 smoke-vs-full comparison" -Command $cmpCmd -StepLog $cmpLog -MaxAttempts 2

    $pkgLog = Join-Path $LogDir "06_package_offline_v2.log"
    $pkgCmd = "& `"$Python`" scripts/package_offline.py --ckpt artifacts/full_800/lora_full_800.safetensors --out mpid_offline_v2"
    Run-ShortStep -Name "Step 6 offline package rebuild" -Command $pkgCmd -StepLog $pkgLog -MaxAttempts 2

    $smokeOfflineLog = Join-Path $LogDir "07_smoke_offline_v2.log"
    $smokeOfflineCmd = "& `"$Python`" scripts/smoke_offline.py --pkg mpid_offline_v2"
    Run-ShortStep -Name "Step 7 offline smoke validation" -Command $smokeOfflineCmd -StepLog $smokeOfflineLog -MaxAttempts 2

    Write-ExecLog "Workflow finished successfully."
}
catch {
    Write-ExecLog "Workflow stopped due to error: $($_.Exception.Message)"
    throw
}
