$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $RepoRoot "logs\phase2_2_full_500"
$ExecLog = Join-Path $RepoRoot "logs\phase2_2_full_500_execution_log.md"
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

function Merge-StepLog {
    param(
        [string]$StepLog,
        [string]$StdoutLog,
        [string]$StderrLog
    )

    if (Test-Path $StdoutLog) {
        Get-Content $StdoutLog -Raw -ErrorAction SilentlyContinue | Add-Content -Path $StepLog -Encoding utf8
    }
    if ((Test-Path $StderrLog) -and ((Get-Item $StderrLog).Length -gt 0)) {
        Add-Content -Path $StepLog -Encoding utf8 -Value "`n[stderr]`n"
        Get-Content $StderrLog -Raw -ErrorAction SilentlyContinue | Add-Content -Path $StepLog -Encoding utf8
    }
}

function Initialize-StepLog {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$StepLog
    )

    $stdoutLog = "$StepLog.stdout"
    $stderrLog = "$StepLog.stderr"
    $cmdLine = $FilePath + " " + ($ArgumentList -join " ")
    "[{0}] COMMAND: {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $cmdLine | Set-Content -Path $StepLog -Encoding utf8
    if (Test-Path $stdoutLog) { Remove-Item $stdoutLog -Force }
    if (Test-Path $stderrLog) { Remove-Item $stderrLog -Force }

    return @{
        StdoutLog = $stdoutLog
        StderrLog = $stderrLog
        StepLog = $StepLog
    }
}

function Run-ShortStep {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$StepLog,
        [int]$MaxAttempts = 2
    )

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Write-ExecLog "$Name started (attempt $attempt/$MaxAttempts)."
        $run = Initialize-StepLog -FilePath $FilePath -ArgumentList $ArgumentList -StepLog $StepLog
        Push-Location $RepoRoot
        try {
            $oldErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            & $FilePath @ArgumentList 1>> $run.StdoutLog 2>> $run.StderrLog
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $oldErrorActionPreference
            Pop-Location
        }
        Merge-StepLog -StepLog $run.StepLog -StdoutLog $run.StdoutLog -StderrLog $run.StderrLog

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
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$StepLog,
        [int]$StatusIntervalMinutes = 20
    )

    Write-ExecLog "$Name started."
    $run = Initialize-StepLog -FilePath $FilePath -ArgumentList $ArgumentList -StepLog $StepLog
    $exitFile = "$StepLog.exitcode"
    if (Test-Path $exitFile) { Remove-Item $exitFile -Force }
    $job = Start-Job -ScriptBlock {
        param($RepoRoot, $FilePath, $ArgumentList, $StdoutLog, $StderrLog, $ExitFile)
        Set-Location $RepoRoot
        $ErrorActionPreference = "Continue"
        & $FilePath @ArgumentList 1>> $StdoutLog 2>> $StderrLog
        Set-Content -Path $ExitFile -Value $LASTEXITCODE
    } -ArgumentList $RepoRoot, $FilePath, $ArgumentList, $run.StdoutLog, $run.StderrLog, $exitFile

    while ($job.State -eq "Running" -or $job.State -eq "NotStarted") {
        Start-Sleep -Seconds ($StatusIntervalMinutes * 60)
        $job = Get-Job -Id $job.Id
        if ($job.State -eq "Running" -or $job.State -eq "NotStarted") {
            $sizeKb = 0
            if (Test-Path $run.StdoutLog) {
                $sizeKb = [math]::Round((Get-Item $run.StdoutLog).Length / 1KB, 1)
            }
            $tail = ""
            if (Test-Path $run.StdoutLog) {
                $tail = (Get-Content $run.StdoutLog -Tail 3 -ErrorAction SilentlyContinue) -join " || "
            }
            Write-ExecLog "$Name status update: still running, job_id=$($job.Id), log_size_kb=$sizeKb, tail=$tail"
        }
    }

    Receive-Job -Id $job.Id -Keep | Out-Null
    Remove-Job -Id $job.Id -Force | Out-Null

    Merge-StepLog -StepLog $run.StepLog -StdoutLog $run.StdoutLog -StderrLog $run.StderrLog

    $exitCode = 1
    if (Test-Path $exitFile) {
        $exitCode = [int](Get-Content $exitFile -Raw)
    }

    if ($exitCode -ne 0) {
        Write-ExecLog "$Name failed with exit code $exitCode."
        throw "$Name failed."
    }
    Write-ExecLog "$Name completed successfully."
}

Set-Content -Path $PidFile -Value $PID
Write-ExecLog "Runner started. PID=$PID"

try {
    Write-ExecLog "Step 1: preflight checks started."
    Test-RequiredPath $Python
    Test-RequiredPath (Join-Path $RepoRoot "configs\full_500_restart.yaml")
    Test-RequiredPath (Join-Path $RepoRoot "configs\benchmark_100.yaml")
    Test-RequiredPath (Join-Path $RepoRoot "data\mpid-v1\train.jsonl")
    Test-RequiredPath (Join-Path $RepoRoot "data\mpid-v1\val.jsonl")
    Test-RequiredPath (Join-Path $RepoRoot "models\smolvlm-500m\config.json")
    Test-RequiredPath (Join-Path $RepoRoot "artifacts\baseline\lora_baseline.safetensors")
    Write-ExecLog "Step 1: preflight checks passed."

    $smokeLog = Join-Path $LogDir "02_smoke_train.log"
    Run-ShortStep `
        -Name "Step 2 smoke training check" `
        -FilePath $Python `
        -ArgumentList @("-X", "utf8", "-u", "scripts/train.py", "--config", "configs/benchmark_100.yaml", "--preload-dataset", "--max-train-steps", "20", "--checkpoint-name", "lora_benchmark_100.safetensors", "--partial-name", "lora_benchmark_100.safetensors") `
        -StepLog $smokeLog `
        -MaxAttempts 2

    $trainLog = Join-Path $LogDir "03_full_500_train.log"
    Run-LongStep `
        -Name "Step 3 full 500-sample training" `
        -FilePath $Python `
        -ArgumentList @("-X", "utf8", "-u", "scripts/train.py", "--config", "configs/full_500_restart.yaml", "--preload-dataset", "--save-every", "100", "--max-train-seconds", "172800", "--checkpoint-name", "lora_full_500_restart.safetensors", "--partial-name", "lora_full_500_restart.safetensors") `
        -StepLog $trainLog `
        -StatusIntervalMinutes 20

    $evalLog = Join-Path $LogDir "04_eval_full_500.log"
    Run-ShortStep `
        -Name "Step 4 single-model evaluation" `
        -FilePath $Python `
        -ArgumentList @("-X", "utf8", "scripts/eval.py", "--config", "configs/full_500_restart.yaml", "--checkpoint", "artifacts/full_500_restart/lora_full_500_restart.safetensors", "--out", "artifacts/full_500_restart") `
        -StepLog $evalLog `
        -MaxAttempts 2

    $cmpLog = Join-Path $LogDir "05_compare_smoke_vs_full.log"
    Run-ShortStep `
        -Name "Step 5 smoke-vs-full comparison" `
        -FilePath $Python `
        -ArgumentList @("-X", "utf8", "scripts/eval.py", "--config", "configs/full_500_restart.yaml", "--compare-smoke-vs-full", "--smoke-checkpoint", "artifacts/baseline/lora_baseline.safetensors", "--full-checkpoint", "artifacts/full_500_restart/lora_full_500_restart.safetensors", "--out", "artifacts/full_500_restart") `
        -StepLog $cmpLog `
        -MaxAttempts 2

    $pkgLog = Join-Path $LogDir "06_package_offline_v2.log"
    Run-ShortStep `
        -Name "Step 6 offline package rebuild" `
        -FilePath $Python `
        -ArgumentList @("-X", "utf8", "scripts/package_offline.py", "--ckpt", "artifacts/full_500_restart/lora_full_500_restart.safetensors", "--out", "mpid_offline_v2") `
        -StepLog $pkgLog `
        -MaxAttempts 2

    $smokeOfflineLog = Join-Path $LogDir "07_smoke_offline_v2.log"
    Run-ShortStep `
        -Name "Step 7 offline smoke validation" `
        -FilePath $Python `
        -ArgumentList @("-X", "utf8", "scripts/smoke_offline.py", "--pkg", "mpid_offline_v2") `
        -StepLog $smokeOfflineLog `
        -MaxAttempts 2

    Write-ExecLog "Workflow finished successfully."
}
catch {
    Write-ExecLog "Workflow stopped due to error: $($_.Exception.Message)"
    throw
}
