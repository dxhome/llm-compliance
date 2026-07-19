param(
    [string]$RunName = "balanced_600",
    [string]$RunDir = "",
    [string]$Config = "",
    [string]$SmokeConfig = "",
    [int]$EvalRecords = 100,
    [int]$EvalSeed = 42,
    [int]$EvalChunkSize = 50,
    [string]$SmokeCheckpoint = "",
    [string]$OfflineDir = "",
    [switch]$PlanOnly,
    [switch]$PreflightOnly,
    [ValidateSet("full", "compare", "direct", "indirect", "package")]
    [string]$StartAt = "full",
    [switch]$SkipSmokeTrain,
    [switch]$SkipCompare,
    [switch]$SkipPackage,
    [int]$StatusIntervalSeconds = 600
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if ($RunDir) {
    $RunDir = if ([System.IO.Path]::IsPathRooted($RunDir)) {
        $RunDir
    }
    else {
        Join-Path $RepoRoot $RunDir
    }
}
else {
    $RunDir = Join-Path $RepoRoot "runs\$RunName"
}
$Config = if ($Config) { $Config } else { Join-Path $RunDir "configs\train.yaml" }
$SmokeConfig = if ($SmokeConfig) { $SmokeConfig } else { Join-Path $RunDir "configs\smoke.yaml" }
$SmokeCheckpoint = if ($SmokeCheckpoint) { $SmokeCheckpoint } else { Join-Path $RunDir "artifacts\smoke\lora_benchmark_100.safetensors" }
$LogDir = Join-Path $RunDir "logs"
$ExecLog = Join-Path $RunDir "execution_log.md"
$PlanJson = Join-Path $RunDir "execution_plan.json"
$PlanMd = Join-Path $RunDir "execution_plan.md"
$StatusJson = Join-Path $RunDir "status.json"
$CurrentStepFile = Join-Path $RunDir "current_step.txt"
$ReadinessJson = Join-Path $RunDir "logs\01_preflight.readiness.json"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RunDir "scripts") | Out-Null
$RunLauncher = Join-Path $RunDir "scripts\launch.ps1"
$RunLauncherContent = @"
param(
    [int]`$EvalRecords = $EvalRecords,
    [int]`$EvalSeed = $EvalSeed,
    [int]`$EvalChunkSize = $EvalChunkSize,
    [switch]`$SkipSmokeTrain,
    [switch]`$SkipCompare,
    [switch]`$SkipPackage,
    [switch]`$PlanOnly,
    [switch]`$PreflightOnly,
    [ValidateSet("full", "compare", "direct", "indirect", "package")]
    [string]`$StartAt = "$StartAt",
    [int]`$StatusIntervalSeconds = 600
)

`$ErrorActionPreference = "Stop"
`$RunDir = Split-Path -Parent `$PSScriptRoot
`$RepoRoot = Split-Path -Parent (Split-Path -Parent `$RunDir)

& (Join-Path `$RepoRoot "scripts\run_phase2_workflow.ps1") ``
    -RunName "$RunName" ``
    -RunDir `$RunDir ``
    -Config (Join-Path `$RunDir "configs\train.yaml") ``
    -SmokeConfig (Join-Path `$RunDir "configs\smoke.yaml") ``
    -EvalRecords `$EvalRecords ``
    -EvalSeed `$EvalSeed ``
    -EvalChunkSize `$EvalChunkSize ``
    -SkipSmokeTrain:`$SkipSmokeTrain ``
    -SkipCompare:`$SkipCompare ``
    -SkipPackage:`$SkipPackage ``
    -PlanOnly:`$PlanOnly ``
    -PreflightOnly:`$PreflightOnly ``
    -StartAt `$StartAt ``
    -StatusIntervalSeconds `$StatusIntervalSeconds
"@
Set-Content -Path $RunLauncher -Value $RunLauncherContent -Encoding utf8

function Write-ExecLog {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "- [$ts] $Message"
    for ($i = 0; $i -lt 10; $i++) {
        try {
            Add-Content -Path $ExecLog -Value $line -Encoding utf8
            return
        }
        catch [System.IO.IOException] {
            Start-Sleep -Milliseconds 300
        }
    }
    Add-Content -Path $ExecLog -Value $line -Encoding utf8
}

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path $RepoRoot $PathValue)
}

function Quote-Arg {
    param([string]$Value)
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Write-Status {
    param(
        [string]$RunStatus,
        [string]$StepId,
        [string]$StepName,
        [string]$StepStatus,
        [string]$LogPath,
        [int]$ExitCode = -999,
        [double]$ElapsedSeconds = 0,
        [double]$EstimateSeconds = 0,
        [string]$Message = ""
    )
    $status = [ordered]@{
        run_name = $RunName
        status = $RunStatus
        step_id = $StepId
        step_name = $StepName
        step_status = $StepStatus
        log = $LogPath
        stdout = "$LogPath.stdout"
        stderr = "$LogPath.stderr"
        exitcode = "$LogPath.exitcode"
        exit_code = $ExitCode
        elapsed_seconds = [math]::Round($ElapsedSeconds, 1)
        estimate_seconds = [math]::Round($EstimateSeconds, 1)
        updated_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        message = $Message
    }
    $status | ConvertTo-Json -Depth 5 | Set-Content -Path $StatusJson -Encoding utf8
    Set-Content -Path $CurrentStepFile -Value "$StepId $StepName"
}

function Invoke-LoggedProcess {
    param(
        [string]$StepId,
        [string]$StepName,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$BaseLog,
        [double]$EstimateSeconds,
        [int]$MaxAttempts = 1
    )

    $stdout = "$BaseLog.stdout"
    $stderr = "$BaseLog.stderr"
    $exitFile = "$BaseLog.exitcode"
    $startedFile = "$BaseLog.started"
    $finishedFile = "$BaseLog.finished"

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        foreach ($path in @($stdout, $stderr, $exitFile, $startedFile, $finishedFile)) {
            if (Test-Path $path) {
                Remove-Item -LiteralPath $path -Force
            }
        }

        $cmdLine = (Quote-Arg $FilePath) + " " + (($ArgumentList | ForEach-Object { Quote-Arg $_ }) -join " ")
        Set-Content -Path $BaseLog -Encoding utf8 -Value @(
            "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] STEP: $StepName",
            "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] ATTEMPT: $attempt/$MaxAttempts",
            "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] COMMAND: $cmdLine",
            "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] ESTIMATE_SECONDS: $EstimateSeconds",
            ""
        )

        $startedAt = Get-Date
        Set-Content -Path $startedFile -Value ($startedAt.ToString("yyyy-MM-dd HH:mm:ss"))
        Write-ExecLog "$StepName started (attempt $attempt/$MaxAttempts). Log: $BaseLog"
        Write-Status -RunStatus "running" -StepId $StepId -StepName $StepName `
            -StepStatus "running" -LogPath $BaseLog -EstimateSeconds $EstimateSeconds

        $outWriter = [System.IO.StreamWriter]::new($stdout, $true, [System.Text.Encoding]::UTF8)
        $errWriter = [System.IO.StreamWriter]::new($stderr, $true, [System.Text.Encoding]::UTF8)
        $outWriter.AutoFlush = $true
        $errWriter.AutoFlush = $true

        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $FilePath
        # Windows PowerShell 5.1 does not reliably expose ProcessStartInfo.ArgumentList.
        # Keep process launching compatible by building the command-line arguments here.
        $psi.Arguments = (($ArgumentList | ForEach-Object { Quote-Arg $_ }) -join " ")
        $psi.WorkingDirectory = $RepoRoot
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $proc = [System.Diagnostics.Process]::new()
        $proc.StartInfo = $psi
        $proc.EnableRaisingEvents = $true

        $outEvent = Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -Action {
            if ($EventArgs.Data -ne $null) {
                $Event.MessageData.WriteLine($EventArgs.Data)
            }
        } -MessageData $outWriter
        $errEvent = Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived -Action {
            if ($EventArgs.Data -ne $null) {
                $Event.MessageData.WriteLine($EventArgs.Data)
            }
        } -MessageData $errWriter

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            [void]$proc.Start()
            $proc.BeginOutputReadLine()
            $proc.BeginErrorReadLine()

            $lastStatusAt = 0.0
            while (-not $proc.HasExited) {
                Start-Sleep -Seconds 2
                $elapsed = $sw.Elapsed.TotalSeconds
                if (($elapsed - $lastStatusAt) -lt $StatusIntervalSeconds) {
                    continue
                }
                $lastStatusAt = $elapsed
                $tail = ""
                if (Test-Path $stdout) {
                    $tail = (Get-Content $stdout -Tail 5 -ErrorAction SilentlyContinue) -join " || "
                }
                Write-ExecLog "$StepName status: running, pid=$($proc.Id), elapsed=$([math]::Round($elapsed, 1))s, estimate=${EstimateSeconds}s, stdout_tail=$tail"
                Write-Status -RunStatus "running" -StepId $StepId -StepName $StepName `
                    -StepStatus "running" -LogPath $BaseLog -ElapsedSeconds $elapsed `
                    -EstimateSeconds $EstimateSeconds -Message $tail
            }

            $proc.WaitForExit()
            Start-Sleep -Milliseconds 300
            $exitCode = $proc.ExitCode
        }
        finally {
            $sw.Stop()
            Unregister-Event -SubscriptionId $outEvent.Id -ErrorAction SilentlyContinue
            Unregister-Event -SubscriptionId $errEvent.Id -ErrorAction SilentlyContinue
            $outWriter.Dispose()
            $errWriter.Dispose()
            $proc.Dispose()
        }

        Set-Content -Path $exitFile -Value $exitCode
        Set-Content -Path $finishedFile -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

        Add-Content -Path $BaseLog -Encoding utf8 -Value "`n[stdout]`n"
        if (Test-Path $stdout) {
            Get-Content $stdout -Raw -ErrorAction SilentlyContinue | Add-Content -Path $BaseLog -Encoding utf8
        }
        if ((Test-Path $stderr) -and ((Get-Item $stderr).Length -gt 0)) {
            Add-Content -Path $BaseLog -Encoding utf8 -Value "`n[stderr]`n"
            Get-Content $stderr -Raw -ErrorAction SilentlyContinue | Add-Content -Path $BaseLog -Encoding utf8
        }

        if ($exitCode -eq 0) {
            Write-ExecLog "$StepName completed successfully in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s. Log: $BaseLog"
            Write-Status -RunStatus "running" -StepId $StepId -StepName $StepName `
                -StepStatus "completed" -LogPath $BaseLog -ExitCode $exitCode `
                -ElapsedSeconds $sw.Elapsed.TotalSeconds -EstimateSeconds $EstimateSeconds
            return
        }

        $stderrTail = ""
        if (Test-Path $stderr) {
            $stderrTail = (Get-Content $stderr -Tail 10 -ErrorAction SilentlyContinue) -join " || "
        }
        Write-ExecLog "$StepName failed with exit code $exitCode after $([math]::Round($sw.Elapsed.TotalSeconds, 1))s. stderr_tail=$stderrTail"
        Write-Status -RunStatus "running" -StepId $StepId -StepName $StepName `
            -StepStatus "failed" -LogPath $BaseLog -ExitCode $exitCode `
            -ElapsedSeconds $sw.Elapsed.TotalSeconds -EstimateSeconds $EstimateSeconds `
            -Message $stderrTail
    }

    throw "$StepName failed after $MaxAttempts attempts. See $BaseLog"
}

function Get-PlanStep {
    param($Plan, [string]$Id)
    return @($Plan.steps | Where-Object { $_.id -eq $Id })[0]
}

try {
    Set-Content -Path $ExecLog -Encoding utf8 -Value "# Phase 2.2 Workflow: $RunName`n"
    Write-ExecLog "Workflow launcher started. PID=$PID"

    if (-not (Test-Path $Python)) {
        throw "Python not found: $Python"
    }
    if (-not (Test-Path (Resolve-RepoPath $Config))) {
        throw "Config not found: $Config"
    }

    $offlineArg = if ($OfflineDir) { $OfflineDir } else { Join-Path $RunDir "artifacts\package\mpid_offline" }
    $planArgs = @(
        "-X", "utf8", "-u",
        "scripts/plan_phase2_workflow.py",
        "--config", $Config,
        "--run-name", $RunName,
        "--run-dir", $RunDir,
        "--eval-records", "$EvalRecords",
        "--log-dir", $LogDir,
        "--offline-dir", $offlineArg,
        "--json-out", $PlanJson,
        "--md-out", $PlanMd
    )
    Invoke-LoggedProcess -StepId "00_plan" -StepName "Build execution plan" `
        -FilePath $Python -ArgumentList $planArgs `
        -BaseLog (Join-Path $LogDir "00_plan.log") -EstimateSeconds 5 -MaxAttempts 1

    $plan = Get-Content $PlanJson -Raw | ConvertFrom-Json
    Write-ExecLog "Execution plan written. Plan: $PlanMd. Total estimate: $($plan.total_estimate_hms)"

    if ($PlanOnly) {
        Write-Status -RunStatus "completed" -StepId "00_plan" -StepName "Plan only" `
            -StepStatus "completed" -LogPath (Join-Path $LogDir "00_plan.log") `
            -ExitCode 0 `
            -Message "Plan-only mode completed. No training/eval/package steps were started."
        Write-ExecLog "Plan-only mode completed. No training/eval/package steps were started."
        return
    }

    $preflight = Get-PlanStep $plan "01_preflight"
    $preflightLog = $preflight.log
    Set-Content -Path $preflightLog -Encoding utf8 -Value "Preflight checks for $RunName"
    Set-Content -Path "$preflightLog.started" -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    Set-Content -Path "$preflightLog.stdout" -Value ""
    Set-Content -Path "$preflightLog.stderr" -Value ""
    Write-Status -RunStatus "running" -StepId "01_preflight" -StepName $preflight.name `
        -StepStatus "running" -LogPath $preflightLog -EstimateSeconds $preflight.estimate_seconds
    $requiredPaths = @($Python, (Resolve-RepoPath $Config), $plan.train_jsonl, $plan.val_jsonl, "runs\_models\smolvlm-500m\config.json")
    if (-not $SkipSmokeTrain) {
        $requiredPaths += (Resolve-RepoPath $SmokeConfig)
    }
    if ($SkipSmokeTrain -and -not $SkipCompare) {
        $requiredPaths += (Resolve-RepoPath $SmokeCheckpoint)
    }
    foreach ($required in $requiredPaths) {
        $resolved = Resolve-RepoPath $required
        if (-not (Test-Path $resolved)) {
            throw "Required path missing: $resolved"
        }
        Add-Content -Path $preflightLog -Value "OK: $resolved"
    }
    Add-Content -Path $preflightLog -Value ""
    Add-Content -Path $preflightLog -Value "[readiness]"
    $readinessArgs = @(
        "-X", "utf8", "-u",
        "scripts/check_phase2_readiness.py",
        "--config", $Config,
        "--run-dir", $RunDir,
        "--eval-records", "$EvalRecords",
        "--json-out", $ReadinessJson,
        "--check-write-access"
    )
    if ($SkipSmokeTrain) {
        $readinessArgs += "--skip-smoke-train"
    }
    if ($SkipCompare) {
        $readinessArgs += "--skip-compare"
    }
    if ($SkipPackage) {
        $readinessArgs += "--skip-package"
    }
    if ($SmokeConfig) {
        $readinessArgs += @("--smoke-config", $SmokeConfig)
    }
    if ($SmokeCheckpoint) {
        $readinessArgs += @("--smoke-checkpoint", $SmokeCheckpoint)
    }
    $readinessOutput = & $Python $readinessArgs 2>&1
    $readinessExit = $LASTEXITCODE
    if ($readinessOutput) {
        foreach ($line in $readinessOutput) {
            Add-Content -Path $preflightLog -Value $line
        }
    }
    if ($readinessExit -ne 0) {
        throw "Readiness check failed with exit code $readinessExit. See $preflightLog"
    }
    Set-Content -Path "$preflightLog.exitcode" -Value 0
    Set-Content -Path "$preflightLog.finished" -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    Write-ExecLog "Preflight checks completed successfully. Log: $preflightLog"

    if ($PreflightOnly) {
        Write-Status -RunStatus "completed" -StepId "01_preflight" -StepName "Preflight only" `
            -StepStatus "completed" -LogPath $preflightLog -ExitCode 0 `
            -Message "Preflight-only mode completed. No training/eval/package steps were started."
        Write-ExecLog "Preflight-only mode completed. No training/eval/package steps were started."
        return
    }

    if ($StartAt -eq "full" -and -not $SkipSmokeTrain) {
        $step = Get-PlanStep $plan "02_smoke_train"
        Invoke-LoggedProcess -StepId $step.id -StepName $step.name -FilePath $Python `
            -BaseLog $step.log -EstimateSeconds $step.estimate_seconds -MaxAttempts 2 `
            -ArgumentList @(
                "-X", "utf8", "-u",
                "scripts/train.py",
                "--config", $SmokeConfig,
                "--preload-dataset",
                "--max-train-steps", "10",
                "--checkpoint-name", "lora_benchmark_100.safetensors",
                "--partial-name", "lora_benchmark_100.safetensors",
                "--save-every", "50"
            )
    }
    else {
        Write-ExecLog "Smoke training check skipped by flag."
    }

    if ($StartAt -eq "full") {
        $step = Get-PlanStep $plan "03_train"
        Invoke-LoggedProcess -StepId $step.id -StepName $step.name -FilePath $Python `
            -BaseLog $step.log -EstimateSeconds $step.estimate_seconds -MaxAttempts 1 `
            -ArgumentList @(
                "-X", "utf8", "-u",
                "scripts/train.py",
                "--config", $plan.config,
                "--preload-dataset",
                "--save-every", "$($plan.save_every)"
            )
    }
    else {
        Write-ExecLog "Training skipped because StartAt=$StartAt."
        if (-not (Test-Path $plan.checkpoint)) {
            throw "Cannot skip training because checkpoint is missing: $($plan.checkpoint)"
        }
    }

    if ($StartAt -ne "package") {
        $step = Get-PlanStep $plan "04_build_eval_sets"
        Invoke-LoggedProcess -StepId $step.id -StepName $step.name -FilePath $Python `
            -BaseLog $step.log -EstimateSeconds $step.estimate_seconds -MaxAttempts 1 `
            -ArgumentList @(
                "-X", "utf8", "-u",
                "scripts/build_label_eval_sets.py",
                "--run-dir", $RunDir,
                "--train-jsonl", $plan.train_jsonl,
                "--out-dir", (Join-Path $RunDir "data"),
                "--records-per-label", "$EvalRecords",
                "--seed", "$EvalSeed",
                "--json-out", (Join-Path $RunDir "data\eval_label_sets_manifest.json")
            )
    }

    if ($StartAt -ne "package" -and -not $SkipCompare) {
        $compareSteps = @("05_compare_clean", "06_compare_direct", "07_compare_indirect")
        if ($StartAt -eq "direct") {
            $compareSteps = @("06_compare_direct", "07_compare_indirect")
        }
        elseif ($StartAt -eq "indirect") {
            $compareSteps = @("07_compare_indirect")
        }
        foreach ($compareId in $compareSteps) {
            $step = Get-PlanStep $plan $compareId
            $label = $step.label
            Invoke-LoggedProcess -StepId $step.id -StepName $step.name -FilePath $Python `
                -BaseLog $step.log -EstimateSeconds $step.estimate_seconds -MaxAttempts 1 `
                -ArgumentList @(
                    "-X", "utf8", "-u",
                    "scripts/eval.py",
                    "--config", $plan.config,
                    "--compare-smoke-vs-full",
                    "--smoke-checkpoint", $SmokeCheckpoint,
                    "--full-checkpoint", $plan.checkpoint,
                    "--val", $step.val_jsonl,
                    "--out", (Join-Path $RunDir "artifacts\comparison\$label"),
                    "--chunk-size", "$EvalChunkSize",
                    "--chunk-output-dir", (Join-Path $RunDir "artifacts\comparison\$label\chunks")
                )
        }
    }
    else {
        Write-ExecLog "Smoke-vs-full comparison skipped by flag."
    }

    if (-not $SkipPackage) {
        $step = Get-PlanStep $plan "08_package"
        Invoke-LoggedProcess -StepId $step.id -StepName $step.name -FilePath $Python `
            -BaseLog $step.log -EstimateSeconds $step.estimate_seconds -MaxAttempts 1 `
            -ArgumentList @(
                "-X", "utf8", "-u",
                "scripts/package_offline.py",
                "--ckpt", $plan.checkpoint,
                "--out", $plan.offline_dir,
                "--report", (Join-Path $RunDir "artifacts\package\package_offline.json")
            )

        $step = Get-PlanStep $plan "09_offline_smoke"
        Invoke-LoggedProcess -StepId $step.id -StepName $step.name -FilePath $Python `
            -BaseLog $step.log -EstimateSeconds $step.estimate_seconds -MaxAttempts 1 `
            -ArgumentList @(
                "-X", "utf8", "-u",
                "scripts/smoke_offline.py",
                "--pkg", $plan.offline_dir,
                "--stage-root", (Join-Path $RunDir "artifacts\offline_smoke_stage")
            )
    }
    else {
        Write-ExecLog "Offline package and smoke validation skipped by flag."
    }

    Write-Status -RunStatus "completed" -StepId "done" -StepName "Workflow complete" `
        -StepStatus "completed" -LogPath $ExecLog -Message "All requested steps completed."
    Write-ExecLog "Workflow finished successfully."
}
catch {
    Write-Status -RunStatus "failed" -StepId "failed" -StepName "Workflow failed" `
        -StepStatus "failed" -LogPath $ExecLog -Message $_.Exception.Message
    try {
        Write-ExecLog "Workflow stopped due to error: $($_.Exception.Message)"
    }
    catch {
        # Avoid masking the original failure when the log file is temporarily locked.
    }
    throw
}
