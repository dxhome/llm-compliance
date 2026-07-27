param(
    [int]$EvalRecords = 500,
    [int]$EvalSeed = 42,
    [int]$EvalChunkSize = 50,
    [switch]$SkipSmokeTrain,
    [switch]$SkipCompare,
    [switch]$SkipPackage,
    [switch]$PlanOnly,
    [switch]$PreflightOnly,
    [int]$StatusIntervalSeconds = 600
)

$ErrorActionPreference = "Stop"
$RunDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RepoRoot = Split-Path -Parent (Split-Path -Parent $RunDir)

& (Join-Path $RepoRoot "scripts\run_phase2_workflow.ps1") `
    -RunName "phase2_2_full_800_20260718_1956" `
    -RunDir $RunDir `
    -Config (Join-Path $RunDir "configs\train.yaml") `
    -SmokeConfig (Join-Path $RunDir "configs\smoke.yaml") `
    -EvalRecords $EvalRecords `
    -EvalSeed $EvalSeed `
    -EvalChunkSize $EvalChunkSize `
    -SkipSmokeTrain:$SkipSmokeTrain `
    -SkipCompare:$SkipCompare `
    -SkipPackage:$SkipPackage `
    -PlanOnly:$PlanOnly `
    -PreflightOnly:$PreflightOnly `
    -StatusIntervalSeconds $StatusIntervalSeconds
