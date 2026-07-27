param(
    [int]$EvalRecords = 500,
    [int]$EvalSeed = 42,
    [int]$EvalChunkSize = 50,
    [int]$StatusIntervalSeconds = 600
)

$ErrorActionPreference = "Stop"
$RunDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RepoRoot = Split-Path -Parent (Split-Path -Parent $RunDir)

& (Join-Path $RepoRoot "scripts\run_phase2_workflow.ps1") `
    -RunName "phase2_2_full_500" `
    -RunDir $RunDir `
    -Config (Join-Path $RunDir "configs\train.yaml") `
    -SmokeConfig (Join-Path $RunDir "configs\smoke.yaml") `
    -SmokeCheckpoint (Join-Path $RepoRoot "runs\_templates\artifacts\checkpoints\lora_baseline.safetensors") `
    -EvalRecords $EvalRecords `
    -EvalSeed $EvalSeed `
    -EvalChunkSize $EvalChunkSize `
    -SkipSmokeTrain `
    -SkipPackage `
    -StatusIntervalSeconds $StatusIntervalSeconds
