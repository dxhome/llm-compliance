$ErrorActionPreference = 'Stop'
$repo = 'C:\work\llm-compliance'
$run = Join-Path $repo 'runs\phase2_3_full_3000_20260729_0958'
Set-Location $repo

& "$repo\.venv\Scripts\python.exe" -X utf8 -u scripts\train.py `
  --config "$run\configs\smoke.yaml" `
  *> "$run\logs\smoke_train.log"

& "$repo\.venv\Scripts\python.exe" -X utf8 -u scripts\eval.py `
  --config "$run\configs\smoke.yaml" `
  --checkpoint "$run\artifacts\smoke\checkpoints\lora_phase2_3_full_3000_smoke.safetensors" `
  --val "$run\data\resume_smoke.jsonl" `
  --out "$run\artifacts\smoke\compare_resume_smoke" `
  --chunk-size 50 `
  --chunk-output-dir "$run\artifacts\smoke\compare_resume_smoke\chunks" `
  *> "$run\logs\smoke_compare.log"
