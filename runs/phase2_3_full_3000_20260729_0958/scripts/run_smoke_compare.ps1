$ErrorActionPreference = 'Stop'
$repo = 'C:\work\llm-compliance'
$run = Join-Path $repo 'runs\phase2_3_full_3000_20260729_0958'
Set-Location $repo

& "$repo\.venv\Scripts\python.exe" -X utf8 -u scripts\eval.py `
  --config "$run\configs\smoke.yaml" `
  --compare-smoke-vs-full `
  --smoke-checkpoint "$repo\runs\phase2_3_full_2000_20260727_1211\artifacts\checkpoints\checkpoint_step_2000.safetensors" `
  --full-checkpoint "$run\artifacts\smoke\checkpoints\checkpoint_step_20.safetensors" `
  --val "$run\data\resume_smoke.jsonl" `
  --out "$run\artifacts\smoke\compare_resume_smoke" `
  --chunk-size 18 `
  --chunk-output-dir "$run\artifacts\smoke\compare_resume_smoke\chunks" `
  *> "$run\logs\smoke_compare.log"
