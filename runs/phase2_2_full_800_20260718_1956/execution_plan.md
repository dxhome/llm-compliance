# Phase 2.2 Execution Plan: phase2_2_full_800_20260718_1956

- config: `C:\work\llm-compliance\runs\phase2_2_full_800_20260718_1956\configs\train.yaml`
- train_jsonl: `C:\work\llm-compliance\runs\phase2_2_full_800_20260718_1956\data\train.jsonl`
- checkpoint: `C:\work\llm-compliance\runs\phase2_2_full_800_20260718_1956\artifacts\checkpoints\lora_full_800.safetensors`
- offline_dir: `C:\work\llm-compliance\runs\phase2_2_full_800_20260718_1956\artifacts\package\mpid_offline`
- train records: **800**
- eval records: **500**
- train steps: **800**
- total estimate: **12h 10m 39s**

## Data Distribution

- train: `{'total': 800, 'labels': {'direct': 649, 'indirect': 63, 'clean': 88}}`
- val: `{'total': 2565, 'labels': {'direct': 2068, 'clean': 297, 'indirect': 200}}`

## Steps

| step | estimate | log |
|---|---:|---|
| Preflight checks | 5s | `C:\work\llm-compliance\runs\phase2_2_full_800_20260718_1956\logs\01_preflight.log` |
| Smoke training check | 12m 0s | `C:\work\llm-compliance\runs\phase2_2_full_800_20260718_1956\logs\02_smoke_train.log` |
| Train phase2_2_full_800_20260718_1956 | 7h 14m 4s | `C:\work\llm-compliance\runs\phase2_2_full_800_20260718_1956\logs\03_train.log` |
| Single-model eval (500 stratified records) | 1h 31m 30s | `C:\work\llm-compliance\runs\phase2_2_full_800_20260718_1956\logs\04_eval.log` |
| Smoke-vs-full comparison (500 stratified records) | 3h 3m 0s | `C:\work\llm-compliance\runs\phase2_2_full_800_20260718_1956\logs\05_compare.log` |
| Offline package rebuild | 5m 0s | `C:\work\llm-compliance\runs\phase2_2_full_800_20260718_1956\logs\06_package.log` |
| Offline smoke validation | 5m 0s | `C:\work\llm-compliance\runs\phase2_2_full_800_20260718_1956\logs\07_offline_smoke.log` |
