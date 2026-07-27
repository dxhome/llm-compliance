# Phase 2.2 Execution Plan: phase2_2_full_500_20260718_1956

- config: `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\configs\train.yaml`
- train_jsonl: `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\data\train.jsonl`
- checkpoint: `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\artifacts\checkpoints\lora_full_500_restart.safetensors`
- offline_dir: `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\artifacts\package\mpid_offline`
- train records: **500**
- eval records: **500**
- train steps: **500**
- total estimate: **9h 28m 51s**

## Data Distribution

- train: `{'total': 500, 'labels': {'direct': 403, 'indirect': 46, 'clean': 51}}`
- val: `{'total': 2565, 'labels': {'direct': 2068, 'clean': 297, 'indirect': 200}}`

## Steps

| step | estimate | log |
|---|---:|---|
| Preflight checks | 5s | `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\logs\01_preflight.log` |
| Smoke training check | 12m 0s | `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\logs\02_smoke_train.log` |
| Train phase2_2_full_500_20260718_1956 | 4h 32m 16s | `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\logs\03_train.log` |
| Single-model eval (500 stratified records) | 1h 31m 30s | `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\logs\04_eval.log` |
| Smoke-vs-full comparison (500 stratified records) | 3h 3m 0s | `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\logs\05_compare.log` |
| Offline package rebuild | 5m 0s | `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\logs\06_package.log` |
| Offline smoke validation | 5m 0s | `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\logs\07_offline_smoke.log` |
