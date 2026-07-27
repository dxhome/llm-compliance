# Phase 2.2 Execution Plan: phase2_2_balanced_600_20260718_1955

- config: `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\configs\train.yaml`
- train_jsonl: `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\data\train.jsonl`
- checkpoint: `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\artifacts\checkpoints\lora_balanced_600.safetensors`
- offline_dir: `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\artifacts\package\mpid_offline`
- train records: **600**
- eval records per label set: **100**
- train steps: **600**
- total estimate: **7h 45m 47s**

## Data Distribution

- train: `{'total': 600, 'labels': {'direct': 200, 'indirect': 200, 'clean': 200}}`
- val: `{'total': 2565, 'labels': {'direct': 2068, 'clean': 297, 'indirect': 200}}`

## Label Eval Sets

- clean: `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\data\eval_clean_100.jsonl`
- direct: `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\data\eval_direct_100.jsonl`
- indirect: `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\data\eval_indirect_100.jsonl`

## Steps

| step | estimate | log |
|---|---:|---|
| Preflight checks | 5s | `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\logs\01_preflight.log` |
| Smoke training check | 12m 0s | `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\logs\02_smoke_train.log` |
| Train phase2_2_balanced_600_20260718_1955 | 5h 26m 12s | `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\logs\03_train.log` |
| Build label-only eval sets (100 records each) | 30s | `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\logs\04_build_eval_sets.log` |
| Clean-only smoke-vs-full comparison (100 records) | 39m 0s | `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\logs\05_compare_clean.log` |
| Direct-only smoke-vs-full comparison (100 records) | 39m 0s | `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\logs\06_compare_direct.log` |
| Indirect-only smoke-vs-full comparison (100 records) | 39m 0s | `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\logs\07_compare_indirect.log` |
| Offline package rebuild | 5m 0s | `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\logs\08_package.log` |
| Offline smoke validation | 5m 0s | `C:\work\llm-compliance\runs\phase2_2_balanced_600_20260718_1955\logs\09_offline_smoke.log` |
