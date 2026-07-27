# Phase 3-5 Balanced-600 Compare Execution Log

- Started: 2026-07-19 18:39:17
- Config: `runs\phase2_2_balanced_600_20260718_1955\configs\train.yaml`
- Checkpoint: `runs\phase2_2_balanced_600_20260718_1955\artifacts\checkpoints\lora_balanced_600.safetensors`
- Pipelines: `VLM only`, `C5 + C6A + VLM/C4 fallback`
- Note: C5/C6A are run before VLM so their real early decisions can reduce VLM calls; C4 is accounted on VLM fallback probabilities.

- Finished: 2026-07-19 18:53:16
- Summary: `runs\phase3_5_compare_balanced_600_20260719_1838\artifacts\compare_summary.md`
- Per-sample: `runs\phase3_5_compare_balanced_600_20260719_1838\artifacts\per_sample.jsonl`
