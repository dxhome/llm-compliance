# MPID Standard Benchmark v1 Cross-Model Evaluation

- Dataset: `C:\work\llm-compliance\benchmarks\mpid_standard_v1`
- Samples: 300 total; 100 each for `clean`, `direct`, and `indirect`
- Pipelines per checkpoint: `MPID LoRA` and `MPID LoRA + C4-C6 optimized`
- Progress logs: each model writes `logs\compare.log` every 5 samples with elapsed time, average latency, ETA, partial metrics, generation count, and stage counts.

## Checkpoints

| Model | Config | Checkpoint |
|---|---|---|
| balanced-600 | `runs\phase2_2_balanced_600_20260718_1955\configs\train.yaml` | `lora_balanced_600.safetensors` |
| full-2000 | `runs\phase2_3_full_2000_20260727_1211\configs\train.yaml` | `lora_phase2_3_best_by_min_class_f1.safetensors` |

## Acceptance Rules

- Run the identical frozen benchmark and inference options for both checkpoints.
- Do not use results for threshold tuning or checkpoint selection.
- Persist per-sample predictions, timing/stage details, JSON summary, and Markdown summary for each model.
- Stop on a failing compare command so later results cannot be mistaken for a complete comparison.
