# Full-2000 C4-C6B-lite Execution Plan

## Frozen inputs

- Checkpoint: `runs/phase2_3_full_2000_20260727_1211/artifacts/checkpoints/lora_phase2_3_best_by_min_class_f1.safetensors`
- Train config: `runs/phase2_3_full_2000_20260727_1211/configs/train.yaml`
- Baseline eval data: `runs/_datasets/mpid-v1/{test.jsonl}` and the Phase 2.3 labeled evaluation sets when present.
- Cross-modal eval data: `runs/_datasets/mpid-v1-crossmodal/test.jsonl`
- Final package: `runs/_artifact/full_2000_c4_c6blite/`

## Steps

1. Smoke the full-2000 checkpoint through both inference pipelines.
2. Measure full-2000 probabilities and select a C4 clean threshold from held-out data.
3. Verify C5 precision and C6A behavior without modifying training code.
4. Implement C6B-lite with local OCR, explainable conflict rules, and regression tests.
5. Compare LoRA-only and optimized generation pipelines on 100 clean, 100 direct, and 100 indirect samples.
6. Package the runtime, full-2000 checkpoint, local backbone, OCR assets, rules, source, and documentation.
7. Smoke the package from an isolated copied directory and record the result.
8. Update the project reference and delivery report.

## Guardrails

- C6B-lite must inspect image pixels. It must not use dataset-only fields such as `ocr_text`, `source`, or `template_id` for runtime decisions.
- Keep the existing training-related working-tree changes untouched.
- Long-running commands write progress at least every five samples; command failures receive at most three bounded repair attempts.
