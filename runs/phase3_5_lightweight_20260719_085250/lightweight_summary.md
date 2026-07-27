# Phase 3-5 Lightweight Execution Summary

Run directory: `C:\work\llm-compliance\runs\phase3_5_lightweight_20260719_085250`

## Scope

- Lightweight Phase 3-5 validation was executed without modifying training code or active Phase 2.2 training run directories.
- Existing completed checkpoint used for C4 smoke: `C:\work\llm-compliance\runs\phase2_2_full_500_20260718_1956\artifacts\checkpoints\lora_full_500_restart.safetensors`
- Active training run `runs\phase2_2_balanced_600_20260718_1955` was not touched.

## Results

| Area | Status | Key result | Logs / artifacts |
| --- | --- | --- | --- |
| C4 early-exit tests | PASS | 13 tests passed | `logs\02_c4_tests.log` |
| C4 text-only eval smoke | PASS | 10 samples, exit_rate=0.0, acc=0.7, macro_f1=0.4333, f1_delta=0.0 | `logs\03c_c4_eval_text_only_10.log`, `artifacts\c4\early_exit_compare.json` |
| C5 rules tests | PASS | 3 tests passed | `logs\04_c5_tests.log` |
| C5 rules smoke | PASS | 20 samples, blocked=10, direct_recall_light=0.625, clean_fpr_light=0.0 | `logs\05_c5_rules_smoke.log`, `artifacts\c5\rules_smoke_report.json` |
| C6 crossmodal tests | PASS | 3 tests passed | `logs\06_c6_tests.log` |
| C6 crossmodal smoke | PASS | 20 samples, suspicious=2, indirect_recall_light=1.0, clean_fpr_light=0.0 | `logs\07_c6_crossmodal_smoke.log`, `artifacts\c6\crossmodal_smoke_report.json` |
| Lightweight pipeline tests | PASS | 4 tests passed after CLI patch | `logs\08b_pipeline_tests_after_cli_patch.log` |
| Lightweight pipeline CLI smoke | PASS | C4 allow, C5 block, C6 block, fallback defer all exercised | `logs\09b_pipeline_cli_smoke.log`, `artifacts\pipeline\pipeline_cli_smoke.jsonl` |
| Offline package build | PASS | Package built: 988.84 MB, 78 files | `logs\10_package_offline.log`, `artifacts\package\mpid_offline_light` |
| Offline package import smoke | PASS | Packaged C4/C5/C6 lightweight pipeline imported and exercised without loading backbone | `logs\11_package_import_smoke.log` |

## Implementation Notes

- Added a lightweight C5 rules engine under `src\mpid\rules`.
- Added a lightweight C6 crossmodal heuristic under `src\mpid\crossmodal`.
- Added a lightweight C4/C5/C6 pipeline under `src\mpid\infer`.
- Added smoke CLIs:
  - `scripts\eval_rules.py`
  - `scripts\eval_crossmodal.py`
  - `scripts\infer_pipeline_light.py`
- `scripts\infer_pipeline_light.py` now supports `--metadata-format` so PowerShell smoke runs do not depend on fragile JSON shell escaping.

## Known Limitations

- C4 smoke used a 10-sample text-only dataset because original validation image paths are unavailable in this workspace snapshot.
- C4 early-exit did not trigger on this tiny sample, so latency savings are not demonstrated yet.
- C5/C6 metrics are lightweight smoke indicators, not final acceptance metrics.
- C6 heuristic does not perform OCR or image-pixel analysis; it checks metadata/path/text signals only.
- Full Phase 3-5 validation should be rerun after the new Phase 2.2 checkpoint from the other session is ready.
