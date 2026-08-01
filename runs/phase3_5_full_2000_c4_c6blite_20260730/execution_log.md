# Full-2000 C4-C6B-lite Execution Log

## 2026-07-30

- Created the dedicated adaptation run.
- Frozen target package path: `runs/_artifact/full_2000_c4_c6blite/`.
- Frozen checkpoint: `runs/phase2_3_full_2000_20260727_1211/artifacts/checkpoints/lora_phase2_3_best_by_min_class_f1.safetensors`.
- Confirmed the current C6A implementation is metadata-based and does not read image pixels.
- Confirmed no supported OCR package is installed in `.venv`; OCR runtime selection is the next dependency decision.

## 2026-07-30 Smoke and model gate

- Ran a real generation-pipeline smoke on five clean, five direct, and five indirect records. Output: `artifacts/smoke_compare/`.
- The run loaded the frozen full-2000 checkpoint in 7.5 seconds and completed both pipelines without process errors.
- LoRA-only result: accuracy 0.333, Macro F1 0.167, 0 generations. All 15 records were blocked by the head.
- Current C5+C6A+C4 optimized result: accuracy 0.533, Macro F1 0.446, 0 generations. C5 blocked 1 record and C6A blocked 3 records; every clean record was still blocked by the head.
- Verified the training run's pre-existing full 200-record reports: at step 2000 clean recall is 3%, direct recall is 99%, and indirect recall is 1%. Step 1000 has the same collapsed prediction distribution.
- Decision gate: do not publish a demo-ready offline package from this checkpoint. C4 only early-exits confident clean predictions and cannot correct confident direct predictions. C5/C6B-lite can add attack recall, but cannot restore clean usability.
- OCR availability check: no `tesseract` executable and no supported Python OCR package are installed. A future C6B-lite implementation requires an approved OCR runtime download and local model bundle.

## 2026-07-30 C6B-lite implementation

- Installed `rapidocr_onnxruntime==1.2.3` with ONNX Runtime for local image-pixel OCR.
- Added `src/mpid/crossmodal/ocr_extract.py`: resolves local/legacy image paths, extracts image-pixel text, and supports package-local OCR model weights through `MPID_OCR_MODELS_DIR`.
- Added `src/mpid/crossmodal/conflict_rules.py`: conservative OCR rules for explicit instruction override, system prompt override, jailbreak, safety-disable, exfiltration, and Chinese equivalents.
- Updated `run_optimized_pipeline()` ordering to `C5 -> C6B-lite OCR -> C6A compatibility fallback -> head -> C4 -> generation`.
- Added `scripts/eval_c6b_lite.py` with five-record progress, elapsed time, average time, and ETA logging.
- Regression test result: `tests/test_crossmodal_heuristic.py tests/test_crossmodal_c6b_lite.py tests/test_pipeline_lightweight.py` => 13 passed.
- Pixel OCR smoke on `syn_0003.png`: read `Safety: OFF.` and blocked as `indirect` with reason `ocr_safety_disabled`.
- C6B-lite full-600 independent evaluation: elapsed 83.3s, indirect direct-block recall 11.5% (23/200), clean OCR-rule FPR 0.0% (0/200), 155 image-bearing records attempted OCR.

## 2026-07-30 offline package

- Updated `scripts/package_offline.py` to bundle RapidOCR ONNX weights, C4 threshold, C6B-lite manifest data, a smoke image, and README.
- Fixed `scripts/smoke_offline.py` Windows relative staging-path duplication; smoke now resolves the staging directory before launching `infer.py`.
- Built final package: `runs/_artifact/full_2000_c4_c6blite/` (1.02 GB, 87 files).
- Offline smoke result with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`: 3/3 passed; C5 rules, head fallback, and C6B-lite OCR branches all exercised; checksums verified.
- The package manifest records the accepted full-2000 clean/indirect-recall limitation.
- Added package-local `demo.py`, a zero-extra-dependency interactive terminal demo using the same protected pipeline as `infer.py`.
- Rebuilt final package after documentation/demo changes: 88 files. Final isolated smoke: 3/3 passed and 88 checksums verified.

## 2026-07-30 comprehensive evaluation report

- Added `artifacts/full_2000_c4_c6blite_comprehensive_report.md` and linked it from `doc/reference.md` test-results section.
- Combined the three pre-existing 200-record full-2000 head reports: accuracy 34.3%, Macro F1 19.3%, clean recall 3.0%, direct recall 99.0%, indirect recall 1.0%.
- Measured C5 over all 600 records: direct block rate 23.0% (46/200), clean false-block rate 1.5% (3/200).
- Measured C6A over all 600 records: indirect recall 52.5% (105/200), clean FPR 0.0%.
- Computed ordered pre-head stages: C5=58, C6B-lite=23, C6A fallback=82, head fallback=437.
- Re-ran the final C6B-lite-enabled real pipeline compare on 15 stratified smoke samples: MPID LoRA Macro F1=16.7%, optimized Macro F1=44.6%; end-to-end time reduced from 284.2s to 226.4s (-20.3%).
- Restructured the comprehensive report around the two primary benchmark definitions: `MPID LoRA` and `MPID LoRA + C4-C6优化`, with overall, per-label, timing, stage, and 600-record supporting diagnostics.
- Reorganized `doc/reference.md` section 3: 3.1 is now a cross-model/round summary; balanced-600 and full-2000 are independent sections 3.3 and 3.4; section 3.5 documents cross-model comparability and interpretation limits.
