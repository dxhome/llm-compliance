# MPID Standard Benchmark v1

Frozen cross-model benchmark for final MPID evaluation. It contains exactly 100 clean, 100 direct, and 100 indirect records. Do not use these records for training, threshold selection, checkpoint selection, or prompt tuning.

Image paths in JSONL are relative to this directory. Consumers must resolve them against the benchmark root before inference. `manifest.json` and `checksums.sha256` are the integrity contract.
