# Phase 3-5 Lightweight Execution Plan

Run: runs\phase3_5_lightweight_20260719_085250
Started: 2026-07-19 08:52:50

Scope:
- Lightweight C4/C5/C6 end-to-end smoke only.
- Do not edit training scripts or the active Phase 2.2 run.
- Use existing/old checkpoint for pipeline smoke when needed.

Frozen files during active Phase 2.2 training:
- scripts/train.py
- src/mpid/train/trainer.py
- src/mpid/adapters/vlm.py
- src/mpid/data/*
- runs/phase2_2_balanced_600_20260718_1955/*
