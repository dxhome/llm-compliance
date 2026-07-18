---
name: phase2-progress-check
description: Monitor long-running Phase 2.2 workflows in llm-compliance and produce compact Chinese progress checks. Use when Codex needs to inspect a running launch.ps1 workflow, judge whether the current phase is healthy, slow, stalled, or failed, and report a short plain-text checklist for smoke, train, eval, compare, package, or offline-smoke phases.
---

# Phase2 Progress Check

Use this skill to turn raw workflow state into a compact operator update.

## Workflow

1. Read `status.json` and the current step log from the run directory.
2. Prefer running `python scripts/render_phase2_progress_check.py --run-dir <run_dir>` when available.
3. Base the verdict on actual movement, phase semantics, and stderr signals, not only on the top-level status field.
4. If there is a traceback, repeated error, or likely stall, say so clearly and propose the next diagnostic action.

## Decision Rules

- Mark healthy when the current phase is still producing progress and stderr only contains low-signal warnings.
- Mark watch when the phase is slow or the latest signal is thin, but there is still evidence of movement.
- Mark intervene when the phase failed, stderr shows a real error, or the process appears stalled.

## Phase-Specific Analysis

- For `smoke` or `train`, include progress, latest loss, loss trend, and whether training looks normal.
- For `eval` or `compare`, include only a concise metric judgement unless a deeper metric read is necessary.
- For `package` or `offline_smoke`, focus on whether the phase is still advancing and whether there is any blocking error.

## Output Style

- Output in Chinese.
- Output as plain text only.
- Use short list-style lines.
- Do not print raw log tails.
- Do not print detailed artifact inventories.
- Only include detailed paths or logs when the run is failing and that detail is needed for action.

## Output Contract

Produce the checkpoint as short plain text in this order:

1. Basic status
   - check time
   - overall status
   - current phase
   - current phase status
   - health verdict

2. Current phase summary
   - elapsed time
   - estimated total time
   - estimated remaining time
   - one-line judgement

3. Phase analysis
   - For train or smoke: progress, latest loss, loss trend, training judgement, ETA if visible
   - For eval or compare: concise metric judgement
   - For other phases: concise high-level judgement

4. Key signals
   - whether there is a clear error signal
   - one short artifact summary only if useful

5. Next action
   - what to do before the next 10-minute check
