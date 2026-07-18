# Phase 2.2 Workflow Ops

This note documents the new non-training operational checks for Phase 2.2 runs.

## Readiness Check

Use the readiness check before starting a run when you want more confidence than
the old "path exists" preflight.

```powershell
.\.venv\Scripts\python.exe scripts\check_phase2_readiness.py `
  --config runs\<run_id>\configs\train.yaml `
  --run-dir runs\<run_id> `
  --check-write-access
```

Optional higher-confidence probe:

```powershell
.\.venv\Scripts\python.exe scripts\check_phase2_readiness.py `
  --config runs\<run_id>\configs\train.yaml `
  --run-dir runs\<run_id> `
  --check-write-access `
  --probe-model-load
```

What it checks:

- core imports
- workflow script presence
- config resolution
- train/val JSONL readability and label sanity
- local backbone files
- run-local output directories and optional write access
- optional offline processor load

## 10-Minute Progress Check

Use the renderer below every 10 minutes after `launch.ps1` starts:

```powershell
.\.venv\Scripts\python.exe scripts\render_phase2_progress_check.py `
  --run-dir runs\<run_id>
```

It prints a Markdown snapshot with:

- current step and status
- elapsed vs estimate
- stdout/stderr tail
- execution log tail
- key artifact presence
- next-action reminders

The generic template now lives in the repository skill reference:

`skills/phase2-progress-check/references/phase2_progress_check_template.md`

When you want Codex to apply the monitoring workflow explicitly, invoke:

`$phase2-progress-check`

Recommended cadence:

1. Run the renderer every 10 minutes.
2. If `stderr_tail` shows a traceback, switch from monitoring to intervention.
3. If a step exceeds estimate with no fresh stdout, treat it as a possible stall.
