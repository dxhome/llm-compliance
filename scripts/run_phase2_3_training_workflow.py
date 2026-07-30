"""Phase 2.3 continuous training workflow launcher.

This script is intentionally safe by default: it performs a dry run unless
``--execute`` is passed.  The real workflow starts one continuous training
process and monitors saved checkpoint files for configured compare points.
It does not stop and resume training just to run compare.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
LABELS = ("clean", "direct", "indirect")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists() or str(value).startswith(".."):
        return candidate
    return (REPO_ROOT / path).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- [{_now()}] {message}\n")


def _run_logged(command: list[str], log_path: Path, dry_run: bool) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {"returncode": 0, "seconds": 0.0, "log": str(log_path), "dry_run": True}

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[workflow] command: " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    return {
        "returncode": proc.returncode,
        "seconds": round(time.monotonic() - started, 1),
        "log": str(log_path),
        "dry_run": False,
    }


def _checkpoint_steps(out_dir: Path) -> list[int]:
    steps: list[int] = []
    for path in out_dir.glob("checkpoint_step_*.safetensors"):
        match = re.search(r"checkpoint_step_(\d+)\.safetensors$", path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def _latest_checkpoint_for_step(out_dir: Path, step: int) -> Path:
    step_ckpt = out_dir / f"checkpoint_step_{step}.safetensors"
    if step_ckpt.exists():
        return step_ckpt
    latest = out_dir / "latest.safetensors"
    if latest.exists():
        return latest
    return step_ckpt


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _metric_from_report(report: dict[str, Any], label: str) -> dict[str, float]:
    per_class = report.get("full", {}).get("per_class", {})
    row = per_class.get(label, {})
    return {
        "precision": float(row.get("precision", 0.0)),
        "recall": float(row.get("recall", 0.0)),
        "f1": float(row.get("f1-score", 0.0)),
        "support": float(row.get("support", 0.0)),
        "accuracy": float(report.get("full", {}).get("accuracy", 0.0)),
        "macro_f1": float(report.get("full", {}).get("macro_f1", 0.0)),
    }


def _baseline_metric_from_report(report: dict[str, Any], label: str) -> dict[str, float]:
    per_class = report.get("smoke", {}).get("per_class", {})
    row = per_class.get(label, {})
    return {
        "precision": float(row.get("precision", 0.0)),
        "recall": float(row.get("recall", 0.0)),
        "f1": float(row.get("f1-score", 0.0)),
        "support": float(row.get("support", 0.0)),
        "accuracy": float(report.get("smoke", {}).get("accuracy", 0.0)),
        "macro_f1": float(report.get("smoke", {}).get("macro_f1", 0.0)),
    }


def _collect_scorecard(run_dir: Path, checkpoint_dir: Path, best_name: str) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    compare_root = run_dir / "artifacts" / "compare"
    for scope in ("quick", "full"):
        for step_dir in sorted((compare_root / scope).glob("step_*")):
            step = int(step_dir.name.replace("step_", ""))
            entry = entries.setdefault(
                str(step),
                {
                    "step": step,
                    "checkpoint": str(_latest_checkpoint_for_step(checkpoint_dir, step)),
                    "quick": {},
                    "full": {},
                },
            )
            for label in LABELS:
                report = _load_json(step_dir / label / "comparison_full_vs_smoke.json")
                if not report:
                    continue
                entry[scope][label] = {
                    "new_model": _metric_from_report(report, label),
                    "baseline": _baseline_metric_from_report(report, label),
                }

    candidates = []
    for entry in entries.values():
        metrics = entry.get("full") if len(entry.get("full", {})) == 3 else entry.get("quick", {})
        if len(metrics) != 3:
            continue
        f1s = {label: metrics[label]["new_model"]["f1"] for label in LABELS}
        recalls = {label: metrics[label]["new_model"]["recall"] for label in LABELS}
        clean_recall = recalls.get("clean", 0.0)
        candidates.append(
            {
                "step": entry["step"],
                "scope": "full" if len(entry.get("full", {})) == 3 else "quick",
                "min_class_f1": min(f1s.values()),
                "macro_target_f1": sum(f1s.values()) / 3.0,
                "indirect_f1": f1s["indirect"],
                "clean_fpr_proxy": 1.0 - clean_recall,
            }
        )

    best = None
    if candidates:
        best = sorted(
            candidates,
            key=lambda row: (
                row["min_class_f1"],
                row["macro_target_f1"],
                row["indirect_f1"],
                -row["clean_fpr_proxy"],
                row["step"],
            ),
            reverse=True,
        )[0]

    return {
        "updated_at": _now(),
        "selection_metric": "best_by_min_class_f1",
        "best_checkpoint_name": best_name,
        "success_threshold": {
            "metric": "clean/direct/indirect full-set target F1",
            "threshold": 0.70,
            "all_classes_must_exceed": True,
        },
        "best": best,
        "entries": sorted(entries.values(), key=lambda row: row["step"]),
    }


def _copy_best(scorecard: dict[str, Any], checkpoint_dir: Path, best_name: str, dry_run: bool) -> None:
    best = scorecard.get("best")
    if not best:
        return
    src = checkpoint_dir / f"checkpoint_step_{best['step']}.safetensors"
    dst = checkpoint_dir / best_name
    if dry_run or not src.exists():
        return
    shutil.copy2(src, dst)


def _best_full_target_f1(scorecard: dict[str, Any]) -> dict[str, float]:
    best = scorecard.get("best")
    if not best:
        return {}
    best_step = int(best["step"])
    for entry in scorecard.get("entries", []):
        if int(entry.get("step", -1)) != best_step:
            continue
        full = entry.get("full", {})
        if len(full) != 3:
            return {}
        return {
            label: float(full[label]["new_model"]["f1"])
            for label in LABELS
            if label in full
        }
    return {}


def _passes_success_threshold(scorecard: dict[str, Any], threshold: float = 0.70) -> bool:
    f1s = _best_full_target_f1(scorecard)
    return len(f1s) == 3 and all(value > threshold for value in f1s.values())


def _build_context(run_dir: Path) -> dict[str, Any]:
    config = run_dir / "configs" / "train.yaml"
    cfg = _read_yaml(config)
    config_dir = config.parent
    training = cfg.get("training", {}) or {}
    compare = cfg.get("compare", {}) or {}
    checkpoint_policy = cfg.get("checkpoint_policy", {}) or {}
    phase = cfg.get("phase2_3", {}) or {}

    total_steps = int(training.get("max_train_steps", 2000))
    quick_cfg = compare.get("quick", {}) or {}
    explicit_quick_steps = quick_cfg.get("schedule_steps")
    if explicit_quick_steps:
        quick_steps = sorted({min(total_steps, max(1, int(step))) for step in explicit_quick_steps})
    else:
        quick_percent = quick_cfg.get(
            "schedule_percent", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        )
        quick_steps = sorted({
            min(total_steps, max(1, math.ceil(total_steps * int(percent) / 100)))
            for percent in quick_percent
        })
    full_steps = sorted({int(x) for x in compare.get("full", {}).get("schedule_steps", [])})
    observe_steps = sorted({*quick_steps, *full_steps, total_steps})
    out_dir = _resolve(cfg.get("io", {}).get("out_dir", "../artifacts/checkpoints"), config_dir)
    package_dir = run_dir / "artifacts" / "package" / "mpid_offline"
    baseline_run = _resolve(phase.get("baseline_run", ""), config_dir)
    baseline_checkpoint = baseline_run / "artifacts" / "checkpoints" / "lora_balanced_600.safetensors"
    return {
        "run_dir": run_dir,
        "config": config,
        "cfg": cfg,
        "total_steps": total_steps,
        "quick_steps": quick_steps,
        "observe_steps": observe_steps,
        "full_steps": full_steps,
        "checkpoint_dir": out_dir,
        "package_dir": package_dir,
        "package_report": run_dir / "artifacts" / "package" / "package_offline.json",
        "offline_smoke_stage": run_dir / "artifacts" / "offline_smoke_stage",
        "baseline_checkpoint": baseline_checkpoint,
        "lora_r": int((cfg.get("lora", {}) or {}).get("r", 16)),
        "lora_alpha": int((cfg.get("lora", {}) or {}).get("alpha", 32)),
        "lora_target": str((cfg.get("lora", {}) or {}).get("target", "q_proj,k_proj,v_proj,o_proj")),
        "best_name": str(checkpoint_policy.get("best_name", "lora_phase2_3_best_by_min_class_f1.safetensors")),
        "quick_sets": {
            label: _resolve(compare["quick"]["sets"][label], config_dir)
            for label in LABELS
        },
        "full_sets": {
            label: _resolve(compare["full"]["sets"][label], config_dir)
            for label in LABELS
        },
    }


def _preflight(ctx: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in ("config", "baseline_checkpoint"):
        if not Path(ctx[key]).exists():
            missing.append(f"{key}: {ctx[key]}")
    for group in ("quick_sets", "full_sets"):
        for label, path in ctx[group].items():
            if not path.exists():
                missing.append(f"{group}.{label}: {path}")
    if not PYTHON.exists():
        missing.append(f"python: {PYTHON}")
    return missing


def _full_train_command(
    ctx: dict[str, Any],
    *,
    resume_from: Path | None = None,
    resume_global_step: int = 0,
    skip_train_batches: int = 0,
) -> list[str]:
    steps_this_run = ctx["total_steps"]
    if resume_global_step > 0:
        steps_this_run = max(0, ctx["total_steps"] - resume_global_step)
    cmd = [
        str(PYTHON), "-X", "utf8", "-u",
        "scripts/train.py",
        "--config", str(ctx["config"]),
        "--save-every", "50",
        "--max-train-steps", str(steps_this_run),
    ]
    if resume_from is not None:
        cmd.extend([
            "--resume-from", str(resume_from),
            "--resume-global-step", str(resume_global_step),
            "--skip-train-batches", str(skip_train_batches),
        ])
    return cmd


def _compare_command(
    ctx: dict[str, Any],
    checkpoint: Path,
    val_path: Path,
    out_dir: Path,
    chunk_size: int,
) -> list[str]:
    return [
        str(PYTHON), "-X", "utf8", "-u",
        "scripts/eval.py",
        "--config", str(ctx["config"]),
        "--compare-smoke-vs-full",
        "--smoke-checkpoint", str(ctx["baseline_checkpoint"]),
        "--full-checkpoint", str(checkpoint),
        "--val", str(val_path),
        "--out", str(out_dir),
        "--chunk-size", str(chunk_size),
        "--chunk-output-dir", str(out_dir / "chunks"),
    ]


def _package_command(ctx: dict[str, Any], checkpoint: Path) -> list[str]:
    return [
        str(PYTHON), "-X", "utf8", "-u",
        "scripts/package_offline.py",
        "--ckpt", str(checkpoint),
        "--out", str(ctx["package_dir"]),
        "--report", str(ctx["package_report"]),
        "--lora-r", str(ctx["lora_r"]),
        "--lora-alpha", str(ctx["lora_alpha"]),
        "--lora-target", ctx["lora_target"],
    ]


def _offline_smoke_command(ctx: dict[str, Any]) -> list[str]:
    return [
        str(PYTHON), "-X", "utf8", "-u",
        "scripts/smoke_offline.py",
        "--pkg", str(ctx["package_dir"]),
        "--stage-root", str(ctx["offline_smoke_stage"]),
    ]


def _summary_command(
    compare_dir: Path,
    val_path: Path,
    scope: str,
    step: int,
    label: str,
) -> list[str]:
    return [
        str(PYTHON), "-X", "utf8",
        "scripts/summarize_phase2_3_compare.py",
        "--compare-dir", str(compare_dir),
        "--val-jsonl", str(val_path),
        "--scope", scope,
        "--step", str(step),
        "--label", label,
    ]


def _compare_done(out_dir: Path) -> bool:
    return (out_dir / "comparison_full_vs_smoke.json").exists()


def _summary_done(out_dir: Path) -> bool:
    return (out_dir / "phase2_3_compare_summary.json").exists()


def _run_compare_and_summary(
    *,
    ctx: dict[str, Any],
    compare_root: Path,
    logs_dir: Path,
    workflow_log: Path,
    plan: dict[str, Any],
    checkpoint: Path,
    scope: str,
    step: int,
    label: str,
    dry_run: bool,
) -> int:
    sets_key = "quick_sets" if scope == "quick" else "full_sets"
    chunk_size = 50
    out_dir = compare_root / scope / f"step_{step:04d}" / label
    if not _compare_done(out_dir):
        log = logs_dir / f"phase2_3_compare_{scope}_step_{step:04d}_{label}.log"
        cmd = _compare_command(ctx, checkpoint, ctx[sets_key][label], out_dir, chunk_size=chunk_size)
        plan["commands"].append({"type": f"{scope}_compare", "step": step, "label": label, "command": cmd, "log": str(log)})
        result = _run_logged(cmd, log, dry_run)
        _append_log(
            workflow_log,
            f"T2.27 {scope} compare {'dry-run' if dry_run else 'run'} step {step} label={label}, returncode={result['returncode']}, log={log}",
        )
        if result["returncode"] != 0:
            return int(result["returncode"])
    if not _summary_done(out_dir):
        summary_log = logs_dir / f"phase2_3_summary_{scope}_step_{step:04d}_{label}.log"
        summary_cmd = _summary_command(out_dir, ctx[sets_key][label], scope, step, label)
        plan["commands"].append({"type": f"{scope}_summary", "step": step, "label": label, "command": summary_cmd, "log": str(summary_log)})
        result = _run_logged(summary_cmd, summary_log, dry_run)
        if result["returncode"] != 0:
            return int(result["returncode"])
    return 0


def run_workflow(
    run_dir: Path,
    execute: bool,
    stop_after_step: int | None,
    skip_package: bool,
    resume_from: Path | None = None,
    resume_global_step: int = 0,
    skip_train_batches: int = 0,
) -> int:
    dry_run = not execute
    ctx = _build_context(run_dir)
    run_dir = ctx["run_dir"]
    logs_dir = run_dir / "logs"
    compare_root = run_dir / "artifacts" / "compare"
    scorecard_path = compare_root / "phase2_3_checkpoint_scorecard.json"
    workflow_plan_path = run_dir / "phase2_3_training_workflow_plan.json"
    workflow_log = run_dir / "execution_log.md"

    missing = _preflight(ctx)
    plan = {
        "run_dir": str(run_dir),
        "dry_run": dry_run,
        "total_steps": ctx["total_steps"],
        "training_mode": "continuous_single_train_process_resume" if resume_from else "continuous_single_train_process",
        "resume": {
            "resume_from": str(resume_from) if resume_from else None,
            "resume_global_step": resume_global_step,
            "skip_train_batches": skip_train_batches,
            "steps_this_run": max(0, ctx["total_steps"] - resume_global_step) if resume_from else ctx["total_steps"],
        },
        "checkpoint_observe_steps": ctx["observe_steps"],
        "quick_compare": {
            "records_per_label": 50,
            "labels": list(LABELS),
            "schedule_steps": ctx["quick_steps"],
        },
        "full_compare": {
            "records_per_label": 200,
            "labels": list(LABELS),
            "schedule_steps": ctx["full_steps"],
            "run_for_best_candidate": True,
        },
        "baseline_checkpoint": str(ctx["baseline_checkpoint"]),
        "checkpoint_dir": str(ctx["checkpoint_dir"]),
        "best_checkpoint": str(ctx["checkpoint_dir"] / ctx["best_name"]),
        "package_dir": str(ctx["package_dir"]),
        "package_after_success_threshold": not skip_package,
        "preflight_missing": missing,
        "commands": [],
    }

    if missing:
        _write_json(workflow_plan_path, plan)
        print("[phase2.3] preflight failed:")
        for item in missing:
            print(f"  - missing {item}")
        return 2

    ctx["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    compare_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_log = logs_dir / (
        f"phase2_3_train_resume_{resume_global_step + 1:04d}_{ctx['total_steps']:04d}.log"
        if resume_from
        else "phase2_3_train_0001_2000.log"
    )
    train_cmd = _full_train_command(
        ctx,
        resume_from=resume_from,
        resume_global_step=resume_global_step,
        skip_train_batches=skip_train_batches,
    )
    plan["commands"].append({"type": "train_continuous", "target_step": ctx["total_steps"], "command": train_cmd, "log": str(train_log)})
    scheduled_steps = sorted({*ctx["quick_steps"], *ctx["full_steps"]})
    if stop_after_step is not None:
        scheduled_steps = [step for step in scheduled_steps if step <= stop_after_step]

    if dry_run:
        for step in scheduled_steps:
            checkpoint = ctx["checkpoint_dir"] / f"checkpoint_step_{step}.safetensors"
            for scope in (("quick",) if step in ctx["quick_steps"] else ()):
                for label in LABELS:
                    code = _run_compare_and_summary(
                        ctx=ctx, compare_root=compare_root, logs_dir=logs_dir,
                        workflow_log=workflow_log, plan=plan, checkpoint=checkpoint,
                        scope=scope, step=step, label=label, dry_run=True,
                    )
                    if code:
                        _write_json(workflow_plan_path, plan)
                        return code
            for scope in (("full",) if step in ctx["full_steps"] else ()):
                for label in LABELS:
                    code = _run_compare_and_summary(
                        ctx=ctx, compare_root=compare_root, logs_dir=logs_dir,
                        workflow_log=workflow_log, plan=plan, checkpoint=checkpoint,
                        scope=scope, step=step, label=label, dry_run=True,
                    )
                    if code:
                        _write_json(workflow_plan_path, plan)
                        return code
    else:
        completed_compare_steps: set[int] = set()
        with train_log.open("w", encoding="utf-8") as log:
            log.write("[workflow] command: " + " ".join(train_cmd) + "\n")
            log.flush()
            proc = subprocess.Popen(
                train_cmd,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            _append_log(workflow_log, f"T2.32 continuous training started PID={proc.pid}, log={train_log}")
            while proc.poll() is None:
                for step in scheduled_steps:
                    if step in completed_compare_steps:
                        continue
                    checkpoint = ctx["checkpoint_dir"] / f"checkpoint_step_{step}.safetensors"
                    if not checkpoint.exists():
                        continue
                    for scope in (("quick",) if step in ctx["quick_steps"] else ()):
                        for label in LABELS:
                            code = _run_compare_and_summary(
                                ctx=ctx, compare_root=compare_root, logs_dir=logs_dir,
                                workflow_log=workflow_log, plan=plan, checkpoint=checkpoint,
                                scope=scope, step=step, label=label, dry_run=False,
                            )
                            if code:
                                _write_json(workflow_plan_path, plan)
                                return code
                    for scope in (("full",) if step in ctx["full_steps"] else ()):
                        for label in LABELS:
                            code = _run_compare_and_summary(
                                ctx=ctx, compare_root=compare_root, logs_dir=logs_dir,
                                workflow_log=workflow_log, plan=plan, checkpoint=checkpoint,
                                scope=scope, step=step, label=label, dry_run=False,
                            )
                            if code:
                                _write_json(workflow_plan_path, plan)
                                return code
                    completed_compare_steps.add(step)
                    scorecard = _collect_scorecard(run_dir, ctx["checkpoint_dir"], ctx["best_name"])
                    _write_json(scorecard_path, scorecard)
                    _copy_best(scorecard, ctx["checkpoint_dir"], ctx["best_name"], dry_run=False)
                time.sleep(60)
            train_returncode = proc.returncode
        _append_log(workflow_log, f"T2.32 continuous training finished returncode={train_returncode}, log={train_log}")
        if train_returncode != 0:
            _write_json(workflow_plan_path, plan)
            return int(train_returncode)

        for step in scheduled_steps:
            if step in completed_compare_steps:
                continue
            checkpoint = ctx["checkpoint_dir"] / f"checkpoint_step_{step}.safetensors"
            if not checkpoint.exists():
                print(f"[phase2.3] expected checkpoint missing for compare: {checkpoint}")
                _write_json(workflow_plan_path, plan)
                return 3
            for scope in (("quick",) if step in ctx["quick_steps"] else ()):
                for label in LABELS:
                    code = _run_compare_and_summary(
                        ctx=ctx, compare_root=compare_root, logs_dir=logs_dir,
                        workflow_log=workflow_log, plan=plan, checkpoint=checkpoint,
                        scope=scope, step=step, label=label, dry_run=False,
                    )
                    if code:
                        _write_json(workflow_plan_path, plan)
                        return code
            for scope in (("full",) if step in ctx["full_steps"] else ()):
                for label in LABELS:
                    code = _run_compare_and_summary(
                        ctx=ctx, compare_root=compare_root, logs_dir=logs_dir,
                        workflow_log=workflow_log, plan=plan, checkpoint=checkpoint,
                        scope=scope, step=step, label=label, dry_run=False,
                    )
                    if code:
                        _write_json(workflow_plan_path, plan)
                        return code
            completed_compare_steps.add(step)
            scorecard = _collect_scorecard(run_dir, ctx["checkpoint_dir"], ctx["best_name"])
            _write_json(scorecard_path, scorecard)
            _copy_best(scorecard, ctx["checkpoint_dir"], ctx["best_name"], dry_run=False)

    scorecard = _collect_scorecard(run_dir, ctx["checkpoint_dir"], ctx["best_name"])
    _write_json(scorecard_path, scorecard)

    best = scorecard.get("best")
    if best:
        best_step = int(best["step"])
        for label in LABELS:
            out_dir = compare_root / "full" / f"step_{best_step:04d}" / label
            checkpoint = ctx["checkpoint_dir"] / f"checkpoint_step_{best_step}.safetensors"
            if not _compare_done(out_dir):
                log = logs_dir / f"phase2_3_compare_full_best_step_{best_step:04d}_{label}.log"
                cmd = _compare_command(ctx, checkpoint, ctx["full_sets"][label], out_dir, chunk_size=50)
                plan["commands"].append({"type": "best_full_compare", "step": best_step, "label": label, "command": cmd, "log": str(log)})
                result = _run_logged(cmd, log, dry_run)
                if result["returncode"] != 0:
                    _write_json(workflow_plan_path, plan)
                    return int(result["returncode"])
            if not _summary_done(out_dir):
                summary_log = logs_dir / f"phase2_3_summary_full_best_step_{best_step:04d}_{label}.log"
                summary_cmd = _summary_command(out_dir, ctx["full_sets"][label], "full", best_step, label)
                plan["commands"].append({"type": "best_full_summary", "step": best_step, "label": label, "command": summary_cmd, "log": str(summary_log)})
                result = _run_logged(summary_cmd, summary_log, dry_run)
                if result["returncode"] != 0:
                    _write_json(workflow_plan_path, plan)
                    return int(result["returncode"])
        scorecard = _collect_scorecard(run_dir, ctx["checkpoint_dir"], ctx["best_name"])
        _write_json(scorecard_path, scorecard)
        _copy_best(scorecard, ctx["checkpoint_dir"], ctx["best_name"], dry_run)

    best_checkpoint = ctx["checkpoint_dir"] / ctx["best_name"]
    package_log = logs_dir / "phase2_3_package_best.log"
    package_cmd = _package_command(ctx, best_checkpoint)
    smoke_log = logs_dir / "phase2_3_offline_smoke_best.log"
    smoke_cmd = _offline_smoke_command(ctx)
    if dry_run:
        plan["commands"].append({"type": "package", "command": package_cmd, "log": str(package_log)})
        plan["commands"].append({"type": "offline_smoke", "command": smoke_cmd, "log": str(smoke_log)})
    elif not skip_package:
        if not _passes_success_threshold(scorecard):
            f1s = _best_full_target_f1(scorecard)
            _write_json(workflow_plan_path, plan)
            print("[phase2.3] success threshold not met; package/offline smoke skipped")
            print(f"[phase2.3] best full target F1: {f1s}")
            return 4
        if not best_checkpoint.exists():
            _write_json(workflow_plan_path, plan)
            print(f"[phase2.3] best checkpoint missing: {best_checkpoint}")
            return 5
        plan["commands"].append({"type": "package", "command": package_cmd, "log": str(package_log)})
        result = _run_logged(package_cmd, package_log, dry_run=False)
        _append_log(
            workflow_log,
            f"T2.32 package best checkpoint run returncode={result['returncode']}, log={package_log}",
        )
        if result["returncode"] != 0:
            _write_json(workflow_plan_path, plan)
            return int(result["returncode"])
        plan["commands"].append({"type": "offline_smoke", "command": smoke_cmd, "log": str(smoke_log)})
        result = _run_logged(smoke_cmd, smoke_log, dry_run=False)
        _append_log(
            workflow_log,
            f"T2.32 offline smoke run returncode={result['returncode']}, log={smoke_log}",
        )
        if result["returncode"] != 0:
            _write_json(workflow_plan_path, plan)
            return int(result["returncode"])

    _write_json(workflow_plan_path, plan)
    print(f"[phase2.3] {'dry-run plan' if dry_run else 'workflow'} complete")
    print(f"[phase2.3] plan: {workflow_plan_path}")
    print(f"[phase2.3] scorecard: {scorecard_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.3 continuous training + checkpoint compare workflow")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "phase2_3_full_2000_20260727_1211",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually start training/compare. Omit this flag for safe dry-run only.",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=None,
        help="Optional debug guard. Executes/dry-runs only up to this scheduled step.",
    )
    parser.add_argument(
        "--skip-package",
        action="store_true",
        help="Stop after training/compare/best selection. By default package and offline smoke run only if full-set F1 threshold passes.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from an existing checkpoint and continue the workflow.",
    )
    parser.add_argument(
        "--resume-global-step",
        type=int,
        default=0,
        help="Logical global step represented by --resume-from.",
    )
    parser.add_argument(
        "--skip-train-batches",
        type=int,
        default=0,
        help="Skip already-consumed deterministic train batches on resume.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_workflow(
        args.run_dir.resolve(),
        execute=args.execute,
        stop_after_step=args.stop_after_step,
        skip_package=args.skip_package,
        resume_from=args.resume_from.resolve() if args.resume_from else None,
        resume_global_step=args.resume_global_step,
        skip_train_batches=args.skip_train_batches,
    )


if __name__ == "__main__":
    raise SystemExit(main())
