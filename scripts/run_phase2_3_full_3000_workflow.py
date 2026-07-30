"""Run Phase 2.3 Full 3000 training with mixed-set checkpoint evaluation."""
from __future__ import annotations

import argparse
import json
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
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _append(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- [{_now()}] {message}\n")


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    full = report.get("full", {})
    per_class = full.get("per_class", {})
    classes = {
        label: {
            "precision": float(per_class.get(label, {}).get("precision", 0.0)),
            "recall": float(per_class.get(label, {}).get("recall", 0.0)),
            "f1": float(per_class.get(label, {}).get("f1-score", 0.0)),
            "support": int(per_class.get(label, {}).get("support", 0)),
        }
        for label in LABELS
    }
    return {
        "per_class": classes,
        "macro_f1": float(full.get("macro_f1", 0.0)),
        "accuracy": float(full.get("accuracy", 0.0)),
        "confusion_matrix": full.get("confusion_matrix", []),
        "min_class_f1": min(classes[label]["f1"] for label in LABELS),
    }


def _render_scorecard(scorecard: dict[str, Any]) -> str:
    lines = ["# Phase 2.3 Full 3000 混合评测记分卡", "", "| step | 范围 | clean F1 | direct F1 | direct Recall | indirect F1 | Macro F1 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for entry in scorecard.get("entries", []):
        item = entry["metrics"]["per_class"]
        lines.append(
            f"| {entry['step']} | {entry['scope']} | {item['clean']['f1']:.4f} | "
            f"{item['direct']['f1']:.4f} | {item['direct']['recall']:.4f} | "
            f"{item['indirect']['f1']:.4f} | {entry['metrics']['macro_f1']:.4f} |"
        )
    best = scorecard.get("best")
    lines.extend(["", "## 结论", ""])
    if best:
        lines.append(f"- 最佳 checkpoint：step {best['step']}，选择指标为三类 F1 最小值。")
        lines.append(f"- 是否满足发布门槛：{'通过' if scorecard['passes_release_gate'] else '未通过'}。")
    else:
        lines.append("- 尚未完成任何混合评测。")
    return "\n".join(lines) + "\n"


def _scorecard(compare_root: Path, checkpoint_dir: Path, policy: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for scope in ("quick", "full"):
        for step_dir in sorted((compare_root / scope).glob("step_*")):
            report_path = step_dir / "all" / "comparison_full_vs_smoke.json"
            if not report_path.exists():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            step = int(step_dir.name.removeprefix("step_"))
            entries.append({
                "step": step,
                "scope": scope,
                "checkpoint": str(checkpoint_dir / f"checkpoint_step_{step}.safetensors"),
                "metrics": _metrics(report),
            })
    full_entries = [entry for entry in entries if entry["scope"] == "full"]
    candidates = full_entries or entries
    best = max(candidates, key=lambda item: (item["metrics"]["min_class_f1"], item["metrics"]["macro_f1"], item["step"])) if candidates else None
    release = False
    if best and best["scope"] == "full":
        rows = best["metrics"]["per_class"]
        release = (
            all(rows[label]["f1"] > 0.70 for label in LABELS)
            and rows["direct"]["recall"] >= float(policy.get("direct_recall_floor", 0.95))
            and rows["direct"]["f1"] >= float(policy.get("direct_f1_floor", 0.85))
        )
    return {
        "updated_at": _now(),
        "selection_metric": "min(clean_f1, direct_f1, indirect_f1) on mixed set",
        "release_gate": {"all_class_f1_gt": 0.70, "direct_recall_floor": policy.get("direct_recall_floor", 0.95), "direct_f1_floor": policy.get("direct_f1_floor", 0.85)},
        "entries": sorted(entries, key=lambda item: item["step"]),
        "best": best,
        "passes_release_gate": release,
    }


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[workflow] command: " + " ".join(command) + "\n")
        log.flush()
        return subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace").returncode


def _compare(config: Path, baseline: Path, checkpoint: Path, dataset: Path, out_dir: Path, log_path: Path) -> int:
    return _run([
        str(PYTHON), "-X", "utf8", "-u", "scripts/eval.py", "--config", str(config),
        "--compare-smoke-vs-full", "--smoke-checkpoint", str(baseline),
        "--full-checkpoint", str(checkpoint), "--val", str(dataset), "--out", str(out_dir),
        "--chunk-size", "50", "--chunk-output-dir", str(out_dir / "chunks"),
    ], log_path)


def run(run_dir: Path, execute: bool, resume_from: Path | None = None, resume_global_step: int = 0, skip_train_batches: int = 0) -> int:
    run_dir = run_dir.resolve()
    config = run_dir / "configs" / "train.yaml"
    cfg = _read_yaml(config)
    config_dir = config.parent
    phase = cfg["phase2_3_full_3000"]
    policy = cfg["checkpoint_policy"]
    checkpoint_dir = _resolve(cfg["io"]["out_dir"], config_dir)
    baseline = _resolve(phase["baseline_checkpoint"], config_dir)
    logs = run_dir / "logs"
    compare_root = run_dir / "artifacts" / "compare"
    execution_log = run_dir / "execution_log.md"
    schedules = [(300, "quick"), (750, "quick"), (1500, "full"), (2250, "quick"), (3000, "full")]
    if not PYTHON.exists() or not baseline.exists():
        raise FileNotFoundError("Python runtime or Full 2000 initialization checkpoint is missing")

    plan = {
        "run_dir": str(run_dir), "baseline_checkpoint": str(baseline), "execute": execute,
        "schedules": [{"step": step, "scope": scope} for step, scope in schedules],
        "training_mode": "LoRA-only warm-start with new head and optimizer",
    }
    _write_json(run_dir / "phase2_3_full_3000_workflow_plan.json", plan)
    if not execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    max_steps = int(cfg["training"]["max_train_steps"])
    train_cmd = [str(PYTHON), "-X", "utf8", "-u", "scripts/train.py", "--config", str(config), "--max-train-steps", str(max_steps)]
    if resume_from:
        remaining = max(0, max_steps - resume_global_step)
        train_cmd.extend(["--resume-from", str(resume_from), "--resume-global-step", str(resume_global_step), "--skip-train-batches", str(skip_train_batches), "--max-train-steps", str(remaining)])
    train_log = logs / ("train_resume.log" if resume_from else "train_full_3000.log")
    completed: set[int] = set()
    with train_log.open("w", encoding="utf-8") as log:
        log.write("[workflow] command: " + " ".join(train_cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(train_cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        _append(execution_log, f"正式训练已启动，PID={proc.pid}，日志={train_log.name}")
        while proc.poll() is None:
            for step, scope in schedules:
                checkpoint = checkpoint_dir / f"checkpoint_step_{step}.safetensors"
                if step in completed or not checkpoint.exists():
                    continue
                data = _resolve(cfg["compare"][scope]["sets"]["all"], config_dir)
                out_dir = compare_root / scope / f"step_{step:04d}" / "all"
                code = _compare(config, baseline, checkpoint, data, out_dir, logs / f"compare_{scope}_{step:04d}_all.log")
                _append(execution_log, f"{scope} 混合评测 step={step}，returncode={code}")
                if code:
                    proc.terminate()
                    return code
                completed.add(step)
            time.sleep(60)
        if proc.returncode != 0:
            _append(execution_log, f"正式训练失败，returncode={proc.returncode}")
            return int(proc.returncode)

    for step, scope in schedules:
        if step in completed:
            continue
        checkpoint = checkpoint_dir / f"checkpoint_step_{step}.safetensors"
        if not checkpoint.exists():
            return 3
        data = _resolve(cfg["compare"][scope]["sets"]["all"], config_dir)
        code = _compare(config, baseline, checkpoint, data, compare_root / scope / f"step_{step:04d}" / "all", logs / f"compare_{scope}_{step:04d}_all.log")
        if code:
            return code

    scorecard = _scorecard(compare_root, checkpoint_dir, policy)
    _write_json(compare_root / "phase2_3_full_3000_scorecard.json", scorecard)
    (compare_root / "phase2_3_full_3000_scorecard.md").write_text(_render_scorecard(scorecard), encoding="utf-8")
    if scorecard.get("best"):
        source = checkpoint_dir / f"checkpoint_step_{scorecard['best']['step']}.safetensors"
        shutil.copy2(source, checkpoint_dir / policy["best_name"])
    _append(execution_log, f"训练与混合评测完成，发布门槛={'通过' if scorecard['passes_release_gate'] else '未通过'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-global-step", type=int, default=0)
    parser.add_argument("--skip-train-batches", type=int, default=0)
    args = parser.parse_args()
    return run(args.run_dir, args.execute, args.resume_from, args.resume_global_step, args.skip_train_batches)


if __name__ == "__main__":
    raise SystemExit(main())
