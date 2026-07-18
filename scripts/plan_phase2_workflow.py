"""Build an execution plan for the Phase 2.2 train/eval/package workflow.

The launcher consumes the JSON output from this script and also writes the
Markdown version for humans and monitoring automation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _resolve_input(path: str | Path, config_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    from_config = (config_dir / p).resolve()
    if from_config.exists():
        return from_config
    return (REPO_ROOT / p).resolve()


def _resolve_output(path: str | Path, config_dir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (config_dir / p).resolve()


def _count_labels(path: Path, max_records: int | None = None) -> dict:
    counts: Counter[str] = Counter()
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if max_records is not None and total >= max_records:
                break
            row = json.loads(line)
            counts[row.get("label", "<missing>")] += 1
            total += 1
    return {"total": total, "labels": dict(counts)}


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _seconds_to_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def build_plan(args: argparse.Namespace) -> dict:
    cfg_path = _resolve(args.config)
    config_dir = cfg_path.resolve().parent
    cfg = _load_config(cfg_path)
    training = cfg.get("training", {}) or {}
    io = cfg.get("io", {}) or {}

    run_dir = _resolve(args.run_dir or f"runs/{args.run_name}")
    train_jsonl = _resolve_input(io["train_jsonl"], config_dir)
    val_jsonl = _resolve_input(io["val_jsonl"], config_dir)
    out_dir = _resolve_output(io.get("out_dir", "artifacts/checkpoints"), config_dir)
    checkpoint_name = training.get("checkpoint_name", f"lora_{args.run_name}.safetensors")
    checkpoint = out_dir / checkpoint_name
    offline_dir = _resolve(args.offline_dir) if args.offline_dir else (run_dir / "artifacts" / "package" / "mpid_offline")

    epochs = int(training.get("epochs", 1))
    max_train_records = int(training.get("max_train_records", 500))
    batch_size = max(1, int(training.get("batch_size", 1)))
    train_steps = ((max_train_records + batch_size - 1) // batch_size) * epochs
    eval_records = int(args.eval_records)

    # Conservative estimates based on the observed CPU-only run on this host.
    preload_records = max_train_records + int(training.get("max_val_records", 100))
    estimates = {
        "preflight": 5,
        "smoke_train": 12 * 60,
        "preload": preload_records * float(args.estimate_preload_seconds),
        "train_loop": train_steps * float(args.estimate_train_step_seconds),
        "single_eval": eval_records * float(args.estimate_eval_sample_seconds) + 90,
        "compare_eval": eval_records * float(args.estimate_eval_sample_seconds) * 2 + 180,
        "package": 5 * 60,
        "offline_smoke": 5 * 60,
    }
    estimates["train"] = estimates["preload"] + estimates["train_loop"] + 120

    log_dir = _resolve(args.log_dir) if args.log_dir else (run_dir / "logs")
    steps = [
        {
            "id": "01_preflight",
            "name": "Preflight checks",
            "estimate_seconds": estimates["preflight"],
            "log": str(log_dir / "01_preflight.log"),
        },
        {
            "id": "02_smoke_train",
            "name": "Smoke training check",
            "estimate_seconds": estimates["smoke_train"],
            "log": str(log_dir / "02_smoke_train.log"),
        },
        {
            "id": "03_train",
            "name": f"Train {args.run_name}",
            "estimate_seconds": estimates["train"],
            "log": str(log_dir / "03_train.log"),
        },
        {
            "id": "04_eval",
            "name": f"Single-model eval ({eval_records} stratified records)",
            "estimate_seconds": estimates["single_eval"],
            "log": str(log_dir / "04_eval.log"),
        },
        {
            "id": "05_compare",
            "name": f"Smoke-vs-full comparison ({eval_records} stratified records)",
            "estimate_seconds": estimates["compare_eval"],
            "log": str(log_dir / "05_compare.log"),
        },
        {
            "id": "06_package",
            "name": "Offline package rebuild",
            "estimate_seconds": estimates["package"],
            "log": str(log_dir / "06_package.log"),
        },
        {
            "id": "07_offline_smoke",
            "name": "Offline smoke validation",
            "estimate_seconds": estimates["offline_smoke"],
            "log": str(log_dir / "07_offline_smoke.log"),
        },
    ]

    total_seconds = sum(step["estimate_seconds"] for step in steps)
    plan = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "config": str(cfg_path),
        "log_dir": str(log_dir),
        "execution_log": str(run_dir / "execution_log.md"),
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "out_dir": str(out_dir),
        "checkpoint": str(checkpoint),
        "offline_dir": str(offline_dir),
        "epochs": epochs,
        "max_train_records": max_train_records,
        "batch_size": batch_size,
        "train_steps": train_steps,
        "eval_records": eval_records,
        "save_every": int(training.get("save_every", 50)),
        "log_every": int(training.get("log_every", 5)),
        "eval_after_epoch": bool(training.get("eval_after_epoch", False)),
        "train_distribution": _count_labels(train_jsonl, max_train_records),
        "val_distribution": _count_labels(val_jsonl, None),
        "steps": steps,
        "total_estimate_seconds": total_seconds,
        "total_estimate_hms": _seconds_to_hms(total_seconds),
    }
    for step in plan["steps"]:
        step["estimate_hms"] = _seconds_to_hms(step["estimate_seconds"])
    return plan


def write_markdown(plan: dict, path: Path) -> None:
    lines = [
        f"# Phase 2.2 Execution Plan: {plan['run_name']}",
        "",
        f"- config: `{plan['config']}`",
        f"- train_jsonl: `{plan['train_jsonl']}`",
        f"- checkpoint: `{plan['checkpoint']}`",
        f"- offline_dir: `{plan['offline_dir']}`",
        f"- train records: **{plan['max_train_records']}**",
        f"- eval records: **{plan['eval_records']}**",
        f"- train steps: **{plan['train_steps']}**",
        f"- total estimate: **{plan['total_estimate_hms']}**",
        "",
        "## Data Distribution",
        "",
        f"- train: `{plan['train_distribution']}`",
        f"- val: `{plan['val_distribution']}`",
        "",
        "## Steps",
        "",
        "| step | estimate | log |",
        "|---|---:|---|",
    ]
    for step in plan["steps"]:
        lines.append(f"| {step['name']} | {step['estimate_hms']} | `{step['log']}` |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--eval-records", type=int, default=500)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--offline-dir", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--estimate-train-step-seconds", type=float, default=32.0)
    parser.add_argument("--estimate-eval-sample-seconds", type=float, default=10.8)
    parser.add_argument("--estimate-preload-seconds", type=float, default=0.36)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(plan, args.md_out)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
