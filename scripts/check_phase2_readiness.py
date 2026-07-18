"""Phase 2.2 readiness check for a run directory.

This script is intentionally stricter than the workflow's old preflight,
but still much cheaper than actually starting train/eval/package steps.
It validates:

  1. Core Python/runtime imports.
  2. Workflow script availability.
  3. YAML config structure and resolved paths.
  4. JSONL readability and label sanity for train/val splits.
  5. Backbone files required for offline loading.
  6. Run-local output directories and optional write access.
  7. Optional model/tokenizer offline probe for higher confidence.

Usage::

    python scripts/check_phase2_readiness.py --config runs/my_run/configs/train.yaml
    python scripts/check_phase2_readiness.py --config runs/my_run/configs/train.yaml --probe-model-load
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mpid.backbones.registry import resolve_local_path  # noqa: E402


ALLOWED_LABELS = {"clean", "direct", "indirect"}
CORE_IMPORTS = [
    "torch",
    "transformers",
    "peft",
    "safetensors",
    "sklearn",
    "yaml",
    "PIL",
]
REQUIRED_WORKFLOW_FILES = [
    REPO_ROOT / "scripts" / "train.py",
    REPO_ROOT / "scripts" / "eval.py",
    REPO_ROOT / "scripts" / "package_offline.py",
    REPO_ROOT / "scripts" / "smoke_offline.py",
    REPO_ROOT / "scripts" / "run_phase2_workflow.ps1",
]
BACKBONE_REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
]


def _resolve_input_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    from_config = (config_dir / path).resolve()
    if from_config.exists():
        return from_config
    return (REPO_ROOT / path).resolve()


def _resolve_output_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _ok(lines: list[str], message: str) -> None:
    lines.append(f"OK   {message}")


def _warn(lines: list[str], message: str) -> None:
    lines.append(f"WARN {message}")


def _fail(lines: list[str], message: str) -> None:
    lines.append(f"FAIL {message}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _count_jsonl(
    path: Path,
    *,
    sample_lines: int = 20,
) -> tuple[int, dict[str, int], list[str]]:
    total = 0
    counts: Counter[str] = Counter()
    bad_lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                bad_lines.append(f"line {idx}: empty")
                continue
            row = json.loads(line)
            label = row.get("label")
            counts[str(label)] += 1
            if idx <= sample_lines:
                if "text" not in row:
                    bad_lines.append(f"line {idx}: missing text")
                elif not isinstance(row.get("text"), str):
                    bad_lines.append(f"line {idx}: text is not a string")
                if label not in ALLOWED_LABELS:
                    bad_lines.append(f"line {idx}: unexpected label={label!r}")
            total += 1
    return total, dict(counts), bad_lines


def _try_imports(lines: list[str]) -> bool:
    ok = True
    for name in CORE_IMPORTS:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            _ok(lines, f"import {name} ({version})")
        except Exception as exc:  # pragma: no cover - environment dependent
            _fail(lines, f"import {name}: {type(exc).__name__}: {exc}")
            ok = False
    return ok


def _probe_model_load(backbone_dir: Path, lines: list[str]) -> bool:
    try:
        from transformers import AutoProcessor
    except Exception as exc:  # pragma: no cover - environment dependent
        _fail(lines, f"probe import AutoProcessor: {type(exc).__name__}: {exc}")
        return False
    env_before = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        processor = AutoProcessor.from_pretrained(
            str(backbone_dir),
            local_files_only=True,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        _fail(lines, f"offline processor load: {type(exc).__name__}: {exc}")
        return False
    finally:
        for key, value in env_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    _ok(lines, f"offline processor load succeeded ({type(processor).__name__})")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.2 readiness check")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--eval-records", type=int, default=500)
    parser.add_argument("--smoke-config", type=Path, default=None)
    parser.add_argument("--smoke-checkpoint", type=Path, default=None)
    parser.add_argument("--skip-smoke-train", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--check-write-access", action="store_true")
    parser.add_argument("--probe-model-load", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lines: list[str] = []
    summary: dict[str, Any] = {
        "ok": True,
        "config": str(args.config),
        "run_dir": None,
        "checks": {},
    }

    config_path = args.config.resolve()
    config_dir = config_path.parent
    run_dir = args.run_dir.resolve() if args.run_dir else config_dir.parent
    summary["run_dir"] = str(run_dir)

    print(f"[readiness] repo_root = {REPO_ROOT}")
    print(f"[readiness] config    = {config_path}")
    print(f"[readiness] run_dir   = {run_dir}")

    if not config_path.exists():
        _fail(lines, f"config missing: {config_path}")
        summary["ok"] = False
        for line in lines:
            print(line)
        return 1

    summary["checks"]["imports"] = _try_imports(lines)
    if not summary["checks"]["imports"]:
        summary["ok"] = False

    workflow_ok = True
    for path in REQUIRED_WORKFLOW_FILES:
        if path.exists():
            _ok(lines, f"workflow file present: {path}")
        else:
            _fail(lines, f"workflow file missing: {path}")
            workflow_ok = False
    summary["checks"]["workflow_files"] = workflow_ok
    if not workflow_ok:
        summary["ok"] = False

    cfg = _load_yaml(config_path)
    defaults = cfg.get("defaults", {}) or {}
    training = cfg.get("training", {}) or {}
    io_cfg = cfg.get("io", {}) or {}
    missing_sections = [name for name in ("defaults", "training", "io") if name not in cfg]
    if missing_sections:
        _warn(lines, f"config missing top-level sections: {missing_sections}")

    try:
        train_jsonl = _resolve_input_path(io_cfg["train_jsonl"], config_dir)
        val_jsonl = _resolve_input_path(io_cfg["val_jsonl"], config_dir)
        out_dir = _resolve_output_path(io_cfg.get("out_dir", "../artifacts/checkpoints"), config_dir)
        checkpoint_name = str(training.get("checkpoint_name", "lora_baseline.safetensors"))
        checkpoint_path = out_dir / checkpoint_name
    except KeyError as exc:
        _fail(lines, f"config missing required io/training key: {exc}")
        summary["ok"] = False
        for line in lines:
            print(line)
        return 1

    _ok(lines, f"resolved train_jsonl: {train_jsonl}")
    _ok(lines, f"resolved val_jsonl: {val_jsonl}")
    _ok(lines, f"resolved out_dir: {out_dir}")
    _ok(lines, f"expected checkpoint: {checkpoint_path}")

    smoke_config = args.smoke_config.resolve() if args.smoke_config else (run_dir / "configs" / "smoke.yaml")
    smoke_checkpoint = (
        args.smoke_checkpoint.resolve()
        if args.smoke_checkpoint
        else run_dir / "artifacts" / "smoke" / "lora_benchmark_100.safetensors"
    )
    if not args.skip_smoke_train:
        if smoke_config.exists():
            _ok(lines, f"smoke config present: {smoke_config}")
        else:
            _fail(lines, f"smoke config missing: {smoke_config}")
            summary["ok"] = False
    elif not args.skip_compare:
        if smoke_checkpoint.exists():
            _ok(lines, f"smoke checkpoint present for compare: {smoke_checkpoint}")
        else:
            _fail(lines, f"smoke checkpoint required by compare but missing: {smoke_checkpoint}")
            summary["ok"] = False

    dataset_ok = True
    for role, path in (("train", train_jsonl), ("val", val_jsonl)):
        if not path.exists():
            _fail(lines, f"{role} jsonl missing: {path}")
            dataset_ok = False
            continue
        try:
            total, counts, bad_lines = _count_jsonl(path)
        except Exception as exc:
            _fail(lines, f"{role} jsonl unreadable: {type(exc).__name__}: {exc}")
            dataset_ok = False
            continue
        _ok(lines, f"{role} jsonl readable: {total} rows labels={counts}")
        if bad_lines:
            _warn(lines, f"{role} jsonl sample issues: {bad_lines[:5]}")
        if role == "train":
            max_train_records = int(training.get("max_train_records", total))
            if total < max_train_records:
                _fail(lines, f"train rows {total} < max_train_records {max_train_records}")
                dataset_ok = False
        if role == "val" and total < args.eval_records:
            _warn(lines, f"val rows {total} < eval_records {args.eval_records}; stratified eval will cap below target")
    summary["checks"]["datasets"] = dataset_ok
    if not dataset_ok:
        summary["ok"] = False

    backbone_name = str(defaults.get("backbone_name", "smolvlm-500m"))
    try:
        backbone_dir = resolve_local_path(backbone_name)
    except Exception as exc:
        _fail(lines, f"backbone resolve failed for {backbone_name!r}: {type(exc).__name__}: {exc}")
        summary["checks"]["backbone"] = False
        summary["ok"] = False
        backbone_dir = None
    else:
        backbone_ok = True
        _ok(lines, f"backbone resolved: {backbone_dir}")
        for rel in BACKBONE_REQUIRED_FILES:
            target = backbone_dir / rel
            if target.exists():
                _ok(lines, f"backbone file present: {target}")
            else:
                _fail(lines, f"backbone file missing: {target}")
                backbone_ok = False
        if args.probe_model_load:
            backbone_ok = _probe_model_load(backbone_dir, lines) and backbone_ok
        summary["checks"]["backbone"] = backbone_ok
        if not backbone_ok:
            summary["ok"] = False

    output_dirs = [
        run_dir / "logs",
        run_dir / "artifacts",
        out_dir,
    ]
    if not args.skip_package:
        output_dirs.append(run_dir / "artifacts" / "package")
    write_ok = True
    for directory in output_dirs:
        if directory.exists():
            _ok(lines, f"output dir present: {directory}")
        else:
            _warn(lines, f"output dir not created yet: {directory}")
        if args.check_write_access:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                probe = directory / ".readiness_write_probe.tmp"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                _ok(lines, f"write probe passed: {directory}")
            except Exception as exc:
                _fail(lines, f"write probe failed for {directory}: {type(exc).__name__}: {exc}")
                write_ok = False
    summary["checks"]["write_access"] = write_ok
    if args.check_write_access and not write_ok:
        summary["ok"] = False

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    **summary,
                    "messages": lines,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    for line in lines:
        print(line)
    print(f"[readiness] overall = {'PASS' if summary['ok'] else 'FAIL'}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
