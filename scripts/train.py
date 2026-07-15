"""MPID training entry point (Phase 2 / T2.7).

Loads ``configs/baseline.yaml`` (or a path given on the CLI), builds a
``TrainConfig`` from it, and calls ``mpid.train.trainer.train``. The
training loop lives in :mod:`mpid.train.trainer`; this script is a
thin CLI shim around it.

Usage::

    # default config
    python scripts/train.py

    # custom config / override out_dir
    python scripts/train.py --config configs/baseline.yaml --out-dir artifacts/run1

The script writes two artefacts to ``out_dir``:

    * ``lora_baseline.safetensors``  — best LoRA + head checkpoint
    * ``train_summary.json``          — per-epoch F1, config, params count
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make ``mpid`` importable when the script is run from a clone of the
# repo (no editable install needed). Mirrors the convention in
# ``scripts/smoke_env.py`` and ``scripts/build_phase1.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mpid.train.trainer import TrainConfig, train  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_train_config(cfg: dict, out_dir_override: str | None) -> TrainConfig:
    """Flatten the YAML into a ``TrainConfig`` dataclass.

    The YAML is grouped by concern (``lora``, ``training``, ``io``) for
    readability; here we collapse those into the flat schema expected
    by the trainer. CLI override of ``out_dir`` is applied last.
    """
    defaults = cfg.get("defaults", {}) or {}
    lora = cfg.get("lora", {}) or {}
    training = cfg.get("training", {}) or {}
    io = cfg.get("io", {}) or {}

    return TrainConfig(
        train_jsonl=io["train_jsonl"],
        val_jsonl=io["val_jsonl"],
        out_dir=out_dir_override or io.get("out_dir", "artifacts/baseline"),
        backbone_name=defaults.get("backbone_name", "smolvlm-500m"),
        dtype=defaults.get("dtype", "float32"),
        device=defaults.get("device", "mps"),
        quantization=defaults.get("quantization"),
        gradient_checkpointing=bool(defaults.get("gradient_checkpointing", True)),
        lora_r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        lora_target=str(lora.get("target", "q_proj,k_proj,v_proj,o_proj")),
        epochs=int(training.get("epochs", 1)),
        max_train_records=int(training.get("max_train_records", 500)),
        max_val_records=int(training.get("max_val_records", 200)),
        batch_size=int(training.get("batch_size", 1)),
        lr=float(training.get("lr", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        class_weighted=bool(training.get("class_weighted", True)),
        early_stop_patience=int(training.get("early_stop_patience", 2)),
        log_every=int(training.get("log_every", 5)),
        seed=int(training.get("seed", 42)),
        # T2.16 真实训练开关
        max_train_seconds=float(training.get("max_train_seconds", 0.0)),
        preload_dataset=bool(training.get("preload_dataset", False)),
        checkpoint_name=str(training.get("checkpoint_name", "lora_baseline.safetensors")),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPID LoRA + 3-class trainer")
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "baseline.yaml",
        help="YAML config path (default: configs/baseline.yaml)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override the output directory from the config",
    )
    p.add_argument(
        "--max-train-seconds",
        type=float,
        default=0.0,
        help="T2.16: wall-clock budget in seconds. 0=no limit. "
             "When set, training stops and saves a partial checkpoint at the deadline.",
    )
    p.add_argument(
        "--preload-dataset",
        action="store_true",
        help="T2.16: pre-encode all records into RAM before training. "
             "Adds ~4 MB × N_records of RAM but eliminates per-step image preprocessing.",
    )
    p.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="T2.16: output safetensors file name (default: lora_baseline.safetensors). "
             "Phase 2.2 uses lora_full.safetensors.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="T2.16: every N training steps, save a partial checkpoint "
             "to <partial_name>. 0 disables (default: only at epoch end "
             "or budget cutoff). Recommended for long runs.",
    )
    p.add_argument(
        "--partial-name",
        type=str,
        default=None,
        help="T2.16: filename for periodic partial checkpoints "
             "(default: lora_partial.safetensors).",
    )
    p.add_argument(
        "-u", "--unbuffered",
        action="store_true",
        help="Force line-buffered stdout (always recommended for long runs)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Force line-buffered stdout when requested (or always — long runs
    # without -u look "stuck" to the user).
    if args.unbuffered:
        import sys as _sys
        _sys.stdout.reconfigure(line_buffering=True)
        _sys.stderr.reconfigure(line_buffering=True)

    raw = load_config(args.config)
    cfg = build_train_config(raw, args.out_dir)

    # CLI overrides.
    if args.max_train_seconds:
        cfg.max_train_seconds = args.max_train_seconds
    if args.preload_dataset:
        cfg.preload_dataset = True
    if args.checkpoint_name:
        cfg.checkpoint_name = args.checkpoint_name
    if args.save_every:
        cfg.save_every = int(args.save_every)
    if args.partial_name:
        cfg.partial_name = args.partial_name

    # Make IO paths absolute relative to the repo root so the script
    # works from any cwd.
    repo = REPO_ROOT
    cfg.train_jsonl = str((repo / cfg.train_jsonl).resolve()) \
        if not Path(cfg.train_jsonl).is_absolute() else cfg.train_jsonl
    cfg.val_jsonl = str((repo / cfg.val_jsonl).resolve()) \
        if not Path(cfg.val_jsonl).is_absolute() else cfg.val_jsonl
    cfg.out_dir = str((repo / cfg.out_dir).resolve()) \
        if not Path(cfg.out_dir).is_absolute() else cfg.out_dir

    print(f"[train] config: {args.config}", flush=True)
    print(f"[train] out_dir: {cfg.out_dir}", flush=True)
    print(f"[train] train: {cfg.train_jsonl}", flush=True)
    print(f"[train] val:   {cfg.val_jsonl}", flush=True)
    print(f"[train] epochs={cfg.epochs}  max_train_records={cfg.max_train_records}  "
          f"batch_size={cfg.batch_size}  lr={cfg.lr}", flush=True)
    if cfg.max_train_seconds:
        print(f"[train] BUDGET: {cfg.max_train_seconds}s", flush=True)
    if cfg.preload_dataset:
        print(f"[train] PRELOAD: enabled (~4 MB × {cfg.max_train_records} ≈ "
              f"{cfg.max_train_records*4/1024:.1f} GB RAM)", flush=True)

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
