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
        log_every=int(training.get("log_every", 10)),
        seed=int(training.get("seed", 42)),
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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw = load_config(args.config)
    cfg = build_train_config(raw, args.out_dir)

    # Make IO paths absolute relative to the repo root so the script
    # works from any cwd.
    repo = REPO_ROOT
    cfg.train_jsonl = str((repo / cfg.train_jsonl).resolve()) \
        if not Path(cfg.train_jsonl).is_absolute() else cfg.train_jsonl
    cfg.val_jsonl = str((repo / cfg.val_jsonl).resolve()) \
        if not Path(cfg.val_jsonl).is_absolute() else cfg.val_jsonl
    cfg.out_dir = str((repo / cfg.out_dir).resolve()) \
        if not Path(cfg.out_dir).is_absolute() else cfg.out_dir

    print(f"[train] config: {args.config}")
    print(f"[train] out_dir: {cfg.out_dir}")
    print(f"[train] train: {cfg.train_jsonl}")
    print(f"[train] val:   {cfg.val_jsonl}")
    print(f"[train] epochs={cfg.epochs}  max_train_records={cfg.max_train_records}  "
          f"batch_size={cfg.batch_size}  lr={cfg.lr}")

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
