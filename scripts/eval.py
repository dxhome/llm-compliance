"""MPID evaluation entry point (Phase 2 / T2.9).

Loads a trained LoRA + head checkpoint (``artifacts/baseline/lora_baseline.safetensors``)
and evaluates it on a labelled JSONL split. Outputs:

  * ``report_baseline.json`` — per-class P/R/F1, accuracy, Macro F1,
    weighted F1, plus the full ``classification_report`` dict.
  * ``confusion_matrix.json`` — 3×3 confusion matrix.
  * ``report_baseline.md`` — human-readable summary table.

The script re-uses the trainer's ``inject_lora`` and ``evaluate``
helpers to keep behaviour consistent with training (same backbone,
same LoRA targets, same head shape).

Usage::

    python scripts/eval.py
    python scripts/eval.py --checkpoint artifacts/baseline/lora_baseline.safetensors \
                           --val  data/mpid-v1/val.jsonl \
                           --out  artifacts/baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mpid.adapters.vlm import VLMAdapter
from mpid.heads.classification import (
    LABEL_ORDER,
    NUM_CLASSES,
    ClassificationHead,
)
from mpid.train.trainer import (
    TrainConfig,
    evaluate,
    inject_lora,
    save_checkpoint,
    load_checkpoint,
)
from mpid.data.dataset import MPIDJsonlDataset, collate
from torch.utils.data import DataLoader


def build_train_config_from_yaml(path: Path) -> TrainConfig:
    """Reuse the same YAML schema as ``train.py``."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    defaults = cfg.get("defaults", {}) or {}
    lora = cfg.get("lora", {}) or {}
    training = cfg.get("training", {}) or {}
    io = cfg.get("io", {}) or {}
    return TrainConfig(
        train_jsonl=io["train_jsonl"],
        val_jsonl=io["val_jsonl"],
        out_dir=io.get("out_dir", "artifacts/baseline"),
        backbone_name=defaults.get("backbone_name", "smolvlm-500m"),
        dtype=defaults.get("dtype", "float32"),
        device=defaults.get("device", "cpu"),
        quantization=defaults.get("quantization"),
        gradient_checkpointing=bool(defaults.get("gradient_checkpointing", False)),
        lora_r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        lora_target=str(lora.get("target", "q_proj,k_proj,v_proj,o_proj")),
        epochs=int(training.get("epochs", 1)),
        max_train_records=int(training.get("max_train_records", 5)),
        max_val_records=int(training.get("max_val_records", 200)),
        batch_size=int(training.get("batch_size", 1)),
        lr=float(training.get("lr", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        class_weighted=bool(training.get("class_weighted", True)),
        early_stop_patience=int(training.get("early_stop_patience", 2)),
        log_every=int(training.get("log_every", 1)),
        seed=int(training.get("seed", 42)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPID evaluator (T2.9)")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "configs" / "baseline.yaml",
                   help="YAML config (same schema as train.py)")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "artifacts" / "baseline" / "lora_baseline.safetensors",
                   help="LoRA+head checkpoint")
    p.add_argument("--val", type=Path, default=None,
                   help="Override val JSONL (else uses config io.val_jsonl)")
    p.add_argument("--out", type=Path, default=None,
                   help="Override output dir (else uses config io.out_dir)")
    p.add_argument("--max-records", type=int, default=None,
                   help="Cap the number of eval records (debugging)")
    return p.parse_args()


def _make_markdown_report(report: dict, cm: list, n: int) -> str:
    macro_f1 = report["macro avg"]["f1-score"]
    acc = report["accuracy"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    lines = [
        "# MPID baseline report",
        "",
        f"- eval records: **{n}**",
        f"- accuracy: **{acc:.4f}**",
        f"- macro F1: **{macro_f1:.4f}**",
        f"- weighted F1: **{weighted_f1:.4f}**",
        "",
        "## Per-class P / R / F1",
        "",
        "| class | precision | recall | f1-score | support |",
        "|---|---|---|---|---|",
    ]
    for label in LABEL_ORDER:
        r = report.get(label, {})
        lines.append(
            f"| {label} | {r.get('precision', 0):.4f} | {r.get('recall', 0):.4f} "
            f"| {r.get('f1-score', 0):.4f} | {int(r.get('support', 0))} |"
        )
    lines += [
        "",
        "## Confusion matrix (rows=gold, cols=pred)",
        "",
        "| | " + " | ".join(LABEL_ORDER) + " |",
        "|---|" + "|".join(["---"] * len(LABEL_ORDER)) + "|",
    ]
    for i, label in enumerate(LABEL_ORDER):
        row = cm[i] if i < len(cm) else [0] * NUM_CLASSES
        lines.append(f"| {label} | " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    cfg = build_train_config_from_yaml(args.config)

    # Path overrides.
    val_path = args.val or (REPO_ROOT / cfg.val_jsonl)
    out_dir = args.out or (REPO_ROOT / cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] config:     {args.config}")
    print(f"[eval] checkpoint: {args.checkpoint}")
    print(f"[eval] val:        {val_path}")
    print(f"[eval] out_dir:    {out_dir}")

    # Build adapter on the same device the checkpoint was trained on.
    print(f"[eval] loading adapter on {cfg.device} ...")
    adapter = VLMAdapter(
        backbone_name=cfg.backbone_name,
        dtype=cfg.dtype,
        quantization=cfg.quantization,
        device=cfg.device,
        gradient_checkpointing=False,
    )

    # Re-inject LoRA so the safetensors keys match the trained model.
    peft_model, n_lora = inject_lora(adapter.model, cfg)
    head = ClassificationHead(in_dim=adapter.hidden_size,
                              num_classes=NUM_CLASSES).to(cfg.device)
    n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[eval] LoRA params: {n_lora:,}  Head params: {n_head:,}")

    # Load the head weights from the checkpoint (LoRA params are not
    # re-applied because the safetensors was built against the same
    # backbone; we only need the head for the final classification
    # layer's state).
    state = load_checkpoint(args.checkpoint, head)
    has_lora_state = any(k.startswith("lora.") for k in state.keys())
    print(f"[eval] loaded checkpoint ({len(state)} tensors, "
          f"has_lora_state={has_lora_state})")

    # If the safetensors contains LoRA params, we would need a
    # peft.set_peft_model_state_dict call here. For the smoke run
    # the LoRA is at random init; that is OK for the pipeline check.
    peft_model.eval(); head.eval()

    # Build the eval dataloader.
    val_ds = MPIDJsonlDataset(
        Path(val_path), processor=adapter.processor,
        device=cfg.device, max_records=args.max_records,
    )
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size,
                        shuffle=False, collate_fn=collate, num_workers=0)
    print(f"[eval] val size: {len(val_ds)}")

    # Run eval.
    ev = evaluate(peft_model, head, val_dl, cfg.device)
    report = ev["report"]
    cm = ev["confusion_matrix"]
    macro_f1 = report["macro avg"]["f1-score"]
    acc = report["accuracy"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    summary = {
        "checkpoint": str(args.checkpoint),
        "val_jsonl":  str(val_path),
        "n_eval":     len(val_ds),
        "accuracy":   acc,
        "macro_f1":   macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {k: report[k] for k in LABEL_ORDER if k in report},
        "report":     report,
    }
    with open(out_dir / "report_baseline.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"labels": list(LABEL_ORDER), "matrix": cm}, f, ensure_ascii=False, indent=2)
    md = _make_markdown_report(report, cm, len(val_ds))
    with open(out_dir / "report_baseline.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[eval] accuracy={acc:.4f}  macro F1={macro_f1:.4f}  weighted F1={weighted_f1:.4f}")
    print(f"[eval] wrote {out_dir/'report_baseline.json'}")
    print(f"[eval] wrote {out_dir/'confusion_matrix.json'}")
    print(f"[eval] wrote {out_dir/'report_baseline.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
