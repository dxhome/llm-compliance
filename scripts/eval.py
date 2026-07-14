"""MPID evaluation entry point (Phase 2 / T2.9 + T2.11 comparison).

Supports two modes:

  1. **Single-model mode** (default, backward-compatible):
     Loads a trained LoRA + head checkpoint and evaluates it on a
     labelled JSONL split. Outputs:

       * ``report_baseline.json`` — per-class P/R/F1, accuracy, Macro F1
       * ``confusion_matrix.json`` — 3×3 confusion matrix
       * ``report_baseline.md`` — human-readable summary table

  2. **Comparison mode** (``--compare`` flag, T2.11):
     Runs BOTH a *baseline* model (LoRA + head at random init) and the
     *modified* (LoRA + head loaded from checkpoint) on the same val
     split, and outputs side-by-side comparison:

       * ``baseline_report.json`` / ``modified_report.json``
       * ``comparison_report.md`` — table + interpretation
       * ``comparison_delta.json`` — quantitative deltas

     The goal is to **prove that the LoRA fine-tuning has given the
     model injection-defence capability**, by showing the modified
     version materially outperforms the un-trained baseline on the
     same samples.

Usage::

    # Single-model (backward-compatible)
    python scripts/eval.py
    python scripts/eval.py --checkpoint artifacts/baseline/lora_baseline.safetensors

    # Comparison (T2.11)
    python scripts/eval.py --compare
    python scripts/eval.py --compare --val data/mpid-v1-crossmodal/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader

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
    load_checkpoint,
)
from mpid.data.dataset import MPIDJsonlDataset, collate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPID evaluator (T2.9 / T2.11)")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "configs" / "baseline.yaml",
                   help="YAML config (same schema as train.py)")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "artifacts" / "baseline" / "lora_baseline.safetensors",
                   help="LoRA+head checkpoint (used by single & compare modes)")
    p.add_argument("--val", type=Path, default=None,
                   help="Override val JSONL (else uses config io.val_jsonl)")
    p.add_argument("--out", type=Path, default=None,
                   help="Override output dir (else uses config io.out_dir)")
    p.add_argument("--max-records", type=int, default=None,
                   help="Cap the number of eval records (debugging)")
    p.add_argument("--compare", action="store_true",
                   help="T2.11: run baseline (untrained) vs modified (LoRA-trained) "
                        "and emit a side-by-side comparison report.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _build_adapter_and_lora(cfg: TrainConfig):
    """Construct VLMAdapter, inject LoRA, return (peft_model, hidden_size, n_lora_params)."""
    adapter = VLMAdapter(
        backbone_name=cfg.backbone_name,
        dtype=cfg.dtype,
        quantization=cfg.quantization,
        device=cfg.device,
        gradient_checkpointing=False,  # inference: no need to checkpoint
    )
    peft_model, n_lora = inject_lora(adapter.model, cfg)
    return peft_model, adapter.hidden_size, n_lora


def _build_dataloader_with_processor(processor, val_path: Path,
                                     device: str, batch_size: int,
                                     max_records: Optional[int]) -> DataLoader:
    val_ds = MPIDJsonlDataset(
        Path(val_path), processor=processor, device=device,
        max_records=max_records,
    )
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=0), val_ds


def _build_probe_processor(cfg: TrainConfig):
    """Build a throwaway VLMAdapter just to expose its processor.

    Both single-model and comparison modes need a processor to build the
    DataLoader. Re-using the eval-time peft_model's processor would mean
    building the backbone twice; using a single probe keeps the flow
    simple and matches the comparison mode's design.
    """
    return VLMAdapter(
        backbone_name=cfg.backbone_name,
        dtype=cfg.dtype,
        quantization=cfg.quantization,
        device=cfg.device,
        gradient_checkpointing=False,
    )


# ---------------------------------------------------------------------------
# Single-model evaluation (backward-compatible)
# ---------------------------------------------------------------------------

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


def run_single_model(cfg: TrainConfig, checkpoint: Path, val_path: Path,
                     out_dir: Path, max_records: Optional[int]) -> int:
    """Original single-model eval — kept for backward compatibility."""
    print(f"[eval] config:     {args_config_str(checkpoint, val_path, out_dir)}")
    print(f"[eval] loading adapter on {cfg.device} ...")

    peft_model, hidden_size, n_lora = _build_adapter_and_lora(cfg)
    head = ClassificationHead(in_dim=hidden_size,
                               num_classes=NUM_CLASSES).to(cfg.device)
    n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[eval] LoRA params: {n_lora:,}  Head params: {n_head:,}")

    state = load_checkpoint(checkpoint, head)
    has_lora_state = any(k.startswith("lora.") for k in state.keys())
    if has_lora_state:
        _apply_lora_state(peft_model, state)
    print(f"[eval] loaded checkpoint ({len(state)} tensors, "
          f"has_lora_state={has_lora_state})")
    peft_model.eval(); head.eval()

    # Build the dataloader via a throwaway probe (cheaper than reusing
    # the peft_model's processor and keeps the path symmetric with the
    # comparison mode).
    probe = _build_probe_processor(cfg)
    val_dl, val_ds = _build_dataloader_with_processor(
        probe.processor, val_path, cfg.device, cfg.batch_size, max_records
    )
    print(f"[eval] val size: {len(val_ds)}")

    ev = evaluate(peft_model, head, val_dl, cfg.device)
    _write_single_model_artifacts(ev, val_ds, cfg, out_dir, checkpoint, val_path)
    return 0


def args_config_str(checkpoint, val_path, out_dir):
    return (f"checkpoint={checkpoint} val={val_path} out={out_dir}")


def _write_single_model_artifacts(ev, val_ds, cfg, out_dir, checkpoint, val_path):
    report = ev["report"]
    cm = ev["confusion_matrix"]
    macro_f1 = report["macro avg"]["f1-score"]
    acc = report["accuracy"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    summary = {
        "checkpoint": str(checkpoint),
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


# ---------------------------------------------------------------------------
# Comparison mode (T2.11)
# ---------------------------------------------------------------------------

def _apply_lora_state(peft_model, full_state: dict) -> None:
    """Apply ``lora.<peft_param_name> -> tensor`` keys onto a peft model."""
    from peft import set_peft_model_state_dict
    lora_state = {k.removeprefix("lora."): v
                  for k, v in full_state.items() if k.startswith("lora.")}
    if lora_state:
        set_peft_model_state_dict(peft_model, lora_state)


def _build_random_model(cfg: TrainConfig) -> tuple:
    """Build a fresh (random LoRA + random head) model. Returns
    ``(peft_model, head, n_lora, n_head)``."""
    peft_model, hidden_size, n_lora = _build_adapter_and_lora(cfg)
    head = ClassificationHead(in_dim=hidden_size,
                               num_classes=NUM_CLASSES).to(cfg.device)
    n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
    peft_model.eval(); head.eval()
    return peft_model, head, n_lora, n_head


def _build_loaded_model(cfg: TrainConfig, checkpoint: Path) -> tuple:
    """Build a model with LoRA + head loaded from a checkpoint. Returns
    ``(peft_model, head, n_lora, n_head)``."""
    peft_model, hidden_size, n_lora = _build_adapter_and_lora(cfg)
    head = ClassificationHead(in_dim=hidden_size,
                               num_classes=NUM_CLASSES).to(cfg.device)
    n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
    state = load_checkpoint(checkpoint, head)
    if any(k.startswith("lora.") for k in state.keys()):
        _apply_lora_state(peft_model, state)
    peft_model.eval(); head.eval()
    return peft_model, head, n_lora, n_head


def _run_one_model(peft_model, head, val_dl, val_ds, cfg) -> dict:
    """Run a single eval pass and return a normalized summary dict."""
    ev = evaluate(peft_model, head, val_dl, cfg.device)
    report = ev["report"]
    cm = ev["confusion_matrix"]
    return {
        "n_eval": len(val_ds),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {k: report.get(k, {}) for k in LABEL_ORDER},
        "confusion_matrix": cm,
        "y_pred": ev["y_pred"],
        "y_gold": ev["y_gold"],
    }


def _compute_delta(baseline: dict, modified: dict) -> dict:
    """Quantitative deltas between the two runs."""
    per_class_recall_delta = {}
    for label in LABEL_ORDER:
        b = baseline["per_class"].get(label, {}).get("recall", 0.0)
        m = modified["per_class"].get(label, {}).get("recall", 0.0)
        per_class_recall_delta[label] = m - b
    per_class_f1_delta = {}
    for label in LABEL_ORDER:
        b = baseline["per_class"].get(label, {}).get("f1-score", 0.0)
        m = modified["per_class"].get(label, {}).get("f1-score", 0.0)
        per_class_f1_delta[label] = m - b
    return {
        "accuracy_delta":      modified["accuracy"] - baseline["accuracy"],
        "macro_f1_delta":      modified["macro_f1"] - baseline["macro_f1"],
        "weighted_f1_delta":   modified["weighted_f1"] - baseline["weighted_f1"],
        "per_class_recall_delta": per_class_recall_delta,
        "per_class_f1_delta":     per_class_f1_delta,
    }


def _make_comparison_markdown(baseline: dict, modified: dict, delta: dict,
                              n: int) -> str:
    """Human-readable side-by-side report."""
    lines = [
        "# MPID Baseline vs Modified Comparison",
        "",
        f"- eval records: **{n}**",
        "",
        "## Headline metrics",
        "",
        "| metric | baseline (untrained) | modified (LoRA-trained) | delta |",
        "|---|---|---|---|",
        f"| accuracy    | {baseline['accuracy']:.4f} | {modified['accuracy']:.4f} | {delta['accuracy_delta']:+.4f} |",
        f"| macro F1    | {baseline['macro_f1']:.4f} | {modified['macro_f1']:.4f} | {delta['macro_f1_delta']:+.4f} |",
        f"| weighted F1 | {baseline['weighted_f1']:.4f} | {modified['weighted_f1']:.4f} | {delta['weighted_f1_delta']:+.4f} |",
        "",
        "## Per-class recall",
        "",
        "| class | baseline | modified | delta |",
        "|---|---|---|---|",
    ]
    for label in LABEL_ORDER:
        b = baseline["per_class"].get(label, {}).get("recall", 0.0)
        m = modified["per_class"].get(label, {}).get("recall", 0.0)
        d = delta["per_class_recall_delta"][label]
        lines.append(f"| {label} | {b:.4f} | {m:.4f} | {d:+.4f} |")
    lines += [
        "",
        "## Per-class F1",
        "",
        "| class | baseline | modified | delta |",
        "|---|---|---|---|",
    ]
    for label in LABEL_ORDER:
        b = baseline["per_class"].get(label, {}).get("f1-score", 0.0)
        m = modified["per_class"].get(label, {}).get("f1-score", 0.0)
        d = delta["per_class_f1_delta"][label]
        lines.append(f"| {label} | {b:.4f} | {m:.4f} | {d:+.4f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        _interpret(delta),
        "",
        "## Pass / fail",
        "",
        f"- macro F1 delta = {delta['macro_f1_delta']:+.4f}  → "
        f"{'PASS' if delta['macro_f1_delta'] >= 0.20 else 'FAIL'} (threshold: ≥ +0.20)",
        f"- per-class recall improved by ≥ +0.20 on "
        f"{sum(1 for v in delta['per_class_recall_delta'].values() if v >= 0.20)}"
        f" / {len(LABEL_ORDER)} classes  → "
        f"{'PASS' if sum(1 for v in delta['per_class_recall_delta'].values() if v >= 0.20) >= 2 else 'FAIL'}"
        f" (threshold: ≥ 2)",
        "",
    ]
    return "\n".join(lines) + "\n"


def _interpret(delta: dict) -> str:
    bits = []
    if delta["macro_f1_delta"] >= 0.20:
        bits.append("- 改造版 Macro F1 显著高于基线 → LoRA 训练生效")
    elif delta["macro_f1_delta"] > 0:
        bits.append("- 改造版 Macro F1 高于基线但差距较小 → 训练不充分或 LoRA 配置有 bug")
    else:
        bits.append("- 改造版 Macro F1 ≤ 基线 → 训练未生效，需要回溯 Step 5")
    improving = [(k, v) for k, v in delta["per_class_recall_delta"].items() if v >= 0.20]
    if improving:
        bits.append(f"- 提升 ≥ 0.20 的类别: {', '.join(k for k, _ in improving)}")
    bits.append("- 间接注入 recall 的提升幅度是判断「端到端学习是否真学到模式」的关键指标")
    return "\n".join(bits)


def run_comparison(cfg: TrainConfig, checkpoint: Path, val_path: Path,
                   out_dir: Path, max_records: Optional[int]) -> int:
    """T2.11: run baseline (random init) vs modified (LoRA-trained) and
    write a side-by-side comparison report."""
    print(f"[eval] config:     cfg=baseline")
    print(f"[eval] checkpoint: {checkpoint}")
    print(f"[eval] val:        {val_path}")
    print(f"[eval] out_dir:    {out_dir}")
    print(f"[eval] mode:       COMPARE (baseline vs LoRA-trained)")

    # Build the dataloader once via a throwaway probe.
    probe = _build_probe_processor(cfg)
    val_dl, val_ds = _build_dataloader_with_processor(
        probe.processor, val_path, cfg.device, cfg.batch_size, max_records
    )
    print(f"[eval] val size: {len(val_ds)}")

    # 1. Baseline (random init)
    print(f"\n[eval] === Running BASELINE (untrained) ===")
    base_peft, base_head, base_lora, base_head_n = _build_random_model(cfg)
    print(f"[eval] baseline: LoRA params: {base_lora:,}  Head params: {base_head_n:,}  (random init)")
    baseline_result = _run_one_model(base_peft, base_head, val_dl, val_ds, cfg)
    print(f"[eval] baseline: acc={baseline_result['accuracy']:.4f}  "
          f"macro F1={baseline_result['macro_f1']:.4f}  "
          f"weighted F1={baseline_result['weighted_f1']:.4f}")
    del base_peft, base_head
    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    # 2. Modified (LoRA + head loaded from checkpoint)
    print(f"\n[eval] === Running MODIFIED (LoRA-trained) ===")
    mod_peft, mod_head, mod_lora, mod_head_n = _build_loaded_model(cfg, checkpoint)
    print(f"[eval] modified: LoRA params: {mod_lora:,}  Head params: {mod_head_n:,}  (loaded)")
    modified_result = _run_one_model(mod_peft, mod_head, val_dl, val_ds, cfg)
    print(f"[eval] modified: acc={modified_result['accuracy']:.4f}  "
          f"macro F1={modified_result['macro_f1']:.4f}  "
          f"weighted F1={modified_result['weighted_f1']:.4f}")
    del mod_peft, mod_head
    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    # 3. Compare
    print(f"\n[eval] === Comparison ===")
    delta = _compute_delta(baseline_result, modified_result)
    print(f"[eval] F1 delta:     {delta['macro_f1_delta']:+.4f}  "
          f"(baseline {baseline_result['macro_f1']:.4f} → "
          f"modified {modified_result['macro_f1']:.4f})")
    print(f"[eval] Acc delta:    {delta['accuracy_delta']:+.4f}")
    for label in LABEL_ORDER:
        print(f"[eval] recall delta  [{label:>9s}]: "
              f"{delta['per_class_recall_delta'][label]:+.4f}")

    # 4. Write artefacts
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_comparison_artifacts(baseline_result, modified_result, delta,
                                val_ds, cfg, checkpoint, val_path, out_dir)
    return 0


def _write_comparison_artifacts(baseline: dict, modified: dict, delta: dict,
                                val_ds, cfg, checkpoint, val_path, out_dir):
    n = len(val_ds)
    base_summary = {
        "checkpoint": str(checkpoint),
        "val_jsonl":  str(val_path),
        "n_eval":     n,
        "accuracy":   baseline["accuracy"],
        "macro_f1":   baseline["macro_f1"],
        "weighted_f1": baseline["weighted_f1"],
        "per_class":  baseline["per_class"],
        "confusion_matrix": baseline["confusion_matrix"],
    }
    mod_summary = {
        "checkpoint": str(checkpoint),
        "val_jsonl":  str(val_path),
        "n_eval":     n,
        "accuracy":   modified["accuracy"],
        "macro_f1":   modified["macro_f1"],
        "weighted_f1": modified["weighted_f1"],
        "per_class":  modified["per_class"],
        "confusion_matrix": modified["confusion_matrix"],
    }
    with open(out_dir / "baseline_report.json", "w", encoding="utf-8") as f:
        json.dump(base_summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "modified_report.json", "w", encoding="utf-8") as f:
        json.dump(mod_summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "comparison_delta.json", "w", encoding="utf-8") as f:
        json.dump(delta, f, ensure_ascii=False, indent=2)
    md = _make_comparison_markdown(baseline, modified, delta, n)
    with open(out_dir / "comparison_report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[eval] wrote {out_dir/'baseline_report.json'}")
    print(f"[eval] wrote {out_dir/'modified_report.json'}")
    print(f"[eval] wrote {out_dir/'comparison_delta.json'}")
    print(f"[eval] wrote {out_dir/'comparison_report.md'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = build_train_config_from_yaml(args.config)
    val_path = args.val or (REPO_ROOT / cfg.val_jsonl)
    out_dir = args.out or (REPO_ROOT / cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        return run_comparison(cfg, args.checkpoint, val_path, out_dir,
                              args.max_records)
    return run_single_model(cfg, args.checkpoint, val_path, out_dir,
                            args.max_records)


if __name__ == "__main__":
    raise SystemExit(main())
