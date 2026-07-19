"""MPID evaluation entry point (Phase 2 / T2.9 + T2.11 + Phase 3 / T3.7).

Supports three modes:

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

  3. **Early-exit comparison mode** (``--early-exit`` flag, T3.7):
     Runs the SAME loaded model TWICE on the val split — once with
     C4 disabled (full path) and once with C4 enabled (early-exit
     path). Outputs:

       * ``early_exit_compare.json`` — exit rate, latency, F1 deltas
       * ``early_exit_compare.md`` — human-readable summary
       * ``early_exit_per_sample.jsonl`` — per-record decision log

     The comparison measures three things:
       - **Exit rate**           : how often C4 triggered
       - **Latency simulation**  : VLM + head + simulated C5/C6 cost
                                   (default 200ms) — with/without C4
       - **F1 impact**           : does C4 cause any wrong decisions?

Usage::

    # Single-model (backward-compatible)
    python scripts/eval.py
    python scripts/eval.py --checkpoint runs/_templates/artifacts/checkpoints/lora_baseline.safetensors

    # Comparison (T2.11)
    python scripts/eval.py --compare
    python scripts/eval.py --compare --val runs/_datasets/mpid-v1-crossmodal/test.jsonl

    # Early-exit comparison (T3.7)
    python scripts/eval.py --early-exit
    python scripts/eval.py --early-exit --clean-threshold 0.90
    python scripts/eval.py --early-exit --simulate-c5-c6-ms 200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mpid.adapters.vlm import VLMAdapter
from mpid.heads.classification import (
    LABEL2IDX,
    LABEL_ORDER,
    NUM_CLASSES,
    ClassificationHead,
)
from mpid.early_exit import (
    EarlyExitConfig,
    EarlyExitStats,
    should_early_exit,
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

def _resolve_input_path(value: str | Path, base_dir: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    from_config = (base_dir / p).resolve()
    if from_config.exists():
        return from_config
    return (REPO_ROOT / p).resolve()


def _resolve_output_path(value: str | Path, base_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def build_train_config_from_yaml(path: Path) -> TrainConfig:
    """Reuse the same YAML schema as ``train.py``."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_dir = path.resolve().parent
    defaults = cfg.get("defaults", {}) or {}
    lora = cfg.get("lora", {}) or {}
    training = cfg.get("training", {}) or {}
    io = cfg.get("io", {}) or {}
    return TrainConfig(
        train_jsonl=str(_resolve_input_path(io["train_jsonl"], base_dir)),
        val_jsonl=str(_resolve_input_path(io["val_jsonl"], base_dir)),
        out_dir=str(_resolve_output_path(io.get("out_dir", "artifacts/eval"), base_dir)),
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
        default=REPO_ROOT / "runs" / "_templates" / "configs" / "baseline.yaml",
                   help="YAML config (same schema as train.py)")
    p.add_argument("--checkpoint", type=Path,
        default=REPO_ROOT / "runs" / "_templates" / "artifacts" / "checkpoints" / "lora_baseline.safetensors",
                   help="LoRA+head checkpoint (used by single & compare modes)")
    p.add_argument("--val", type=Path, default=None,
                   help="Override val JSONL (else uses config io.val_jsonl)")
    p.add_argument("--out", type=Path, default=None,
                   help="Override output dir (else uses config io.out_dir)")
    p.add_argument("--max-records", type=int, default=None,
                   help="Cap the number of eval records (debugging)")
    p.add_argument("--stratified-max-records", type=int, default=None,
                   help="Cap eval records with deterministic stratified sampling "
                        "across the 3 labels.")
    p.add_argument("--sample-seed", type=int, default=42,
                   help="Random seed for stratified sampling (default: 42).")
    p.add_argument("--chunk-size", type=int, default=0,
                   help="Evaluate in independent chunks of this many records; "
                        "each chunk writes its own complete artifacts.")
    p.add_argument("--chunk-output-dir", type=Path, default=None,
                   help="Directory for per-chunk artifacts (default: out/chunks).")
    p.add_argument("--compare", action="store_true",
                   help="T2.11: run baseline (untrained) vs modified (LoRA-trained) "
                        "and emit a side-by-side comparison report.")
    p.add_argument("--compare-smoke-vs-full", action="store_true",
                   help="T2.18: compare the smoke model (lora_baseline.safetensors) "
                        "vs the full model (lora_full.safetensors) on the same "
                        "test set. Both are loaded and evaluated.")
    p.add_argument("--smoke-checkpoint", type=Path, default=None,
                   help="T2.18: smoke checkpoint. Default is the run-local "
                        "artifacts/checkpoints/lora_baseline.safetensors when available.")
    p.add_argument("--full-checkpoint", type=Path, default=None,
                   help="T2.18: full checkpoint. Default is --checkpoint.")
    p.add_argument("--early-exit", action="store_true",
                   help="T3.7: run the SAME model twice — once with C4 disabled "
                        "(full path) and once with C4 enabled (early-exit path) — "
                        "and emit a comparison of exit rate, latency, F1.")
    p.add_argument("--clean-threshold", type=float, default=0.95,
                   help="T3.7: C4 clean threshold (default: 0.95).")
    p.add_argument("--simulate-c5-c6-ms", type=float, default=200.0,
                   help="T3.7: simulated C5+C6 cost in ms (default: 200). "
                        "Used to estimate the latency savings of skipping C5/C6.")
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
                                     max_records: Optional[int],
                                     stratified_max_records: Optional[int] = None,
                                     sample_seed: int = 42) -> DataLoader:
    val_ds = MPIDJsonlDataset(
        Path(val_path), processor=processor, device=device,
        max_records=max_records,
    )
    effective_ds = val_ds
    if stratified_max_records is not None and 0 < stratified_max_records < len(val_ds):
        sampled_indices = _build_stratified_indices(
            val_ds.records,
            stratified_max_records,
            seed=sample_seed,
        )
        effective_ds = Subset(val_ds, sampled_indices)
        sampled_counts: dict[str, int] = {}
        for idx in sampled_indices:
            label = val_ds.records[idx]["label"]
            sampled_counts[label] = sampled_counts.get(label, 0) + 1
        print(
            "[eval] stratified sample enabled: "
            f"{len(sampled_indices)}/{len(val_ds)} records "
            f"(seed={sample_seed}, counts={sampled_counts})"
        )
    return DataLoader(effective_ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=0), effective_ds


def _build_stratified_indices(records: list[dict], target_n: int, seed: int) -> list[int]:
    """Return deterministic stratified sample indices with all labels covered."""
    by_label: dict[str, list[int]] = {label: [] for label in LABEL_ORDER}
    for idx, record in enumerate(records):
        label = record.get("label")
        if label in by_label:
            by_label[label].append(idx)

    available_labels = [label for label in LABEL_ORDER if by_label[label]]
    if not available_labels:
        return list(range(min(target_n, len(records))))

    rng = random.Random(seed)
    for indices in by_label.values():
        rng.shuffle(indices)

    total_available = sum(len(by_label[label]) for label in available_labels)
    target_n = min(target_n, total_available)
    if target_n <= len(available_labels):
        chosen_labels = available_labels[:target_n]
        selected = [by_label[label][0] for label in chosen_labels]
        rng.shuffle(selected)
        return selected

    counts = {label: 1 for label in available_labels}
    remaining = target_n - len(available_labels)
    quotas: list[tuple[float, str]] = []
    for label in available_labels:
        pool_size = len(by_label[label])
        extra_capacity = pool_size - 1
        if extra_capacity <= 0:
            continue
        ideal_extra = remaining * (pool_size / total_available)
        allocated = min(extra_capacity, int(ideal_extra))
        counts[label] += allocated
        quotas.append((ideal_extra - allocated, label))

    assigned = sum(counts.values())
    remaining = target_n - assigned
    if remaining > 0:
        for _, label in sorted(quotas, reverse=True):
            if remaining <= 0:
                break
            extra_capacity = len(by_label[label]) - counts[label]
            if extra_capacity <= 0:
                continue
            take = min(extra_capacity, remaining)
            counts[label] += take
            remaining -= take

    if remaining > 0:
        for label in available_labels:
            if remaining <= 0:
                break
            extra_capacity = len(by_label[label]) - counts[label]
            if extra_capacity <= 0:
                continue
            take = min(extra_capacity, remaining)
            counts[label] += take
            remaining -= take

    selected: list[int] = []
    for label in available_labels:
        selected.extend(by_label[label][:counts[label]])
    rng.shuffle(selected)
    return selected


def _iter_chunk_datasets(dataset, chunk_size: int):
    """Yield (1-based chunk_index, chunk_dataset) pairs."""
    if chunk_size <= 0:
        yield 1, dataset
        return
    n = len(dataset)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        yield (start // chunk_size) + 1, Subset(dataset, list(range(start, stop)))


def _eval_from_predictions(y_gold: list[int], y_pred: list[int]) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    cm = confusion_matrix(y_gold, y_pred, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        y_gold,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "confusion_matrix": cm.tolist(),
        "report": report,
        "y_pred": y_pred,
        "y_gold": y_gold,
    }


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


def run_single_model(
    cfg: TrainConfig,
    checkpoint: Path,
    val_path: Path,
    out_dir: Path,
    max_records: Optional[int],
    stratified_max_records: Optional[int] = None,
    sample_seed: int = 42,
    chunk_size: int = 0,
    chunk_output_dir: Optional[Path] = None,
) -> int:
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
        probe.processor,
        val_path,
        cfg.device,
        cfg.batch_size,
        max_records,
        stratified_max_records=stratified_max_records,
        sample_seed=sample_seed,
    )
    print(f"[eval] val size: {len(val_ds)}")

    if chunk_size > 0:
        chunk_output_dir = chunk_output_dir or (out_dir / "chunks")
        chunk_output_dir.mkdir(parents=True, exist_ok=True)
        all_gold: list[int] = []
        all_pred: list[int] = []
        for chunk_idx, chunk_ds in _iter_chunk_datasets(val_ds, chunk_size):
            chunk_dir = chunk_output_dir / f"single_model_group_{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[eval] === chunk {chunk_idx} single-model eval "
                f"({len(chunk_ds)} records) -> {chunk_dir} ==="
            )
            chunk_dl = DataLoader(
                chunk_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=0,
            )
            chunk_ev = evaluate(
                peft_model,
                head,
                chunk_dl,
                cfg.device,
                progress_every=5,
                progress_label=f"eval-g{chunk_idx:03d}",
            )
            _write_single_model_artifacts(
                chunk_ev,
                chunk_ds,
                cfg,
                chunk_dir,
                checkpoint,
                val_path,
            )
            all_gold.extend(chunk_ev["y_gold"])
            all_pred.extend(chunk_ev["y_pred"])
            print(f"[eval] === chunk {chunk_idx} single-model eval done ===")
        ev = _eval_from_predictions(all_gold, all_pred)
    else:
        ev = evaluate(
            peft_model,
            head,
            val_dl,
            cfg.device,
            progress_every=5,
            progress_label="eval",
        )
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


def _prediction_counts(result: dict) -> dict:
    counts = {label: 0 for label in LABEL_ORDER}
    for pred in result.get("y_pred", []):
        counts[LABEL_ORDER[int(pred)]] += 1
    return counts


def _gold_counts(result: dict) -> dict:
    counts = {label: 0 for label in LABEL_ORDER}
    for gold in result.get("y_gold", []):
        counts[LABEL_ORDER[int(gold)]] += 1
    return counts


def _dominant_gold_label(result: dict) -> str | None:
    counts = _gold_counts(result)
    nonzero = [(label, count) for label, count in counts.items() if count > 0]
    if len(nonzero) == 1:
        return nonzero[0][0]
    return None


def _label_specific_metrics(result: dict) -> dict:
    label = _dominant_gold_label(result)
    pred_counts = _prediction_counts(result)
    metrics = {
        "gold_counts": _gold_counts(result),
        "prediction_counts": pred_counts,
        "dominant_gold_label": label,
    }
    if label is None:
        return metrics
    n_eval = max(1, int(result.get("n_eval", 0)))
    correct = pred_counts.get(label, 0)
    metrics.update(
        {
            "target_label": label,
            "target_accuracy": correct / n_eval,
            "target_recall": result["per_class"].get(label, {}).get("recall", 0.0),
            "target_precision": result["per_class"].get(label, {}).get("precision", 0.0),
            "target_f1": result["per_class"].get(label, {}).get("f1-score", 0.0),
            "miss_rate_to_clean": (
                0.0 if label == "clean" else pred_counts.get("clean", 0) / n_eval
            ),
            "false_alarm_rate": (
                0.0 if label != "clean"
                else (pred_counts.get("direct", 0) + pred_counts.get("indirect", 0)) / n_eval
            ),
        }
    )
    return metrics


def _write_full_model_artifacts(summary: dict, out_dir: Path) -> None:
    full = summary["full"]
    report = {
        "mode": "full_model_absolute",
        "full_checkpoint": summary["full_checkpoint"],
        "val_jsonl": summary["val_jsonl"],
        "n_eval": summary["n_eval"],
        "accuracy": full["accuracy"],
        "macro_f1": full["macro_f1"],
        "weighted_f1": full["weighted_f1"],
        "per_class": full["per_class"],
        "confusion_matrix": full.get("confusion_matrix"),
        "label_specific": _label_specific_metrics(full),
    }
    with open(out_dir / "full_model_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "full_model_report.md", "w", encoding="utf-8") as f:
        f.write(_make_full_model_markdown(report))
    print(f"[eval] wrote {out_dir/'full_model_report.json'}")
    print(f"[eval] wrote {out_dir/'full_model_report.md'}")


def _make_full_model_markdown(report: dict) -> str:
    label_specific = report["label_specific"]
    target = label_specific.get("target_label")
    lines = [
        "# Full Model Absolute Evaluation",
        "",
        f"- val records: **{report['n_eval']}**",
        f"- full checkpoint: `{report['full_checkpoint']}`",
        f"- val jsonl: `{report['val_jsonl']}`",
        "",
        "## Headline Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| accuracy | {report['accuracy']:.4f} |",
        f"| macro F1 | {report['macro_f1']:.4f} |",
        f"| weighted F1 | {report['weighted_f1']:.4f} |",
        "",
        "## Label-Specific Metrics",
        "",
    ]
    if target:
        lines += [
            f"- target label: `{target}`",
            f"- target recall: **{label_specific['target_recall']:.4f}**",
            f"- target precision: **{label_specific['target_precision']:.4f}**",
            f"- target F1: **{label_specific['target_f1']:.4f}**",
        ]
        if target == "clean":
            lines.append(f"- false alarm rate: **{label_specific['false_alarm_rate']:.4f}**")
        else:
            lines.append(f"- miss rate to clean: **{label_specific['miss_rate_to_clean']:.4f}**")
    else:
        lines.append("- mixed-label dataset: use per-class table below.")
    lines += [
        "",
        "## Prediction Distribution",
        "",
        "| label | gold | predicted |",
        "|---|---:|---:|",
    ]
    gold_counts = label_specific["gold_counts"]
    pred_counts = label_specific["prediction_counts"]
    for label in LABEL_ORDER:
        lines.append(f"| {label} | {gold_counts.get(label, 0)} | {pred_counts.get(label, 0)} |")
    lines += [
        "",
        "## Per-Class Metrics",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABEL_ORDER:
        row = report["per_class"].get(label, {})
        lines.append(
            f"| {label} | {row.get('precision', 0):.4f} | "
            f"{row.get('recall', 0):.4f} | {row.get('f1-score', 0):.4f} | "
            f"{int(row.get('support', 0))} |"
        )
    return "\n".join(lines) + "\n"


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


def _run_one_model(peft_model, head, val_dl, val_ds, cfg, progress_label: str = "eval") -> dict:
    """Run a single eval pass and return a normalized summary dict."""
    ev = evaluate(
        peft_model,
        head,
        val_dl,
        cfg.device,
        progress_every=5,
        progress_label=progress_label,
    )
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


def run_comparison(
    cfg: TrainConfig,
    checkpoint: Path,
    val_path: Path,
    out_dir: Path,
    max_records: Optional[int],
    stratified_max_records: Optional[int] = None,
    sample_seed: int = 42,
) -> int:
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
        probe.processor,
        val_path,
        cfg.device,
        cfg.batch_size,
        max_records,
        stratified_max_records=stratified_max_records,
        sample_seed=sample_seed,
    )
    print(f"[eval] val size: {len(val_ds)}")

    # 1. Baseline (random init)
    print(f"\n[eval] === Running BASELINE (untrained) ===")
    base_peft, base_head, base_lora, base_head_n = _build_random_model(cfg)
    print(f"[eval] baseline: LoRA params: {base_lora:,}  Head params: {base_head_n:,}  (random init)")
    baseline_result = _run_one_model(
        base_peft, base_head, val_dl, val_ds, cfg, progress_label="eval-baseline"
    )
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
    modified_result = _run_one_model(
        mod_peft, mod_head, val_dl, val_ds, cfg, progress_label="eval-modified"
    )
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


def run_smoke_vs_full(
    cfg: TrainConfig,
    smoke_ckpt: Path, full_ckpt: Path,
    val_path: Path, out_dir: Path, max_records: Optional[int],
    stratified_max_records: Optional[int] = None,
    sample_seed: int = 42,
    chunk_size: int = 0,
    chunk_output_dir: Optional[Path] = None,
) -> int:
    """T2.18: compare smoke (5-record) vs full (200-record) trained models.

    Both checkpoints are loaded and evaluated on the same val set. The
    output ``comparison_full_vs_smoke.{json,md}`` shows:
      - Whether the full model **strictly improves** on every metric
        (the success criterion of T2.18).
      - Per-class recall / F1 deltas.
      - The "lift" of real training vs smoke training.
    """
    print(f"[eval] mode:       COMPARE-SMOKE-VS-FULL (T2.18)")
    print(f"[eval] smoke:      {smoke_ckpt}")
    print(f"[eval] full:       {full_ckpt}")
    print(f"[eval] val:        {val_path}")
    print(f"[eval] out_dir:    {out_dir}")

    probe = _build_probe_processor(cfg)
    val_dl, val_ds = _build_dataloader_with_processor(
        probe.processor,
        val_path,
        cfg.device,
        cfg.batch_size,
        max_records,
        stratified_max_records=stratified_max_records,
        sample_seed=sample_seed,
    )
    print(f"[eval] val size: {len(val_ds)}")

    if chunk_size > 0:
        chunk_output_dir = chunk_output_dir or (out_dir / "chunks")
        chunk_output_dir.mkdir(parents=True, exist_ok=True)
        smoke_all_gold: list[int] = []
        smoke_all_pred: list[int] = []
        full_all_gold: list[int] = []
        full_all_pred: list[int] = []
        for chunk_idx, chunk_ds in _iter_chunk_datasets(val_ds, chunk_size):
            chunk_dir = chunk_output_dir / f"smoke_vs_full_group_{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"\n[eval] === chunk {chunk_idx} smoke-vs-full compare "
                f"({len(chunk_ds)} records) -> {chunk_dir} ==="
            )
            chunk_dl = DataLoader(
                chunk_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=0,
            )

            print(f"[eval] chunk {chunk_idx}: building SMOKE model")
            smoke_peft, smoke_head, smoke_lora, smoke_head_n = _build_loaded_model(cfg, smoke_ckpt)
            print(f"[eval] chunk {chunk_idx}: smoke LoRA={smoke_lora:,} Head={smoke_head_n:,}")
            smoke_result = _run_one_model(
                smoke_peft,
                smoke_head,
                chunk_dl,
                chunk_ds,
                cfg,
                progress_label=f"eval-smoke-g{chunk_idx:03d}",
            )
            del smoke_peft, smoke_head
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

            print(f"[eval] chunk {chunk_idx}: building FULL model")
            full_peft, full_head, full_lora, full_head_n = _build_loaded_model(cfg, full_ckpt)
            print(f"[eval] chunk {chunk_idx}: full LoRA={full_lora:,} Head={full_head_n:,}")
            full_result = _run_one_model(
                full_peft,
                full_head,
                chunk_dl,
                chunk_ds,
                cfg,
                progress_label=f"eval-full-g{chunk_idx:03d}",
            )
            del full_peft, full_head
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

            delta = _compute_delta(smoke_result, full_result)
            all_improved = (
                delta["macro_f1_delta"] > 0
                and delta["accuracy_delta"] > 0
                and all(v > 0 for v in delta["per_class_recall_delta"].values())
            )
            chunk_summary = {
                "mode": "compare_smoke_vs_full_chunk",
                "chunk_index": chunk_idx,
                "smoke_checkpoint": str(smoke_ckpt),
                "full_checkpoint": str(full_ckpt),
                "val_jsonl": str(val_path),
                "n_eval": len(chunk_ds),
                "smoke": {
                    "accuracy": smoke_result["accuracy"],
                    "macro_f1": smoke_result["macro_f1"],
                    "weighted_f1": smoke_result["weighted_f1"],
                    "per_class": smoke_result["per_class"],
                    "confusion_matrix": smoke_result["confusion_matrix"],
                    "y_pred": smoke_result["y_pred"],
                    "y_gold": smoke_result["y_gold"],
                },
                "full": {
                    "accuracy": full_result["accuracy"],
                    "macro_f1": full_result["macro_f1"],
                    "weighted_f1": full_result["weighted_f1"],
                    "per_class": full_result["per_class"],
                    "confusion_matrix": full_result["confusion_matrix"],
                    "y_pred": full_result["y_pred"],
                    "y_gold": full_result["y_gold"],
                },
                "delta": delta,
                "verdict": {
                    "all_metrics_improved": all_improved,
                    "macro_f1_improved": delta["macro_f1_delta"] > 0,
                    "all_per_class_recall_improved":
                        all(v > 0 for v in delta["per_class_recall_delta"].values()),
                },
            }
            with open(chunk_dir / "comparison_full_vs_smoke.json", "w", encoding="utf-8") as f:
                json.dump(chunk_summary, f, ensure_ascii=False, indent=2)
            with open(chunk_dir / "comparison_full_vs_smoke.md", "w", encoding="utf-8") as f:
                f.write(_make_smoke_vs_full_markdown(chunk_summary))
            print(
                f"[eval] chunk {chunk_idx}: smoke macro F1={smoke_result['macro_f1']:.4f}, "
                f"full macro F1={full_result['macro_f1']:.4f}, "
                f"delta={delta['macro_f1_delta']:+.4f}"
            )
            print(f"[eval] chunk {chunk_idx}: wrote {chunk_dir/'comparison_full_vs_smoke.json'}")

            smoke_all_gold.extend(smoke_result["y_gold"])
            smoke_all_pred.extend(smoke_result["y_pred"])
            full_all_gold.extend(full_result["y_gold"])
            full_all_pred.extend(full_result["y_pred"])

        smoke_ev = _eval_from_predictions(smoke_all_gold, smoke_all_pred)
        full_ev = _eval_from_predictions(full_all_gold, full_all_pred)
        smoke_report = smoke_ev["report"]
        full_report = full_ev["report"]
        smoke_result = {
            "n_eval": len(val_ds),
            "accuracy": smoke_report["accuracy"],
            "macro_f1": smoke_report["macro avg"]["f1-score"],
            "weighted_f1": smoke_report["weighted avg"]["f1-score"],
            "per_class": {k: smoke_report.get(k, {}) for k in LABEL_ORDER},
            "confusion_matrix": smoke_ev["confusion_matrix"],
            "y_pred": smoke_ev["y_pred"],
            "y_gold": smoke_ev["y_gold"],
        }
        full_result = {
            "n_eval": len(val_ds),
            "accuracy": full_report["accuracy"],
            "macro_f1": full_report["macro avg"]["f1-score"],
            "weighted_f1": full_report["weighted avg"]["f1-score"],
            "per_class": {k: full_report.get(k, {}) for k in LABEL_ORDER},
            "confusion_matrix": full_ev["confusion_matrix"],
            "y_pred": full_ev["y_pred"],
            "y_gold": full_ev["y_gold"],
        }
        print("[eval] === chunked smoke-vs-full aggregate ===")
    else:
        smoke_result = None
        full_result = None

    if chunk_size > 0:
        # Reuse the common summary writer below with aggregate chunked predictions.
        pass
    else:
        # 1. Smoke model
        print(f"\n[eval] === Running SMOKE model (5 records) ===")
        smoke_peft, smoke_head, smoke_lora, smoke_head_n = _build_loaded_model(cfg, smoke_ckpt)
        print(f"[eval] smoke: LoRA={smoke_lora:,} Head={smoke_head_n:,}")
        smoke_result = _run_one_model(
            smoke_peft, smoke_head, val_dl, val_ds, cfg, progress_label="eval-smoke"
        )
        print(f"[eval] smoke: acc={smoke_result['accuracy']:.4f}  "
              f"macro F1={smoke_result['macro_f1']:.4f}  "
              f"weighted F1={smoke_result['weighted_f1']:.4f}")
        del smoke_peft, smoke_head
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

        # 2. Full model
        print(f"\n[eval] === Running FULL model (200 records) ===")
        full_peft, full_head, full_lora, full_head_n = _build_loaded_model(cfg, full_ckpt)
        print(f"[eval] full: LoRA={full_lora:,} Head={full_head_n:,}")
        full_result = _run_one_model(
            full_peft, full_head, val_dl, val_ds, cfg, progress_label="eval-full"
        )
        print(f"[eval] full: acc={full_result['accuracy']:.4f}  "
              f"macro F1={full_result['macro_f1']:.4f}  "
              f"weighted F1={full_result['weighted_f1']:.4f}")
        del full_peft, full_head
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    # 3. Delta (full - smoke)
    delta = _compute_delta(smoke_result, full_result)
    print(f"\n[eval] === Delta (full - smoke) ===")
    print(f"[eval] F1 delta:     {delta['macro_f1_delta']:+.4f}  "
          f"(smoke {smoke_result['macro_f1']:.4f} → "
          f"full {full_result['macro_f1']:.4f})")
    print(f"[eval] Acc delta:    {delta['accuracy_delta']:+.4f}")
    for label in LABEL_ORDER:
        print(f"[eval] recall delta  [{label:>9s}]: "
              f"{delta['per_class_recall_delta'][label]:+.4f}")

    # 4. Verdict: real training must strictly beat smoke on every
    #    headline metric.
    all_improved = (
        delta["macro_f1_delta"] > 0
        and delta["accuracy_delta"] > 0
        and all(v > 0 for v in delta["per_class_recall_delta"].values())
    )
    print(f"\n[eval] PASS = full strictly better than smoke on all metrics: "
          f"{'YES' if all_improved else 'NO'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "compare_smoke_vs_full",
        "smoke_checkpoint": str(smoke_ckpt),
        "full_checkpoint":  str(full_ckpt),
        "val_jsonl":        str(val_path),
        "n_eval":           len(val_ds),
        "smoke": {
            "accuracy":   smoke_result["accuracy"],
            "macro_f1":   smoke_result["macro_f1"],
            "weighted_f1": smoke_result["weighted_f1"],
            "per_class":  smoke_result["per_class"],
            "confusion_matrix": smoke_result["confusion_matrix"],
            "y_pred": smoke_result["y_pred"],
            "y_gold": smoke_result["y_gold"],
        },
        "full": {
            "accuracy":   full_result["accuracy"],
            "macro_f1":   full_result["macro_f1"],
            "weighted_f1": full_result["weighted_f1"],
            "per_class":  full_result["per_class"],
            "confusion_matrix": full_result["confusion_matrix"],
            "y_pred": full_result["y_pred"],
            "y_gold": full_result["y_gold"],
        },
        "delta": delta,
        "verdict": {
            "all_metrics_improved": all_improved,
            "macro_f1_improved":    delta["macro_f1_delta"] > 0,
            "all_per_class_recall_improved":
                all(v > 0 for v in delta["per_class_recall_delta"].values()),
        },
    }
    with open(out_dir / "comparison_full_vs_smoke.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = _make_smoke_vs_full_markdown(summary)
    with open(out_dir / "comparison_full_vs_smoke.md", "w", encoding="utf-8") as f:
        f.write(md)
    _write_full_model_artifacts(summary, out_dir)

    print(f"\n[eval] wrote {out_dir/'comparison_full_vs_smoke.json'}")
    print(f"[eval] wrote {out_dir/'comparison_full_vs_smoke.md'}")
    return 0


def _make_smoke_vs_full_markdown(s: dict) -> str:
    """Human-readable smoke-vs-full comparison report."""
    smoke = s["smoke"]
    full = s["full"]
    delta = s["delta"]
    v = s["verdict"]
    lines = [
        "# Phase 2.2: Baseline vs Full Model Comparison",
        "",
        f"- val records: **{s['n_eval']}**",
        f"- baseline checkpoint: `{s['smoke_checkpoint']}`",
        f"- full checkpoint:  `{s['full_checkpoint']}`",
        "",
        "## Headline metrics",
        "",
        "| metric | baseline/smoke | full/new | delta (full - baseline) |",
        "|---|---|---|---|",
        f"| accuracy    | {smoke['accuracy']:.4f} | {full['accuracy']:.4f} | "
        f"{delta['accuracy_delta']:+.4f} |",
        f"| macro F1    | {smoke['macro_f1']:.4f} | {full['macro_f1']:.4f} | "
        f"{delta['macro_f1_delta']:+.4f} |",
        f"| weighted F1 | {smoke['weighted_f1']:.4f} | {full['weighted_f1']:.4f} | "
        f"{delta['weighted_f1_delta']:+.4f} |",
        "",
        "## Per-class recall",
        "",
        "| class | baseline/smoke | full/new | delta |",
        "|---|---|---|---|",
    ]
    for label in LABEL_ORDER:
        s_r = smoke["per_class"].get(label, {}).get("recall", 0.0)
        f_r = full["per_class"].get(label, {}).get("recall", 0.0)
        d = delta["per_class_recall_delta"][label]
        lines.append(f"| {label} | {s_r:.4f} | {f_r:.4f} | {d:+.4f} |")
    lines += [
        "",
        "## Per-class F1",
        "",
        "| class | baseline/smoke | full/new | delta |",
        "|---|---|---|---|",
    ]
    for label in LABEL_ORDER:
        s_f = smoke["per_class"].get(label, {}).get("f1-score", 0.0)
        f_f = full["per_class"].get(label, {}).get("f1-score", 0.0)
        d = delta["per_class_f1_delta"][label]
        lines.append(f"| {label} | {s_f:.4f} | {f_f:.4f} | {d:+.4f} |")
    lines += [
        "",
        "## Verdict (T2.18 验收标准)",
        "",
        f"- macro F1 提升: {'✅' if v['macro_f1_improved'] else '❌'} "
        f"({delta['macro_f1_delta']:+.4f})",
        f"- 全部 per-class recall 提升: "
        f"{'✅' if v['all_per_class_recall_improved'] else '❌'}",
        f"- **总评: {'PASS' if v['all_metrics_improved'] else 'FAIL'}** — "
        f"全指标严格优于 smoke = {v['all_metrics_improved']}",
        "",
        "## Interpretation",
        "",
    ]
    if v["all_metrics_improved"]:
        lines.append("- 真实训练 (200 样本 × 3 epoch) 在所有指标上都严格优于 smoke (5 样本)")
        lines.append("- 这证明 200 样本训练**有真东西**，不只是回归到「预测多数类」")
    else:
        lines.append("- 真实训练在某些指标上未严格优于 smoke，可能原因:")
        lines.append("  - 200 样本仍不够（任务原计划 2000，受 Mac 硬件限制降规模）")
        lines.append("  - 训练不充分（仅 1-3 epoch）")
        lines.append("  - MPS+LoRA 的 NaN 防护跳过了部分 step")
        lines.append("  - 数据不均衡，class_weighted 后 indirect 类权重过大导致训练波动")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Early-exit comparison (T3.7)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _run_with_early_exit_per_sample(
    peft_model, head, val_dl, device: str,
    cfg: EarlyExitConfig, simulate_c5_c6_ms: float,
) -> tuple:
    """Run val set, applying C4 per sample, and return per-sample decisions
    plus aggregate stats.

    The "total latency" is the **VLM + head + (conditional C5/C6) time**.
    We measure the actual VLM + head time, and add ``simulate_c5_c6_ms``
    to every sample that did NOT trigger C4. This is a deliberate
    simulation — the real Phase 4/5 C5/C6 modules are not yet built,
    but the saving from skipping them is the headline metric for C4.

    Returns:
        (samples_list, stats_no_exit, stats_with_exit, f1_no_exit, f1_with_exit)
    """
    import time
    import torch.nn.functional as F

    peft_model.eval(); head.eval()
    samples = []
    stats_no_exit = EarlyExitStats()   # would have happened (just for ref)
    stats_with_exit = EarlyExitStats() # actually happened

    y_pred_no_exit = []
    y_pred_with_exit = []
    y_gold = []

    for batch in val_dl:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        t0 = time.perf_counter()
        outputs = peft_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            pixel_attention_mask=batch.get("pixel_attention_mask"),
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]
        last_idx = batch["attention_mask"].sum(dim=1) - 1
        b = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[b, last_idx]
        logits = head(pooled)
        probs_t = F.softmax(logits, dim=-1)
        latency_vlm_head_ms = (time.perf_counter() - t0) * 1000.0

        for i in range(probs_t.size(0)):
            probs_i = probs_t[i]
            p_clean = float(probs_i[LABEL2IDX["clean"]].item())
            label_no_exit = LABEL_ORDER[int(probs_i.argmax(dim=-1).item())]
            early = should_early_exit(probs_i, cfg)
            label_with_exit = early if early is not None else label_no_exit

            gold = LABEL_ORDER[int(batch["label"][i].item())]

            # No-exit path: full VLM + head + C5/C6 simulated
            total_no_exit_ms = latency_vlm_head_ms + simulate_c5_c6_ms
            # With-exit path: VLM + head + (C5/C6 if not exited)
            if early is not None:
                total_with_exit_ms = latency_vlm_head_ms
            else:
                total_with_exit_ms = latency_vlm_head_ms + simulate_c5_c6_ms

            # Update stats
            stats_no_exit.n_total += 1
            stats_no_exit.latency_full_ms += total_no_exit_ms
            stats_no_exit.per_class_total[gold] += 1

            stats_with_exit.n_total += 1
            if early is not None:
                stats_with_exit.n_exited += 1
                stats_with_exit.latency_exit_ms += total_with_exit_ms
                if gold == "clean":
                    stats_with_exit.n_clean_exited += 1
                else:
                    stats_with_exit.n_clean_wrong_exit += 1
            else:
                stats_with_exit.latency_full_ms += total_with_exit_ms
            stats_with_exit.per_class_total[gold] += 1
            if early is not None:
                stats_with_exit.per_class_exits[gold] += 1

            samples.append({
                "id":       batch.get("id", ["?"] * probs_t.size(0))[i] if isinstance(batch.get("id"), list) else str(i),
                "gold":     gold,
                "p_clean":  p_clean,
                "p_direct": float(probs_i[LABEL2IDX["direct"]].item()),
                "p_indirect": float(probs_i[LABEL2IDX["indirect"]].item()),
                "label_no_exit":  label_no_exit,
                "label_with_exit": label_with_exit,
                "exited":   early is not None,
                "label_changed": label_no_exit != label_with_exit,
                "latency_vlm_head_ms": latency_vlm_head_ms,
                "latency_no_exit_ms":  total_no_exit_ms,
                "latency_with_exit_ms": total_with_exit_ms,
            })
            y_pred_no_exit.append(LABEL2IDX[label_no_exit])
            y_pred_with_exit.append(LABEL2IDX[label_with_exit])
            y_gold.append(LABEL2IDX[gold])

    from sklearn.metrics import classification_report, confusion_matrix
    rep_no_exit = classification_report(
        y_gold, y_pred_no_exit, labels=list(range(NUM_CLASSES)),
        target_names=LABEL_ORDER, output_dict=True, zero_division=0,
    )
    rep_with_exit = classification_report(
        y_gold, y_pred_with_exit, labels=list(range(NUM_CLASSES)),
        target_names=LABEL_ORDER, output_dict=True, zero_division=0,
    )
    cm_no_exit = confusion_matrix(y_gold, y_pred_no_exit, labels=list(range(NUM_CLASSES))).tolist()
    cm_with_exit = confusion_matrix(y_gold, y_pred_with_exit, labels=list(range(NUM_CLASSES))).tolist()

    return (samples, stats_no_exit, stats_with_exit,
            {"report": rep_no_exit, "confusion_matrix": cm_no_exit},
            {"report": rep_with_exit, "confusion_matrix": cm_with_exit})


def run_early_exit_compare(
    cfg: TrainConfig, checkpoint: Path, val_path: Path, out_dir: Path,
    max_records: Optional[int], clean_threshold: float = 0.95,
    simulate_c5_c6_ms: float = 200.0,
) -> int:
    """T3.7: Run same model twice (with/without C4) and compare.

    Goal: prove C4 saves latency while not hurting F1.
    """
    print(f"[eval] mode:       EARLY-EXIT COMPARE (C4 on/off)")
    print(f"[eval] threshold:  P(clean) > {clean_threshold}  → exit as 'clean'")
    print(f"[eval] simulated C5+C6 cost: {simulate_c5_c6_ms} ms per non-exit sample")
    print(f"[eval] checkpoint: {checkpoint}")
    print(f"[eval] val:        {val_path}")
    print(f"[eval] out_dir:    {out_dir}")

    # Build the model once
    peft_model, hidden_size, n_lora = _build_adapter_and_lora(cfg)
    head = ClassificationHead(in_dim=hidden_size,
                               num_classes=NUM_CLASSES).to(cfg.device)
    n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
    state = load_checkpoint(checkpoint, head)
    if any(k.startswith("lora.") for k in state.keys()):
        _apply_lora_state(peft_model, state)
    peft_model.eval(); head.eval()
    print(f"[eval] LoRA params: {n_lora:,}  Head params: {n_head:,}  (loaded)")

    # Build dataloader once
    probe = _build_probe_processor(cfg)
    val_dl, val_ds = _build_dataloader_with_processor(
        probe.processor, val_path, cfg.device, cfg.batch_size, max_records
    )
    print(f"[eval] val size: {len(val_ds)}")

    # Run with C4 enabled (and use the same output for "C4 disabled" since
    # C4 OFF is just "always use argmax" which is what VLM+head produces
    # before the C4 check; both paths share the VLM forward).
    cfg_e = EarlyExitConfig(enabled=True, clean_threshold=clean_threshold)
    samples, _, stats_with, f1_no, f1_with = _run_with_early_exit_per_sample(
        peft_model, head, val_dl, cfg.device, cfg_e, simulate_c5_c6_ms,
    )
    n = len(samples)
    print(f"[eval] C4 exit rate: {stats_with.n_exited}/{n} = "
          f"{stats_with.n_exited / max(1, n):.1%}")
    print(f"[eval] C4 wrong-exit: {stats_with.n_clean_wrong_exit} "
          f"(direct/indirect wrongly labeled as clean)")

    # Aggregate the comparison
    avg_total_no  = sum(s["latency_no_exit_ms"]  for s in samples) / max(1, n)
    avg_total_with = sum(s["latency_with_exit_ms"] for s in samples) / max(1, n)
    speedup = avg_total_no / max(1e-9, avg_total_with)
    saved_ms_per_sample = avg_total_no - avg_total_with
    saved_pct = (saved_ms_per_sample / max(1e-9, avg_total_no)) * 100.0

    # F1 deltas (early-exit might cause some wrong decisions)
    f1_no_macro = f1_no["report"]["macro avg"]["f1-score"]
    f1_with_macro = f1_with["report"]["macro avg"]["f1-score"]
    f1_delta = f1_with_macro - f1_no_macro
    acc_no = f1_no["report"]["accuracy"]
    acc_with = f1_with["report"]["accuracy"]

    # FPR (clean) — should be ~0 for high threshold
    fpr_clean_no = 1.0 - f1_no["report"].get("clean", {}).get("recall", 0.0)
    fpr_clean_with = 1.0 - f1_with["report"].get("clean", {}).get("recall", 0.0)

    summary = {
        "checkpoint": str(checkpoint),
        "val_jsonl":  str(val_path),
        "n_eval":     n,
        "clean_threshold": clean_threshold,
        "simulate_c5_c6_ms": simulate_c5_c6_ms,
        "exit_rate":  stats_with.n_exited / max(1, n),
        "n_exited":   stats_with.n_exited,
        "n_clean_wrong_exit": stats_with.n_clean_wrong_exit,
        # Per-class
        "per_class_exits":   dict(stats_with.per_class_exits),
        "per_class_total":   dict(stats_with.per_class_total),
        "per_class_exit_rate": {
            lbl: stats_with.per_class_exits[lbl] / max(1, stats_with.per_class_total[lbl])
            for lbl in LABEL_ORDER
        },
        # Latency
        "avg_latency_no_exit_ms":  avg_total_no,
        "avg_latency_with_exit_ms": avg_total_with,
        "saved_ms_per_sample":    saved_ms_per_sample,
        "saved_pct":              saved_pct,
        "speedup":                speedup,
        # F1 / accuracy
        "f1_no_exit":   f1_no_macro,
        "f1_with_exit": f1_with_macro,
        "f1_delta":     f1_delta,
        "acc_no_exit":  acc_no,
        "acc_with_exit": acc_with,
        # FPR
        "fpr_clean_no_exit":  fpr_clean_no,
        "fpr_clean_with_exit": fpr_clean_with,
        # Reports for reference
        "report_no_exit":  f1_no["report"],
        "report_with_exit": f1_with["report"],
        "confusion_matrix_no_exit":  f1_no["confusion_matrix"],
        "confusion_matrix_with_exit": f1_with["confusion_matrix"],
    }

    # Write artefacts
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "early_exit_compare.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "early_exit_per_sample.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    md = _make_early_exit_markdown(summary)
    with open(out_dir / "early_exit_compare.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n[eval] === Early-Exit Summary ===")
    print(f"[eval] Exit rate:    {summary['exit_rate']:.1%}  ({summary['n_exited']}/{n})")
    print(f"[eval] Wrong exit:   {summary['n_clean_wrong_exit']} (non-clean → clean)")
    print(f"[eval] Latency:      {avg_total_no:.1f} ms → {avg_total_with:.1f} ms "
          f"({saved_pct:.1f}% saved, {speedup:.2f}x speedup)")
    print(f"[eval] F1 delta:     {f1_delta:+.4f}  ({f1_no_macro:.4f} → {f1_with_macro:.4f})")
    print(f"[eval] Acc delta:    {acc_with - acc_no:+.4f}")
    print(f"[eval] wrote {out_dir/'early_exit_compare.json'}")
    print(f"[eval] wrote {out_dir/'early_exit_compare.md'}")
    print(f"[eval] wrote {out_dir/'early_exit_per_sample.jsonl'}")
    return 0


def _make_early_exit_markdown(s: dict) -> str:
    """Human-readable early-exit comparison report."""
    lines = [
        "# C4 Early-Exit Comparison Report",
        "",
        f"- eval records: **{s['n_eval']}**",
        f"- clean threshold: **P(clean) > {s['clean_threshold']}**",
        f"- simulated C5+C6 cost: **{s['simulate_c5_c6_ms']} ms** per non-exit sample",
        "",
        "## Exit statistics",
        "",
        f"- Exit rate: **{s['exit_rate']:.1%}**  ({s['n_exited']}/{s['n_eval']})",
        f"- Wrong exits (non-clean → clean): **{s['n_clean_wrong_exit']}**",
        "",
        "### Per-class exit rate",
        "",
        "| class | exits | total | exit rate |",
        "|---|---|---|---|",
    ]
    for lbl in LABEL_ORDER:
        ex = s["per_class_exits"][lbl]
        to = s["per_class_total"][lbl]
        rate = s["per_class_exit_rate"][lbl]
        lines.append(f"| {lbl} | {ex} | {to} | {rate:.1%} |")
    lines += [
        "",
        "## Latency",
        "",
        f"- Average per-sample latency (no C4):  **{s['avg_latency_no_exit_ms']:.1f} ms**",
        f"- Average per-sample latency (with C4): **{s['avg_latency_with_exit_ms']:.1f} ms**",
        f"- Saved per sample: **{s['saved_ms_per_sample']:.1f} ms** ({s['saved_pct']:.1f}%)",
        f"- Speedup: **{s['speedup']:.2f}x**",
        "",
        "## F1 / accuracy",
        "",
        f"- Macro F1 (no C4):  **{s['f1_no_exit']:.4f}**",
        f"- Macro F1 (with C4): **{s['f1_with_exit']:.4f}**  (delta {s['f1_delta']:+.4f})",
        f"- Accuracy (no C4):  **{s['acc_no_exit']:.4f}**",
        f"- Accuracy (with C4): **{s['acc_with_exit']:.4f}**",
        f"- FPR clean (no C4):  **{s['fpr_clean_no_exit']:.4f}**",
        f"- FPR clean (with C4): **{s['fpr_clean_with_exit']:.4f}**",
        "",
        "## Interpretation",
        "",
    ]
    # Auto-interpret
    if s["f1_delta"] >= -0.02:
        lines.append(f"- C4 在 F1 上几乎无影响 (delta {s['f1_delta']:+.4f})：早退判定没有引入显著误判")
    else:
        lines.append(f"- C4 引入 F1 下降 {s['f1_delta']:.4f}：需要调高阈值或检查 clean 误判")
    if s["n_clean_wrong_exit"] == 0:
        lines.append("- 没有出现「非 clean → clean」误判：C4 不会漏报")
    else:
        lines.append(f"- 有 {s['n_clean_wrong_exit']} 个非 clean 样本被误判为 clean：建议调高阈值")
    if s["saved_pct"] >= 30:
        lines.append(f"- 延迟节省 {s['saved_pct']:.1f}%，效果显著：clean 样本快速放行")
    elif s["saved_pct"] >= 10:
        lines.append(f"- 延迟节省 {s['saved_pct']:.1f}%，效果适中")
    else:
        lines.append(f"- 延迟节省仅 {s['saved_pct']:.1f}%：早退命中率偏低")
    lines += [
        "",
        "## Pass / fail",
        "",
        f"- F1 退化 ≤ 0.02: {'PASS' if s['f1_delta'] >= -0.02 else 'FAIL'} "
        f"(actual: {s['f1_delta']:+.4f})",
        f"- 无 clean 漏报: {'PASS' if s['n_clean_wrong_exit'] == 0 else 'FAIL'} "
        f"(actual: {s['n_clean_wrong_exit']})",
        f"- 节省 ≥ 10%: {'PASS' if s['saved_pct'] >= 10 else 'FAIL'} "
        f"(actual: {s['saved_pct']:.1f}%)",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = build_train_config_from_yaml(args.config)
    config_dir = args.config.resolve().parent
    val_path = args.val if args.val else Path(cfg.val_jsonl)
    if not val_path.is_absolute():
        val_path = _resolve_input_path(val_path, config_dir)
    out_dir = args.out if args.out else Path(cfg.out_dir)
    if not out_dir.is_absolute():
        out_dir = _resolve_output_path(out_dir, config_dir)
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = _resolve_input_path(checkpoint, config_dir)
    chunk_output_dir = args.chunk_output_dir
    if chunk_output_dir is not None and not chunk_output_dir.is_absolute():
        chunk_output_dir = _resolve_output_path(chunk_output_dir, config_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        return run_comparison(
            cfg,
            checkpoint,
            val_path,
            out_dir,
            args.max_records,
            stratified_max_records=args.stratified_max_records,
            sample_seed=args.sample_seed,
        )
    if args.compare_smoke_vs_full:
        smoke_ckpt = args.smoke_checkpoint or (config_dir.parent / "artifacts" / "checkpoints" / "lora_baseline.safetensors")
        full_ckpt = args.full_checkpoint or checkpoint
        if not smoke_ckpt.is_absolute():
            smoke_ckpt = _resolve_input_path(smoke_ckpt, config_dir)
        if not full_ckpt.is_absolute():
            full_ckpt = _resolve_input_path(full_ckpt, config_dir)
        return run_smoke_vs_full(
            cfg, smoke_ckpt, full_ckpt,
            val_path, out_dir, args.max_records,
            stratified_max_records=args.stratified_max_records,
            sample_seed=args.sample_seed,
            chunk_size=args.chunk_size,
            chunk_output_dir=chunk_output_dir,
        )
    if args.early_exit:
        return run_early_exit_compare(
            cfg, checkpoint, val_path, out_dir, args.max_records,
            clean_threshold=args.clean_threshold,
            simulate_c5_c6_ms=args.simulate_c5_c6_ms,
        )
    return run_single_model(
        cfg,
        checkpoint,
        val_path,
        out_dir,
        args.max_records,
        stratified_max_records=args.stratified_max_records,
        sample_seed=args.sample_seed,
        chunk_size=args.chunk_size,
        chunk_output_dir=chunk_output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
