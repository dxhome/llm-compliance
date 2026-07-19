"""Compare balanced VLM-only inference with the lightweight C4/C5/C6A pipeline.

The runner is intentionally operational rather than benchmark-perfect:

* load the fine-tuned VLM checkpoint once;
* sample a small, balanced validation slice from class-specific JSONL files;
* run two real inference paths:
  - VLM only: every sample goes through the VLM + head;
  - C5/C6A/C4 + VLM: rules and cross-modal heuristics can decide early,
    otherwise the sample falls back to a real VLM pass and optional C4 clean
    early-exit accounting.
* emit frequent progress logs with elapsed time, average latency, ETA, stage
  counts, and partial F1.

This script does not touch training code and is safe to run alongside long
Phase 2.2 fine-tuning sessions.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mpid.adapters.vlm import VLMAdapter
from mpid.crossmodal import check_crossmodal
from mpid.data.prompt import build_prompt
from mpid.early_exit import EarlyExitConfig, should_early_exit
from mpid.heads.classification import IDX2LABEL, LABEL2IDX, LABEL_ORDER, ClassificationHead
from mpid.rules import scan_text
from mpid.train.trainer import TrainConfig, apply_lora_state, inject_lora, load_checkpoint


@dataclass
class LoadedModel:
    adapter: VLMAdapter
    model: torch.nn.Module
    head: ClassificationHead
    device: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3-5 lightweight compare runner")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--clean-jsonl", type=Path, required=True)
    p.add_argument("--direct-jsonl", type=Path, required=True)
    p.add_argument("--indirect-jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--per-class", type=int, default=20)
    p.add_argument("--class-batch-size", type=int, default=10)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--clean-threshold", type=float, default=0.95)
    p.add_argument("--limit-total", type=int, default=0, help="Optional smoke cap after sampling.")
    return p.parse_args()


def resolve_input_path(value: str | Path, base_dir: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    from_config = (base_dir / p).resolve()
    if from_config.exists():
        return from_config
    return (REPO_ROOT / p).resolve()


def resolve_output_path(value: str | Path, base_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def build_train_config_from_yaml(path: Path) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_dir = path.resolve().parent
    defaults = cfg.get("defaults", {}) or {}
    lora = cfg.get("lora", {}) or {}
    training = cfg.get("training", {}) or {}
    io = cfg.get("io", {}) or {}
    return TrainConfig(
        train_jsonl=str(resolve_input_path(io["train_jsonl"], base_dir)),
        val_jsonl=str(resolve_input_path(io["val_jsonl"], base_dir)),
        out_dir=str(resolve_output_path(io.get("out_dir", "artifacts/eval"), base_dir)),
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


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(path: Path, message: str) -> None:
    text = f"[{now()}] {message}"
    print(text, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    sources = [
        ("clean", args.clean_jsonl),
        ("direct", args.direct_jsonl),
        ("indirect", args.indirect_jsonl),
    ]
    rng = random.Random(args.sample_seed)
    samples: list[dict[str, Any]] = []
    for label, path in sources:
        rows = read_jsonl(path)
        rows = [dict(r, label=label) for r in rows]
        rng.shuffle(rows)
        picked = rows[: args.per_class]
        for i, rec in enumerate(picked):
            rec["_eval_label_set"] = label
            rec["_eval_class_order"] = i
        samples.extend(picked)

    samples.sort(key=lambda r: (LABEL2IDX[r["label"]], r["_eval_class_order"]))
    if args.limit_total and args.limit_total > 0:
        samples = samples[: args.limit_total]
    return samples


def load_model(cfg_path: Path, checkpoint: Path, log_path: Path) -> LoadedModel:
    cfg = build_train_config_from_yaml(cfg_path)
    log_line(log_path, f"load model: config={cfg_path} checkpoint={checkpoint}")
    t0 = time.perf_counter()
    adapter = VLMAdapter(
        backbone_name=cfg.backbone_name,
        dtype=cfg.dtype,
        quantization=cfg.quantization,
        device=cfg.device,
        gradient_checkpointing=False,
    )
    model, n_lora = inject_lora(adapter.model, cfg)
    head = ClassificationHead(in_dim=adapter.hidden_size).to(cfg.device)
    state = load_checkpoint(checkpoint, head)
    n_state = apply_lora_state(model, state)
    model.eval()
    head.eval()
    log_line(
        log_path,
        f"model ready in {time.perf_counter() - t0:.1f}s; "
        f"device={cfg.device}; lora_params={n_lora}; lora_tensors={n_state}",
    )
    return LoadedModel(adapter=adapter, model=model, head=head, device=cfg.device)


def image_arg(record: dict[str, Any]) -> str | None:
    image = record.get("image")
    if not image:
        return None
    p = Path(str(image))
    return str(p) if p.exists() else None


@torch.inference_mode()
def run_vlm(model_bundle: LoadedModel, record: dict[str, Any]) -> dict[str, Any]:
    prompt = build_prompt(str(record.get("text") or ""))
    encoded = model_bundle.adapter.preprocess(prompt, image_arg(record))
    encoded = {
        k: v.to(model_bundle.device) if torch.is_tensor(v) else v
        for k, v in encoded.items()
    }
    outputs = model_bundle.model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        pixel_values=encoded["pixel_values"],
        pixel_attention_mask=encoded.get("pixel_attention_mask"),
        output_hidden_states=True,
    )
    last_hidden = outputs.hidden_states[-1]
    last_idx = encoded["attention_mask"].sum(dim=1) - 1
    b = torch.arange(last_hidden.size(0), device=last_hidden.device)
    pooled = last_hidden[b, last_idx]
    logits = model_bundle.head(pooled)
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
    pred_idx = int(probs.argmax().item())
    return {
        "label": IDX2LABEL[pred_idx],
        "probs": [float(x) for x in probs.tolist()],
        "risk": float(probs.max().item()),
    }


def f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    total = len(gold)
    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / total if total else 0.0
    by_label: dict[str, Any] = {}
    weighted_f1 = 0.0
    for label in LABEL_ORDER:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        support = sum(1 for g in gold if g == label)
        one = f1_from_counts(tp, fp, fn)
        one["support"] = support
        by_label[label] = one
        weighted_f1 += one["f1"] * support
    macro_f1 = sum(by_label[label]["f1"] for label in LABEL_ORDER) / len(LABEL_ORDER)
    weighted_f1 = weighted_f1 / total if total else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "by_label": by_label,
        "confusion": {
            g: {p: sum(1 for gg, pp in zip(gold, pred) if gg == g and pp == p) for p in LABEL_ORDER}
            for g in LABEL_ORDER
        },
    }


def progress_message(
    pipeline: str,
    seen: int,
    total: int,
    started: float,
    gold: list[str],
    pred: list[str],
    stage_counts: Counter[str] | None = None,
) -> str:
    elapsed = time.perf_counter() - started
    avg = elapsed / max(1, seen)
    eta = avg * max(0, total - seen)
    partial = metrics(gold, pred)
    suffix = ""
    if stage_counts is not None:
        suffix = f" stage_counts={dict(stage_counts)}"
    return (
        f"{pipeline} progress {seen}/{total}; elapsed={elapsed:.1f}s; "
        f"avg={avg:.2f}s/sample; ETA={eta:.0f}s; "
        f"acc={partial['accuracy']:.3f}; macro_f1={partial['macro_f1']:.3f};"
        f"{suffix}"
    )


def run_vlm_only(
    model_bundle: LoadedModel,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    log_path: Path,
    per_sample_path: Path,
) -> dict[str, Any]:
    log_line(log_path, "pipeline start: VLM only")
    started = time.perf_counter()
    gold: list[str] = []
    pred: list[str] = []
    latencies: list[float] = []
    for i, rec in enumerate(samples, start=1):
        t0 = time.perf_counter()
        out = run_vlm(model_bundle, rec)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        gold.append(rec["label"])
        pred.append(out["label"])
        append_sample(per_sample_path, "vlm_only", rec, out["label"], "vlm_head", dt, out)
        if i % args.log_every == 0 or i == len(samples):
            log_line(log_path, progress_message("VLM only", i, len(samples), started, gold, pred))
        maybe_class_batch_log("VLM only", i, samples, gold, pred, args, log_path)

    return {
        "metrics": metrics(gold, pred),
        "latency": latency_summary(latencies),
        "stage_counts": {"vlm_head": len(samples)},
    }


def run_full_pipeline(
    model_bundle: LoadedModel,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    log_path: Path,
    per_sample_path: Path,
) -> dict[str, Any]:
    log_line(log_path, "pipeline start: C5 + C6A + VLM/C4 fallback")
    started = time.perf_counter()
    gold: list[str] = []
    pred: list[str] = []
    latencies: list[float] = []
    stage_counts: Counter[str] = Counter()
    early_cfg = EarlyExitConfig(enabled=True, clean_threshold=args.clean_threshold)

    for i, rec in enumerate(samples, start=1):
        t0 = time.perf_counter()
        c5 = scan_text(str(rec.get("text") or ""))
        if c5.blocked:
            label = c5.label
            stage = "c5_rules"
            details = c5.to_dict()
        else:
            c6 = check_crossmodal(rec)
            if c6.suspicious:
                label = c6.label
                stage = "c6_crossmodal"
                details = c6.to_dict()
            else:
                vlm = run_vlm(model_bundle, rec)
                probs_t = torch.tensor(vlm["probs"], dtype=torch.float32)
                early = should_early_exit(probs_t, early_cfg)
                if early is not None:
                    label = "clean"
                    stage = "c4_early_exit"
                    details = {"vlm": vlm, "threshold": args.clean_threshold}
                else:
                    label = vlm["label"]
                    stage = "vlm_head_fallback"
                    details = {"vlm": vlm, "threshold": args.clean_threshold}

        dt = time.perf_counter() - t0
        stage_counts[stage] += 1
        latencies.append(dt)
        gold.append(rec["label"])
        pred.append(label)
        append_sample(per_sample_path, "c5_c6a_c4_vlm", rec, label, stage, dt, details)
        if i % args.log_every == 0 or i == len(samples):
            log_line(log_path, progress_message("C5+C6A+C4+VLM", i, len(samples), started, gold, pred, stage_counts))
        maybe_class_batch_log("C5+C6A+C4+VLM", i, samples, gold, pred, args, log_path)

    return {
        "metrics": metrics(gold, pred),
        "latency": latency_summary(latencies),
        "stage_counts": dict(stage_counts),
    }


def append_sample(
    path: Path,
    pipeline: str,
    record: dict[str, Any],
    pred: str,
    stage: str,
    latency_s: float,
    details: dict[str, Any],
) -> None:
    row = {
        "pipeline": pipeline,
        "id": record.get("id"),
        "gold": record.get("label"),
        "pred": pred,
        "stage": stage,
        "latency_s": latency_s,
        "correct": pred == record.get("label"),
        "source": record.get("source"),
        "metadata": record.get("metadata") or {},
        "details": details,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "total_s": 0.0, "avg_s": 0.0, "min_s": 0.0, "max_s": 0.0}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "total_s": sum(values),
        "avg_s": sum(values) / len(values),
        "min_s": min(values),
        "p50_s": sorted_values[len(values) // 2],
        "max_s": max(values),
    }


def maybe_class_batch_log(
    pipeline: str,
    seen: int,
    samples: list[dict[str, Any]],
    gold: list[str],
    pred: list[str],
    args: argparse.Namespace,
    log_path: Path,
) -> None:
    if args.class_batch_size <= 0:
        return
    current = samples[seen - 1]["label"]
    class_seen = sum(1 for s in samples[:seen] if s["label"] == current)
    if class_seen % args.class_batch_size != 0:
        return
    idxs = [i for i, s in enumerate(samples[:seen]) if s["label"] == current]
    recent = idxs[-args.class_batch_size :]
    batch_gold = [gold[i] for i in recent]
    batch_pred = [pred[i] for i in recent]
    batch_metrics = metrics(batch_gold, batch_pred)
    log_line(
        log_path,
        f"{pipeline} class-batch label={current} completed={class_seen}; "
        f"batch_size={len(recent)}; acc={batch_metrics['accuracy']:.3f}; "
        f"macro_f1={batch_metrics['macro_f1']:.3f}",
    )


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    artifacts = out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    with open(artifacts / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# Phase 3-5 Compare Summary",
        "",
        f"- Run dir: `{out_dir}`",
        f"- Samples: {summary['sample_counts']}",
        f"- Clean threshold: {summary['clean_threshold']}",
        "",
        "| Pipeline | Accuracy | Macro F1 | Weighted F1 | Total Time(s) | Avg(s/sample) | Stage Counts |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, res in summary["pipelines"].items():
        lat = res["latency"]
        m = res["metrics"]
        lines.append(
            f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
            f"{m['weighted_f1']:.3f} | {lat['total_s']:.1f} | {lat['avg_s']:.2f} | "
            f"`{res['stage_counts']}` |"
        )
    lines.extend(["", "## Per-Class F1", ""])
    lines.append("| Pipeline | Label | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name, res in summary["pipelines"].items():
        for label in LABEL_ORDER:
            row = res["metrics"]["by_label"][label]
            lines.append(
                f"| {name} | {label} | {row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['f1']:.3f} | {row['support']} |"
            )
    with open(artifacts / "compare_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "data").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    log_path = args.out_dir / "logs" / "compare.log"
    execution_log = args.out_dir / "execution_log.md"
    per_sample_path = args.out_dir / "artifacts" / "per_sample.jsonl"
    for p in (log_path, execution_log, per_sample_path):
        if p.exists():
            p.unlink()

    log_line(log_path, "phase3-5 compare run started")
    with open(execution_log, "w", encoding="utf-8") as f:
        f.write("# Phase 3-5 Balanced-600 Compare Execution Log\n\n")
        f.write(f"- Started: {now()}\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Checkpoint: `{args.checkpoint}`\n")
        f.write("- Pipelines: `VLM only`, `C5 + C6A + VLM/C4 fallback`\n")
        f.write("- Note: C5/C6A are run before VLM so their real early decisions can reduce VLM calls; C4 is accounted on VLM fallback probabilities.\n\n")

    samples = sample_records(args)
    sample_path = args.out_dir / "data" / "sampled_eval.jsonl"
    with open(sample_path, "w", encoding="utf-8") as f:
        for rec in samples:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sample_counts = dict(Counter(r["label"] for r in samples))
    missing_images = sum(1 for r in samples if r.get("image") and not Path(str(r["image"])).exists())
    log_line(log_path, f"samples ready: total={len(samples)} counts={sample_counts} missing_images={missing_images}")

    model_bundle = load_model(args.config, args.checkpoint, log_path)
    vlm_res = run_vlm_only(model_bundle, samples, args, log_path, per_sample_path)
    full_res = run_full_pipeline(model_bundle, samples, args, log_path, per_sample_path)

    summary = {
        "run_dir": str(args.out_dir),
        "sample_path": str(sample_path),
        "sample_counts": sample_counts,
        "clean_threshold": args.clean_threshold,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "pipelines": {
            "VLM only": vlm_res,
            "C5+C6A+C4+VLM": full_res,
        },
    }
    write_summary(args.out_dir, summary)
    with open(execution_log, "a", encoding="utf-8") as f:
        f.write(f"- Finished: {now()}\n")
        f.write(f"- Summary: `{args.out_dir / 'artifacts' / 'compare_summary.md'}`\n")
        f.write(f"- Per-sample: `{per_sample_path}`\n")
    log_line(log_path, f"DONE summary={args.out_dir / 'artifacts' / 'compare_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
