"""Compare LoRA-only generation pipeline against optimized C4-C6 pipeline.

This runner evaluates the two current demo/offline inference paths:

* LoRA only: MPID head -> block/allow -> LoRA generation.
* Optimized: C5 -> C6 -> MPID head -> C4 -> block/allow -> LoRA generation.

Blocked samples do not generate text. Allowed samples invoke the same LoRA
backbone generation path used by the demo.
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
from mpid.data.prompt import build_prompt
from mpid.heads.classification import IDX2LABEL, LABEL2IDX, LABEL_ORDER, ClassificationHead
from mpid.infer import PipelineResult, run_lora_only_pipeline, run_optimized_pipeline
from mpid.train.trainer import TrainConfig, apply_lora_state, inject_lora, load_checkpoint


@dataclass
class LoadedModel:
    adapter: VLMAdapter
    model: torch.nn.Module
    head: ClassificationHead
    device: str
    max_new_tokens: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare LoRA generation pipelines")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--clean-jsonl", type=Path, required=True)
    p.add_argument("--direct-jsonl", type=Path, required=True)
    p.add_argument("--indirect-jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--class-batch-size", type=int, default=50)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--clean-threshold", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--limit-total", type=int, default=0)
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
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.sample_seed)
    samples: list[dict[str, Any]] = []
    for label, path in [
        ("clean", args.clean_jsonl),
        ("direct", args.direct_jsonl),
        ("indirect", args.indirect_jsonl),
    ]:
        rows = [dict(r, label=label) for r in read_jsonl(path)]
        rng.shuffle(rows)
        for i, rec in enumerate(rows[: args.per_class]):
            rec["_eval_label_set"] = label
            rec["_eval_class_order"] = i
            samples.append(rec)
    samples.sort(key=lambda r: (LABEL2IDX[r["label"]], r["_eval_class_order"]))
    if args.limit_total > 0:
        samples = samples[: args.limit_total]
    return samples


def image_arg(image: Any) -> str | None:
    if not image:
        return None
    p = Path(str(image))
    return str(p) if p.exists() else None


def load_model(args: argparse.Namespace, log_path: Path) -> LoadedModel:
    cfg = build_train_config_from_yaml(args.config)
    log_line(log_path, f"load model: config={args.config} checkpoint={args.checkpoint}")
    t0 = time.perf_counter()
    adapter = VLMAdapter(
        backbone_name=cfg.backbone_name,
        dtype=cfg.dtype,
        quantization=cfg.quantization,
        device=cfg.device,
        gradient_checkpointing=False,
    )
    model, n_lora = inject_lora(adapter.model, cfg)
    adapter.model = model
    head = ClassificationHead(in_dim=adapter.hidden_size).to(cfg.device)
    state = load_checkpoint(args.checkpoint, head)
    n_state = apply_lora_state(model, state)
    model.eval()
    head.eval()
    log_line(
        log_path,
        f"model ready in {time.perf_counter() - t0:.1f}s; "
        f"device={cfg.device}; lora_params={n_lora}; lora_tensors={n_state}; "
        f"max_new_tokens={args.max_new_tokens}",
    )
    return LoadedModel(
        adapter=adapter,
        model=model,
        head=head,
        device=cfg.device,
        max_new_tokens=args.max_new_tokens,
    )


@torch.inference_mode()
def classify(model_bundle: LoadedModel, text: str, image: Any) -> dict[str, Any]:
    prompt = build_prompt(text)
    encoded = model_bundle.adapter.preprocess(prompt, image_arg(image))
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


@torch.inference_mode()
def generate(model_bundle: LoadedModel, text: str, image: Any) -> str:
    return model_bundle.adapter.generate(
        text,
        image_arg(image),
        max_new_tokens=model_bundle.max_new_tokens,
        do_sample=False,
    )


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
    return {
        "accuracy": accuracy,
        "macro_f1": sum(by_label[label]["f1"] for label in LABEL_ORDER) / len(LABEL_ORDER),
        "weighted_f1": weighted_f1 / total if total else 0.0,
        "by_label": by_label,
        "confusion": {
            g: {p: sum(1 for gg, pp in zip(gold, pred) if gg == g and pp == p) for p in LABEL_ORDER}
            for g in LABEL_ORDER
        },
    }


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "total_s": 0.0, "avg_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "max_s": 0.0}
    xs = sorted(values)
    p50 = xs[len(xs) // 2]
    p95 = xs[min(len(xs) - 1, int(0.95 * (len(xs) - 1)))]
    return {
        "count": len(xs),
        "total_s": sum(xs),
        "avg_s": sum(xs) / len(xs),
        "min_s": xs[0],
        "p50_s": p50,
        "p95_s": p95,
        "max_s": xs[-1],
    }


def append_sample(path: Path, pipeline: str, rec: dict[str, Any], result: PipelineResult) -> None:
    row = {
        "pipeline": pipeline,
        "id": rec.get("id"),
        "gold": rec.get("label"),
        "pred": result.label,
        "action": result.action,
        "stage": result.stage,
        "correct": result.label == rec.get("label"),
        "timings": result.timings,
        "head": result.head,
        "explanation": result.explanation,
        "generated_text": result.output,
        "generated_chars": len(result.output or ""),
        "source": rec.get("source"),
        "metadata": rec.get("metadata") or {},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def progress_message(
    pipeline: str,
    seen: int,
    total: int,
    started: float,
    gold: list[str],
    pred: list[str],
    stage_counts: Counter[str],
    gen_count: int,
) -> str:
    elapsed = time.perf_counter() - started
    avg = elapsed / max(1, seen)
    eta = avg * max(0, total - seen)
    partial = metrics(gold, pred)
    return (
        f"{pipeline} progress {seen}/{total}; elapsed={elapsed:.1f}s; "
        f"avg={avg:.2f}s/sample; ETA={eta:.0f}s; "
        f"acc={partial['accuracy']:.3f}; macro_f1={partial['macro_f1']:.3f}; "
        f"generations={gen_count}; stage_counts={dict(stage_counts)}"
    )


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
    batch_metrics = metrics([gold[i] for i in recent], [pred[i] for i in recent])
    log_line(
        log_path,
        f"{pipeline} class-batch label={current} completed={class_seen}; "
        f"batch_size={len(recent)}; acc={batch_metrics['accuracy']:.3f}; "
        f"macro_f1={batch_metrics['macro_f1']:.3f}",
    )


def run_pipeline(
    *,
    name: str,
    fn,
    model_bundle: LoadedModel,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    log_path: Path,
    per_sample_path: Path,
) -> dict[str, Any]:
    log_line(log_path, f"pipeline start: {name}")
    started = time.perf_counter()
    gold: list[str] = []
    pred: list[str] = []
    total_latencies: list[float] = []
    head_latencies: list[float] = []
    gen_latencies: list[float] = []
    stage_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    gen_count = 0

    classify_fn = lambda text, image: classify(model_bundle, text, image)
    generate_fn = lambda text, image: generate(model_bundle, text, image)

    for i, rec in enumerate(samples, start=1):
        result: PipelineResult = fn(
            rec,
            classify_fn=classify_fn,
            generate_fn=generate_fn,
            clean_threshold=args.clean_threshold,
        ) if name == "LoRA + C4-C6 optimized" else fn(
            rec,
            classify_fn=classify_fn,
            generate_fn=generate_fn,
        )
        gold.append(rec["label"])
        pred.append(result.label)
        stage_counts[result.stage] += 1
        action_counts[result.action] += 1
        total_latencies.append(float(result.timings.get("total_seconds", 0.0)))
        if "head_seconds" in result.timings:
            head_latencies.append(float(result.timings["head_seconds"]))
        if "generate_seconds" in result.timings:
            gen_latencies.append(float(result.timings["generate_seconds"]))
            gen_count += 1
        append_sample(per_sample_path, name, rec, result)
        if i % args.log_every == 0 or i == len(samples):
            log_line(log_path, progress_message(name, i, len(samples), started, gold, pred, stage_counts, gen_count))
        maybe_class_batch_log(name, i, samples, gold, pred, args, log_path)

    return {
        "metrics": metrics(gold, pred),
        "latency": latency_summary(total_latencies),
        "head_latency": latency_summary(head_latencies),
        "generation_latency": latency_summary(gen_latencies),
        "stage_counts": dict(stage_counts),
        "action_counts": dict(action_counts),
        "generation_count": gen_count,
    }


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    artifacts = out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    with open(artifacts / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# LoRA Pipeline Compare Summary",
        "",
        f"- Samples: {summary['sample_counts']}",
        f"- Clean threshold: {summary['clean_threshold']}",
        f"- Max new tokens: {summary['max_new_tokens']}",
        "",
        "| Pipeline | Accuracy | Macro F1 | Weighted F1 | Total Time(s) | Avg(s/sample) | Generations | Stage Counts |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, res in summary["pipelines"].items():
        m = res["metrics"]
        lat = res["latency"]
        lines.append(
            f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} | "
            f"{lat['total_s']:.1f} | {lat['avg_s']:.2f} | {res['generation_count']} | "
            f"`{res['stage_counts']}` |"
        )
    lines.extend(["", "## Per-Class Metrics", ""])
    lines.append("| Pipeline | Label | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name, res in summary["pipelines"].items():
        for label in LABEL_ORDER:
            row = res["metrics"]["by_label"][label]
            lines.append(
                f"| {name} | {label} | {row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['f1']:.3f} | {row['support']} |"
            )
    lines.extend(["", "## Runtime Details", ""])
    lines.append("| Pipeline | Head Avg(s) | Generation Avg(s) | Generation Total(s) | Actions |")
    lines.append("|---|---:|---:|---:|---|")
    for name, res in summary["pipelines"].items():
        lines.append(
            f"| {name} | {res['head_latency']['avg_s']:.2f} | "
            f"{res['generation_latency']['avg_s']:.2f} | "
            f"{res['generation_latency']['total_s']:.1f} | `{res['action_counts']}` |"
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
    per_sample_path = args.out_dir / "artifacts" / "per_sample.jsonl"
    execution_log = args.out_dir / "execution_log.md"
    for p in (log_path, per_sample_path):
        if p.exists():
            p.unlink()

    log_line(log_path, "generation pipeline compare run started")
    with open(execution_log, "w", encoding="utf-8") as f:
        f.write("# LoRA Pipeline Compare Execution Log\n\n")
        f.write(f"- Started: {now()}\n")
        f.write("- Pipelines: `LoRA only`, `LoRA + C4-C6 optimized`\n")
        f.write("- Semantics: blocked samples do not run generation; allowed samples run LoRA generation.\n")

    samples = sample_records(args)
    with open(args.out_dir / "data" / "sampled_eval.jsonl", "w", encoding="utf-8") as f:
        for rec in samples:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sample_counts = dict(Counter(r["label"] for r in samples))
    log_line(log_path, f"samples ready: total={len(samples)} counts={sample_counts}")

    model_bundle = load_model(args, log_path)
    lora_only = run_pipeline(
        name="LoRA only",
        fn=run_lora_only_pipeline,
        model_bundle=model_bundle,
        samples=samples,
        args=args,
        log_path=log_path,
        per_sample_path=per_sample_path,
    )
    optimized = run_pipeline(
        name="LoRA + C4-C6 optimized",
        fn=run_optimized_pipeline,
        model_bundle=model_bundle,
        samples=samples,
        args=args,
        log_path=log_path,
        per_sample_path=per_sample_path,
    )
    summary = {
        "sample_counts": sample_counts,
        "clean_threshold": args.clean_threshold,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "pipelines": {
            "LoRA only": lora_only,
            "LoRA + C4-C6 optimized": optimized,
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
