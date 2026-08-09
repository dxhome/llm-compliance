"""Time the frozen F-3000 LoRA-only classifier by gold class on V2.

This is a measurement-only evaluator.  It loads the same F-3000 checkpoint as
the formal baseline, then measures one full LoRA/head decision per record after
model load.  Labels are retained solely for report aggregation.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as functional
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from mpid.adapters.vlm import VLMAdapter
from mpid.heads.classification import IDX2LABEL, NUM_CLASSES, ClassificationHead
from mpid.data.prompt import build_prompt
from mpid.train.trainer import apply_lora_state, inject_lora, load_checkpoint


LABELS = ["clean", "direct", "indirect"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time F-3000 LoRA-only on frozen V2")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    defaults = raw_cfg.get("defaults", {})
    lora = raw_cfg.get("lora", {})

    class Config:
        backbone_name = defaults.get("backbone_name", "smolvlm-500m")
        dtype = defaults.get("dtype", "float32")
        device = defaults.get("device", "cpu")
        quantization = defaults.get("quantization")
        lora_r = int(lora.get("r", 32))
        lora_alpha = int(lora.get("alpha", 64))
        lora_dropout = float(lora.get("dropout", 0.0))
        lora_target = str(lora.get("target", "q_proj,k_proj,v_proj,o_proj"))

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("input has no records")

    adapter = VLMAdapter(
        backbone_name=Config.backbone_name,
        dtype=Config.dtype,
        quantization=Config.quantization,
        device=Config.device,
        gradient_checkpointing=False,
    )
    peft_model, _ = inject_lora(adapter.model, Config())
    head = ClassificationHead(in_dim=adapter.hidden_size, num_classes=NUM_CLASSES).to(Config.device)
    state = load_checkpoint(args.checkpoint, head)
    apply_lora_state(peft_model, state)
    peft_model.eval()
    head.eval()

    def predict(text: str, image_path: str | None) -> dict:
        image = Image.new("RGB", (512, 512), (235, 235, 235))
        if image_path:
            image = Image.open(image_path).convert("RGB")
        encoded = adapter.preprocess(build_prompt(text), image)
        with torch.inference_mode():
            output = peft_model(**encoded, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        last_index = encoded["attention_mask"].sum(dim=1) - 1
        batch = torch.arange(hidden.size(0), device=hidden.device)
        logits = head(hidden[batch, last_index])
        probabilities = functional.softmax(logits, dim=-1)
        label_index = probabilities.argmax(dim=-1)
        return {
            "label": IDX2LABEL[int(label_index[0].item())],
            "risk": float(probabilities.max(dim=-1).values[0].item()),
        }

    # Keep model loading outside the timed boundary, matching formal eval_elapsed.
    started = time.perf_counter()
    predictions = []
    for index, row in enumerate(rows, start=1):
        record_started = time.perf_counter()
        output = predict(row.get("text", ""), row.get("image"))
        record_elapsed = time.perf_counter() - record_started
        predictions.append({
            "id": row.get("id"),
            "gold": row.get("label"),
            "pred": output["label"],
            "has_image": bool(row.get("has_image")),
            "elapsed_seconds": record_elapsed,
            "output": output,
        })
        if index % args.progress_every == 0 or index == len(rows):
            elapsed = time.perf_counter() - started
            average = elapsed / index
            print(
                f"[lora-only-timed] progress: {index}/{len(rows)} "
                f"step_dt={average:.2f}s eval_elapsed={elapsed:.1f}s ETA={average * (len(rows) - index):.0f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    y_true = [item["gold"] for item in predictions]
    y_pred = [item["pred"] for item in predictions]
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    per_class_timing = {}
    for label in LABELS:
        values = [item["elapsed_seconds"] for item in predictions if item["gold"] == label]
        per_class_timing[label] = {
            "records": len(values),
            "elapsed_seconds": sum(values),
            "average_seconds_per_record": sum(values) / len(values),
        }
    summary = {
        "n_eval": len(predictions),
        "timing_scope": "post_model_load_per_record_lora_head",
        "elapsed_seconds": elapsed,
        "average_seconds_per_record": elapsed / len(predictions),
        "per_class_timing": per_class_timing,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()},
        "prediction_counts": dict(Counter(y_pred)),
        "checkpoint": str(args.checkpoint.resolve()),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions), encoding="utf-8"
    )
    (args.out_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# F-3000 LoRA-only Timed V2/full500",
        "",
        f"- Records: {summary['n_eval']}",
        "- Timing scope: post-model-load, one LoRA/head decision per record",
        f"- Elapsed: {summary['elapsed_seconds']:.1f}s",
        f"- Average decision time: {summary['average_seconds_per_record']:.3f}s/record",
        "",
        "| Gold class | Records | Average decision time |",
        "|---|---:|---:|",
    ]
    for label in LABELS:
        item = summary["per_class_timing"][label]
        lines.append(f"| {label} | {item['records']} | {item['average_seconds_per_record']:.3f}s |")
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "n_eval", "elapsed_seconds", "average_seconds_per_record", "accuracy", "macro_f1", "weighted_f1"
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
