"""Evaluate and time the released offline package on a frozen V2 split.

Timing begins only after the packaged model and policy have loaded, matching the
existing LoRA-only evaluator's per-record timing boundary.  Gold labels remain
inside this evaluator and are never passed to the offline inference function.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


LABELS = ["clean", "direct", "indirect"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time a released MPID offline package on frozen V2")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def load_package_predictor(package: Path):
    infer_path = package / "infer.py"
    if not infer_path.exists():
        raise FileNotFoundError(f"offline infer.py not found: {infer_path}")
    # The artifact's infer module sets its own import and OCR model paths.
    spec = importlib.util.spec_from_file_location("released_mpid_infer", infer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load released infer module: {infer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.optimized_predict


def main() -> int:
    args = parse_args()
    package = args.package.resolve()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("input has no records")

    # Model/package initialization is deliberately outside the timed boundary.
    predict = load_package_predictor(package)
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
            "stage": output.get("stage"),
            "mcr_active": bool((output.get("head") or {}).get("mcr_active", False)),
            "elapsed_seconds": record_elapsed,
            "output": output,
        })
        if index % args.progress_every == 0 or index == len(rows):
            elapsed = time.perf_counter() - started
            avg = elapsed / index
            eta = avg * (len(rows) - index)
            print(
                f"[offline-timed] progress: {index}/{len(rows)} "
                f"step_dt={avg:.2f}s eval_elapsed={elapsed:.1f}s ETA={eta:.0f}s",
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
        "timing_scope": "post_package_load_per_record_pipeline",
        "elapsed_seconds": elapsed,
        "average_seconds_per_record": elapsed / len(predictions),
        "per_class_timing": per_class_timing,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()},
        "prediction_counts": dict(Counter(y_pred)),
        "stage_counts": dict(Counter(item["stage"] for item in predictions)),
        "mcr_activated": sum(item["mcr_active"] for item in predictions),
        "package_manifest": json.loads((package / "MANIFEST.json").read_text(encoding="utf-8")),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions), encoding="utf-8"
    )
    (args.out_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# F-3000-MCR-SBC Timed Offline Package V2/full500",
        "",
        f"- Records: {summary['n_eval']}",
        "- Timing scope: post-package-load, one complete pipeline decision per record",
        f"- Elapsed: {summary['elapsed_seconds']:.1f}s",
        f"- Average decision time: {summary['average_seconds_per_record']:.3f}s/record",
        f"- Accuracy: {summary['accuracy']:.4%}",
        f"- Macro F1: {summary['macro_f1']:.4%}",
        f"- Weighted F1: {summary['weighted_f1']:.4%}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        item = summary["per_class"][label]
        lines.append(
            f"| {label} | {item['precision']:.4%} | {item['recall']:.4%} | "
            f"{item['f1-score']:.4%} | {int(item['support'])} |"
        )
    lines.extend(["", "| Gold class | Records | Average decision time |", "|---|---:|---:|"])
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
