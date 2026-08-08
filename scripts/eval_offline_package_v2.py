"""Evaluate a movable MPID offline package on a frozen V2 JSONL split.

The package process is started once and receives NDJSON, so every sample uses
the exact packaged C4-C6 and F-3000-MCR-SBC runtime without reloading weights.
Labels stay in this evaluator and are never sent to the offline package.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


LABELS = ["clean", "direct", "indirect"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen V2 evaluation through an MPID offline package")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    package = args.package.resolve()
    infer = package / "infer.py"
    if not infer.exists():
        raise FileNotFoundError(f"offline infer.py not found: {infer}")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = "".join(json.dumps({"text": row.get("text", ""), "image": row.get("image")}, ensure_ascii=False) + "\n" for row in rows)
    env = dict(os.environ)
    # V2 includes multilingual and punctuation-heavy prompts. Force the
    # package boundary to UTF-8 instead of inheriting the Windows GBK locale.
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, str(infer)], input=payload, text=True, capture_output=True,
        cwd=str(package), timeout=max(1800, len(rows) * 30),
        encoding="utf-8", errors="strict", env=env,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "package_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (args.out_dir / "package_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"offline package exited {proc.returncode}; see {args.out_dir / 'package_stderr.log'}")
    outputs = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            outputs.append(json.loads(line))
    if len(outputs) != len(rows):
        raise RuntimeError(f"expected {len(rows)} package predictions, got {len(outputs)}")
    predictions = []
    for row, output in zip(rows, outputs, strict=True):
        predictions.append({
            "id": row.get("id"), "gold": row.get("label"), "pred": output["label"],
            "has_image": bool(row.get("has_image")), "stage": output.get("stage"),
            "mcr_active": bool((output.get("head") or {}).get("mcr_active", False)),
            "output": output,
        })
    y_true = [item["gold"] for item in predictions]
    y_pred = [item["pred"] for item in predictions]
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    summary = {
        "n_eval": len(predictions),
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
    (args.out_dir / "predictions.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions), encoding="utf-8")
    (args.out_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# F-3000-MCR-SBC Offline Package V2/full500", "", f"- Records: {summary['n_eval']}", f"- Accuracy: {summary['accuracy']:.4%}", f"- Macro F1: {summary['macro_f1']:.4%}", f"- Weighted F1: {summary['weighted_f1']:.4%}", f"- MCR activated: {summary['mcr_activated']}", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label in LABELS:
        item = summary["per_class"][label]
        lines.append(f"| {label} | {item['precision']:.4%} | {item['recall']:.4%} | {item['f1-score']:.4%} | {int(item['support'])} |")
    lines.extend(["", "## Stage Counts", "", *[f"- {key}: {value}" for key, value in summary["stage_counts"].items()]])
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("n_eval", "accuracy", "macro_f1", "weighted_f1", "mcr_activated")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
