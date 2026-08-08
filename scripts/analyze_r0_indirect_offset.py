"""Explore a fixed indirect-logit offset from saved evaluation predictions.

Selection uses only the public expected class prior, never per-record gold labels.
Gold labels are read only after the offset is locked to report exploratory metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


LABELS = ("clean", "direct", "indirect")


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any("log_probs" not in row for row in rows):
        raise ValueError(f"{path} must contain non-empty saved log_probs")
    return rows


def predict(rows: list[dict], offset: float) -> list[str]:
    predictions: list[str] = []
    for row in rows:
        scores = dict(row["log_probs"])
        scores["indirect"] += offset
        predictions.append(max(LABELS, key=scores.__getitem__))
    return predictions


def summarize(rows: list[dict], predictions: list[str]) -> dict:
    gold = [row["gold"] for row in rows]
    report = classification_report(
        gold, predictions, labels=list(LABELS), output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(gold, predictions, labels=list(LABELS)).tolist()
    return {
        "n_eval": len(rows),
        "accuracy": accuracy_score(gold, predictions),
        "macro_f1": f1_score(gold, predictions, labels=list(LABELS), average="macro", zero_division=0),
        "weighted_f1": f1_score(gold, predictions, labels=list(LABELS), average="weighted", zero_division=0),
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": {"labels": list(LABELS), "matrix": matrix},
        "prediction_counts": {label: predictions.count(label) for label in LABELS},
    }


def write_result(path: Path, rows: list[dict], predictions: list[str], offset: float) -> dict:
    adjusted_rows = []
    for row, prediction in zip(rows, predictions):
        adjusted = dict(row)
        adjusted["prediction_raw"] = row["prediction"]
        adjusted["prediction"] = prediction
        adjusted["r0_indirect_logit_offset"] = offset
        adjusted_rows.append(adjusted)
    with path.with_suffix(".predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in adjusted_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows, predictions)
    with path.with_suffix(".report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", type=Path, help="Saved logits used only for prior-based offset selection.")
    parser.add_argument("--apply", type=Path, action="append", required=True, help="Saved logits to report after applying the locked offset.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fixed-offset", type=float, help="Apply an already locked indirect offset without re-running selection.")
    parser.add_argument("--expected-indirect-rate", type=float, default=0.15)
    parser.add_argument("--min-offset", type=float, default=-1.0)
    parser.add_argument("--max-offset", type=float, default=3.0)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    if not 0 < args.expected_indirect_rate < 1 or args.step <= 0:
        raise ValueError("expected rate must be in (0, 1), and step must be positive")
    if args.fixed_offset is not None:
        if args.select is not None:
            raise ValueError("--fixed-offset cannot be combined with --select")
        offset = args.fixed_offset
        selection = {
            "method": "previously-locked offset; no selection or gold-label access",
            "selected_indirect_logit_offset": offset,
        }
    else:
        if args.select is None:
            raise ValueError("--select is required unless --fixed-offset is provided")
        selected_rows = load_rows(args.select)
        candidates = []
        steps = int(round((args.max_offset - args.min_offset) / args.step))
        for index in range(steps + 1):
            candidate_offset = round(args.min_offset + index * args.step, 8)
            predicted = predict(selected_rows, candidate_offset)
            indirect_rate = predicted.count("indirect") / len(predicted)
            # The sort key deliberately contains no gold-label-derived value.
            candidates.append((abs(indirect_rate - args.expected_indirect_rate), candidate_offset, indirect_rate))
        _, offset, indirect_rate = min(candidates)
        selection = {
            "method": "public-prior-only; no per-record gold labels used for selection",
            "select_source": str(args.select),
            "expected_indirect_rate": args.expected_indirect_rate,
            "selected_indirect_logit_offset": offset,
            "selected_prediction_indirect_rate": indirect_rate,
            "search": {"min": args.min_offset, "max": args.max_offset, "step": args.step},
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "r0_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)

    all_reports = {}
    for input_path in args.apply:
        rows = load_rows(input_path)
        stem = input_path.parent.name
        all_reports[stem] = write_result(args.out_dir / stem, rows, predict(rows, offset), offset)
    with (args.out_dir / "r0_reports.json").open("w", encoding="utf-8") as handle:
        json.dump(all_reports, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"selection": selection, "reports": all_reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
