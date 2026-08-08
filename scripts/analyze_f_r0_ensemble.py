"""Evaluate a two-checkpoint logits ensemble with the already locked R0 offset.

Selection is limited to the V2 smoke development set.  Full500 is only
reported after the smoke-selected weight is fixed.
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


def align(left: list[dict], right: list[dict]) -> list[tuple[dict, dict]]:
    if len(left) != len(right):
        raise ValueError("ensemble inputs must have the same number of records")
    pairs = list(zip(left, right))
    if any(a["id"] != b["id"] or a["gold"] != b["gold"] for a, b in pairs):
        raise ValueError("ensemble inputs are not aligned by id and gold")
    return pairs


def predict(pairs: list[tuple[dict, dict]], alpha_3000: float, offset: float) -> tuple[list[str], list[dict]]:
    predictions: list[str] = []
    rows: list[dict] = []
    for row_2250, row_3000 in pairs:
        scores = {
            label: (1.0 - alpha_3000) * row_2250["log_probs"][label]
            + alpha_3000 * row_3000["log_probs"][label]
            for label in LABELS
        }
        scores["indirect"] += offset
        prediction = max(LABELS, key=scores.__getitem__)
        predictions.append(prediction)
        result = dict(row_3000)
        result["prediction_raw_2250"] = row_2250["prediction"]
        result["prediction_raw_3000"] = row_3000["prediction"]
        result["ensemble_alpha_3000"] = alpha_3000
        result["r0_indirect_logit_offset"] = offset
        result["ensemble_log_probs"] = scores
        result["prediction"] = prediction
        rows.append(result)
    return predictions, rows


def summarize(rows: list[dict], predictions: list[str]) -> dict:
    gold = [row["gold"] for row in rows]
    report = classification_report(gold, predictions, labels=list(LABELS), output_dict=True, zero_division=0)
    per_class = {label: report[label] for label in LABELS}
    return {
        "n_eval": len(rows),
        "accuracy": accuracy_score(gold, predictions),
        "macro_f1": f1_score(gold, predictions, labels=list(LABELS), average="macro", zero_division=0),
        "weighted_f1": f1_score(gold, predictions, labels=list(LABELS), average="weighted", zero_division=0),
        "per_class": per_class,
        "min_class_f1": min(per_class[label]["f1-score"] for label in LABELS),
        "confusion_matrix": {"labels": list(LABELS), "matrix": confusion_matrix(gold, predictions, labels=list(LABELS)).tolist()},
        "prediction_counts": {label: predictions.count(label) for label in LABELS},
    }


def write_result(out_dir: Path, name: str, rows: list[dict], predictions: list[str]) -> dict:
    with (out_dir / f"{name}.predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows, predictions)
    with (out_dir / f"{name}.report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-2250", type=Path, required=True)
    parser.add_argument("--smoke-3000", type=Path, required=True)
    parser.add_argument("--full-2250", type=Path, required=True)
    parser.add_argument("--full-3000", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fixed-indirect-offset", type=float, default=0.55)
    args = parser.parse_args()

    smoke_pairs = align(load_rows(args.smoke_2250), load_rows(args.smoke_3000))
    full_pairs = align(load_rows(args.full_2250), load_rows(args.full_3000))
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    smoke_results: dict[float, tuple[list[dict], list[str], dict]] = {}
    for alpha in candidates:
        predictions, rows = predict(smoke_pairs, alpha, args.fixed_indirect_offset)
        smoke_results[alpha] = (rows, predictions, summarize(rows, predictions))

    # The official checkpoint rule prefers the weakest class, then macro F1.
    selected_alpha = max(
        candidates,
        key=lambda alpha: (
            smoke_results[alpha][2]["min_class_f1"],
            smoke_results[alpha][2]["macro_f1"],
            smoke_results[alpha][2]["per_class"]["indirect"]["f1-score"],
        ),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "method": "V2 smoke150 development-only selection; full500 labels not used for selection",
        "candidate_alpha_3000": list(candidates),
        "fixed_indirect_logit_offset": args.fixed_indirect_offset,
        "selected_alpha_3000": selected_alpha,
        "selection_priority": ["min_class_f1", "macro_f1", "indirect_f1"],
    }
    with (args.out_dir / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)

    reports = {}
    for alpha in (0.5, selected_alpha):
        suffix = "midpoint" if alpha == 0.5 else "smoke_selected"
        smoke_rows, smoke_predictions, _ = smoke_results[alpha]
        reports[f"smoke_{suffix}"] = write_result(args.out_dir, f"smoke_{suffix}", smoke_rows, smoke_predictions)
        full_predictions, full_rows = predict(full_pairs, alpha, args.fixed_indirect_offset)
        reports[f"full_{suffix}"] = write_result(args.out_dir, f"full_{suffix}", full_rows, full_predictions)
    with (args.out_dir / "reports.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"selection": selection, "reports": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
