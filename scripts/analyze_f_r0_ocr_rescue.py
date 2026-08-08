"""Evaluate a development-selected image/OCR-only Indirect rescue bias.

The global R0 offset remains fixed.  Only the extra bias for records marked as
image or OCR is selected on V2 smoke150; full500 is report-only.
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


def predict(rows: list[dict], global_offset: float, image_ocr_extra: float) -> tuple[list[str], list[dict]]:
    predictions: list[str] = []
    adjusted_rows: list[dict] = []
    for row in rows:
        scores = dict(row["log_probs"])
        is_image_ocr = bool(row.get("has_image") or row.get("ocr_present"))
        scores["indirect"] += global_offset + (image_ocr_extra if is_image_ocr else 0.0)
        prediction = max(LABELS, key=scores.__getitem__)
        adjusted = dict(row)
        adjusted["prediction_raw"] = row["prediction"]
        adjusted["prediction"] = prediction
        adjusted["r0_indirect_logit_offset"] = global_offset
        adjusted["c1_image_ocr_extra_indirect_offset"] = image_ocr_extra if is_image_ocr else 0.0
        adjusted["c1_is_image_ocr"] = is_image_ocr
        adjusted["c1_log_probs"] = scores
        predictions.append(prediction)
        adjusted_rows.append(adjusted)
    return predictions, adjusted_rows


def summarize(rows: list[dict], predictions: list[str]) -> dict:
    gold = [row["gold"] for row in rows]
    report = classification_report(gold, predictions, labels=list(LABELS), output_dict=True, zero_division=0)
    per_class = {label: report[label] for label in LABELS}
    subsets = {}
    for name, predicate in {
        "image_ocr": lambda row: row.get("has_image") or row.get("ocr_present"),
        "text_only": lambda row: not row.get("has_image") and not row.get("ocr_present"),
    }.items():
        indexes = [index for index, row in enumerate(rows) if predicate(row)]
        subset_gold = [gold[index] for index in indexes]
        subset_predictions = [predictions[index] for index in indexes]
        indirect_support = subset_gold.count("indirect")
        indirect_tp = sum(gold[index] == "indirect" and predictions[index] == "indirect" for index in indexes)
        indirect_predicted = subset_predictions.count("indirect")
        subsets[name] = {
            "n_eval": len(indexes),
            "indirect_support": indirect_support,
            "indirect_true_positive": indirect_tp,
            "indirect_prediction_count": indirect_predicted,
            "indirect_recall": indirect_tp / indirect_support if indirect_support else 0.0,
            "indirect_precision": indirect_tp / indirect_predicted if indirect_predicted else 0.0,
        }
    return {
        "n_eval": len(rows),
        "accuracy": accuracy_score(gold, predictions),
        "macro_f1": f1_score(gold, predictions, labels=list(LABELS), average="macro", zero_division=0),
        "weighted_f1": f1_score(gold, predictions, labels=list(LABELS), average="weighted", zero_division=0),
        "per_class": per_class,
        "min_class_f1": min(per_class[label]["f1-score"] for label in LABELS),
        "confusion_matrix": {"labels": list(LABELS), "matrix": confusion_matrix(gold, predictions, labels=list(LABELS)).tolist()},
        "prediction_counts": {label: predictions.count(label) for label in LABELS},
        "subsets": subsets,
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
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fixed-global-offset", type=float, default=0.55)
    parser.add_argument("--candidates", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    args = parser.parse_args()

    smoke_rows = load_rows(args.smoke)
    full_rows = load_rows(args.full)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_reports = {}
    cache = {}
    for extra in args.candidates:
        predictions, adjusted_rows = predict(smoke_rows, args.fixed_global_offset, extra)
        report = summarize(adjusted_rows, predictions)
        cache[extra] = (adjusted_rows, predictions, report)
        candidate_reports[str(extra)] = report

    selected_extra = max(
        args.candidates,
        key=lambda extra: (
            cache[extra][2]["min_class_f1"],
            cache[extra][2]["macro_f1"],
            cache[extra][2]["per_class"]["indirect"]["f1-score"],
        ),
    )
    selection = {
        "method": "V2 smoke150 development-only selection; full500 labels not used for selection",
        "fixed_global_r0_indirect_offset": args.fixed_global_offset,
        "candidate_image_ocr_extra_offsets": args.candidates,
        "selected_image_ocr_extra_offset": selected_extra,
        "selection_priority": ["min_class_f1", "macro_f1", "indirect_f1"],
    }
    with (args.out_dir / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)
    with (args.out_dir / "smoke_candidate_reports.json").open("w", encoding="utf-8") as handle:
        json.dump(candidate_reports, handle, ensure_ascii=False, indent=2)

    reports = {}
    for name, extra in (("baseline", 0.0), ("smoke_selected", selected_extra)):
        smoke_adjusted, smoke_predictions, _ = cache[extra]
        reports[f"smoke_{name}"] = write_result(args.out_dir, f"smoke_{name}", smoke_adjusted, smoke_predictions)
        full_predictions, full_adjusted = predict(full_rows, args.fixed_global_offset, extra)
        reports[f"full_{name}"] = write_result(args.out_dir, f"full_{name}", full_adjusted, full_predictions)
    with (args.out_dir / "reports.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"selection": selection, "reports": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
