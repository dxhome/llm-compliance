"""Render Phase 2.3 compare outputs into compact Chinese reports."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("clean", "direct", "indirect")
SLICE_FIELDS = (
    "source",
    "template",
    "lang",
    "has_image",
    "ocr_present",
    "cross_modal_attack_type",
    "hard_negative_type",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _binary_metrics(y_gold: list[str], y_pred: list[str], target: str) -> dict[str, float]:
    tp = sum(1 for g, p in zip(y_gold, y_pred) if g == target and p == target)
    fp = sum(1 for g, p in zip(y_gold, y_pred) if g != target and p == target)
    fn = sum(1 for g, p in zip(y_gold, y_pred) if g == target and p != target)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": sum(1 for g in y_gold if g == target),
    }


def _idx_to_label(values: list[int]) -> list[str]:
    return [LABELS[int(value)] for value in values]


def _prediction_distribution(labels: list[str]) -> dict[str, int]:
    counts = Counter(labels)
    return {label: counts.get(label, 0) for label in LABELS}


def _slice_metrics(records: list[dict[str, Any]], y_gold: list[str], y_pred: list[str], target: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in SLICE_FIELDS:
        buckets: dict[str, list[int]] = defaultdict(list)
        for idx, record in enumerate(records):
            value = record.get(field)
            if value in (None, ""):
                value = "unknown"
            buckets[str(value)].append(idx)
        field_rows = {}
        for value, indices in sorted(buckets.items()):
            if len(indices) < 3:
                continue
            gold = [y_gold[i] for i in indices]
            pred = [y_pred[i] for i in indices]
            field_rows[value] = {
                "n": len(indices),
                "target": _binary_metrics(gold, pred, target),
                "prediction_distribution": _prediction_distribution(pred),
            }
        out[field] = field_rows
    return out


def _misclassified(records: list[dict[str, Any]], y_gold: list[str], y_pred: list[str], limit: int) -> list[dict[str, Any]]:
    rows = []
    for record, gold, pred in zip(records, y_gold, y_pred):
        if gold == pred:
            continue
        rows.append({
            "id": record.get("id"),
            "gold": gold,
            "pred": pred,
            "source": record.get("source"),
            "template": record.get("template"),
            "lang": record.get("lang"),
            "has_image": record.get("has_image"),
            "ocr_present": record.get("ocr_present"),
            "cross_modal_attack_type": record.get("cross_modal_attack_type"),
            "hard_negative_type": record.get("hard_negative_type"),
            "text_preview": str(record.get("text", ""))[:160],
        })
        if len(rows) >= limit:
            break
    return rows


def _target_from_val(records: list[dict[str, Any]], fallback: str) -> str:
    counts = Counter(str(record.get("label")) for record in records)
    if counts:
        return counts.most_common(1)[0][0]
    return fallback


def _render_md(summary: dict[str, Any]) -> str:
    target = summary["target_label"]
    new_m = summary["new_model"]["target_metrics"]
    base_m = summary["baseline"]["target_metrics"]
    delta = summary["delta"]["target_f1_delta"]
    lines = [
        f"# Phase 2.3 compare 汇总：{summary['scope']} step {summary['step']} {target}",
        "",
        f"- 数据集：{summary['val_jsonl']}",
        f"- 样本数：{summary['n_eval']}",
        f"- 目标类：{target}",
        f"- 新模型目标类 P/R/F1：{new_m['precision']:.4f} / {new_m['recall']:.4f} / {new_m['f1']:.4f}",
        f"- baseline 目标类 P/R/F1：{base_m['precision']:.4f} / {base_m['recall']:.4f} / {base_m['f1']:.4f}",
        f"- 目标类 F1 提升：{delta:+.4f}",
        f"- 新模型预测分布：{summary['new_model']['prediction_distribution']}",
        f"- baseline 预测分布：{summary['baseline']['prediction_distribution']}",
        "",
        "## 切片指标",
        "",
    ]
    for field, values in summary["new_model"]["slice_metrics"].items():
        lines.append(f"### {field}")
        for value, row in values.items():
            m = row["target"]
            lines.append(
                f"- {value}: n={row['n']}, target_f1={m['f1']:.4f}, "
                f"target_recall={m['recall']:.4f}, pred={row['prediction_distribution']}"
            )
        lines.append("")
    lines.extend([
        "## 新模型误判样本",
        "",
    ])
    if summary["new_model"]["misclassified"]:
        for row in summary["new_model"]["misclassified"]:
            lines.append(
                f"- {row['id']}: gold={row['gold']}, pred={row['pred']}, "
                f"source={row['source']}, template={row['template']}, text={row['text_preview']}"
            )
    else:
        lines.append("- 无")
    lines.append("")
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    compare_dir = args.compare_dir.resolve()
    val_jsonl = args.val_jsonl.resolve()
    records = _load_jsonl(val_jsonl)
    report = _load_json(compare_dir / "comparison_full_vs_smoke.json")
    target = args.target_label or _target_from_val(records, args.label)

    base_gold = _idx_to_label(report["smoke"]["y_gold"])
    base_pred = _idx_to_label(report["smoke"]["y_pred"])
    new_gold = _idx_to_label(report["full"]["y_gold"])
    new_pred = _idx_to_label(report["full"]["y_pred"])

    summary = {
        "scope": args.scope,
        "step": args.step,
        "label": args.label,
        "target_label": target,
        "val_jsonl": str(val_jsonl),
        "compare_dir": str(compare_dir),
        "n_eval": len(records),
        "baseline": {
            "checkpoint": report.get("smoke_checkpoint"),
            "target_metrics": _binary_metrics(base_gold, base_pred, target),
            "accuracy": report["smoke"]["accuracy"],
            "macro_f1": report["smoke"]["macro_f1"],
            "prediction_distribution": _prediction_distribution(base_pred),
            "slice_metrics": _slice_metrics(records, base_gold, base_pred, target),
            "misclassified": _misclassified(records, base_gold, base_pred, args.max_misclassified),
        },
        "new_model": {
            "checkpoint": report.get("full_checkpoint"),
            "target_metrics": _binary_metrics(new_gold, new_pred, target),
            "accuracy": report["full"]["accuracy"],
            "macro_f1": report["full"]["macro_f1"],
            "prediction_distribution": _prediction_distribution(new_pred),
            "slice_metrics": _slice_metrics(records, new_gold, new_pred, target),
            "misclassified": _misclassified(records, new_gold, new_pred, args.max_misclassified),
        },
    }
    summary["delta"] = {
        "target_f1_delta": summary["new_model"]["target_metrics"]["f1"] - summary["baseline"]["target_metrics"]["f1"],
        "accuracy_delta": summary["new_model"]["accuracy"] - summary["baseline"]["accuracy"],
        "macro_f1_delta": summary["new_model"]["macro_f1"] - summary["baseline"]["macro_f1"],
    }

    out_json = compare_dir / "phase2_3_compare_summary.json"
    out_md = compare_dir / "phase2_3_compare_summary.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-dir", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--scope", choices=("quick", "full"), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--label", choices=LABELS, required=True)
    parser.add_argument("--target-label", choices=LABELS, default=None)
    parser.add_argument("--max-misclassified", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    summary = summarize(parse_args())
    print(json.dumps({
        "scope": summary["scope"],
        "step": summary["step"],
        "label": summary["label"],
        "target_label": summary["target_label"],
        "new_target_f1": summary["new_model"]["target_metrics"]["f1"],
        "baseline_target_f1": summary["baseline"]["target_metrics"]["f1"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
