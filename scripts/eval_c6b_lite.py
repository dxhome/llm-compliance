"""Evaluate C6B-lite OCR rules with progress and timing output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mpid.crossmodal import check_ocr_conflict, extract_ocr_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate C6B-lite local OCR guard")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--max-records", type=int, default=0)
    return p.parse_args()


def records(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    args = parse_args()
    rows = list(records(args.input))
    if args.max_records:
        rows = rows[: args.max_records]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sample: list[dict] = []
    t0 = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        ocr = extract_ocr_text(row.get("image"))
        decision = check_ocr_conflict(ocr)
        per_sample.append({
            "id": row.get("id"),
            "gold": row.get("label"),
            "pred": decision.label,
            "blocked": decision.suspicious,
            "details": decision.to_dict(),
        })
        if index % args.log_every == 0 or index == len(rows):
            elapsed = time.perf_counter() - t0
            avg = elapsed / index
            eta = avg * (len(rows) - index)
            print(f"[c6b-lite] progress {index}/{len(rows)} elapsed={elapsed:.1f}s avg={avg:.2f}s/sample ETA={eta:.0f}s", flush=True)

    labels = Counter(str(x["gold"]) for x in per_sample)
    blocked = Counter(str(x["gold"]) for x in per_sample if x["blocked"])
    unavailable = sum(not x["details"]["ocr_available"] for x in per_sample)
    summary = {
        "input": str(args.input),
        "records": len(per_sample),
        "elapsed_seconds": time.perf_counter() - t0,
        "gold_counts": dict(labels),
        "blocked_counts": dict(blocked),
        "indirect_recall": blocked["indirect"] / labels["indirect"] if labels["indirect"] else None,
        "clean_fpr": blocked["clean"] / labels["clean"] if labels["clean"] else None,
        "ocr_unavailable": unavailable,
    }
    (args.out_dir / "c6b_lite_per_sample.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in per_sample) + "\n", encoding="utf-8"
    )
    (args.out_dir / "c6b_lite_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = ["# C6B-lite OCR Report", "", f"- Records: {summary['records']}", f"- Elapsed: {summary['elapsed_seconds']:.1f}s", f"- Indirect recall: {summary['indirect_recall']}", f"- Clean FPR: {summary['clean_fpr']}", f"- OCR unavailable: {unavailable}"]
    (args.out_dir / "c6b_lite_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[c6b-lite] DONE report={args.out_dir / 'c6b_lite_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
