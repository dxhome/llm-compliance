"""Lightweight C6 cross-modal smoke evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mpid.crossmodal import check_crossmodal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C6 lightweight cross-modal smoke evaluator")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-records", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.perf_counter()
    with args.input.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if args.max_records and i > args.max_records:
                break
            rec = json.loads(line)
            result = check_crossmodal(rec)
            rows.append(
                {
                    "id": rec.get("id", str(i)),
                    "gold": rec.get("label", ""),
                    "prediction": result.label,
                    "suspicious": result.suspicious,
                    "reasons": result.reasons,
                }
            )

    n = len(rows)
    indirect = [r for r in rows if r["gold"] == "indirect"]
    clean = [r for r in rows if r["gold"] == "clean"]
    suspicious = sum(1 for r in rows if r["suspicious"])
    indirect_hit = sum(1 for r in indirect if r["suspicious"])
    clean_fp = sum(1 for r in clean if r["suspicious"])
    elapsed = time.perf_counter() - t0
    summary = {
        "n": n,
        "suspicious": suspicious,
        "suspicious_rate": suspicious / n if n else 0.0,
        "indirect_recall_light": indirect_hit / len(indirect) if indirect else 0.0,
        "clean_fpr_light": clean_fp / len(clean) if clean else 0.0,
        "elapsed_seconds": elapsed,
    }

    (args.out_dir / "crossmodal_smoke_per_sample.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "crossmodal_smoke_report.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "# C6 Lightweight Cross-Modal Smoke",
        "",
        f"- records: {n}",
        f"- suspicious: {suspicious}",
        f"- suspicious rate: {summary['suspicious_rate']:.3f}",
        f"- indirect recall light: {summary['indirect_recall_light']:.3f}",
        f"- clean FPR light: {summary['clean_fpr_light']:.3f}",
        f"- elapsed seconds: {elapsed:.3f}",
    ]
    (args.out_dir / "crossmodal_smoke_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[crossmodal] records={n} suspicious={suspicious} indirect_recall={summary['indirect_recall_light']:.3f} clean_fpr={summary['clean_fpr_light']:.3f}")
    print(f"[crossmodal] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
