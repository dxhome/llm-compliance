"""Lightweight C5 rule smoke evaluator."""

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

from mpid.rules import scan_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C5 lightweight rule smoke evaluator")
    p.add_argument("--input", type=Path, required=True, help="Input JSONL with text/label fields")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--max-records", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t0 = time.perf_counter()
    with args.input.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if args.max_records and i > args.max_records:
                break
            rec = json.loads(line)
            result = scan_text(rec.get("text", ""))
            rows.append(
                {
                    "id": rec.get("id", str(i)),
                    "gold": rec.get("label", ""),
                    "prediction": result.label,
                    "blocked": result.blocked,
                    "matches": [m.__dict__ for m in result.matches],
                }
            )

    n = len(rows)
    blocked = sum(1 for r in rows if r["blocked"])
    direct = [r for r in rows if r["gold"] == "direct"]
    clean = [r for r in rows if r["gold"] == "clean"]
    direct_hit = sum(1 for r in direct if r["blocked"])
    clean_fp = sum(1 for r in clean if r["blocked"])
    elapsed = time.perf_counter() - t0

    summary = {
        "n": n,
        "blocked": blocked,
        "block_rate": blocked / n if n else 0.0,
        "direct_recall_light": direct_hit / len(direct) if direct else 0.0,
        "clean_fpr_light": clean_fp / len(clean) if clean else 0.0,
        "elapsed_seconds": elapsed,
    }

    (out_dir / "rules_smoke_per_sample.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "rules_smoke_report.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "# C5 Lightweight Rules Smoke",
        "",
        f"- records: {n}",
        f"- blocked: {blocked}",
        f"- block rate: {summary['block_rate']:.3f}",
        f"- direct recall light: {summary['direct_recall_light']:.3f}",
        f"- clean FPR light: {summary['clean_fpr_light']:.3f}",
        f"- elapsed seconds: {elapsed:.3f}",
    ]
    (out_dir / "rules_smoke_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[rules] records={n} blocked={blocked} direct_recall={summary['direct_recall_light']:.3f} clean_fpr={summary['clean_fpr_light']:.3f}")
    print(f"[rules] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
