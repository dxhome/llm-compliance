"""End-to-end smoke test for the Phase 2.5 demo pipeline (T2.5.6).

This script does NOT start the Gradio server; it exercises the same
``DemoPipeline`` class that the Gradio app uses, on all 8 preset
samples, and writes a JSON report. It's the offline half of the
"端到端冒烟" acceptance criterion — the online half (browser screenshot)
is done by hand after the server is up.

Usage::

    python demo/smoke_pipeline.py
    python demo/smoke_pipeline.py --device cpu
    python demo/smoke_pipeline.py --out demo/screenshots/smoke_report.json

The script writes:

  * ``smoke_report.json`` — per-sample {label_gt, label_pred, risk, probs,
    base_generation} plus a pass/fail summary.
  * optional ``smoke_<i>.md``  — markdown per-sample, easier to read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO = DEMO_DIR.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(DEMO_DIR))  # so we can import gradio_app

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path,
                   default=REPO / "runs" / "_models" / "smolvlm-500m")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO / "runs" / "_templates" / "artifacts" /
                   "checkpoints" / "lora_baseline.safetensors")
    p.add_argument("--samples", type=Path, default=DEMO_DIR / "samples.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Smaller default than UI to keep smoke fast.")
    p.add_argument("--out", type=Path, default=DEMO_DIR / "screenshots" / "smoke_report.json")
    args = p.parse_args()

    # Import here so the argparse above is fast.
    from gradio_app import DemoPipeline, resolve_demo_asset

    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    assert len(samples) == 8, f"expected 8 samples, got {len(samples)}"

    t0 = time.perf_counter()
    pipeline = DemoPipeline(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    t_load = time.perf_counter() - t0
    print(f"[smoke] pipeline ready in {t_load:.1f}s")

    report = {
        "model_dir": str(args.model_dir),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "load_seconds": t_load,
        "samples": [],
    }

    for s in samples:
        t0 = time.perf_counter()
        img = None
        if s.get("image"):
            abs_p = resolve_demo_asset(s["image"])
            if abs_p and abs_p.exists():
                img = str(abs_p)

        # 1. classification
        cls = pipeline.classify(s["text"], img)
        t_cls = time.perf_counter() - t0

        # 2. generation (truncate text to keep smoke fast)
        gen_text = s["text"]
        t1 = time.perf_counter()
        base_text = pipeline.generate(gen_text, img)
        t_gen = time.perf_counter() - t1

        sample_report = {
            "index": s["index"],
            "id": s["id"],
            "title": s["title"],
            "label_gt": s["label"],
            "label_pred": cls["label"],
            "label_match": cls["label"] == s["label"],
            "risk": cls["risk"],
            "probs": cls["probs"],
            "base_text": base_text,
            "classify_seconds": t_cls,
            "generate_seconds": t_gen,
        }
        report["samples"].append(sample_report)
        agree = "OK" if sample_report["label_match"] else "MISMATCH"
        print(f"[smoke] #{s['index']} gt={s['label']:>8} "
              f"pred={cls['label']:>8} risk={cls['risk']:.2f} "
              f"[{agree}]   cls={t_cls:.1f}s  gen={t_gen:.1f}s")

    # Summary.
    n_total = len(report["samples"])
    n_match = sum(1 for s in report["samples"] if s["label_match"])
    n_correct = {l: 0 for l in ("clean", "direct", "indirect")}
    n_total_by = {l: 0 for l in ("clean", "direct", "indirect")}
    for s in report["samples"]:
        n_total_by[s["label_gt"]] += 1
        if s["label_match"]:
            n_correct[s["label_gt"]] += 1
    report["summary"] = {
        "n_total": n_total,
        "n_match": n_match,
        "n_match_by_class": n_correct,
        "n_total_by_class": n_total_by,
        "per_class_recall": {
            l: (n_correct[l] / n_total_by[l] if n_total_by[l] else 0.0)
            for l in n_total_by
        },
    }
    print(f"\n[smoke] summary: {n_match}/{n_total} matched")
    for l in n_total_by:
        rec = report["summary"]["per_class_recall"][l]
        print(f"  recall[{l:>8}] = {n_correct[l]}/{n_total_by[l]} = {rec:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[smoke] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
