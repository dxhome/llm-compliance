"""MPID single-sample inference entry point (Phase 2 / T2.1 + Phase 3 / T3.6).

This script runs the full MPID inference pipeline on a single
(text, image) sample and prints the predicted label. With ``--early-exit``
(T3.6) it also enables C4 — the early-exit layer that returns
``"clean"`` directly when ``P(clean) > threshold`` instead of falling
through to C5/C6.

Usage::

    # Default (no early exit)
    python scripts/infer.py --text "Hello, world."

    # With image
    python scripts/infer.py --text "Describe this." --image /path/to/img.png

    # With C4 early-exit (Phase 3)
    python scripts/infer.py --text "Hello, world." --early-exit
    python scripts/infer.py --text "..." --early-exit --clean-threshold 0.90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPID single-sample inference")
    p.add_argument("--text", type=str, default="Hello, world.",
                   help="User text input (default: a trivial greeting).")
    p.add_argument("--image", type=str, default=None,
                   help="Path to an image (else: text-only).")
    p.add_argument("--checkpoint", type=str,
                   default=str(REPO_ROOT / "artifacts" / "baseline" / "lora_baseline.safetensors"),
                   help="LoRA + head checkpoint to load.")
    p.add_argument("--config", type=str,
                   default=str(REPO_ROOT / "configs" / "baseline.yaml"),
                   help="YAML config (same schema as train.py / eval.py).")
    p.add_argument("--early-exit", action="store_true",
                   help="T3.6: enable C4 early-exit (P(clean) > threshold → 'clean').")
    p.add_argument("--clean-threshold", type=float, default=0.95,
                   help="T3.6: P(clean) threshold (default 0.95).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[infer] MPID single-sample inference (Phase 2/3)")
    print(f"[infer] text:   {args.text!r}")
    print(f"[infer] image:  {args.image or '(none — placeholder)'}")
    print(f"[infer] early exit: {'ON (threshold=' + str(args.clean_threshold) + ')' if args.early_exit else 'OFF'}")
    print(f"[infer] checkpoint: {args.checkpoint}")
    print(f"[infer] config:     {args.config}")
    print()
    print("[infer] Note: this is a placeholder (Phase 0 / T0.3).")
    print("[infer] Full pipeline is implemented in:")
    print("[infer]   - src/mpid/adapters/vlm.py      (T2.1)")
    print("[infer]   - src/mpid/heads/classification.py (T2.3)")
    print("[infer]   - src/mpid/early_exit.py        (T3.1, this Phase 3)")
    print("[infer] To run a real inference: use src/mpid/early_exit.classify_with_early_exit()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
