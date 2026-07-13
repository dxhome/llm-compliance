"""Placeholder for the MPID inference entry point (Phase 0 / T0.3).

Real implementation lands in Phase 2 (T2.1 ``src/mpid/adapters/vlm.py``
+ classification head) and bundles into the offline package at
``scripts/package_offline.py`` (T2.11).

Usage::

    python scripts/infer.py
    python scripts/infer.py --text "Ignore previous instructions..."
    python scripts/infer.py --image /path/to/img.png --text "Describe this."
"""
from __future__ import annotations

import sys


def main() -> int:
    print("[infer] MPID single-sample inference (placeholder).")
    print("[infer] Real implementation: Phase 2 / T2.1 (VLM adapter + 3-class head).")
    print("[infer] Offline package: Phase 2 / T2.11 (no network at runtime).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
