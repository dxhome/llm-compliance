"""Lightweight C4/C5/C6 single-record inference smoke CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mpid.infer import run_lightweight_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight C4/C5/C6 pipeline smoke")
    p.add_argument("--text", default="")
    p.add_argument("--image", default=None)
    p.add_argument("--source", default="")
    p.add_argument("--metadata-json", default="{}")
    p.add_argument("--metadata-format", default=None,
                   help="Convenience metadata format, e.g. figstep, to avoid shell JSON quoting issues")
    p.add_argument("--probs", default=None, help="Optional clean,direct,indirect probabilities")
    p.add_argument("--clean-threshold", type=float, default=0.95)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    probs = None
    if args.probs:
        probs = [float(x.strip()) for x in args.probs.split(",")]
    try:
        metadata = json.loads(args.metadata_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Invalid --metadata-json: {exc}. "
            "Use valid JSON or pass --metadata-format for common smoke cases."
        ) from exc
    if args.metadata_format:
        metadata["format"] = args.metadata_format
    record = {
        "text": args.text,
        "image": args.image,
        "source": args.source,
        "metadata": metadata,
    }
    result = run_lightweight_pipeline(
        record,
        probs=probs,
        clean_threshold=args.clean_threshold,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
