"""Single-record optimized pipeline smoke CLI.

This script simulates the MPID head with ``--probs`` and ``--label`` so the
C5/C6/C4 orchestration can be exercised without loading the VLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mpid.infer import run_optimized_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimized C5/C6/head/C4 pipeline smoke")
    p.add_argument("--text", default="")
    p.add_argument("--image", default=None)
    p.add_argument("--source", default="")
    p.add_argument("--metadata-json", default="{}")
    p.add_argument("--metadata-format", default=None,
                   help="Convenience metadata format, e.g. figstep")
    p.add_argument("--probs", default="0.34,0.33,0.33",
                   help="Simulated clean,direct,indirect head probabilities")
    p.add_argument("--label", default=None,
                   help="Optional simulated head label; defaults to argmax(probs)")
    p.add_argument("--risk", type=float, default=None,
                   help="Optional simulated risk; defaults to max non-clean prob")
    p.add_argument("--clean-threshold", type=float, default=0.95)
    return p.parse_args()


def _parse_probs(value: str) -> list[float]:
    probs = [float(x.strip()) for x in value.split(",")]
    if len(probs) != 3:
        raise SystemExit("--probs must contain clean,direct,indirect")
    return probs


def main() -> int:
    args = parse_args()
    probs = _parse_probs(args.probs)
    labels = ["clean", "direct", "indirect"]
    label = args.label or labels[max(range(3), key=lambda i: probs[i])]
    risk = args.risk if args.risk is not None else max(probs[1], probs[2])

    try:
        metadata = json.loads(args.metadata_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --metadata-json: {exc}") from exc
    if args.metadata_format:
        metadata["format"] = args.metadata_format

    record = {
        "text": args.text,
        "image": args.image,
        "source": args.source,
        "metadata": metadata,
    }

    def classify(_text, _image):
        return {"label": label, "risk": risk, "probs": probs}

    def generate(_text, _image):
        return "[simulated LoRA generation]"

    result = run_optimized_pipeline(
        record,
        classify_fn=classify,
        generate_fn=generate,
        clean_threshold=args.clean_threshold,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
