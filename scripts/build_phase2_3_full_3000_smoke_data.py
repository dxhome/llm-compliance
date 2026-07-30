"""Build a deterministic, class-balanced smoke subset for Phase 2.3."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    groups: dict[str, list[dict]] = {"clean": [], "direct": [], "indirect": []}
    for line in args.input.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("label") in groups:
            groups[record["label"]].append(record)

    rng = random.Random(args.seed)
    selected: list[dict] = []
    for label, records in groups.items():
        if len(records) < args.per_class:
            raise ValueError(f"{label} has only {len(records)} records")
        rng.shuffle(records)
        selected.extend(records[:args.per_class])
    rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print({label: sum(r["label"] == label for r in selected) for label in groups})


if __name__ == "__main__":
    main()
