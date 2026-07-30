"""Build deterministic Phase 2.3 multi-strategy smoke datasets."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


RUN = Path("runs/phase2_3_full_3000_20260729_0958")
LABELS = ("clean", "direct", "indirect")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def main() -> None:
    rng = random.Random(43)
    train = read_jsonl(RUN / "data" / "train.jsonl")
    selected: list[dict] = []
    for label in LABELS:
        paired = [r for r in train if r["label"] == label and r["source"] == "phase2_3_context_pairs"]
        other = [r for r in train if r["label"] == label and r["source"] != "phase2_3_context_pairs"]
        rng.shuffle(paired); rng.shuffle(other)
        selected.extend(paired[:40] + other[:60])
    rng.shuffle(selected)
    write_jsonl(RUN / "data" / "smoke_train_300.jsonl", selected)

    quick = {label: read_jsonl(RUN / "data" / f"compare_quick_{label}.jsonl") for label in LABELS}
    mixed90 = [r for label in LABELS for r in quick[label][:30]]
    rng.shuffle(mixed90)
    write_jsonl(RUN / "data" / "smoke_mixed_90.jsonl", mixed90)
    mixed150 = [r for label in LABELS for r in quick[label][:50]]
    rng.shuffle(mixed150)
    write_jsonl(RUN / "data" / "smoke_quick_150.jsonl", mixed150)
    write_jsonl(RUN / "data" / "smoke_direct_replay.jsonl", quick["direct"][:30])

    summary = {
        "smoke_train_300": Counter(r["label"] for r in selected),
        "smoke_mixed_90": Counter(r["label"] for r in mixed90),
        "smoke_quick_150": Counter(r["label"] for r in mixed150),
        "paired_train": Counter(r["label"] for r in selected if r["source"] == "phase2_3_context_pairs"),
    }
    print({name: dict(counts) for name, counts in summary.items()})


if __name__ == "__main__":
    main()
