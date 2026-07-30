"""Build the counterfactual-boundary smoke datasets for Phase 2.3."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


RUN = Path("runs/phase2_3_full_3000_20260729_0958")
LABELS = ("clean", "direct", "indirect")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def complete_triplets(records: list[dict]) -> list[list[dict]]:
    groups: dict[object, dict[str, dict]] = defaultdict(dict)
    for record in records:
        if record.get("source") != "phase2_3_context_pairs":
            continue
        pair_id = (record.get("metadata") or {}).get("pair_id")
        if pair_id is not None:
            groups[pair_id][record["label"]] = record
    return [
        [group[label] for label in LABELS]
        for group in groups.values()
        if all(label in group for label in LABELS)
    ]


def main() -> None:
    rng = random.Random(307)
    data_dir = RUN / "data"
    train = read_jsonl(data_dir / "train.jsonl")
    triplets = complete_triplets(train)
    rng.shuffle(triplets)
    if len(triplets) < 240:
        raise RuntimeError(f"Expected at least 240 training triplets, found {len(triplets)}")

    # 240 matched triplets teach the trusted-boundary distinction directly.
    boundary_train = [record for triplet in triplets[:240] for record in triplet]
    used_ids = {record["id"] for record in boundary_train}
    for label in LABELS:
        supplements = [
            record for record in train
            if record["label"] == label
            and record["id"] not in used_ids
            and record.get("source") != "phase2_3_context_pairs"
        ]
        rng.shuffle(supplements)
        if len(supplements) < 60:
            raise RuntimeError(f"Expected at least 60 {label} supplemental records")
        boundary_train.extend(supplements[:60])
    rng.shuffle(boundary_train)
    if Counter(record["label"] for record in boundary_train) != Counter({label: 300 for label in LABELS}):
        raise RuntimeError("Boundary train set is not 300/300/300 balanced")
    write_jsonl(data_dir / "smoke_boundary_train_900.jsonl", boundary_train)

    quick = read_jsonl(data_dir / "compare_quick_all.jsonl")
    # Preserve exactly 50 examples per label while randomising evaluation order.
    mixed150 = []
    for label in LABELS:
        subset = [record for record in quick if record["label"] == label]
        rng.shuffle(subset)
        mixed150.extend(subset[:50])
    rng.shuffle(mixed150)
    write_jsonl(data_dir / "smoke_boundary_mixed_150.jsonl", mixed150)

    used_eval_keys = {record.get("dedup_key") for record in mixed150}
    pair_candidates = [
        triplet for triplet in complete_triplets(read_jsonl(data_dir / "compare_full_all.jsonl"))
        if not any(record.get("dedup_key") in used_eval_keys for record in triplet)
    ]
    rng.shuffle(pair_candidates)
    if len(pair_candidates) < 40:
        raise RuntimeError(f"Expected at least 40 disjoint evaluation triplets, found {len(pair_candidates)}")
    pair_suite = [record for triplet in pair_candidates[:40] for record in triplet]
    write_jsonl(data_dir / "smoke_boundary_pair_suite_120.jsonl", pair_suite)

    summary = {
        "train_records": len(boundary_train),
        "train_labels": dict(Counter(record["label"] for record in boundary_train)),
        "train_triplets": 240,
        "supplemental_records": 180,
        "mixed_records": len(mixed150),
        "mixed_labels": dict(Counter(record["label"] for record in mixed150)),
        "pair_suite_records": len(pair_suite),
        "pair_suite_triplets": len(pair_suite) // 3,
        "pair_suite_labels": dict(Counter(record["label"] for record in pair_suite)),
    }
    (RUN / "smoke_boundary_v2_data_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
