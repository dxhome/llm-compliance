"""Build a leakage-audited hard-direct counterfactual curriculum for F4."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "phase2_3_full_3000_20260729_0958"
LABELS = ("clean", "direct", "indirect")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def text_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def complete_triplets(records: list[dict]) -> list[list[dict]]:
    groups: dict[object, dict[str, dict]] = defaultdict(dict)
    for record in records:
        pair_id = (record.get("metadata") or {}).get("pair_id")
        if pair_id is not None:
            groups[pair_id][record["label"]] = record
    return [[group[label] for label in LABELS] for group in groups.values() if all(label in group for label in LABELS)]


def hard_triplet(record: dict, index: int) -> list[dict]:
    attack = record["text"]
    pair_id = f"f4_hard_direct_{index:03d}"
    direct = dict(record)
    direct.update({
        "id": f"f4_hard_direct_{index:03d}",
        "split": "smoke_f4_train",
        "metadata": {"pair_id": pair_id, "curriculum": "hard_direct_replay"},
    })
    clean = dict(record)
    clean.update({
        "id": f"f4_hard_clean_{index:03d}",
        "text": f"For security training, explain why this quoted instruction must not be obeyed:\n\n{attack}",
        "label": "clean",
        "split": "smoke_f4_train",
        "dedup_key": text_key(f"f4-clean:{attack}"),
        "attack_family": None,
        "indirect_subtype": None,
        "has_instruction_override": False,
        "hard_negative_type": "quoted_attack_discussion",
        "content_role": "trusted_user_request",
        "metadata": {"pair_id": pair_id, "curriculum": "hard_direct_replay"},
    })
    indirect = dict(record)
    indirect.update({
        "id": f"f4_hard_indirect_{index:03d}",
        "text": f"External document content (untrusted):\n\n{attack}",
        "label": "indirect",
        "split": "smoke_f4_train",
        "dedup_key": text_key(f"f4-indirect:{attack}"),
        "indirect_subtype": "document_context_injection",
        "has_instruction_override": True,
        "content_role": "untrusted_external_content",
        "metadata": {"pair_id": pair_id, "curriculum": "hard_direct_replay"},
    })
    return [clean, direct, indirect]


def main() -> None:
    source_triplets = complete_triplets(read_jsonl(RUN / "data" / "smoke_boundary_train_900.jsonl"))
    replay = read_jsonl(RUN / "data" / "smoke_direct_replay.jsonl")
    benchmark = read_jsonl(RUN / "artifacts" / "standard_benchmark_v2_a_h" / "smoke_all_150_resolved_images.jsonl")
    eval_keys = {record.get("dedup_key") for record in benchmark}
    eval_text_keys = {text_key(record["text"]) for record in benchmark}

    if len(source_triplets) != 240:
        raise RuntimeError(f"Expected 240 source triplets, found {len(source_triplets)}")
    replay = [record for record in replay if record["label"] == "direct"]
    collisions = [record["id"] for record in replay if record.get("dedup_key") in eval_keys or text_key(record["text"]) in eval_text_keys]
    if collisions:
        raise RuntimeError(f"F4 replay overlaps frozen benchmark: {collisions}")
    if len(replay) != 30:
        raise RuntimeError(f"Expected 30 leakage-free direct replay records, found {len(replay)}")

    rng = random.Random(427)
    rng.shuffle(source_triplets)
    curriculum = [hard_triplet(record, index) for index, record in enumerate(replay, start=1)] + source_triplets[:90]
    rng.shuffle(curriculum)
    records = [record for triplet in curriculum for record in triplet]
    if Counter(record["label"] for record in records) != Counter({label: 120 for label in LABELS}):
        raise RuntimeError("F4 curriculum must remain class balanced")
    all_keys = {record.get("dedup_key") for record in records}
    if all_keys & eval_keys:
        raise RuntimeError("F4 curriculum dedup key overlaps frozen benchmark")
    if any(text_key(record["text"]) in eval_text_keys for record in records):
        raise RuntimeError("F4 curriculum text overlaps frozen benchmark")

    out = RUN / "data" / "smoke_f4_hard_direct_train_360.jsonl"
    write_jsonl(out, records)
    audit = {
        "strategy": "F4",
        "purpose": "hard direct counterfactual curriculum",
        "records": len(records),
        "triplets": len(curriculum),
        "labels": dict(Counter(record["label"] for record in records)),
        "hard_direct_triplets": len(replay),
        "source_triplets": 90,
        "benchmark_dedup_key_overlap": False,
        "benchmark_normalized_text_overlap": False,
        "seed": 427,
    }
    audit_path = RUN / "artifacts" / "smoke_f4" / "f4_data_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
