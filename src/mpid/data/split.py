"""Stratified 8:1:1 split for MPID (Phase 1 / T1.5).

Reads unified records from ``src.mpid.data.public_loaders.load_all``,
splits them into train / val / test in a stratified manner (preserving
the ``label`` distribution across the three splits), and writes
JSONL files to ``runs/_datasets/mpid-v1/``.

Stratification is per-label, NOT per-(label, source). The reason is
that the test set is supposed to reflect the real attack distribution;
keeping the per-label ratios equal across splits is more important
than per-source balance. We DO print a per-source distribution so the
operator can spot-check (e.g. test set should not be 100% JailbreakV).

Usage::

    from mpid.data.split import split_and_dump

    split_and_dump(
        raw_dir=Path("runs/_datasets/raw"),
        out_dir=Path("runs/_datasets/mpid-v1"),
        seed=42,
        max_per_dataset={
            "safe_guard_prompt_injection": 1500,
            "jailbreakv_28k":               1500,
            "nlphuji_flickr30k":            1500,
        },
    )
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from mpid.data.public_loaders import Record, load_all


def _stratified_split(
    records: list[Record],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[Record], list[Record], list[Record]]:
    """Stratify by ``label``, shuffle within each stratum deterministically."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0; got {sum(ratios)}")
    rng = random.Random(seed)

    buckets: dict[str, list[Record]] = {}
    for r in records:
        buckets.setdefault(r.label, []).append(r)
    for v in buckets.values():
        rng.shuffle(v)

    train, val, test = [], [], []
    for label, items in buckets.items():
        n = len(items)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # Remainder goes to test to keep the total = n.
        n_test = n - n_train - n_val
        if n_test < 0:
            # Tiny stratum: fall back to greedy split.
            n_train = max(1, n - 2)
            n_val = max(1, (n - n_train) // 2)
            n_test = n - n_train - n_val
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _write_jsonl(records: Iterable[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def _distribution(records: list[Record]) -> dict:
    by_label = Counter(r.label for r in records)
    by_source = Counter(r.source for r in records)
    by_lang = Counter(r.lang for r in records)
    return {
        "by_label": dict(by_label),
        "by_source": dict(by_source),
        "by_lang": dict(by_lang),
        "total": len(records),
    }


def split_and_dump(
    raw_dir: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    max_per_dataset: dict[str, int] | None = None,
    datasets: list[str] | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """End-to-end: load, split, write JSONL, return distributions."""
    records = list(load_all(raw_dir,
                            max_per_dataset=max_per_dataset,
                            datasets=datasets))
    if not records:
        raise RuntimeError("no records loaded — check raw_dir and dataset names")

    train, val, test = _stratified_split(records, ratios=ratios, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train, out_dir / "train.jsonl")
    _write_jsonl(val,   out_dir / "val.jsonl")
    _write_jsonl(test,  out_dir / "test.jsonl")

    dist = {
        "total":      len(records),
        "train":      _distribution(train),
        "val":        _distribution(val),
        "test":       _distribution(test),
        "ratios":     list(ratios),
        "seed":       seed,
        "max_per_dataset": max_per_dataset or {},
    }
    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(dist, f, ensure_ascii=False, indent=2)
    return dist


__all__ = ["split_and_dump", "_stratified_split", "_distribution"]
