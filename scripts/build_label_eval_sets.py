"""Build label-specific Phase 2.2 eval sets.

The generated files are intentionally independent of the training sample:
records that overlap the run-local train JSONL by id/source or text/image hash
are excluded before sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("clean", "direct", "indirect")
DEFAULT_SOURCES = (
    "runs/_datasets/mpid-v1/train.jsonl",
    "runs/_datasets/mpid-v1/val.jsonl",
    "runs/_datasets/mpid-v1/test.jsonl",
    "runs/_datasets/mpid-v1-crossmodal/train.jsonl",
    "runs/_datasets/mpid-v1-crossmodal/val.jsonl",
    "runs/_datasets/mpid-v1-crossmodal/test.jsonl",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _record_keys(record: dict) -> set[str]:
    source = str(record.get("source", ""))
    rec_id = str(record.get("id", ""))
    text = str(record.get("text", ""))
    image = str(record.get("image", ""))
    text_hash = hashlib.sha256(f"{text}\n{image}".encode("utf-8")).hexdigest()
    return {
        f"id::{rec_id}",
        f"source_id::{source}::{rec_id}",
        f"text_image::{text_hash}",
    }


def _normalize_image_path(record: dict) -> dict:
    image = record.get("image")
    if not image:
        return record
    image_path = Path(str(image))
    if image_path.exists():
        return record
    fallback = REPO_ROOT / "runs" / "_datasets" / "mpid-v1-crossmodal" / "images" / image_path.name
    if fallback.exists():
        record = dict(record)
        metadata = dict(record.get("metadata") or {})
        metadata["phase2_eval_original_image"] = str(image)
        record["metadata"] = metadata
        record["image"] = str(fallback)
    return record


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_eval_sets(args: argparse.Namespace) -> dict:
    run_dir = _resolve(args.run_dir)
    train_jsonl = _resolve(args.train_jsonl) if args.train_jsonl else run_dir / "data" / "train.jsonl"
    out_dir = _resolve(args.out_dir) if args.out_dir else run_dir / "data"
    source_paths = [_resolve(p) for p in (args.source_jsonl or DEFAULT_SOURCES)]

    train_keys: set[str] = set()
    for record in _read_jsonl(train_jsonl):
        train_keys.update(_record_keys(record))

    by_label: dict[str, list[dict]] = defaultdict(list)
    seen_keys: set[str] = set()
    source_counts: Counter[str] = Counter()
    excluded_train_overlap = 0
    excluded_duplicate = 0

    for source_path in source_paths:
        for record in _read_jsonl(source_path):
            label = record.get("label")
            if label not in LABELS:
                continue
            keys = _record_keys(record)
            if keys & train_keys:
                excluded_train_overlap += 1
                continue
            primary_key = sorted(keys)[0]
            if primary_key in seen_keys or keys & seen_keys:
                excluded_duplicate += 1
                continue
            seen_keys.update(keys)
            rec = _normalize_image_path(dict(record))
            metadata = dict(rec.get("metadata") or {})
            metadata.update(
                {
                    "phase2_eval_source_file": str(source_path),
                    "phase2_eval_excluded_train_overlap": False,
                }
            )
            rec["metadata"] = metadata
            by_label[label].append(rec)
            source_counts[f"{label}:{rec.get('source', '<missing>')}"] += 1

    rng = random.Random(args.seed)
    manifest = {
        "run_dir": str(run_dir),
        "train_jsonl": str(train_jsonl),
        "source_jsonl": [str(p) for p in source_paths],
        "records_per_label": args.records_per_label,
        "seed": args.seed,
        "excluded_train_overlap": excluded_train_overlap,
        "excluded_duplicate": excluded_duplicate,
        "available_after_exclusion": {label: len(by_label[label]) for label in LABELS},
        "source_counts_after_exclusion": dict(source_counts),
        "sets": {},
    }

    for label in LABELS:
        candidates = by_label[label]
        if len(candidates) < args.records_per_label:
            raise RuntimeError(
                f"Not enough {label} records after train-overlap exclusion: "
                f"{len(candidates)} < {args.records_per_label}"
            )
        selected = list(candidates)
        rng.shuffle(selected)
        selected = selected[: args.records_per_label]
        for idx, record in enumerate(selected):
            metadata = dict(record.get("metadata") or {})
            metadata.update(
                {
                    "phase2_eval_set": f"{label}_only",
                    "phase2_eval_sample_seed": args.seed,
                    "phase2_eval_sample_index": idx,
                }
            )
            record["metadata"] = metadata
        out_path = out_dir / f"eval_{label}_{args.records_per_label}.jsonl"
        _write_jsonl(out_path, selected)
        manifest["sets"][label] = {
            "path": str(out_path),
            "records": len(selected),
            "labels": dict(Counter(r.get("label") for r in selected)),
            "sources": dict(Counter(str(r.get("source", "<missing>")) for r in selected)),
        }

    manifest_path = out_dir / "eval_label_sets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--records-per-label", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--source-jsonl", action="append", default=[])
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_eval_sets(args)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
