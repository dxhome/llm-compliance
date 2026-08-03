"""Freeze and audit MPID Standard Benchmark v2 inputs for the F formal run.

This script is deliberately validation-only: it verifies the frozen split
contents, materializes absolute image paths for evaluation, and checks that
the formal training data is disjoint from both V2 splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "runs" / "phase2_3_full_3000_20260729_0958"
LABELS = ("clean", "direct", "indirect")
EXPECTED = {
    "smoke": {"total": 150, "labels": {"clean": 75, "direct": 53, "indirect": 22}, "images": 30},
    "full": {"total": 500, "labels": {"clean": 250, "direct": 175, "indirect": 75}, "images": 80},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def media_content_fingerprint(row: dict) -> str:
    """Fingerprint the full multimodal item, not a reused prompt template."""
    image = row.get("image")
    image_hash = ""
    if image:
        image_path = Path(str(image))
        if image_path.is_file():
            image_hash = sha256(image_path)
        else:
            image_hash = f"unresolved:{image_path}"
    payload = "\n".join((
        normalized_text(row.get("text")),
        normalized_text(row.get("ocr_text")),
        image_hash,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprints(rows: list[dict]) -> dict[str, set]:
    return {
        "dedup_key": {row["dedup_key"] for row in rows if row.get("dedup_key")},
        "source_record_id": {
            (row.get("source"), row.get("source_record_id"))
            for row in rows
            if row.get("source") and row.get("source_record_id")
        },
        "content_fingerprint": {media_content_fingerprint(row) for row in rows},
        "normalized_text_template_sha256": {
            hashlib.sha256(normalized_text(row.get("text")).encode("utf-8")).hexdigest()
            for row in rows
            if normalized_text(row.get("text"))
        },
    }


def verify_split(name: str, split_root: Path, output_dir: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((split_root / "manifest.json").read_text(encoding="utf-8"))
    expected = EXPECTED[name]
    if manifest.get("total") != expected["total"] or manifest.get("by_label") != expected["labels"]:
        raise ValueError(f"{name}: manifest counts do not match expected frozen definition")

    checksum_lines = (split_root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    checksum_map = {}
    for line in checksum_lines:
        digest, relative = line.split(maxsplit=1)
        checksum_map[relative.strip()] = digest
    if checksum_map != manifest.get("files", {}):
        raise ValueError(f"{name}: checksums.sha256 differs from manifest files")

    failed_hashes: list[str] = []
    for relative, expected_hash in checksum_map.items():
        target = split_root / relative
        if not target.is_file() or sha256(target) != expected_hash:
            failed_hashes.append(relative)
    if failed_hashes:
        raise ValueError(f"{name}: checksum failure: {failed_hashes}")

    source_path = split_root / f"{name}_all_{expected['total']}.jsonl"
    rows = read_jsonl(source_path)
    labels = Counter(row.get("label") for row in rows)
    image_rows = [row for row in rows if row.get("has_image")]
    if len(rows) != expected["total"] or dict(labels) != expected["labels"] or len(image_rows) != expected["images"]:
        raise ValueError(f"{name}: JSONL count or label distribution mismatch")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError(f"{name}: duplicate record id")

    resolved_rows = []
    missing_images: list[str] = []
    for row in rows:
        copy = dict(row)
        image = copy.get("image")
        if image:
            image_path = (split_root / image).resolve()
            if not image_path.is_file():
                missing_images.append(image)
            copy["image"] = str(image_path)
        resolved_rows.append(copy)
    if missing_images:
        raise ValueError(f"{name}: missing image files: {missing_images}")

    resolved_path = output_dir / f"{name}_all_{expected['total']}_resolved_images.jsonl"
    with resolved_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in resolved_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "name": name,
        "split_root": str(split_root.resolve()),
        "source_jsonl": str(source_path.resolve()),
        "resolved_jsonl": str(resolved_path.resolve()),
        "manifest_sha256": sha256(split_root / "manifest.json"),
        "checksums_sha256": sha256(split_root / "checksums.sha256"),
        "verified_files": len(checksum_map),
        "total": len(rows),
        "by_label": dict(labels),
        "with_image": len(image_rows),
        "with_ocr": sum(bool(row.get("ocr_present")) for row in rows),
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "artifacts" / "formal_f_benchmark_v2" / "inputs",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_root = REPO_ROOT / "benchmarks" / "mpid_standard_v2"
    smoke_rows, smoke_report = verify_split("smoke", benchmark_root / "smoke", output_dir)
    full_rows, full_report = verify_split("full", benchmark_root / "full", output_dir)

    smoke_fingerprints = fingerprints(smoke_rows)
    full_fingerprints = fingerprints(full_rows)
    mutual_overlap = {
        key: len(smoke_fingerprints[key] & full_fingerprints[key])
        for key in ("dedup_key", "source_record_id", "content_fingerprint")
    }
    if any(mutual_overlap.values()):
        raise ValueError(f"V2 smoke/full overlap detected: {mutual_overlap}")
    template_overlap = len(
        smoke_fingerprints["normalized_text_template_sha256"]
        & full_fingerprints["normalized_text_template_sha256"]
    )

    train_rows = read_jsonl(RUN_ROOT / "data" / "train.jsonl")
    train_labels = Counter(row.get("label") for row in train_rows)
    train_fingerprints = fingerprints(train_rows)
    benchmark_overlap = {}
    for split_name, split_fingerprints in (("smoke", smoke_fingerprints), ("full", full_fingerprints)):
        benchmark_overlap[split_name] = {
            key: len(train_fingerprints[key] & split_fingerprints[key])
            for key in ("dedup_key", "source_record_id", "content_fingerprint")
        }
    if any(count for split in benchmark_overlap.values() for count in split.values()):
        raise ValueError(f"training/V2 overlap detected: {benchmark_overlap}")

    groups: dict[str, set[str]] = {}
    for row in train_rows:
        pair_id = (row.get("metadata") or {}).get("pair_id")
        if pair_id:
            groups.setdefault(pair_id, set()).add(str(row.get("label")))
    complete_triplets = sum(labels == set(LABELS) for labels in groups.values())
    train_report = {
        "train_jsonl": str((RUN_ROOT / "data" / "train.jsonl").resolve()),
        "total": len(train_rows),
        "by_label": dict(train_labels),
        "pair_groups": len(groups),
        "complete_triplets": complete_triplets,
        "records_in_complete_triplets": complete_triplets * 3,
        "v2_overlap": benchmark_overlap,
    }
    if len(train_rows) != 3000 or dict(train_labels) != {label: 1000 for label in LABELS} or complete_triplets != 240:
        raise ValueError(f"unexpected F training facts: {train_report}")

    integrity = {
        "status": "pass",
        "benchmark_root": str(benchmark_root.resolve()),
        "smoke": smoke_report,
        "full": full_report,
        "smoke_full_overlap": mutual_overlap,
        "smoke_full_reused_text_templates": template_overlap,
        "train_v2_audit": train_report,
    }
    (output_dir / "benchmark_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(integrity, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
