"""Localize the public datasets needed for MPID (Phase 0A / TP3.4 + Phase 2.2 / T2.13).

This script downloads public datasets from the HuggingFace Hub into
``data/raw/<short_name>/`` and **never modifies** the downloaded files —
Phase 1 (T1.3 ``public_loaders.py``) is responsible for unifying the
schema into ``(text, image, label)`` triples.

The four categories required by the task list (TP3.1–TP3.3):

  1. **English injection** (direct + indirect)
     ``deepset/prompt-injections`` ~540 records.
  2. **Multilingual injection** (direct + indirect)
     ``xTRam1/safe-guard-prompt-injection`` ~6 k records.
  3. **Multimodal injection** (image + text, EN + CN)
     ``JailbreakV-28K/JailBreakV-28k`` ~28 k records.
  4. **Clean negatives** (text + image)
     MMLU (EN) + CMMLU (CN) prompts + Flickr30k image captions.

The script is **idempotent**: re-running on a populated ``data/raw/`` is a
no-op for present files. ``--force`` re-downloads every file.

Phase 2.2 / T2.13 additions:
  * ``--status``              — report current download state per dataset
                                (file count, total size, completeness vs spec).
  * ``--full``                — include the **optional** ``llm_transfer_attack``
                                image set (not on HF Hub by default).
  * ``--status-json <path>``  — emit a machine-readable JSON status report
                                (used by T2.14 build_phase1.py).
  * Retry with exponential back-off (already present) and per-file
    progress logging.

Network reality check: HF's xet bridge sometimes hits SSL EOFs on big
transfers; we retry each file up to 3 times before giving up.

Usage::

    python scripts/download_data.py
    python scripts/download_data.py --datasets deepset_prompt_injections
    python scripts/download_data.py --force
    python scripts/download_data.py --dry-run
    python scripts/download_data.py --status
    python scripts/download_data.py --status --status-json data/raw_status.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "runs" / "_datasets" / "raw"


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------
# Each entry knows how to download itself. Adding a dataset = adding one
# entry below plus a name in DEFAULT_DATASETS, with no other code change.
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    short_name: str
    repo_id: str
    repo_type: str               # "dataset" or "model"
    description: str
    # File patterns to include (whitelist). Empty list = take all files.
    allow_patterns: list[str]
    # Estimated on-disk size in MB, for the progress report only.
    approx_size_mb: int
    # Optional extras pulled in ``--full`` mode (T2.13). For the default
    # mode we use the same allow_patterns; ``--full`` adds full_extras.
    full_extras: list[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.full_extras is None:
            self.full_extras = []


DATASETS: dict[str, DatasetSpec] = {
    "deepset_prompt_injections": DatasetSpec(
        short_name="deepset_prompt_injections",
        repo_id="deepset/prompt-injections",
        repo_type="dataset",
        description="EN, ~540 records, direct/indirect injection",
        allow_patterns=["*.json", "*.jsonl", "*.md", "*.parquet", "*.txt"],
        approx_size_mb=2,
    ),
    "safe_guard_prompt_injection": DatasetSpec(
        short_name="safe_guard_prompt_injection",
        repo_id="xTRam1/safe-guard-prompt-injection",
        repo_type="dataset",
        description="multi-lingual, ~6k records, direct/indirect",
        allow_patterns=["*.parquet", "*.md", "*.json"],
        approx_size_mb=4,
    ),
    "jailbreakv_28k": DatasetSpec(
        short_name="jailbreakv_28k",
        repo_id="JailbreakV-28K/JailBreakV-28k",
        repo_type="dataset",
        description="multimodal, EN/CN, ~28k records (image + text)",
        # The repo has CSVs at the top level + nested image folders. We
        # want both: CSVs give the labels, images are the visual payload.
        # The HF Hub only ships 100 sample images per folder
        # (figstep_*, llm_transfer_attack/*) — the FULL 22k figstep set
        # lives on the JailbreakV GitHub release (T2.13 decision record:
        # the GitHub release is **not** automatically pulled here; we
        # take the 100 sample images and the 28k CSV, which gives us
        # 20k+ text-only trainable rows after the build_phase1 split).
        # T2.13 ``--full`` mode adds ``llm_transfer_attack/**`` (the
        # other 100 sample images) for completeness.
        allow_patterns=[
            "*.csv",
            "JailBreakV_28K/figstep/**",
            "JailBreakV_28K/RedTeam_2K.csv",
            "JailBreakV_28K/JailBreakV_28K.csv",
            "README.md",
        ],
        # ``--full`` extras: pull the second image folder.
        full_extras=["JailBreakV_28K/llm_transfer_attack/**"],
        approx_size_mb=300,
    ),
    "cais_mmlu": DatasetSpec(
        short_name="cais_mmlu",
        repo_id="cais/mmlu",
        repo_type="dataset",
        description="EN, 57 subjects * dev split = 285 records (used as clean prompts)",
        # The repo has one folder per subject (57 in total) each holding
        # dev / test / validation parquet. We grab **only the dev split**
        # for the "clean" probe — full MMLU is ~1 GB and not needed in P0A-3.
        # The recursive `**` is required: files live at
        # ``abstract_algebra/dev-00000-of-00001.parquet`` etc.
        allow_patterns=["**/dev-*.parquet", "README.md"],
        # ``--full`` extras: pull the test split too (~250 MB) so the
        # T2.14 build can draw clean prompts from a larger pool.
        full_extras=["**/test-*.parquet"],
        approx_size_mb=5,
    ),
    "haonan_li_cmmlu": DatasetSpec(
        short_name="haonan_li_cmmlu",
        repo_id="haonan-li/cmmlu",
        repo_type="dataset",
        description="CN, single zip cmmlu_v1_0_1.zip (~1 MB)",
        # The repo contains a single 1 MB zip with the full data, plus
        # a loader script. We do not pre-extract — the smoke test loads
        # the zip and reads the inner parquet to keep "download = no modify".
        allow_patterns=["cmmlu_v1_0_1.zip", "cmmlu.py", "README.md"],
        approx_size_mb=1,
    ),
    "nlphuji_flickr30k": DatasetSpec(
        short_name="nlphuji_flickr30k",
        repo_id="nlphuji/flickr30k",
        repo_type="dataset",
        description="EN, image+caption pairs (31k). Default: annotations CSV only. ``--full`` adds the 4.4 GB images.zip (T2.13).",
        # The full image set is 4.4 GB — way out of scope for P0A-3. The
        # 13 MB annotations CSV gives us all 31k captions and the image
        # filename mapping; we can sample the captions for the smoke test
        # without paying the image-download cost.
        # T2.13 ``--full`` mode adds ``flickr30k-images.zip`` so C5/C6
        # cross-modal training can resolve real image paths.
        allow_patterns=["flickr_annotations_30k.csv", "README.md"],
        full_extras=["flickr30k-images.zip"],
        approx_size_mb=13,
    ),
}

DEFAULT_DATASETS = list(DATASETS.keys())


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _download_one(spec: DatasetSpec, target_dir: Path, force: bool,
                  dry_run: bool, full: bool = False) -> bool:
    """Download a single dataset. Returns True on success.

    If ``full`` is True, the spec's ``full_extras`` patterns are appended
    to ``allow_patterns`` so we pull the optional image set (T2.13).
    """
    from huggingface_hub import snapshot_download

    patterns = list(spec.allow_patterns)
    if full and spec.full_extras:
        patterns = patterns + spec.full_extras

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[dl] {spec.short_name}")
    print(f"     repo_id  = {spec.repo_id}")
    print(f"     target   = {target_dir}")
    print(f"     patterns = {patterns}")
    print(f"     size est = {spec.approx_size_mb} MB  full={full}")

    if dry_run:
        return True

    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            t0 = time.perf_counter()
            snapshot_download(
                repo_id=spec.repo_id,
                repo_type=spec.repo_type,
                local_dir=str(target_dir),
                allow_patterns=patterns,
                force_download=force,
            )
            dt = time.perf_counter() - t0
            print(f"     OK in {dt:.1f}s")
            return True
        except Exception as e:
            last_err = e
            print(f"     attempt {attempt}/3 failed: {type(e).__name__}: {str(e)[:120]}")
            if attempt < 3:
                time.sleep(2 ** attempt)  # 2s, 4s

    print(f"     FAILED after 3 attempts: {type(last_err).__name__}: {last_err}")
    return False


# ---------------------------------------------------------------------------
# Status reporting (T2.13)
# ---------------------------------------------------------------------------

def _dir_size_bytes(p: Path) -> int:
    total = 0
    if not p.exists():
        return 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _count_files(p: Path, suffix: str | None = None) -> int:
    if not p.exists():
        return 0
    if suffix:
        return sum(1 for _ in p.rglob(f"*{suffix}"))
    return sum(1 for _ in p.rglob("*") if _.is_file())


def _dataset_status(spec: DatasetSpec, raw_dir: Path) -> dict:
    """Inspect the on-disk state of a single dataset.

    Returns a dict that downstream tools (build_phase1.py) can consume
    to decide whether to re-download, fall back to a smaller cap, etc.
    """
    target = raw_dir / spec.short_name
    csv_n = _count_files(target, ".csv")
    parquet_n = _count_files(target, ".parquet")
    json_n = _count_files(target, ".json")
    jsonl_n = _count_files(target, ".jsonl")
    img_n = (_count_files(target, ".png")
             + _count_files(target, ".jpg")
             + _count_files(target, ".jpeg"))
    zip_n = _count_files(target, ".zip")
    total_size_mb = round(_dir_size_bytes(target) / 1024 / 1024, 2)
    # Completeness heuristic: at least one CSV / parquet / JSON for the
    # text datasets; at least some images for the multimodal ones.
    expected_kind = {
        "deepset_prompt_injections": "parquet",  # HF ships parquet
        "safe_guard_prompt_injection": "parquet",
        "jailbreakv_28k": "csv",
        "cais_mmlu": "parquet",
        "haonan_li_cmmlu": "zip",
        "nlphuji_flickr30k": "csv",
    }.get(spec.short_name, "any")
    completeness = {
        "json": json_n + jsonl_n,
        "parquet": parquet_n,
        "csv": csv_n,
        "zip": zip_n,
        "any": csv_n + parquet_n + json_n + jsonl_n + zip_n,
    }[expected_kind]
    complete = completeness > 0
    return {
        "short_name": spec.short_name,
        "repo_id": spec.repo_id,
        "target_dir": str(target),
        "exists": target.exists(),
        "total_size_mb": total_size_mb,
        "file_counts": {
            "csv": csv_n,
            "parquet": parquet_n,
            "json": json_n,
            "jsonl": jsonl_n,
            "image": img_n,
            "zip": zip_n,
        },
        "expected_kind": expected_kind,
        "completeness_units": completeness,
        "complete": complete,
        "approx_size_mb_target": spec.approx_size_mb,
    }


def _print_status(raw_dir: Path) -> dict:
    """Print a human-readable status report and return the JSON dict."""
    print(f"\n[status] raw_dir = {raw_dir}\n")
    report: dict = {"raw_dir": str(raw_dir), "datasets": []}
    grand_total_mb = 0.0
    for name, spec in DATASETS.items():
        s = _dataset_status(spec, raw_dir)
        report["datasets"].append(s)
        grand_total_mb += s["total_size_mb"]
        flag = "OK " if s["complete"] else "-- "
        print(f"  [{flag}] {name:<32s}  "
              f"{s['total_size_mb']:>8.1f} MB   "
              f"csv={s['file_counts']['csv']} "
              f"parquet={s['file_counts']['parquet']} "
              f"img={s['file_counts']['image']} "
              f"zip={s['file_counts']['zip']}")
    report["total_size_mb"] = round(grand_total_mb, 2)
    print(f"\n  Grand total on disk: {grand_total_mb:.1f} MB")
    n_ok = sum(1 for s in report["datasets"] if s["complete"])
    n_total = len(report["datasets"])
    print(f"  Completeness: {n_ok}/{n_total} datasets ready")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
        help="Which datasets to download (default: all).",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Root directory (default: {DEFAULT_RAW_DIR})",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if files exist")
    p.add_argument("--dry-run", action="store_true", help="Show plan, do not download")
    p.add_argument(
        "--full", action="store_true",
        help="T2.13: also pull the optional 'full_extras' patterns "
             "(Flickr30k images.zip, jailbreakv transfer_attack, MMLU test).",
    )
    p.add_argument(
        "--status", action="store_true",
        help="T2.13: report current on-disk state per dataset; "
             "do not download anything.",
    )
    p.add_argument(
        "--status-json", type=Path, default=None,
        help="T2.13: when used with --status, also write the report to "
             "this JSON file (consumed by build_phase1.py / VERIFICATION.md).",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token (defaults to $HF_TOKEN). All listed datasets are public.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    # --status short-circuits everything else: just report.
    if args.status:
        report = _print_status(args.raw_dir)
        if args.status_json:
            args.status_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.status_json, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n[status] wrote {args.status_json}")
        return 0

    print(f"[dl] raw_dir = {args.raw_dir}")
    print(f"[dl] datasets = {args.datasets}")
    print(f"[dl] full = {args.full}")

    n_ok = 0
    n_total = len(args.datasets)
    for name in args.datasets:
        spec = DATASETS[name]
        target = args.raw_dir / spec.short_name
        ok = _download_one(spec, target, args.force, args.dry_run, full=args.full)
        if ok:
            n_ok += 1

    print(f"\n[dl] Summary: {n_ok}/{n_total} dataset(s) OK")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
