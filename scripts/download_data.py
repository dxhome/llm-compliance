"""Localize the public datasets needed for MPID (Phase 0A / TP3.4).

This script downloads six public datasets from the HuggingFace Hub into
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

Network reality check: HF's xet bridge sometimes hits SSL EOFs on big
transfers; we retry each file up to 3 times before giving up.

Usage::

    python scripts/download_data.py
    python scripts/download_data.py --datasets deepset_prompt_injections
    python scripts/download_data.py --force
    python scripts/download_data.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"


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
        allow_patterns=[
            "*.csv",
            "JailBreakV_28K/figstep/**",
            "JailBreakV_28K/RedTeam_2K.csv",
            "JailBreakV_28K/JailBreakV_28K.csv",
            "README.md",
        ],
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
        description="EN, image+caption pairs (31k). We only fetch the 13 MB annotations CSV; the 4.4 GB images.zip is deferred to Phase 2.",
        # The full image set is 4.4 GB — way out of scope for P0A-3. The
        # 13 MB annotations CSV gives us all 31k captions and the image
        # filename mapping; we can sample the captions for the smoke test
        # without paying the image-download cost.
        allow_patterns=["flickr_annotations_30k.csv", "README.md"],
        approx_size_mb=13,
    ),
}

DEFAULT_DATASETS = list(DATASETS.keys())


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _download_one(spec: DatasetSpec, target_dir: Path, force: bool, dry_run: bool) -> bool:
    """Download a single dataset. Returns True on success."""
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[dl] {spec.short_name}")
    print(f"     repo_id  = {spec.repo_id}")
    print(f"     target   = {target_dir}")
    print(f"     patterns = {spec.allow_patterns}")
    print(f"     size est = {spec.approx_size_mb} MB")

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
                allow_patterns=spec.allow_patterns,
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
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token (defaults to $HF_TOKEN). All listed datasets are public.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dl] raw_dir = {args.raw_dir}")
    print(f"[dl] datasets = {args.datasets}")

    n_ok = 0
    n_total = len(args.datasets)
    for name in args.datasets:
        spec = DATASETS[name]
        target = args.raw_dir / spec.short_name
        ok = _download_one(spec, target, args.force, args.dry_run)
        if ok:
            n_ok += 1

    print(f"\n[dl] Summary: {n_ok}/{n_total} dataset(s) OK")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
