"""Localize the SmolVLM-500M model (Phase 0A / TP2.2).

Downloads every file the model needs (config, weights, tokenizer, processor,
chat template) into ``models/smolvlm-500m/`` so the rest of the pipeline can
load with ``local_files_only=True`` and have **zero network dependency** at
inference time (TP2.5).

Why not stream and load from the hub? Two reasons:

  1. The Phase 6 / Phase 2 acceptance explicitly requires the offline package
     to be redistributable (``scripts/package_offline.py``). Streaming would
     bake in a runtime HF dependency.
  2. We want the smoke to fail *loudly* if the local copy is incomplete,
     rather than silently re-downloading a corrupt file.

The script is **idempotent**: re-running it on an up-to-date cache is a no-op.
Pass ``--force`` to re-download even when files are present.

Usage::

    python scripts/download_models.py                       # default 500M
    python scripts/download_models.py --model-id <hf-id>    # override
    python scripts/download_models.py --force               # re-download
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEFAULT_LOCAL_DIR = REPO_ROOT / "models" / "smolvlm-500m"

# File patterns to fetch. Using allow-list (not ignore-patterns) so we never
# accidentally miss a required file. The list is validated against the hub
# before download; missing files are reported as warnings, not errors, in
# case the model card has been updated.
REQUIRED_PATTERNS = [
    "*.json",       # config / generation_config / preprocessor_config
    "*.txt",        # chat templates, license, use
    "*.md",         # model card
    "*.safetensors",# weights
    "*.tiktoken",   # tokenizer assets
    "*.py",         # custom modeling code (Idefics3 etc.)
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id (default: {DEFAULT_MODEL_ID})",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Local directory to populate (default: {DEFAULT_LOCAL_DIR})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local files are present.",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token (defaults to $HF_TOKEN). Most SmolVLM models are public.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually fetching.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    target: Path = args.local_dir
    target.mkdir(parents=True, exist_ok=True)

    # Heavy imports are deferred so ``--help`` is fast.
    from huggingface_hub import snapshot_download, HfApi

    api = HfApi(token=args.token)
    print(f"[download] model_id = {args.model_id}")
    print(f"[download] local    = {target}")
    print(f"[download] allow    = {len(REQUIRED_PATTERNS)} patterns")

    # List remote files so the user can see what we are about to pull.
    try:
        files = api.list_repo_files(args.model_id, token=args.token)
    except Exception as e:
        print(f"[download] ERROR: cannot list files for {args.model_id}: {e}")
        return 1

    matched = [f for f in files if any(Path(f).match(pat) for pat in REQUIRED_PATTERNS)]
    print(f"[download] remote matched {len(matched)}/{len(files)} files:")
    for f in matched[:10]:
        size_note = ""
        print(f"             - {f}{size_note}")
    if len(matched) > 10:
        print(f"             ... and {len(matched) - 10} more")

    if args.dry_run:
        print("[download] dry-run mode, exiting before snapshot_download")
        return 0

    # Idempotent: if a previous run already wrote all matched files, we are done
    # unless --force is set.
    if not args.force and target.exists():
        local_files = {p.name for p in target.rglob("*") if p.is_file()}
        missing = [f for f in matched if Path(f).name not in local_files]
        if not missing and matched:
            print(f"[download] local copy already complete ({len(matched)} files), nothing to do")
            return 0
        if missing:
            print(f"[download] local copy incomplete, missing {len(missing)} file(s):")
            for m in missing[:5]:
                print(f"             - {m}")
            if len(missing) > 5:
                print(f"             ... and {len(missing) - 5} more")

    try:
        snapshot_download(
            repo_id=args.model_id,
            local_dir=str(target),
            allow_patterns=REQUIRED_PATTERNS,
            token=args.token,
            force_download=args.force,
        )
    except Exception as e:
        print(f"[download] ERROR during snapshot_download: {type(e).__name__}: {e}")
        return 1

    # Final report.
    n_files = sum(1 for _ in target.rglob("*") if _.is_file())
    total_bytes = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(f"[download] OK: {n_files} files, {total_bytes / 1e6:.1f} MB on disk at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
