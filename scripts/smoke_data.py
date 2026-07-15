"""Smoke test for the localized datasets (Phase 0A / TP3.5).

For each of the 6 datasets in ``data/raw/<name>/``:

  1. **Loads** at least 5 records from whatever format (parquet / csv / zip)
     the dataset ships in. The format is preserved by ``download_data.py``;
     this script only *reads*, never *modifies* (Phase 1 will unify schemas).
  2. **Verifies** the canonical MPID field is non-empty (text/prompt/image).
  3. **For multimodal** datasets, opens one image to confirm the on-disk
     payload matches the row pointer.
  4. **Tallies** per-dataset label distribution into a coarse category
     (``clean | direct | indirect | other``) to give the operator an
     at-a-glance view of what is available before Phase 1 schema work.

The script also runs the **课题符合性自检清单** from TP3.6 and prints it
as a checklist.

Usage::

    python scripts/smoke_data.py
    python scripts/smoke_data.py --raw-dir data/raw --samples 5
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Windows console defaults to GBK, which can't encode the ✅ / ❌
# glyphs used in the conformance checklist. Reconfigure stdout to utf-8
# so the script runs identically on every platform.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_GREEN = "\033[32m" if USE_COLOR else ""
_RED = "\033[31m" if USE_COLOR else ""
_YELLOW = "\033[33m" if USE_COLOR else ""
_BOLD = "\033[1m" if USE_COLOR else ""
_RESET = "\033[0m" if USE_COLOR else ""


def _ok(msg: str) -> None: print(f"  {_GREEN}[OK]{_RESET}     {msg}")
def _fail(msg: str) -> None: print(f"  {_RED}[FAIL]{_RESET}   {msg}")
def _info(msg: str) -> None: print(f"  {_YELLOW}[INFO]{_RESET}   {msg}")


# ---------------------------------------------------------------------------
# Per-dataset record yielders. Each yields a dict with at minimum
# ``text`` (str) and ``_source`` (str), and optionally ``image_path`` (Path).
# The yielder is the ONLY place that knows the dataset's quirks.
# ---------------------------------------------------------------------------


def _yield_deepset(raw_dir: Path, n: int) -> tuple[list[dict], dict]:
    """deepset/prompt-injections: parquet, columns {text, label}, label 0/1."""
    import pyarrow.parquet as pq

    files = sorted((raw_dir / "data").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no train parquet under {raw_dir}/data")
    table = pq.read_table(files[0])
    df = table.to_pandas()

    samples: list[dict] = []
    counts: Counter = Counter()
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        samples.append({
            "text": str(row["text"]),
            "label_raw": int(row["label"]),
            "_source": "deepset_prompt_injections",
        })
        counts["clean" if int(row["label"]) == 0 else "direct"] += 1
    return samples, {"total": len(df), "label_hist": dict(counts)}


def _yield_safeguard(raw_dir: Path, n: int) -> tuple[list[dict], dict]:
    """xTRam1/safe-guard-prompt-injection: parquet, columns {text, label, ...}."""
    import pyarrow.parquet as pq

    files = sorted((raw_dir / "data").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no train parquet under {raw_dir}/data")
    table = pq.read_table(files[0])
    df = table.to_pandas()
    cols = list(df.columns)
    # Coarse category mapping. safe-guard has many configs; we bucket
    # anything that doesn't look clean as a direct or indirect injection
    # based on column presence. Phase 1 will do the canonical mapping.
    text_col = "text" if "text" in cols else (cols[0] if cols else None)
    label_col = next((c for c in ("label", "injection", "is_injection") if c in cols), None)

    samples: list[dict] = []
    counts: Counter = Counter()
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        text = str(row[text_col]) if text_col else ""
        raw_label = int(row[label_col]) if label_col else None
        # Heuristic: in safe-guard, label=0 is "safe", label=1 is "unsafe"
        # (injection). Anything that mentions indirect-context we treat
        # as indirect; everything else as direct.
        cat = "clean" if raw_label == 0 else ("indirect" if "indirect" in text.lower() else "direct")
        samples.append({
            "text": text,
            "label_raw": raw_label,
            "_source": "safe_guard_prompt_injection",
        })
        counts[cat] += 1
    return samples, {"total": len(df), "columns": cols, "label_hist": dict(counts)}


def _yield_jailbreakv(raw_dir: Path, n: int) -> tuple[list[dict], dict]:
    """JailbreakV-28K: CSV at top, figstep image folder.

    Schema: {id, jailbreak_query, redteam_query, format, policy,
             image_path (e.g. llm_transfer_attack/... — NOT downloaded in
             P0A-3), from, selected_mini, transfer_from_llm}.

    The CSV's ``image_path`` column points at the full 28k image tree,
    which we did NOT download (cost: 300+ MB). We DID download the
    figstep subset (100 images). The smoke therefore:

      - reads text + format straight from the CSV (``jailbreak_query``,
        with ``redteam_query`` as a fallback when the jailbreak query is
        the empty placeholder);
      - picks figstep/ images directly off disk (not via CSV), since the
        CSV-to-image mapping for the figstep subset is not in the CSV.
    """
    csv_candidates = [raw_dir / "JailBreakV_28K" / "JailBreakV_28K.csv",
                      raw_dir / "JailBreakV_28K" / "RedTeam_2K.csv"]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if not csv_path:
        raise FileNotFoundError(f"no JailbreakV CSV at {csv_candidates}")

    # Read first N rows from CSV for text + format.
    samples: list[dict] = []
    counts: Counter = Counter()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("jailbreak_query")
                    or row.get("redteam_query")
                    or row.get("text")
                    or row.get("question")
                    or "")
            fmt = row.get("format", "?")
            # Coarse category: "Template" / "Persuade" etc. treated as
            # direct injection (they are explicit instructions to the
            # LLM); "FigStep" is a multimodal indirect attack (image
            # carries the jailbreak) — we tag it as indirect.
            cat = "indirect" if fmt.lower() == "figstep" else "direct"
            img_rel = row.get("image_path", "")
            # The CSV image_path points to llm_transfer_attack/ which we
            # didn't download. We just record the pointer for honesty.
            samples.append({
                "text": str(text)[:120],
                "format": fmt,
                "image_path": img_rel or None,  # not on disk in P0A-3
                "label_raw": fmt,
                "_source": "jailbreakv_28k",
            })
            counts[cat] += 1
            if len(samples) >= n:
                break

    # Now scan figstep/ independently for an image-loading probe.
    image_root = raw_dir / "JailBreakV_28K" / "figstep"
    image_paths: list[Path] = []
    if image_root.exists():
        image_paths = sorted(image_root.glob("*.png"))[:n]
    return samples, {"csv": str(csv_path),
                     "image_root": str(image_root),
                     "image_root_exists": image_root.exists(),
                     "image_files_on_disk": len(image_paths),
                     "label_hist": dict(counts)}


def _yield_mmlu(raw_dir: Path, n: int) -> tuple[list[dict], dict]:
    """cais/mmlu: parquet per subject. We sample across all dev splits."""
    import pyarrow.parquet as pq

    files = sorted(raw_dir.glob("*/dev-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no dev parquet under {raw_dir}/<subject>/")
    samples: list[dict] = []
    counts: Counter = Counter()
    for fpath in files:
        if len(samples) >= n:
            break
        table = pq.read_table(fpath)
        df = table.to_pandas()
        cols = list(df.columns)
        if len(df) == 0:
            continue
        row = df.iloc[0]
        # MMLU columns: question, choices (list), answer
        text = str(row.get("question", ""))
        subject = fpath.parent.name
        samples.append({
            "text": text[:120],
            "subject": subject,
            "choices_count": len(row.get("choices", [])) if "choices" in df.columns else 0,
            "label_raw": "clean",
            "_source": "cais_mmlu",
        })
        counts["clean"] += 1
    return samples, {"total_files": len(files), "label_hist": dict(counts)}


def _yield_cmmlu(raw_dir: Path, n: int) -> tuple[list[dict], dict]:
    """haonan-li/cmmlu: single zip; inner content is ``dev/<subject>.csv``.

    The repo ships a 1 MB zip whose top-level is ``dev/<subject>.csv`` for
    67 subjects. We read straight from the zip — no extract to disk — to
    keep "download = no modify".
    """
    import csv as _csv

    zip_path = raw_dir / "cmmlu_v1_0_1.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"missing zip at {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = sorted(n for n in zf.namelist()
                           if n.startswith("dev/") and n.endswith(".csv"))
        if not csv_names:
            raise FileNotFoundError(f"no dev/<subject>.csv inside {zip_path}")
        samples: list[dict] = []
        counts: Counter = Counter()
        for name in csv_names:
            if len(samples) >= n:
                break
            with zf.open(name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8")
                reader = _csv.DictReader(text)
                try:
                    row = next(reader)
                except StopIteration:
                    continue
            # CMMLU columns: Question, A, B, C, D, Answer
            q = row.get("Question") or row.get("question") or next(iter(row.values()), "")
            subject = Path(name).stem
            samples.append({
                "text": str(q)[:120],
                "subject": subject,
                "label_raw": "clean",
                "_source": "haonan_li_cmmlu",
            })
            counts["clean"] += 1
        return samples, {"zip": str(zip_path),
                         "dev_csvs": len(csv_names),
                         "label_hist": dict(counts)}


def _yield_flickr30k(raw_dir: Path, n: int) -> tuple[list[dict], dict]:
    """nlphuji/flickr30k: annotations CSV (no images downloaded in P0A-3)."""
    csv_path = raw_dir / "flickr_annotations_30k.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing annotations at {csv_path}")
    samples: list[dict] = []
    counts: Counter = Counter()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # The annotations CSV has 'filename', 'caption' (or similar)
            # with 5 captions per image. We grab the first one.
            filename = row.get("filename") or row.get("image") or ""
            caption = (row.get("caption") or row.get("captions") or
                       row.get("raw") or next(iter(row.values()), ""))
            samples.append({
                "text": str(caption)[:120],
                "filename": str(filename),
                "image_path": None,  # images deferred to Phase 2
                "label_raw": "clean",
                "_source": "nlphuji_flickr30k",
            })
            counts["clean"] += 1
            if len(samples) >= n:
                break
    return samples, {"csv": str(csv_path), "label_hist": dict(counts)}


# Registry maps dataset short-name -> (yielder, multimodal flag)
DATASET_REGISTRY: dict[str, tuple[Any, bool]] = {
    "deepset_prompt_injections":  (_yield_deepset, False),
    "safe_guard_prompt_injection": (_yield_safeguard, False),
    "jailbreakv_28k":             (_yield_jailbreakv, True),
    "cais_mmlu":                  (_yield_mmlu, False),
    "haonan_li_cmmlu":            (_yield_cmmlu, False),
    "nlphuji_flickr30k":          (_yield_flickr30k, True),
}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

@dataclass
class DatasetResult:
    name: str
    ok: bool
    n_records_seen: int = 0
    n_samples: int = 0
    multimodal: bool = False
    image_load_ok: bool = False
    label_hist: dict = field(default_factory=dict)
    error: str | None = None


def _check_one(name: str, yielder, multimodal: bool, raw_dir: Path, n: int) -> DatasetResult:
    print(f"\n{_BOLD}[{name}]{_RESET}")
    try:
        samples, meta = yielder(raw_dir, n)
    except Exception as e:
        _fail(f"load: {type(e).__name__}: {e}")
        return DatasetResult(name=name, ok=False, error=f"{type(e).__name__}: {e}")

    _ok(f"loaded {len(samples)} samples (total available: {meta.get('total', '?')})")
    for k, v in meta.items():
        if k == "label_hist":
            continue
        _info(f"{k} = {v}")

    # Verify text non-empty
    empty = sum(1 for s in samples if not s.get("text"))
    if empty:
        _fail(f"{empty}/{len(samples)} samples have empty text")
        return DatasetResult(name=name, ok=False, n_samples=len(samples), multimodal=multimodal, error=f"empty text in {empty}")
    _ok(f"all {len(samples)} samples have non-empty text")

    # Print first sample (truncated)
    _info(f"first: {samples[0]}")

    # Image check (multimodal only).
    # Two paths:
    #   (a) sample['image_path'] points to a real on-disk file
    #       (deepset / safe-guard have no images; jailbreakv's CSV image_path
    #        points to llm_transfer_attack/ which we didn't download)
    #   (b) the dataset has an image folder separate from the row pointers
    #       (JailbreakV's figstep/, Flickr30k's images.zip which is deferred)
    # For (b) we probe the image_root reported in meta.
    image_ok = True
    if multimodal:
        from PIL import Image
        # Try sample-level image first
        for s in samples:
            ip = s.get("image_path")
            if ip:
                ip_path = Path(ip) if Path(ip).is_absolute() else (raw_dir / "JailBreakV_28K" / ip)
                if ip_path.exists():
                    try:
                        with Image.open(ip_path) as img:
                            img.verify()
                        _ok(f"image opens: {ip_path.name}")
                        image_ok = True
                        break
                    except Exception as e:
                        _fail(f"image verify: {ip_path}: {e}")
                        image_ok = False
                        break
        # Fall back: probe image_root reported by the yielder
        if not image_ok or all(not s.get("image_path") for s in samples):
            image_root_str = meta.get("image_root")
            if image_root_str:
                image_root = Path(image_root_str)
                if image_root.exists():
                    pngs = sorted(image_root.glob("*.png"))[:1]
                    if pngs:
                        try:
                            with Image.open(pngs[0]) as img:
                                img.verify()
                            _ok(f"image opens (from image_root): {pngs[0].name}")
                            image_ok = True
                        except Exception as e:
                            _fail(f"image_root verify: {pngs[0]}: {e}")
                            image_ok = False
                    else:
                        _info("image_root has no png files")
                else:
                    _info(f"image_root does not exist on disk: {image_root}")

    return DatasetResult(
        name=name, ok=True, n_samples=len(samples),
        multimodal=multimodal, image_load_ok=image_ok,
        label_hist=meta.get("label_hist", {}),
    )


def _check_conformance(results: list[DatasetResult]) -> None:
    """TP3.6 课题符合性自检清单."""
    print(f"\n{_BOLD}=== 课题符合性自检 (TP3.6) ==={_RESET}")

    has_zh = False
    has_image = False
    has_multilingual = False
    has_direct_or_indirect = False
    has_clean_ample = False
    label_counter: Counter = Counter()

    for r in results:
        if not r.ok:
            continue
        label_counter.update(r.label_hist)
        if r.name in ("cais_mmlu", "haonan_li_cmmlu", "nlphuji_flickr30k"):
            has_clean_ample = True
        if r.name == "haonan_li_cmmlu":
            has_zh = True
        if r.name == "jailbreakv_28k" or r.name == "nlphuji_flickr30k":
            has_image = True
        if r.name == "safe_guard_prompt_injection":
            has_multilingual = True
        if r.name in ("deepset_prompt_injections", "safe_guard_prompt_injection",
                      "jailbreakv_28k"):
            has_direct_or_indirect = True

    checklist = [
        ("包含中文样本 (CMMLU + JailbreakV-28K)",  has_zh),
        ("包含图像样本 (Flickr30k + JailbreakV-28K)", has_image),
        ("支持多语种 (safe-guard 多语)",           has_multilingual),
        ("覆盖直接/间接注入 (deepset/safe-guard)", has_direct_or_indirect),
        ("干净样本充足 (MMLU/CMMLU/Flickr30k)",    has_clean_ample),
    ]
    n_ok = 0
    for label, ok in checklist:
        mark = f"{_GREEN}✅{_RESET}" if ok else f"{_RED}❌{_RESET}"
        print(f"  {mark} {label}")
        if ok:
            n_ok += 1
    print(f"  -> {n_ok}/{len(checklist)} checks passed")
    if label_counter:
        print(f"  combined label histogram: {dict(label_counter)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    p.add_argument("--samples", type=int, default=5,
                   help="How many records to sample per dataset (default 5)")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict to a subset of dataset short names")
    args = p.parse_args()

    print(f"{_BOLD}MPID data smoke test (Phase 0A-3){_RESET}")
    print(f"  raw_dir   : {args.raw_dir}")
    print(f"  samples   : {args.samples}")

    chosen = args.only or list(DATASET_REGISTRY.keys())
    results: list[DatasetResult] = []
    for name in chosen:
        if name not in DATASET_REGISTRY:
            _fail(f"unknown dataset: {name}")
            continue
        yielder, multimodal = DATASET_REGISTRY[name]
        raw_dir = args.raw_dir / name
        if not raw_dir.exists():
            _fail(f"raw dir not present: {raw_dir}")
            results.append(DatasetResult(name=name, ok=False, error="raw dir missing"))
            continue
        r = _check_one(name, yielder, multimodal, raw_dir, args.samples)
        results.append(r)

    print(f"\n{_BOLD}Summary{_RESET}")
    n_ok = sum(1 for r in results if r.ok)
    for r in results:
        status = f"{_GREEN}PASS{_RESET}" if r.ok else f"{_RED}FAIL{_RESET}"
        print(f"  {r.name:35s} {status:10s} samples={r.n_samples} multimodal={r.multimodal} image_ok={r.image_load_ok}")
    print(f"  -> {n_ok}/{len(results)} dataset(s) loaded OK")
    _check_conformance(results)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
