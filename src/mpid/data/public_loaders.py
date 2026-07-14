"""Unified public-dataset loaders (Phase 1 / T1.3).

Each function reads from ``data/raw/<short_name>/`` (preserved verbatim
by ``scripts/download_data.py``) and yields records in the canonical
MPID schema::

    {
        "id":       str,            # unique within dataset
        "text":     str,            # the prompt / query
        "image":    Path | None,    # absolute path or None
        "label":    "clean" | "direct" | "indirect",
        "source":   "<short_name>",
        "lang":     "en" | "zh" | "multi" | "unknown",
        "metadata": dict,           # raw fields, for debugging
    }

The label mapping follows ``doc/reference.md`` § 2.1.6 (威胁模型) and § 2.2.3 (数据集对应).
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Public schema
# ---------------------------------------------------------------------------

VALID_LABELS = frozenset({"clean", "direct", "indirect"})


@dataclass
class Record:
    """Canonical MPID record (mirrors the dict schema in the docstring)."""

    id: str
    text: str
    image: Optional[Path]
    label: str
    source: str
    lang: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in VALID_LABELS:
            raise ValueError(
                f"invalid label {self.label!r} for id={self.id!r}; "
                f"expected one of {sorted(VALID_LABELS)}"
            )
        if not self.text or not isinstance(self.text, str):
            raise ValueError(f"text must be non-empty str for id={self.id!r}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "image": str(self.image) if self.image else None,
            "label": self.label,
            "source": self.source,
            "lang": self.lang,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Language detection (heuristic; no langdetect dep)
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def detect_lang(text: str) -> str:
    """Coarse language tag from text content.

    Returns ``"zh"`` if any CJK character is present, otherwise ``"en"``.
    The safe-guard dataset is genuinely multi-lingual; its language is
    tagged separately in ``load_safeguard`` based on metadata, not here.
    """
    if not text:
        return "unknown"
    return "zh" if _CJK_RE.search(text) else "en"


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------

def load_deepset(raw_dir: Path, max_records: int | None = None) -> Iterator[Record]:
    """deepset/prompt-injections: parquet, columns {text, label ∈ {0,1}}."""
    import pyarrow.parquet as pq

    files = sorted((raw_dir / "data").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no train parquet under {raw_dir}/data")
    table = pq.read_table(files[0])
    df = table.to_pandas()
    n = len(df)
    for i, row in df.iterrows():
        if max_records is not None and i >= max_records:
            break
        text = str(row["text"]).strip()
        if not text:
            continue
        yield Record(
            id=f"deepset_{i:06d}",
            text=text,
            image=None,
            label="clean" if int(row["label"]) == 0 else "direct",
            source="deepset_prompt_injections",
            lang=detect_lang(text),
            metadata={"label_raw": int(row["label"])},
        )


def load_safeguard(raw_dir: Path, max_records: int | None = None) -> Iterator[Record]:
    """xTRam1/safe-guard-prompt-injection: parquet, columns {text, label ∈ {0,1}}.

    The dataset does NOT expose an ``injection_type`` field, so the
    direct/indirect split is a text-keyword heuristic ("indirect" in the
    prompt). The EDA report (T1.6) surfaces the resulting distribution so
    that downstream phases can decide whether to drop or re-label.
    """
    import pyarrow.parquet as pq

    files = sorted((raw_dir / "data").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no train parquet under {raw_dir}/data")
    table = pq.read_table(files[0])
    df = table.to_pandas()
    cols = list(df.columns)
    text_col = "text" if "text" in cols else cols[0]
    label_col = next((c for c in ("label", "injection", "is_injection") if c in cols), None)

    for i, row in df.iterrows():
        if max_records is not None and i >= max_records:
            break
        text = str(row[text_col]).strip() if text_col else ""
        if not text:
            continue
        raw_label = int(row[label_col]) if label_col else None
        if raw_label == 0:
            label = "clean"
        elif "indirect" in text.lower():
            label = "indirect"
        else:
            label = "direct"
        yield Record(
            id=f"safeguard_{i:06d}",
            text=text,
            image=None,
            label=label,
            source="safe_guard_prompt_injection",
            lang=detect_lang(text),
            metadata={"label_raw": raw_label},
        )


def load_jailbreakv(
    raw_dir: Path,
    max_records: int | None = None,
    link_figstep: bool = True,
) -> Iterator[Record]:
    """JailbreakV-28K: CSV at top, figstep images on disk.

    The CSV's ``image_path`` points to ``llm_transfer_attack/...`` which
    was not downloaded in P0A-3. We independently sample figstep images
    for those rows where ``format=FigStep`` (the genuine indirect-injection
    subset) so that the VLM has a real image payload for cross-modal
    training (C6). All other rows get ``image=None`` in this phase.
    """
    csv_candidates = [raw_dir / "JailBreakV_28K" / "JailBreakV_28K.csv",
                      raw_dir / "JailBreakV_28K" / "RedTeam_2K.csv"]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if not csv_path:
        raise FileNotFoundError(f"no JailbreakV CSV at {csv_candidates}")

    figstep_dir = raw_dir / "JailBreakV_28K" / "figstep"
    figstep_pool: list[Path] = (sorted(figstep_dir.glob("*.png")) if figstep_dir.exists() else [])

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_records is not None and i >= max_records:
                break
            text = (row.get("jailbreak_query")
                    or row.get("redteam_query")
                    or row.get("text")
                    or row.get("question")
                    or "").strip()
            if not text:
                continue
            fmt = (row.get("format") or "").strip()
            is_indirect = fmt.lower() == "figstep"
            image: Optional[Path] = None
            if link_figstep and is_indirect and figstep_pool:
                # Round-robin over the 100 figstep images. The mapping
                # is intentionally non-deterministic across runs unless
                # we sort by name; for reproducibility we just take the
                # i-th file. This is fine for P0A-3 (we only have 100
                # images for ~28k rows; coverage is < 1% anyway).
                image = figstep_pool[i % len(figstep_pool)]
            yield Record(
                id=f"jailbreakv_{i:06d}",
                text=text,
                image=image,
                label="indirect" if is_indirect else "direct",
                source="jailbreakv_28k",
                lang=detect_lang(text),
                metadata={"format": fmt,
                          "csv_image_path": row.get("image_path", "")},
            )


def load_mmlu(raw_dir: Path, max_records: int | None = None) -> Iterator[Record]:
    """cais/mmlu: parquet per subject, dev split only.

    Yields each (subject, question) pair as a clean record. The full
    ``choices`` list is preserved in ``metadata`` for Phase 1 EDA but
    is not fed to the 3-class head.
    """
    import pyarrow.parquet as pq

    files = sorted(raw_dir.glob("*/dev-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no dev parquet under {raw_dir}/<subject>/")
    counter = 0
    for fpath in files:
        table = pq.read_table(fpath)
        df = table.to_pandas()
        subject = fpath.parent.name
        for _, row in df.iterrows():
            if max_records is not None and counter >= max_records:
                return
            text = str(row.get("question", "")).strip()
            if not text:
                continue
            choices = list(row.get("choices", [])) if "choices" in df.columns else []
            yield Record(
                id=f"mmlu_{counter:06d}",
                text=text,
                image=None,
                label="clean",
                source="cais_mmlu",
                lang=detect_lang(text),
                metadata={"subject": subject,
                          "choices_count": len(choices)},
            )
            counter += 1


def load_cmmlu(raw_dir: Path, max_records: int | None = None) -> Iterator[Record]:
    """haonan-li/cmmlu: single zip; inner content is ``dev/<subject>.csv``."""
    zip_path = raw_dir / "cmmlu_v1_0_1.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"missing zip at {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = sorted(n for n in zf.namelist()
                           if n.startswith("dev/") and n.endswith(".csv"))
        if not csv_names:
            raise FileNotFoundError(f"no dev/<subject>.csv inside {zip_path}")
        counter = 0
        for name in csv_names:
            if max_records is not None and counter >= max_records:
                return
            subject = Path(name).stem
            with zf.open(name) as fh:
                text_stream = io.TextIOWrapper(fh, encoding="utf-8")
                reader = csv.DictReader(text_stream)
                for row in reader:
                    if max_records is not None and counter >= max_records:
                        return
                    q = (row.get("Question")
                         or row.get("question")
                         or next(iter(row.values()), "")
                         or "").strip()
                    if not q:
                        continue
                    yield Record(
                        id=f"cmmlu_{counter:06d}",
                        text=q,
                        image=None,
                        label="clean",
                        source="haonan_li_cmmlu",
                        lang=detect_lang(q),
                        metadata={"subject": subject},
                    )
                    counter += 1


def load_flickr30k(raw_dir: Path, max_records: int | None = None) -> Iterator[Record]:
    """nlphuji/flickr30k: annotations CSV only (images.zip deferred to Phase 2)."""
    csv_path = raw_dir / "flickr_annotations_30k.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing annotations at {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_records is not None and i >= max_records:
                break
            filename = (row.get("filename")
                        or row.get("image")
                        or row.get("image_name")
                        or "")
            # Some HF mirrors put the captions in a list-typed cell; we
            # always take the first one for Phase 1.
            caption = (row.get("caption")
                       or row.get("captions")
                       or row.get("raw")
                       or row.get("comment")
                       or next(iter(row.values()), ""))
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            text = str(caption).strip()
            if not text:
                continue
            yield Record(
                id=f"flickr30k_{i:06d}",
                text=text,
                image=None,  # images.zip (4.4 GB) deferred to Phase 2
                label="clean",
                source="nlphuji_flickr30k",
                lang=detect_lang(text),
                metadata={"filename": str(filename)},
            )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOADERS = {
    "deepset_prompt_injections":   load_deepset,
    "safe_guard_prompt_injection": load_safeguard,
    "jailbreakv_28k":              load_jailbreakv,
    "cais_mmlu":                   load_mmlu,
    "haonan_li_cmmlu":             load_cmmlu,
    "nlphuji_flickr30k":           load_flickr30k,
}


def load_all(
    raw_dir: Path,
    *,
    max_per_dataset: dict[str, int] | None = None,
    datasets: list[str] | None = None,
) -> Iterator[Record]:
    """Yield records from every requested dataset in ``raw_dir``.

    ``max_per_dataset`` lets the caller cap a single dataset (e.g. to
    avoid pulling 28k JailbreakV rows). If unset, all rows are yielded.
    """
    selected = datasets or list(LOADERS.keys())
    for name in selected:
        loader = LOADERS[name]
        cap = (max_per_dataset or {}).get(name)
        per_dir = raw_dir / name
        if not per_dir.exists():
            raise FileNotFoundError(f"raw dir not present: {per_dir}")
        yield from loader(per_dir, max_records=cap)


__all__ = [
    "Record",
    "VALID_LABELS",
    "LOADERS",
    "load_deepset",
    "load_safeguard",
    "load_jailbreakv",
    "load_mmlu",
    "load_cmmlu",
    "load_flickr30k",
    "load_all",
    "detect_lang",
]
