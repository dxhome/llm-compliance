"""Local image OCR used by the C6B-lite cross-modal guard.

The extractor intentionally receives only an image value. Runtime decisions
must never consume dataset annotation fields such as ``ocr_text`` or
``source``; this keeps evaluation and deployment on the same evidence path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OCRText:
    text: str
    confidence: float | None
    image_path: str | None
    backend: str | None
    available: bool
    reason: str | None = None


def resolve_image_path(image: Any) -> Path | None:
    """Resolve packaged and legacy dataset image paths without annotations."""

    if not isinstance(image, (str, Path)):
        return None
    candidate = Path(image)
    if candidate.exists() and candidate.is_file():
        return candidate

    # Phase 1 manifests originally carried an absolute data/ path. The actual
    # offline dataset is stored under runs/_datasets, so recover by filename.
    if candidate.name:
        repo_root = Path(__file__).resolve().parents[3]
        fallback = repo_root / "runs" / "_datasets" / "mpid-v1-crossmodal" / "images" / candidate.name
        if fallback.exists() and fallback.is_file():
            return fallback
    return None


@lru_cache(maxsize=1)
def _rapid_ocr():
    from rapidocr_onnxruntime import RapidOCR

    models_dir = os.environ.get("MPID_OCR_MODELS_DIR")
    if models_dir:
        root = Path(models_dir)
        det = root / "ch_PP-OCRv3_det_infer.onnx"
        rec = root / "ch_PP-OCRv3_rec_infer.onnx"
        cls = root / "ch_ppocr_mobile_v2.0_cls_infer.onnx"
        if det.exists() and rec.exists() and cls.exists():
            return RapidOCR(
                det_model_path=str(det),
                rec_model_path=str(rec),
                cls_model_path=str(cls),
            )
    return RapidOCR()


def extract_ocr_text(image: Any) -> OCRText:
    """Read image pixels with RapidOCR and return normalized text evidence."""

    path = resolve_image_path(image)
    if path is None:
        return OCRText("", None, None, None, False, "image_unavailable")
    try:
        result, _timings = _rapid_ocr()(str(path))
    except ModuleNotFoundError:
        return OCRText("", None, str(path), None, False, "rapidocr_not_installed")
    except Exception as exc:  # OCR must fail open to the VLM fallback.
        return OCRText("", None, str(path), "rapidocr", False, f"ocr_error:{type(exc).__name__}")

    lines: list[str] = []
    confidences: list[float] = []
    for item in result or []:
        if len(item) < 3:
            continue
        value = str(item[1]).strip()
        if value:
            lines.append(value)
        try:
            confidences.append(float(item[2]))
        except (TypeError, ValueError):
            pass
    confidence = sum(confidences) / len(confidences) if confidences else None
    return OCRText("\n".join(lines), confidence, str(path), "rapidocr", True)
