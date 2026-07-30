"""Lightweight C6 cross-modal consistency package."""

from __future__ import annotations

from .heuristic import CrossModalResult, check_crossmodal
from .conflict_rules import OCRConflictResult, check_ocr_conflict
from .ocr_extract import OCRText, extract_ocr_text

__all__ = [
    "CrossModalResult",
    "OCRConflictResult",
    "OCRText",
    "check_crossmodal",
    "check_ocr_conflict",
    "extract_ocr_text",
]
