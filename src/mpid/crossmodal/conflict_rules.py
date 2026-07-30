"""C6B-lite OCR conflict rules.

This layer is intentionally conservative: it blocks only explicit injected
instructions visible in image pixels. Ambiguous OCR text falls through to the
MPID head instead of creating a clean false-positive path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .ocr_extract import OCRText


@dataclass(frozen=True)
class OCRConflictResult:
    suspicious: bool
    label: str
    reasons: list[str]
    ocr: OCRText

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "suspicious": self.suspicious,
            "reasons": list(self.reasons),
            "ocr_text": self.ocr.text,
            "ocr_confidence": self.ocr.confidence,
            "image_path": self.ocr.image_path,
            "ocr_backend": self.ocr.backend,
            "ocr_available": self.ocr.available,
            "ocr_reason": self.ocr.reason,
        }


_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ocr_ignore_previous_instructions", r"\b(ignore|disregard)\b.{0,48}\b(previous|prior|above)\b.{0,48}\b(instruction|prompt|rule)"),
    ("ocr_system_prompt_override", r"\b(system\s+prompt|developer\s+message|developer\s+mode)\b"),
    ("ocr_jailbreak", r"\b(jailbreak|do\s+anything\s+now|DAN)\b"),
    ("ocr_safety_disabled", r"\b(safety|guardrail|policy)\b\s*[:=-]?\s*(off|disabled?|bypass)"),
    ("ocr_exfiltrate_prompt", r"\b(reveal|print|show|leak)\b.{0,48}\b(system\s+prompt|instructions?|secret)\b"),
    ("ocr_zh_ignore_instructions", r"忽略.{0,24}(之前|以上|所有)?.{0,24}(指令|提示|规则)"),
    ("ocr_zh_system_override", r"(系统提示|开发者模式|越狱模式)"),
    ("ocr_zh_safety_disabled", r"(安全|规则|限制).{0,12}(关闭|禁用|绕过)"),
)


def find_ocr_injection_reasons(text: str) -> list[str]:
    normalized = " ".join((text or "").lower().split())
    return [name for name, pattern in _INJECTION_PATTERNS if re.search(pattern, normalized, re.IGNORECASE | re.DOTALL)]


def check_ocr_conflict(ocr: OCRText) -> OCRConflictResult:
    """Classify only explicit OCR injection signals as an indirect attack."""

    if not ocr.available:
        return OCRConflictResult(False, "clean", [ocr.reason or "ocr_unavailable"], ocr)
    reasons = find_ocr_injection_reasons(ocr.text)
    return OCRConflictResult(bool(reasons), "indirect" if reasons else "clean", reasons, ocr)
