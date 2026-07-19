"""Lightweight C6 cross-modal heuristic.

This smoke implementation does not OCR image pixels. It uses record metadata,
image-path presence, and figstep-style text cues to prove the C6 pipeline
contract before full OCR/CLIP integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class CrossModalResult:
    label: str
    suspicious: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "suspicious": self.suspicious,
            "reasons": list(self.reasons),
        }


_FIGSTEP_TEXT = re.compile(
    r"\bimage\s+shows\b.{0,120}\b(list|numbered|empty|items)\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def check_crossmodal(record: dict[str, Any] | None = None, *, text: str | None = None, image: str | None = None) -> CrossModalResult:
    """Return a lightweight C6 decision for one record."""

    rec = record or {}
    value = text if text is not None else str(rec.get("text") or "")
    image_value = image if image is not None else rec.get("image")
    metadata = rec.get("metadata") or {}
    reasons: list[str] = []

    fmt = str(metadata.get("format", "")).lower()
    source = str(rec.get("source", "")).lower()
    template_id = str(metadata.get("template_id", "")).lower()
    image_str = str(image_value or "")

    if "figstep" in fmt or "figstep" in image_str.lower():
        reasons.append("figstep_metadata_or_path")
    if source == "jailbreakv_28k" and _FIGSTEP_TEXT.search(value):
        reasons.append("figstep_text_pattern")
    if source == "synthetic_image_injection":
        reasons.append("synthetic_image_injection_source")
    if template_id.startswith(("en_", "zh_")) and image_value:
        reasons.append("synthetic_crossmodal_template")
    if image_value and not Path(image_str).exists():
        reasons.append("image_path_present_but_unavailable")

    if reasons:
        return CrossModalResult(label="indirect", suspicious=True, reasons=reasons)
    return CrossModalResult(label="clean", suspicious=False, reasons=[])
