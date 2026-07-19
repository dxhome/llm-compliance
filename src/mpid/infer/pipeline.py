"""Lightweight C4/C5/C6 inference pipeline.

This module is a smoke-friendly orchestrator. It does not load the VLM; callers
may pass precomputed class probabilities to exercise C4, while C5/C6 run as
deterministic Python checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mpid.crossmodal import check_crossmodal
from mpid.early_exit import EarlyExitConfig, should_early_exit
from mpid.rules import scan_text


@dataclass(frozen=True)
class PipelineResult:
    label: str
    action: str
    stage: str
    explanation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "action": self.action,
            "stage": self.stage,
            "explanation": self.explanation,
        }


def _parse_probs(probs: list[float] | None) -> torch.Tensor | None:
    if probs is None:
        return None
    if len(probs) != 3:
        raise ValueError("probs must contain exactly 3 values: clean,direct,indirect")
    return torch.tensor(probs, dtype=torch.float32)


def run_lightweight_pipeline(
    record: dict[str, Any],
    *,
    probs: list[float] | None = None,
    clean_threshold: float = 0.95,
) -> PipelineResult:
    """Run C4 -> C5 -> C6 -> fallback for one record."""

    probs_t = _parse_probs(probs)
    if probs_t is not None:
        early = should_early_exit(
            probs_t,
            EarlyExitConfig(enabled=True, clean_threshold=clean_threshold),
        )
        if early is not None:
            return PipelineResult(
                label="clean",
                action="allow",
                stage="c4_early_exit",
                explanation={"p_clean": float(probs_t[0]), "threshold": clean_threshold},
            )

    c5 = scan_text(record.get("text", ""))
    if c5.blocked:
        return PipelineResult(
            label=c5.label,
            action="block",
            stage="c5_rules",
            explanation=c5.to_dict(),
        )

    c6 = check_crossmodal(record)
    if c6.suspicious:
        return PipelineResult(
            label=c6.label,
            action="block",
            stage="c6_crossmodal",
            explanation=c6.to_dict(),
        )

    return PipelineResult(
        label="fallback",
        action="defer_to_vlm",
        stage="vlm_head_fallback",
        explanation={"reason": "no C4/C5/C6 lightweight decision"},
    )
