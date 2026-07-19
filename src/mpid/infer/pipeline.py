"""Inference pipeline orchestrators.

This module exposes two comparable runtime paths:

* ``run_lora_only_pipeline``: MPID head + LoRA generation, without C4/C5/C6.
* ``run_optimized_pipeline``: C5 -> C6 -> MPID head -> C4 -> generation.

Both functions accept caller-provided ``classify_fn`` and optional
``generate_fn`` callables so the orchestration can be reused by the demo,
offline package, and future compare scripts without reloading models here.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import torch

from mpid.crossmodal import check_crossmodal
from mpid.early_exit import EarlyExitConfig, should_early_exit
from mpid.rules import scan_text


ClassifyFn = Callable[[str, Any], dict[str, Any]]
GenerateFn = Callable[[str, Any], str]


@dataclass(frozen=True)
class PipelineResult:
    label: str
    action: str
    stage: str
    explanation: dict[str, Any]
    timings: dict[str, float]
    output: str | None = None
    head: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "action": self.action,
            "stage": self.stage,
            "explanation": self.explanation,
            "timings": self.timings,
            "output": self.output,
            "head": self.head,
        }


def _parse_probs(probs: list[float] | None) -> torch.Tensor | None:
    if probs is None:
        return None
    if len(probs) != 3:
        raise ValueError("probs must contain exactly 3 values: clean,direct,indirect")
    return torch.tensor(probs, dtype=torch.float32)


def _call_timed(fn, *args, **kwargs) -> tuple[Any, float]:
    t0 = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, time.perf_counter() - t0


def _text_image(record: dict[str, Any]) -> tuple[str, Any]:
    return str(record.get("text") or ""), record.get("image")


def _generate_if_allowed(
    generate_fn: GenerateFn | None,
    text: str,
    image: Any,
    timings: dict[str, float],
) -> str | None:
    if generate_fn is None:
        return None
    output, timings["generate_seconds"] = _call_timed(generate_fn, text, image)
    return output


def run_lora_only_pipeline(
    record: dict[str, Any],
    *,
    classify_fn: ClassifyFn,
    generate_fn: GenerateFn | None = None,
) -> PipelineResult:
    """Run the baseline LoRA-only path: head -> block/allow -> generation."""

    text, image = _text_image(record)
    timings: dict[str, float] = {}

    head, timings["head_seconds"] = _call_timed(classify_fn, text, image)
    label = str(head.get("label") or "fallback")
    if label == "clean":
        output = _generate_if_allowed(generate_fn, text, image, timings)
        timings["total_seconds"] = sum(timings.values())
        return PipelineResult(
            label="clean",
            action="allow",
            stage="lora_head_clean",
            explanation={"reason": "head allowed clean"},
            timings=timings,
            output=output,
            head=head,
        )

    timings["total_seconds"] = sum(timings.values())
    return PipelineResult(
        label=label,
        action="block",
        stage="lora_head_injection",
        explanation={"reason": "head blocked injection"},
        timings=timings,
        head=head,
    )


def run_optimized_pipeline(
    record: dict[str, Any],
    *,
    classify_fn: ClassifyFn,
    generate_fn: GenerateFn | None = None,
    clean_threshold: float = 0.95,
) -> PipelineResult:
    """Run the optimized C5 -> C6 -> head -> C4 -> generation path."""

    text, image = _text_image(record)
    timings: dict[str, float] = {}

    c5, timings["c5_seconds"] = _call_timed(scan_text, text)
    if c5.blocked:
        timings["total_seconds"] = sum(timings.values())
        return PipelineResult(
            label=c5.label,
            action="block",
            stage="c5_rules",
            explanation=c5.to_dict(),
            timings=timings,
        )

    c6, timings["c6_seconds"] = _call_timed(check_crossmodal, record)
    if c6.suspicious:
        timings["total_seconds"] = sum(timings.values())
        return PipelineResult(
            label=c6.label,
            action="block",
            stage="c6_crossmodal",
            explanation=c6.to_dict(),
            timings=timings,
        )

    head, timings["head_seconds"] = _call_timed(classify_fn, text, image)
    probs = list(head.get("probs") or [])
    probs_t = _parse_probs(probs)
    if probs_t is None:
        raise ValueError("classify_fn must return probs with clean,direct,indirect")

    early, timings["c4_seconds"] = _call_timed(
        should_early_exit,
        probs_t,
        EarlyExitConfig(enabled=True, clean_threshold=clean_threshold),
    )
    if early is not None:
        output = _generate_if_allowed(generate_fn, text, image, timings)
        timings["total_seconds"] = sum(timings.values())
        return PipelineResult(
            label="clean",
            action="allow",
            stage="c4_early_exit",
            explanation={"p_clean": float(probs_t[0]), "threshold": clean_threshold},
            timings=timings,
            output=output,
            head=head,
        )

    label = str(head.get("label") or "fallback")
    if label == "clean":
        output = _generate_if_allowed(generate_fn, text, image, timings)
        timings["total_seconds"] = sum(timings.values())
        return PipelineResult(
            label="clean",
            action="allow",
            stage="head_clean_fallback",
            explanation={"reason": "C5/C6/C4 did not decide; head allowed clean"},
            timings=timings,
            output=output,
            head=head,
        )

    timings["total_seconds"] = sum(timings.values())
    return PipelineResult(
        label=label,
        action="block",
        stage="head_injection_fallback",
        explanation={"reason": "C5/C6/C4 did not decide; head blocked injection"},
        timings=timings,
        head=head,
    )
