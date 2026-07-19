"""Lightweight C5 rule pre-filter package."""

from __future__ import annotations

from .engine import RuleMatch, RuleResult, scan_text

__all__ = ["RuleMatch", "RuleResult", "scan_text"]
