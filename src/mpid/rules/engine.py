"""Lightweight C5 rule pre-filter.

The lightweight version is intentionally deterministic and dependency-free:
it catches obvious prompt-injection strings before the VLM fallback and
returns explainable rule hits for smoke validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class RuleMatch:
    """One explainable rule hit."""

    rule_id: str
    category: str
    label: str
    severity: str
    snippet: str


@dataclass(frozen=True)
class RuleResult:
    """Aggregated C5 decision for a single text input."""

    label: str
    action: str
    matches: list[RuleMatch] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "action": self.action,
            "blocked": self.blocked,
            "matches": [m.__dict__ for m in self.matches],
        }


_DIRECT_PATTERNS: list[tuple[str, str]] = [
    ("direct.ignore_previous", r"\bignore\s+(all\s+)?(previous|prior)\s+instructions?\b"),
    ("direct.jailbreak", r"\b(jailbreak|developer mode|dan|do anything now)\b"),
    ("direct.role_hijack", r"\b(you are now|simulate|act as)\b.{0,80}\b(unrestricted|uncensored|no restrictions|oppo|livegpt)\b"),
    ("direct.policy_bypass", r"\b(bypass|disable|ignore)\b.{0,80}\b(policy|rules|safety|guidelines)\b"),
]

_SENSITIVE_PATTERNS: list[tuple[str, str]] = [
    ("sensitive.malware", r"\b(malware|trojan|phishing|keylogger|payload|exploit)\b"),
    ("sensitive.fraud", r"\b(scam|fraud|launder money|fake warranty|phishing toolkit)\b"),
]

_STRUCTURE_PATTERNS: list[tuple[str, str]] = [
    ("structure.role_prefix", r"\b(system|assistant|developer)\s*:"),
    ("structure.sudo", r"<<\s*sudo\s*>>|/system|\\system"),
]

_UNICODE_PATTERNS: list[tuple[str, str]] = [
    ("unicode.zero_width", r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]"),
]


def _snippet(text: str, match: re.Match, width: int = 48) -> str:
    start = max(match.start() - width // 2, 0)
    end = min(match.end() + width // 2, len(text))
    return text[start:end].replace("\n", "\\n")


def _scan_patterns(
    text: str,
    patterns: list[tuple[str, str]],
    *,
    category: str,
    label: str,
    severity: str,
) -> list[RuleMatch]:
    hits: list[RuleMatch] = []
    for rule_id, pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            hits.append(
                RuleMatch(
                    rule_id=rule_id,
                    category=category,
                    label=label,
                    severity=severity,
                    snippet=_snippet(text, m),
                )
            )
    return hits


def scan_text(text: str | None) -> RuleResult:
    """Scan one text input and return an explainable C5 result."""

    value = text or ""
    matches: list[RuleMatch] = []
    matches.extend(
        _scan_patterns(
            value,
            _DIRECT_PATTERNS,
            category="keyword",
            label="direct",
            severity="high",
        )
    )
    matches.extend(
        _scan_patterns(
            value,
            _SENSITIVE_PATTERNS,
            category="sensitive",
            label="direct",
            severity="medium",
        )
    )
    matches.extend(
        _scan_patterns(
            value,
            _STRUCTURE_PATTERNS,
            category="structure",
            label="direct",
            severity="medium",
        )
    )
    matches.extend(
        _scan_patterns(
            value,
            _UNICODE_PATTERNS,
            category="unicode",
            label="direct",
            severity="medium",
        )
    )

    if matches:
        return RuleResult(label="direct", action="block", matches=matches)
    return RuleResult(label="clean", action="pass", matches=[])
