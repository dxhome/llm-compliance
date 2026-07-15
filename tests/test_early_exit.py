"""Unit tests for ``mpid.early_exit`` (Phase 3 / T3.4).

These tests are pure-Python: they only need ``torch`` and exercise the
threshold logic + stats tracking. They do **not** require a real VLM
backbone, so they run on any host (CI / mac / x86 CPU).

Run with::

    .venv/bin/pytest tests/test_early_exit.py -v
"""
from __future__ import annotations

import torch

from mpid.early_exit import (
    EarlyExitConfig,
    EarlyExitResult,
    EarlyExitStats,
    should_early_exit,
)
from mpid.heads.classification import LABEL2IDX


# ---------------------------------------------------------------------------
# should_early_exit
# ---------------------------------------------------------------------------

def test_should_early_exit_disabled_returns_none():
    """Disabled C4 should never exit, regardless of P(clean)."""
    cfg = EarlyExitConfig(enabled=False, clean_threshold=0.0)
    probs = torch.tensor([0.99, 0.005, 0.005])  # all "clean"
    assert should_early_exit(probs, cfg) is None


def test_should_early_exit_high_clean_returns_clean():
    """P(clean) = 0.99, threshold = 0.95 → exit with 'clean'."""
    cfg = EarlyExitConfig(enabled=True, clean_threshold=0.95)
    probs = torch.tensor([0.99, 0.005, 0.005])
    assert should_early_exit(probs, cfg) == "clean"


def test_should_early_exit_low_clean_returns_none():
    """P(clean) = 0.50, threshold = 0.95 → no exit."""
    cfg = EarlyExitConfig(enabled=True, clean_threshold=0.95)
    probs = torch.tensor([0.50, 0.30, 0.20])
    assert should_early_exit(probs, cfg) is None


def test_should_early_exit_boundary_above():
    """P(clean) = 0.951, threshold = 0.95 → exit (strict '>' is per spec)."""
    cfg = EarlyExitConfig(enabled=True, clean_threshold=0.95)
    probs = torch.tensor([0.951, 0.04, 0.009])
    assert should_early_exit(probs, cfg) == "clean"


def test_should_early_exit_boundary_at_threshold_no_exit():
    """P(clean) = 0.95, threshold = 0.95 → no exit (strict '>' is per spec).

    Boundary semantics: P(clean) > threshold, so P == threshold does NOT
    trigger exit. This is intentional — at exactly the threshold the
    confidence is "not enough" by the strict-greater definition.
    """
    cfg = EarlyExitConfig(enabled=True, clean_threshold=0.95)
    probs = torch.tensor([0.95, 0.03, 0.02])
    assert should_early_exit(probs, cfg) is None


def test_should_early_exit_batched_input():
    """Input shape (1, 3) should be handled the same as (3,)."""
    cfg = EarlyExitConfig(enabled=True, clean_threshold=0.90)
    probs = torch.tensor([[0.95, 0.03, 0.02]])
    assert should_early_exit(probs, cfg) == "clean"


def test_should_early_exit_direct_does_not_exit():
    """Even with a high clean threshold, a direct injection (low P(clean)) should not exit."""
    cfg = EarlyExitConfig(enabled=True, clean_threshold=0.80)
    probs = torch.tensor([0.10, 0.85, 0.05])  # clearly direct
    assert should_early_exit(probs, cfg) is None


def test_should_early_exit_indirect_does_not_exit():
    """An indirect injection (low P(clean)) should not exit either."""
    cfg = EarlyExitConfig(enabled=True, clean_threshold=0.80)
    probs = torch.tensor([0.15, 0.05, 0.80])
    assert should_early_exit(probs, cfg) is None


# ---------------------------------------------------------------------------
# EarlyExitStats
# ---------------------------------------------------------------------------

def test_stats_to_dict_basic():
    """A stats object with no observations should serialize to a valid dict."""
    stats = EarlyExitStats()
    d = stats.to_dict()
    assert d["n_total"] == 0
    assert d["n_exited"] == 0
    assert d["exit_rate"] == 0.0
    assert d["n_clean_wrong_exit"] == 0
    # All per-class fields should exist with 0 counts.
    for lbl in ("clean", "direct", "indirect"):
        assert lbl in d["per_class_exits"]
        assert lbl in d["per_class_total"]
        assert d["per_class_exit_rate"][lbl] == 0.0


def test_stats_manual_update():
    """A stats object should accumulate counts correctly."""
    stats = EarlyExitStats()
    # 10 clean samples, 8 of which exit (correct)
    stats.n_total = 10
    stats.n_exited = 8
    stats.n_clean_exited = 8
    stats.n_clean_wrong_exit = 0
    stats.per_class_exits["clean"] = 8
    stats.per_class_total["clean"] = 10
    stats.latency_full_ms = 1500.0  # 2 non-exit samples * 750ms
    stats.latency_exit_ms = 240.0   # 8 exit samples * 30ms

    d = stats.to_dict()
    assert d["n_total"] == 10
    assert d["n_exited"] == 8
    assert d["exit_rate"] == 0.8
    # 10 samples, 8 exited, 2 went through full
    assert d["avg_latency_full_ms"] == 750.0
    assert d["avg_latency_exit_ms"] == 30.0
    # Per-class
    assert d["per_class_exits"]["clean"] == 8
    assert d["per_class_exit_rate"]["clean"] == 0.8


def test_stats_per_class_independent():
    """Per-class exit rates should be tracked independently."""
    stats = EarlyExitStats()
    stats.n_total = 30
    stats.n_exited = 18
    # 20 clean, 16 exit
    stats.per_class_exits["clean"] = 16
    stats.per_class_total["clean"] = 20
    # 5 direct, 2 exit (false positive — direct was wrongly exited as clean)
    stats.per_class_exits["direct"] = 2
    stats.per_class_total["direct"] = 5
    # 5 indirect, 0 exit
    stats.per_class_exits["indirect"] = 0
    stats.per_class_total["indirect"] = 5
    stats.n_clean_wrong_exit = 2  # 2 direct samples were exited as clean (漏报)

    d = stats.to_dict()
    assert d["per_class_exit_rate"]["clean"] == 0.8
    assert d["per_class_exit_rate"]["direct"] == 0.4
    assert d["per_class_exit_rate"]["indirect"] == 0.0
    assert d["n_clean_wrong_exit"] == 2


# ---------------------------------------------------------------------------
# EarlyExitResult
# ---------------------------------------------------------------------------

def test_result_construction():
    """EarlyExitResult should be a plain dataclass — all fields settable."""
    r = EarlyExitResult(
        label="clean",
        probs={"clean": 0.98, "direct": 0.01, "indirect": 0.01},
        exited=True,
        latency_ms=42.0,
        p_clean=0.98,
    )
    assert r.label == "clean"
    assert r.exited is True
    assert r.p_clean == 0.98
    assert r.latency_ms == 42.0
    assert sum(r.probs.values()) == pytest_approx(1.0)


def test_result_label_order_matches_classification():
    """probs dict keys should match LABEL_ORDER."""
    cfg = EarlyExitConfig(enabled=True, clean_threshold=0.95)
    probs = torch.tensor([0.99, 0.005, 0.005])
    r = EarlyExitResult(
        label=should_early_exit(probs, cfg),
        probs={"clean": 0.99, "direct": 0.005, "indirect": 0.005},
        exited=should_early_exit(probs, cfg) is not None,
        latency_ms=10.0,
        p_clean=0.99,
    )
    assert r.label == "clean"
    # The classification module's LABEL2IDX is the canonical ordering.
    assert LABEL2IDX["clean"] == 0
    assert LABEL2IDX["direct"] == 1
    assert LABEL2IDX["indirect"] == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pytest_approx(x, tol=1e-6):
    """Tiny replacement for pytest.approx — keeps the test file self-contained."""
    class _A:
        def __init__(self, val, tol): self.val, self.tol = val, tol
        def __eq__(self, other): return abs(self.val - other) < self.tol
        def __repr__(self): return f"approx({self.val})"
    return _A(x, tol)
