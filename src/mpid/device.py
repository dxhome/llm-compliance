"""Device abstraction for MPID.

Provides a single source of truth for the compute device used by every
component (backbone, head, trainer, inference). The behaviour depends on the
host platform:

  * macOS (Apple Silicon)            -> mps   (Metal Performance Shaders)
  * Linux / Windows + NVIDIA CUDA    -> cuda  (with explicit index)
  * Fallback                         -> cpu

The selection is fully automatic, but callers may force a specific device via
the ``prefer`` argument. We deliberately do NOT silently fall back from cuda
to cpu when CUDA is requested but unavailable, because that hides OOM issues
and breaks the cross-platform F1-consistency requirement from Phase 2.

This module is intentionally pure-Python and dependency-free at import time so
that ``python -c "from mpid.device import get_device"`` works even before
torch is installed (used by the env smoke test).
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Optional

# Lazy torch import — see module docstring.
_torch = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _is_apple_silicon() -> bool:
    """Detect Apple Silicon (M1/M2/.../M4). Works on macOS, Linux, and Windows.

    Uses ``platform`` stdlib (no ``os.uname``) so it is safe to call on Windows.
    """
    if platform.system() != "Darwin":
        return False
    return platform.machine() in ("arm64", "aarch64")


def _has_mps() -> bool:
    """True if MPS is compiled in and the OS supports it (>= 12.3)."""
    torch = _import_torch()
    if not hasattr(torch.backends, "mps"):
        return False
    if not torch.backends.mps.is_built():
        return False
    # torch 2.5+ raises if OS < 13 — guard against that too.
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _has_cuda() -> bool:
    torch = _import_torch()
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_count() -> int:
    torch = _import_torch()
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def detect_device(prefer: Optional[str] = None) -> str:
    """Return the best available torch device string.

    Resolution order when ``prefer`` is None:
        1. MPS on Apple Silicon
        2. CUDA on NVIDIA hosts
        3. cpu (fallback)

    When ``prefer`` is set ("mps" | "cuda" | "cuda:0" | "cpu"), we honour it
    strictly and raise if it is not actually usable. This avoids silent
    fallbacks that would skew cross-platform benchmarks.
    """
    if prefer is not None:
        p = prefer.lower()
        if p == "mps":
            if not _is_apple_silicon() or not _has_mps():
                raise RuntimeError(
                    f"prefer='mps' requested but MPS is not usable on this host"
                )
            return "mps"
        if p == "cuda":
            if not _has_cuda():
                raise RuntimeError(
                    f"prefer='cuda' requested but CUDA is not available"
                )
            return "cuda"
        if p.startswith("cuda:"):
            idx = int(p.split(":", 1)[1])
            if not _has_cuda() or idx >= _cuda_count():
                raise RuntimeError(
                    f"prefer='{p}' but only {_cuda_count()} CUDA device(s) present"
                )
            return p
        if p == "cpu":
            return "cpu"
        raise ValueError(f"Unknown prefer device: {prefer!r}")

    # Auto-detect.
    if _is_apple_silicon() and _has_mps():
        return "mps"
    if _has_cuda():
        return "cuda"
    return "cpu"


# Module-level cache so repeated calls are cheap.
_cached: Optional[str] = None


def get_device(prefer: Optional[str] = None, use_cache: bool = True) -> str:
    """Public entry point. Cached after the first successful resolve."""
    global _cached
    if use_cache and prefer is None and _cached is not None:
        return _cached
    dev = detect_device(prefer)
    if use_cache and prefer is None:
        _cached = dev
    return dev


def reset_cache() -> None:
    """Drop the cached auto-detected device (used by tests)."""
    global _cached
    _cached = None


# ---------------------------------------------------------------------------
# Introspection helpers (useful for the smoke script and for the report)
# ---------------------------------------------------------------------------

def device_summary() -> dict:
    """Return a dict describing the host compute environment."""
    torch = _import_torch()
    info = {
        "python": torch.__version__,
        "platform": platform.system(),
        "machine": platform.machine(),
        "is_apple_silicon": _is_apple_silicon(),
        "mps_built": _has_mps(),
        "mps_available": _has_mps(),
        "cuda_available": _has_cuda(),
        "cuda_count": _cuda_count() if _has_cuda() else 0,
        "selected": get_device(),
    }
    return info


def _git_head() -> Optional[str]:
    """Best-effort git HEAD SHA, used for the env fingerprint in reports."""
    if not shutil.which("git"):
        return None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


__all__ = ["get_device", "detect_device", "device_summary", "reset_cache"]
