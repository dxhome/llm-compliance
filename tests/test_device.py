"""Unit tests for ``mpid.device``.

These tests don't require an actual GPU/MPS device; they patch the low-level
detection helpers so the resolution logic can be exercised on any host.

Run with::

    .venv/bin/pytest tests/test_device.py -v
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from mpid import device as device_mod


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the module-level cache between tests."""
    device_mod.reset_cache()
    yield
    device_mod.reset_cache()


# ---------------------------------------------------------------------------
# Pure-Python detection helpers
# ---------------------------------------------------------------------------

def test_is_apple_silicon_uses_uname(monkeypatch):
    """We can fake the platform via monkeypatching the stdlib ``platform``.

    The production code now uses ``platform.system()`` / ``platform.machine()``
    (cross-platform, unlike ``os.uname`` which does not exist on Windows),
    so the test patches those two module-level callables.
    """
    import platform as _platform

    monkeypatch.setattr(device_mod, "platform", _platform)
    monkeypatch.setattr(_platform, "system", lambda: "Linux")
    monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
    assert device_mod._is_apple_silicon() is False
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_platform, "machine", lambda: "arm64")
    assert device_mod._is_apple_silicon() is True


def test_detect_device_picks_mps_on_apple_silicon(monkeypatch):
    """Apple Silicon + mps_available -> 'mps'."""
    monkeypatch.setattr(device_mod, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(device_mod, "_has_mps", lambda: True)
    monkeypatch.setattr(device_mod, "_has_cuda", lambda: False)
    assert device_mod.detect_device() == "mps"


def test_detect_device_falls_back_to_cpu(monkeypatch):
    """No MPS, no CUDA -> 'cpu'."""
    monkeypatch.setattr(device_mod, "_is_apple_silicon", lambda: False)
    monkeypatch.setattr(device_mod, "_has_mps", lambda: False)
    monkeypatch.setattr(device_mod, "_has_cuda", lambda: False)
    assert device_mod.detect_device() == "cpu"


def test_detect_device_picks_cuda_over_cpu(monkeypatch):
    """On x86 with CUDA, we prefer cuda over cpu."""
    monkeypatch.setattr(device_mod, "_is_apple_silicon", lambda: False)
    monkeypatch.setattr(device_mod, "_has_mps", lambda: False)
    monkeypatch.setattr(device_mod, "_has_cuda", lambda: True)
    assert device_mod.detect_device() == "cuda"


# ---------------------------------------------------------------------------
# Strict ``prefer`` resolution
# ---------------------------------------------------------------------------

def test_prefer_mps_strict_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(device_mod, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(device_mod, "_has_mps", lambda: True)
    assert device_mod.detect_device("mps") == "mps"


def test_prefer_mps_rejected_on_intel(monkeypatch):
    """On an Intel host we must NOT silently fall back when mps is requested."""
    monkeypatch.setattr(device_mod, "_is_apple_silicon", lambda: False)
    monkeypatch.setattr(device_mod, "_has_mps", lambda: False)
    with pytest.raises(RuntimeError, match="MPS is not usable"):
        device_mod.detect_device("mps")


def test_prefer_cuda_rejected_when_no_gpu(monkeypatch):
    monkeypatch.setattr(device_mod, "_has_cuda", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        device_mod.detect_device("cuda")


def test_prefer_cuda_index_bounds_check(monkeypatch):
    monkeypatch.setattr(device_mod, "_has_cuda", lambda: True)
    monkeypatch.setattr(device_mod, "_cuda_count", lambda: 1)
    with pytest.raises(RuntimeError, match="only 1 CUDA"):
        device_mod.detect_device("cuda:3")
    assert device_mod.detect_device("cuda:0") == "cuda:0"


def test_prefer_cpu_always_works():
    assert device_mod.detect_device("cpu") == "cpu"


def test_prefer_unknown_raises():
    with pytest.raises(ValueError, match="Unknown prefer device"):
        device_mod.detect_device("tpu")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_get_device_caches(monkeypatch):
    monkeypatch.setattr(device_mod, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(device_mod, "_has_mps", lambda: True)
    monkeypatch.setattr(device_mod, "_has_cuda", lambda: False)
    device_mod.get_device()  # populate cache
    # Flip the underlying detection; cache should still report the old value.
    monkeypatch.setattr(device_mod, "_has_mps", lambda: False)
    assert device_mod.get_device() == "mps"
    # After reset, the new detection takes effect.
    device_mod.reset_cache()
    assert device_mod.get_device() == "cpu"


def test_get_device_cache_bypassed_when_prefer_set(monkeypatch):
    """``prefer`` must always re-resolve (no cache)."""
    monkeypatch.setattr(device_mod, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(device_mod, "_has_mps", lambda: True)
    device_mod.get_device()  # caches mps
    # Now change mps availability; prefer=mps should still go through the check.
    monkeypatch.setattr(device_mod, "_has_mps", lambda: False)
    with pytest.raises(RuntimeError):
        device_mod.get_device("mps")
