"""Cross-platform environment smoke test (Phase 0A / TP1.6).

Goal: one command, both platforms, that proves the tool-chain is healthy
before we start writing any model code.

It checks, in order:

  1. All hard-required imports succeed.
  2. ``mpid.device.get_device()`` resolves to the expected device family
     (mps on Apple Silicon, cuda on NVIDIA, cpu otherwise).
  3. A 2x2 tensor can be allocated and reduced on the selected device.
  4. The SmolVLM-500M tokenizer can be loaded *offline* from
     ``models/smolvlm-500m/`` if present (this only passes after P0A-2
     downloads the model; before that, step 4 is reported as ``SKIPPED``).

Exit code is non-zero if any step in 1–3 fails, so this can be wired into CI.

Usage::

    python scripts/smoke_env.py            # auto-detect device
    python scripts/smoke_env.py --prefer mps
    python scripts/smoke_env.py --prefer cpu
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make `src/` importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mpid.device import device_summary, get_device  # noqa: E402

# ---------------------------------------------------------------------------
# Pretty printing helpers (no external deps)
# ---------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_GREEN = "\033[32m" if USE_COLOR else ""
_RED = "\033[31m" if USE_COLOR else ""
_YELLOW = "\033[33m" if USE_COLOR else ""
_BOLD = "\033[1m" if USE_COLOR else ""
_RESET = "\033[0m" if USE_COLOR else ""


def _ok(msg: str) -> None:
    print(f"  {_GREEN}[OK]{_RESET}     {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}[FAIL]{_RESET}   {msg}")


def _skip(msg: str) -> None:
    print(f"  {_YELLOW}[SKIP]{_RESET}   {msg}")


# ---------------------------------------------------------------------------
# Step 1: hard import sweep
# ---------------------------------------------------------------------------

REQUIRED_IMPORTS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("tokenizers", "tokenizers"),
    ("peft", "peft"),
    ("accelerate", "accelerate"),
    ("bitsandbytes", "bitsandbytes"),
    ("datasets", "datasets"),
    ("sklearn", "sklearn"),
    ("PIL", "PIL"),
    ("yaml", "yaml"),
    ("tqdm", "tqdm"),
    ("huggingface_hub", "huggingface_hub"),
]


def step1_imports() -> bool:
    print(f"\n{_BOLD}[1/4] Required imports{_RESET}")
    all_ok = True
    for name, mod in REQUIRED_IMPORTS:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            _ok(f"{name:18s} {ver}")
        except Exception as e:  # pragma: no cover
            _fail(f"{name:18s} -> {type(e).__name__}: {e}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Step 2: device resolution
# ---------------------------------------------------------------------------

def step2_device(prefer: str | None) -> bool:
    print(f"\n{_BOLD}[2/4] Device resolution{_RESET}")
    info = device_summary()
    for k in ("python", "platform", "machine", "is_apple_silicon",
              "mps_built", "mps_available", "cuda_available",
              "cuda_count", "selected"):
        _ok(f"{k:18s} = {info[k]}")

    try:
        dev = get_device(prefer)
    except Exception as e:
        _fail(f"get_device(prefer={prefer!r}) -> {type(e).__name__}: {e}")
        return False

    _ok(f"get_device(prefer={prefer!r}) -> {dev}")
    # When ``prefer`` is set we are pinning the device; it must match the
    # explicit request exactly. When ``prefer`` is None we compare against
    # the auto-detected value cached in ``device_summary()``.
    expected = prefer if prefer is not None else info["selected"]
    if dev != expected:
        _fail(f"device mismatch: got {dev!r}, expected {expected!r}")
        return False
    return True


# ---------------------------------------------------------------------------
# Step 3: tiny tensor round-trip on the chosen device
# ---------------------------------------------------------------------------

def step3_tensor() -> bool:
    print(f"\n{_BOLD}[3/4] Tensor round-trip on selected device{_RESET}")
    try:
        import torch
    except Exception as e:
        _fail(f"import torch: {e}")
        return False

    dev = get_device()
    try:
        t0 = time.perf_counter()
        x = torch.randn(2, 2, device=dev)
        s = x.sum().item()
        dt_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        _fail(f"torch.randn(2,2) on {dev}: {type(e).__name__}: {e}")
        return False
    _ok(f"tensor device={dev} sum={s:.6f} alloc+sum={dt_ms:.2f} ms")
    return True


# ---------------------------------------------------------------------------
# Step 4: SmolVLM-500M tokenizer (only if local model is on disk)
# ---------------------------------------------------------------------------

SMOLVLM_LOCAL_DIR = REPO_ROOT / "models" / "smolvlm-500m"


def step4_tokenizer() -> bool:
    print(f"\n{_BOLD}[4/4] SmolVLM-500M tokenizer (offline){_RESET}")
    if not SMOLVLM_LOCAL_DIR.exists():
        _skip(f"local model not found at {SMOLVLM_LOCAL_DIR} (P0A-2 will provide)")
        return True
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        _fail(f"import AutoTokenizer: {e}")
        return False
    try:
        tok = AutoTokenizer.from_pretrained(str(SMOLVLM_LOCAL_DIR), local_files_only=True)
    except Exception as e:
        _fail(f"load tokenizer from {SMOLVLM_LOCAL_DIR}: {type(e).__name__}: {e}")
        return False
    out = tok("Hello, world!", return_tensors="pt")
    n_tok = int(out["input_ids"].shape[-1])
    _ok(f"tokenizer loaded; encoded 'Hello, world!' -> {n_tok} token(s)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefer",
        default=None,
        choices=[None, "mps", "cuda", "cpu"],
        help="Force a specific device (default: auto-detect).",
    )
    args = parser.parse_args()

    print(f"{_BOLD}MPID environment smoke test{_RESET}")
    print(f"  repo root : {REPO_ROOT}")
    print(f"  python    : {sys.executable}")

    results = {
        "imports":  step1_imports(),
        "device":   step2_device(args.prefer),
        "tensor":   step3_tensor(),
        "tokenizer": step4_tokenizer(),
    }

    print(f"\n{_BOLD}Summary{_RESET}")
    n_pass = sum(1 for v in results.values() if v)
    for k, v in results.items():
        status = f"{_GREEN}PASS{_RESET}" if v else f"{_RED}FAIL{_RESET}"
        print(f"  {k:10s} {status}")
    print(f"  -> {n_pass}/{len(results)} steps passed")

    # Steps 1-3 must pass; step 4 is allowed to be skipped (model not yet downloaded).
    required_pass = results["imports"] and results["device"] and results["tensor"]
    return 0 if required_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
