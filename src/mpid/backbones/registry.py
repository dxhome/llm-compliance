"""Backbone registry (Phase 2 / T2.2).

A flat dict mapping a short name to the local directory containing the
model weights. The VLM adapter (``mpid.adapters.vlm.VLMAdapter``) calls
``resolve_local_path()`` to translate ``"smolvlm-500m"`` →
``models/smolvlm-500m/`` without ever touching the network.

Adding a new backbone is one line of code: append to ``REGISTRY``.

We deliberately keep this minimal (no auto-download, no metadata
schema) because Phase 2 only needs one model. The C4 / C5 phases
add an ``EarlyExitHead`` that reads ``num_layers`` from the same
config — that's the only reason the registry exposes per-backbone
metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_ROOT = REPO_ROOT / "runs" / "_models"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# A backbone entry is just a directory name. We keep the structure
# deliberately flat so that an operator can drop a new model in
# ``models/<short_name>/`` and add one line below.
REGISTRY: dict[str, str] = {
    # Phase 2 default. ~5e8 params, Apache 2.0, HF repo:
    # HuggingFaceTB/SmolVLM-500M-Instruct.
    "smolvlm-500m": "smolvlm-500m",
}


def list_backbones() -> list[str]:
    """Return the registered backbone names."""
    return list(REGISTRY.keys())


def resolve_local_path(
    name: str,
    *,
    models_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Return the absolute path to the local copy of ``name``.

    Raises ``FileNotFoundError`` if the directory is missing; the
    operator should re-run ``scripts/download_models.py`` to fetch it.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"unknown backbone {name!r}; known: {list_backbones()}"
        )
    root = Path(models_root) if models_root else DEFAULT_MODELS_ROOT
    p = root / REGISTRY[name]
    if not p.exists():
        raise FileNotFoundError(
            f"backbone {name!r} not found at {p}. "
            f"Run `python scripts/download_models.py` to fetch it."
        )
    # Sanity: at least config.json should be present.
    if not (p / "config.json").exists():
        raise FileNotFoundError(
            f"backbone dir {p} exists but is missing config.json — "
            f"the download is corrupted; re-run download_models.py."
        )
    return p


__all__ = ["REGISTRY", "list_backbones", "resolve_local_path",
           "DEFAULT_MODELS_ROOT"]
