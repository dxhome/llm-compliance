"""MPID data layer (Phase 1+).

Submodules:

  - ``public_loaders``           — unified schema loaders (T1.3)
  - ``synthetic_image_injection`` — cross-modal generator (T1.4)
  - ``split``                    — 8:1:1 stratified split (T1.5)
  - ``dataset``                  — PyTorch dataset over JSONL splits (Phase 2)
  - ``prompt``                   — 3-class prompt template (T2.4)
"""
from __future__ import annotations

from mpid.data.dataset import MPIDJsonlDataset, collate
from mpid.data.prompt import (
    ANSWER_CHOICES,
    ANSWER_CLEAN,
    ANSWER_DIRECT,
    ANSWER_INDIRECT,
    PROMPT_TEMPLATE,
    build_prompt,
)
from mpid.data.public_loaders import (
    LOADERS,
    Record,
    VALID_LABELS,
    detect_lang,
    load_all,
    load_cmmlu,
    load_deepset,
    load_flickr30k,
    load_jailbreakv,
    load_mmlu,
    load_safeguard,
)

__all__ = [
    "LOADERS",
    "MPIDJsonlDataset",
    "PROMPT_TEMPLATE",
    "ANSWER_CHOICES",
    "ANSWER_CLEAN",
    "ANSWER_DIRECT",
    "ANSWER_INDIRECT",
    "Record",
    "VALID_LABELS",
    "build_prompt",
    "collate",
    "detect_lang",
    "load_all",
    "load_cmmlu",
    "load_deepset",
    "load_flickr30k",
    "load_jailbreakv",
    "load_mmlu",
    "load_safeguard",
]
