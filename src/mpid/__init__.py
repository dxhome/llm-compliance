"""MPID — Multimodal Prompt Injection Detection.

A defense stack against prompt-injection attacks on Vision-Language Models.
Combines a SmolVLM-500M backbone with rule pre-filter, early-exit, and
cross-modal consistency checks (see doc/VERIFICATION.md for details).

Public surface:

  - ``mpid.device.get_device(prefer=None)`` — device resolver
    (MPS / CUDA / CPU, with strict ``--prefer`` semantics).
  - ``mpid.device.device_summary()`` — read-only host diagnostic dict.
  - ``mpid.adapters.vlm.VLMAdapter`` — SmolVLM-500M adapter (Phase 2).
  - ``mpid.backbones.registry.resolve_local_path`` — model path resolver.

Phase 2 also adds:

  - ``mpid.heads.classification``   — 3-class (clean / direct / indirect) head
  - ``mpid.data.prompt``            — 3-class prompt template
  - ``mpid.data.dataset``           — PyTorch dataset wrapping JSONL splits
  - ``mpid.train.trainer``          — LoRA + eval-callback trainer
  - ``mpid.rules.engine``           — C5 rule pre-filter (Phase 4)
  - ``mpid.infer.early_exit``       — C4 adaptive inference (Phase 3)
  - ``mpid.rules.crossmodal``       — C6 cross-modal consistency (Phase 5)
"""
from __future__ import annotations

__version__ = "0.2.0"
__phase__ = "2"  # 0=脚手架, 1=威胁模型, 2=基线, 3-5=C4/C5/C6, 6=评测, 7=整理

__all__ = [
    "__version__",
    "__phase__",
    "device",
    "adapters",
    "backbones",
    "data",
    "heads",
]
