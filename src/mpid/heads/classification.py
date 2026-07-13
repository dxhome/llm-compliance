"""3-class detection head (Phase 2 / T2.3).

The head is a thin linear layer (with dropout) on top of the VLM
adapter's pooled hidden state. It maps the language-model's last
hidden state to ``[clean, direct, indirect]`` logits. The risk score
is the softmax-max probability (``max(softmax(logits))``) which lies
in ``[1/3, 1]``.

Why a plain Linear head (not an MLP, not an attention pooler)?
  * The task explicitly asks for a "head + risk score" abstraction;
    Linear is the smallest thing that satisfies it and trains fast
    on a 25k-record dataset.
  * The C4 early-exit layer (Phase 3) will reuse this pattern with
    a different ``in_dim`` (intermediate hidden size).
  * A 2-layer MLP variant is easy to add later if Linear under-fits
    on the val set.

The head is intentionally NOT registered as a ``peft`` / LoRA module.
LoRA is applied inside the backbone (Phase 2 / T2.5); the head is
trained in full because it has only 960 × 3 + 3 ≈ 3 k parameters.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical label order. The index of each label is its logit position.
LABEL_ORDER: tuple[str, ...] = ("clean", "direct", "indirect")
LABEL2IDX: dict[str, int] = {l: i for i, l in enumerate(LABEL_ORDER)}
IDX2LABEL: dict[int, str] = {i: l for i, l in enumerate(LABEL_ORDER)}
NUM_CLASSES: int = len(LABEL_ORDER)


class ClassificationHead(nn.Module):
    """Linear head mapping backbone hidden state → 3-class logits."""

    def __init__(
        self,
        in_dim: int = 960,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)
        # Kaiming init — MPS fp16 has shown NaN with default init in
        # earlier experiments (P0A-2), so we explicitly init to a
        # well-conditioned distribution.
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``(B, in_dim)`` or ``(B, T, in_dim)`` → ``(B, num_classes)``.

        If a 3-D tensor is passed, the **last token's** hidden state is
        classified (we assume a left-padded causal mask).
        """
        if hidden.dim() == 3:
            hidden = hidden[:, -1, :]
        return self.fc(self.dropout(hidden))

    def predict(self, hidden: torch.Tensor) -> dict:
        """Return ``{"logits", "probs", "label", "label_idx", "risk"}``."""
        logits = self.forward(hidden)
        probs = F.softmax(logits, dim=-1)
        label_idx = probs.argmax(dim=-1)
        risk = probs.max(dim=-1).values  # confidence in the chosen class
        label = [IDX2LABEL[int(i)] for i in label_idx.cpu().tolist()]
        return {
            "logits": logits,
            "probs": probs,
            "label_idx": label_idx,
            "label": label,
            "risk": risk,
        }


__all__ = [
    "LABEL_ORDER",
    "LABEL2IDX",
    "IDX2LABEL",
    "NUM_CLASSES",
    "ClassificationHead",
]
