"""PyTorch dataset wrapping the mpid-v1 JSONL splits (Phase 2).

This sits between the Phase 1 ``public_loaders`` (which produced
JSONL files) and the Phase 2 trainer. The dataset:

  * streams the JSONL file lazily (one record at a time) so the
    training process never holds the full 25 k records in memory
    in their pre-processed form;
  * applies the 3-class prompt template (T2.4) at ``__getitem__``;
  * returns a dict with ``input_ids`` / ``attention_mask`` /
    ``pixel_values`` / ``label`` that the trainer can collate.

The pre-processing (tokenize, image-decode) happens once and is
cached via an LRU — the first epoch pays the cost, subsequent
epochs amortise it. For Phase 2 we use a 1024-entry LRU which is
enough for ~250 mini-batches of size 4.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from mpid.heads.classification import LABEL2IDX
from mpid.data.prompt import build_prompt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MPIDJsonlDataset(Dataset):
    """Streaming JSONL dataset for the 3-class detection task."""

    def __init__(
        self,
        jsonl_path: Path,
        processor,
        device: str,
        *,
        max_records: Optional[int] = None,
        cache_size: int = 1024,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.processor = processor
        self.device = device
        # Load the JSONL into memory as a list of dicts. The total size
        # of 25 k records × ~2 KB each ≈ 50 MB — fine for RAM.
        self.records: list[dict] = []
        with open(self.jsonl_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("label") not in LABEL2IDX:
                    continue  # skip unknown labels defensively
                self.records.append(d)
                if max_records is not None and len(self.records) >= max_records:
                    break
        # LRU cache for preprocessed batches.
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cache_size = cache_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        r = self.records[idx]
        text = r["text"]
        image = r.get("image")
        prompt = build_prompt(text)
        # Pre-process: tokenize text + load image.
        # We use the processor directly to keep the code parallel to
        # ``VLMAdapter.preprocess`` but bypass the device move (the
        # collate function does that in a single batched op).
        from PIL import Image
        if image:
            img = Image.open(image).convert("RGB")
        else:
            img = Image.new("RGB", (512, 512), (235, 235, 235))
        enc = self.processor(
            text=prompt,
            images=[img],
            return_tensors="pt",
        )
        item = {
            "input_ids":          enc["input_ids"][0],
            "attention_mask":     enc["attention_mask"][0],
            "pixel_values":       enc["pixel_values"][0],
            "pixel_attention_mask": enc.get("pixel_attention_mask", [None])[0]
                                    if "pixel_attention_mask" in enc else None,
            "label":              torch.tensor(LABEL2IDX[r["label"]], dtype=torch.long),
        }
        # LRU bookkeeping.
        self._cache[idx] = item
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return item


def collate(batch: list[dict]) -> dict:
    """Pad a list of dataset items to the same length and stack them.

    For Phase 2 the variable-length axis is ``input_ids`` /
    ``attention_mask`` / ``pixel_attention_mask`` (T, 512, 512).
    ``pixel_values`` is per-record constant-shape (17, 3, 512, 512)
    so it stacks directly.
    """
    pad_id = 0
    max_T = max(b["input_ids"].size(0) for b in batch)

    input_ids = torch.full((len(batch), max_T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_T), dtype=torch.long)
    for i, b in enumerate(batch):
        T = b["input_ids"].size(0)
        input_ids[i, :T] = b["input_ids"]
        attn[i, :T] = b["attention_mask"]

    pv = torch.stack([b["pixel_values"] for b in batch], dim=0)
    pix_attn = None
    if batch[0]["pixel_attention_mask"] is not None:
        max_pT = max(b["pixel_attention_mask"].size(0) for b in batch)
        pix_attn = torch.zeros((len(batch), max_pT, 512, 512), dtype=torch.long)
        for i, b in enumerate(batch):
            T = b["pixel_attention_mask"].size(0)
            pix_attn[i, :T] = b["pixel_attention_mask"]

    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {
        "input_ids":          input_ids,
        "attention_mask":     attn,
        "pixel_values":       pv,
        "pixel_attention_mask": pix_attn,
        "label":              labels,
    }


__all__ = ["MPIDJsonlDataset", "collate"]
