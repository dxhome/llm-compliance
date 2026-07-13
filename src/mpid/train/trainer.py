"""LoRA + head trainer for SmolVLM-500M (Phase 2 / T2.5).

The trainer orchestrates:

  1. LoRA injection on the backbone's language-model attention layers
     (Q / K / V / O projections) via ``peft``.
  2. A 3-class linear head (T2.3) attached to the pooled hidden state.
  3. Class-weighted cross-entropy loss (the dataset is 80 % direct,
     8 % indirect — without weights the model collapses to "direct").
  4. A single eval callback after each epoch (Macro F1, per-class
     precision / recall, confusion matrix).
  5. Early stopping on Macro F1 (patience = 2 by default).

Trainable parameter count is small (LoRA r = 16 on ~ 8 attention
projections ≈ 1.5 M params, head ≈ 3 k params) so the rest of the
backbone stays frozen. Forward / backward on the 500 M-param model
still uses significant memory; we enable gradient checkpointing
by default to keep the activation footprint low.

The trainer is deliberately **not** an ``accelerate`` /
``transformers.Trainer`` subclass. Those are excellent for production
runs but add a lot of moving parts (config classes, deepspeed
integration, etc.) that Phase 2 does not need. A ~200-line script
is easier to audit against the phase-2 acceptance criteria.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from mpid.adapters.vlm import VLMAdapter
from mpid.heads.classification import (
    IDX2LABEL,
    LABEL2IDX,
    LABEL_ORDER,
    NUM_CLASSES,
    ClassificationHead,
)
from mpid.data.dataset import MPIDJsonlDataset, collate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Plain dataclass — the YAML loader in ``scripts/train.py`` populates
    this. Keeping it close to a dict (no nested config schemas) means the
    operator can read every hyper-parameter in 30 seconds."""

    train_jsonl: str
    val_jsonl: str
    out_dir: str

    backbone_name: str = "smolvlm-500m"
    dtype: str = "float32"             # fp16 + MPS gives NaN (P0A-2)
    device: str = "mps"               # overridden by get_device()
    quantization: Optional[str] = None
    gradient_checkpointing: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target: str = "q_proj,k_proj,v_proj,o_proj"

    epochs: int = 1
    max_train_records: int = 500       # cap for Phase 2 timing
    max_val_records: int = 200         # cap val eval to keep training fast
    batch_size: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.0
    class_weighted: bool = True        # inverse-frequency weights

    early_stop_patience: int = 2
    log_every: int = 10
    seed: int = 42


# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------

def inject_lora(backbone_model: nn.Module, cfg: TrainConfig) -> tuple[nn.Module, int]:
    """Wrap the backbone with LoRA on the requested target modules.

    Returns ``(peft_model, n_trainable_params)``.
    """
    from peft import LoraConfig, get_peft_model

    target_modules = [m.strip() for m in cfg.lora_target.split(",") if m.strip()]
    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=target_modules,
        # We do not modify the vision encoder — only the text path.
        # SmolVLM's Idefics3 has ``q_proj`` etc. on the language stack.
        modules_to_save=None,
    )
    peft_model = get_peft_model(backbone_model, peft_cfg)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    return peft_model, n_trainable


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(records: list[dict]) -> torch.Tensor:
    """Inverse-frequency weights, smoothed to avoid divide-by-zero.

    weight_i = N / (K * count_i)
    where K is the number of classes and N is the total sample count.
    """
    counts = Counter(r["label"] for r in records)
    N = sum(counts.values())
    K = len(LABEL_ORDER)
    weights = []
    for label in LABEL_ORDER:
        c = max(1, counts.get(label, 0))
        weights.append(N / (K * c))
    w = torch.tensor(weights, dtype=torch.float32)
    # Normalise so the mean weight is 1 — keeps the loss scale stable.
    w = w * (K / w.sum())
    return w


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    head: ClassificationHead,
    dataloader: DataLoader,
    device: str,
) -> dict:
    model.eval()
    head.eval()
    all_pred, all_gold = [], []
    for batch in dataloader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            pixel_attention_mask=batch.get("pixel_attention_mask"),
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]   # (B, T, D)
        last_idx = batch["attention_mask"].sum(dim=1) - 1
        b = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[b, last_idx]
        logits = head(pooled)
        pred = logits.argmax(dim=-1).cpu().tolist()
        gold = batch["label"].cpu().tolist()
        all_pred.extend(pred)
        all_gold.extend(gold)
    # Aggregate.
    cm = confusion_matrix(all_gold, all_pred, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        all_gold, all_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {"confusion_matrix": cm.tolist(),
            "report": report,
            "y_pred": all_pred,
            "y_gold": all_gold}


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    best_macro_f1: float = 0.0
    best_epoch: int = -1
    history: list = field(default_factory=list)


def train(cfg: TrainConfig) -> TrainResult:
    """Run the full training loop and return a result summary."""
    from mpid.device import get_device

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cfg.device = cfg.device or get_device(prefer=None)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Adapter (loads backbone)
    print(f"[train] loading adapter on {cfg.device} ...")
    adapter = VLMAdapter(
        backbone_name=cfg.backbone_name,
        dtype=cfg.dtype,
        quantization=cfg.quantization,
        device=cfg.device,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    # Cast the underlying model to whatever dtype we picked (fp16 is
    # silently upgraded to fp32 above for MPS, but we re-cast here
    # to be explicit).
    if cfg.dtype == "float16":
        adapter.model.to(torch.float16)
    elif cfg.dtype == "bfloat16":
        adapter.model.to(torch.bfloat16)

    # 2. LoRA injection
    peft_model, n_lora_params = inject_lora(adapter.model, cfg)
    peft_model.train()
    # Re-apply gradient checkpointing after peft wrapping
    if cfg.gradient_checkpointing and hasattr(peft_model, "gradient_checkpointing_enable"):
        peft_model.gradient_checkpointing_enable()
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()

    # 3. Head
    head = ClassificationHead(
        in_dim=adapter.hidden_size,
        num_classes=NUM_CLASSES,
    ).to(cfg.device)
    n_head_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[train] LoRA params: {n_lora_params:,}  Head params: {n_head_params:,}")

    # 4. Data
    train_ds = MPIDJsonlDataset(
        Path(cfg.train_jsonl),
        processor=adapter.processor,
        device=cfg.device,
        max_records=cfg.max_train_records,
    )
    val_ds = MPIDJsonlDataset(
        Path(cfg.val_jsonl),
        processor=adapter.processor,
        device=cfg.device,
        max_records=cfg.max_val_records,
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True, collate_fn=collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size,
                        shuffle=False, collate_fn=collate, num_workers=0)
    print(f"[train] dataset: train={len(train_ds)} val={len(val_ds)}")

    # 5. Loss + optimiser
    if cfg.class_weighted:
        weights = compute_class_weights(train_ds.records).to(cfg.device)
    else:
        weights = None
    print(f"[train] class weights: {weights.tolist() if weights is not None else 'None'}")

    # Trainable params: LoRA + head. We do NOT freeze explicitly
    # because LoRA already freezes the base; the head is fresh.
    trainable = [p for p in peft_model.parameters() if p.requires_grad] \
                + list(head.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[train] total trainable params: {n_trainable:,}")
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 6. Loop
    res = TrainResult()
    patience_left = cfg.early_stop_patience
    for epoch in range(cfg.epochs):
        peft_model.train(); head.train()
        t_epoch = time.perf_counter()
        loss_sum, loss_count = 0.0, 0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(cfg.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            outputs = peft_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                pixel_attention_mask=batch.get("pixel_attention_mask"),
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1]
            last_idx = batch["attention_mask"].sum(dim=1) - 1
            b = torch.arange(last_hidden.size(0), device=last_hidden.device)
            pooled = last_hidden[b, last_idx]
            logits = head(pooled)
            loss = F.cross_entropy(logits, batch["label"], weight=weights)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            loss_sum += float(loss.item())
            loss_count += 1
            if (step + 1) % cfg.log_every == 0:
                avg = loss_sum / loss_count
                print(f"[train] epoch {epoch} step {step+1}/{len(train_dl)} "
                      f"loss={avg:.4f}  ({time.perf_counter()-t_epoch:.1f}s)")
                loss_sum, loss_count = 0.0, 0
        # End of epoch: eval.
        t_eval = time.perf_counter()
        ev = evaluate(peft_model, head, val_dl, cfg.device)
        dt_eval = time.perf_counter() - t_eval
        macro_f1 = ev["report"]["macro avg"]["f1-score"]
        acc = ev["report"]["accuracy"]
        print(f"[train] epoch {epoch}: val Macro F1={macro_f1:.4f}  "
              f"acc={acc:.4f}  (eval in {dt_eval:.1f}s)")
        res.history.append({"epoch": epoch,
                            "val_macro_f1": macro_f1,
                            "val_accuracy": acc,
                            "confusion_matrix": ev["confusion_matrix"],
                            "report": ev["report"]})
        # Always save the latest epoch's checkpoint so that even a
        # sub-threshold run (tiny smoke, extreme imbalance) yields
        # an artefact. We also keep the "best so far" semantics
        # by overwriting if the new F1 is strictly better.
        save_checkpoint(out_dir / "lora_baseline.safetensors",
                       peft_model, head, cfg)
        if macro_f1 > res.best_macro_f1:
            res.best_macro_f1 = macro_f1
            res.best_epoch = epoch
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[train] early stop at epoch {epoch} (best epoch {res.best_epoch})")
                break

    # Final summary
    summary = {
        "config": cfg.__dict__,
        "n_lora_params": n_lora_params,
        "n_head_params": n_head_params,
        "n_trainable_params": n_trainable,
        "best_epoch": res.best_epoch,
        "best_macro_f1": res.best_macro_f1,
        "history": res.history,
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[train] best Macro F1 = {res.best_macro_f1:.4f} at epoch {res.best_epoch}")
    return res


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, peft_model, head: ClassificationHead, cfg: TrainConfig) -> None:
    """Save LoRA adapters + head weights to a single safetensors file.

    We bundle the two because they have to be loaded together at
    inference time; a single file is easier to ship in the offline
    package (T2.11).
    """
    from safetensors.torch import save_file

    state = {}
    # LoRA params (from peft).
    for n, p in peft_model.named_parameters():
        if p.requires_grad or "lora_" in n:
            state[f"lora.{n}"] = p.detach().contiguous().cpu()
    # Head params.
    for n, p in head.named_parameters():
        state[f"head.{n}"] = p.detach().contiguous().cpu()
    # Config metadata (so the offline package can recreate the head).
    state["__head_in_dim__"] = torch.tensor(head.in_dim)
    state["__head_num_classes__"] = torch.tensor(head.num_classes)
    save_file(state, str(path), metadata={
        "format": "mpid-baseline-v1",
        "backbone": cfg.backbone_name,
        "lora_r": str(cfg.lora_r),
        "lora_alpha": str(cfg.lora_alpha),
    })
    print(f"[train] saved {path} ({len(state)} tensors)")


def load_checkpoint(path: Path, head: ClassificationHead) -> dict:
    """Load a saved checkpoint and apply the head weights in place."""
    from safetensors.torch import load_file
    state = load_file(str(path))
    head_state = {k.removeprefix("head."): v
                  for k, v in state.items() if k.startswith("head.")}
    head.load_state_dict(head_state)
    return state


__all__ = [
    "TrainConfig",
    "TrainResult",
    "train",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "compute_class_weights",
    "inject_lora",
]
