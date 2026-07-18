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
import sys
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


def _log(msg: str, flush: bool = True) -> None:
    """A single point for progress output. Always uses ``flush=True`` by
    default so that long-running training loops show progress in real
    time even when stdout is line-buffered (e.g. when launched via
    subprocess)."""
    if flush:
        print(msg, flush=True)
    else:
        print(msg)


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
    log_every: int = 5             # default tighter so progress is visible on CPU
    eval_after_epoch: bool = False # Phase 2.2 eval runs as a separate workflow step
    seed: int = 42
    # T2.16: phase 2.2 真实训练开关
    max_train_seconds: float = 0.0  # 0=不限制；>0 时单次 run 总时长上限（秒）
                                     # 防止 benchmark 把整夜跑完
    preload_dataset: bool = False   # 预编码所有样本到 RAM（牺牲 RAM 换时间）
    checkpoint_name: str = "lora_baseline.safetensors"  # T2.16 改为 lora_full.safetensors
    flush: bool = True              # 所有 print 强制 flush（避免缓冲）
    save_every: int = 50           # >0 时每 N step 保存一次 partial checkpoint
                                    # 0 = 仅 epoch 结束或 budget 超时时保存
    partial_name: str = "lora_partial.safetensors"  # partial checkpoint 文件名


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


def apply_lora_state(peft_model: nn.Module, full_state: dict) -> int:
    """Apply saved LoRA tensors onto a PEFT model."""
    from peft import set_peft_model_state_dict

    lora_state = {
        k.removeprefix("lora."): v
        for k, v in full_state.items()
        if k.startswith("lora.")
    }
    if not lora_state:
        return 0
    set_peft_model_state_dict(peft_model, lora_state)
    return len(lora_state)


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
    progress_every: int = 0,
    progress_label: str = "eval",
) -> dict:
    model.eval()
    head.eval()
    all_pred, all_gold = [], []
    seen = 0
    total = len(dataloader.dataset) if hasattr(dataloader, "dataset") else None
    t_eval0 = time.perf_counter()
    t_warmup = None
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
        seen += len(gold)
        if progress_every > 0 and seen % progress_every == 0:
            t_now = time.perf_counter()
            if t_warmup is None:
                t_warmup = t_now
                per_sample_s = float("nan")
                eta_s = float("nan")
            else:
                elapsed_eval = t_now - t_warmup
                per_sample_s = elapsed_eval / max(1, seen - progress_every)
                eta_s = float("nan")
                if total is not None:
                    eta_s = per_sample_s * max(0, total - seen)
            if total is not None:
                _log(
                    f"[{progress_label}] progress: {seen}/{total} samples  "
                    f"step_dt={per_sample_s:.2f}s  "
                    f"eval_elapsed={t_now-t_eval0:.1f}s  "
                    f"ETA={eta_s:.0f}s"
                )
            else:
                _log(
                    f"[{progress_label}] progress: {seen} samples  "
                    f"step_dt={per_sample_s:.2f}s  "
                    f"eval_elapsed={t_now-t_eval0:.1f}s"
                )
    if progress_every > 0 and seen > 0 and seen % progress_every != 0:
        t_now = time.perf_counter()
        if t_warmup is None:
            per_sample_s = float("nan")
            eta_s = float("nan")
        else:
            elapsed_eval = t_now - t_warmup
            per_sample_s = elapsed_eval / max(1, seen - progress_every)
            eta_s = float("nan")
            if total is not None:
                eta_s = per_sample_s * max(0, total - seen)
        if total is not None:
            _log(
                f"[{progress_label}] progress: {seen}/{total} samples  "
                f"step_dt={per_sample_s:.2f}s  "
                f"eval_elapsed={t_now-t_eval0:.1f}s  "
                f"ETA={eta_s:.0f}s"
            )
        else:
            _log(
                f"[{progress_label}] progress: {seen} samples  "
                f"step_dt={per_sample_s:.2f}s  "
                f"eval_elapsed={t_now-t_eval0:.1f}s"
            )
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

    # Helpers -------------------------------------------------------------
    log = lambda m: _log(m, flush=cfg.flush)
    t0_total = time.perf_counter()

    def phase(stage: str) -> None:
        dt = time.perf_counter() - t0_total
        log(f"[train][+{dt:6.1f}s] === {stage} ===")

    # Signal handler: save partial checkpoint on SIGTERM/SIGINT.
    # The ``peft_model`` and ``head`` will be assigned after phase 2/3;
    # we use a small mutable holder so the handler can reach them
    # even if the user kills the run mid-epoch.
    state_holder: dict = {}
    _interrupted = {"flag": False}

    def _save_partial_and_exit(signum, frame):
        sig_name = {2: "SIGINT", 15: "SIGTERM"}.get(signum, f"signal-{signum}")
        log(f"[train] !! {sig_name} received - saving partial checkpoint ...")
        pm = state_holder.get("peft_model")
        hd = state_holder.get("head")
        if pm is not None and hd is not None:
            try:
                save_checkpoint(out_dir / cfg.partial_name, pm, hd, cfg)
                log(f"[train] !! partial saved to {out_dir / cfg.partial_name}")
            except Exception as e:
                log(f"[train] !! partial save FAILED: {e}")
        _interrupted["flag"] = True

    import signal as _signal
    try:
        _signal.signal(_signal.SIGTERM, _save_partial_and_exit)
        _signal.signal(_signal.SIGINT, _save_partial_and_exit)
    except (ValueError, OSError):
        # Not on main thread (e.g. embedded); skip signal hooks.
        pass

    phase("phase 1/6 加载 backbone")
    # 1. Adapter (loads backbone)
    log(f"[train] loading adapter on {cfg.device} ...")
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
    log(f"[train]   backbone loaded in {time.perf_counter()-t0_total:.1f}s")

    phase("phase 2/6 LoRA + head 注入")
    # 2. LoRA injection
    peft_model, n_lora_params = inject_lora(adapter.model, cfg)
    # Disable KV cache BEFORE training: the IDEFICS3 model defaults
    # to ``use_cache=True``, which is *incompatible* with gradient
    # checkpointing (the checkpoint layer re-evaluates attention and
    # would clobber the cache). Without this, MPS's autograd goes
    # off the rails around step 2-3 and produces NaN losses.
    try:
        peft_model.config.use_cache = False
    except AttributeError:
        pass
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
    log(f"[train] LoRA params: {n_lora_params:,}  Head params: {n_head_params:,}")

    # Expose to signal handler for mid-epoch partial saves.
    state_holder["peft_model"] = peft_model
    state_holder["head"] = head

    resume_from = getattr(cfg, "resume_from", None)
    if resume_from:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        state = load_checkpoint(ckpt_path, head)
        n_loaded_lora = apply_lora_state(peft_model, state)
        n_head_tensors = len([k for k in state if k.startswith("head.")])
        log(f"[train] resumed from {ckpt_path} "
            f"(lora tensors={n_loaded_lora}, head tensors={n_head_tensors})")

    phase("phase 3/6 数据集加载")
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
    log(f"[train] dataset: train={len(train_ds)} val={len(val_ds)}")
    log(f"[train] batch_size={cfg.batch_size}  steps_per_epoch={len(train_ds)//cfg.batch_size}")

    # Optional: preload dataset into RAM. For 2k records each with
    # 17 patches × 512x512 pixel_values this is ~9 GB on RAM, which
    # is fine on a 16 GB Mac. The payoff is no per-step image
    # preprocessing on the main thread.
    if cfg.preload_dataset:
        phase("phase 3b/6 预编码数据到 RAM (--preload-dataset)")
        t_pre = time.perf_counter()
        train_ds.preload(log_every=200)
        val_ds.preload(log_every=200)
        log(f"[train] preload done in {time.perf_counter()-t_pre:.1f}s")

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True, collate_fn=collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size,
                        shuffle=False, collate_fn=collate, num_workers=0)

    phase("phase 4/6 优化器 + class weights")
    # 5. Loss + optimiser
    if cfg.class_weighted:
        weights = compute_class_weights(train_ds.records).to(cfg.device)
    else:
        weights = None
    log(f"[train] class weights: {weights.tolist() if weights is not None else 'None'}")

    # Trainable params: LoRA + head. We do NOT freeze explicitly
    # because LoRA already freezes the base; the head is fresh.
    trainable = [p for p in peft_model.parameters() if p.requires_grad] \
                + list(head.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    log(f"[train] total trainable params: {n_trainable:,}")
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    phase(f"phase 5/6 训练循环  ({cfg.epochs} epoch × {len(train_ds)} sample × bs={cfg.batch_size})")
    # 6. Loop
    res = TrainResult()
    patience_left = cfg.early_stop_patience
    n_steps_per_epoch = max(1, len(train_dl))
    total_steps = n_steps_per_epoch * cfg.epochs
    t_warmup = None
    step_global = int(getattr(cfg, "resume_global_step", 0))
    resumed_this_run = 0
    skip_train_batches = int(getattr(cfg, "skip_train_batches", 0))
    budget_deadline = (t0_total + cfg.max_train_seconds) if cfg.max_train_seconds > 0 else None
    max_train_steps = int(getattr(cfg, "max_train_steps", 0))

    for epoch in range(cfg.epochs):
        peft_model.train(); head.train()
        t_epoch = time.perf_counter()
        loss_sum, loss_count = 0.0, 0
        for step, batch in enumerate(train_dl):
            if epoch == 0 and skip_train_batches > 0 and step < skip_train_batches:
                if step == 0:
                    log(f"[train] skipping first {skip_train_batches} batches "
                        f"to approximate resume position")
                continue
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
            # NaN / Inf guard BEFORE clipping. MPS + LoRA + grad-ckpt
            # has a known issue where the loss occasionally diverges to
            # NaN (FP16 / BF16 gradient underflow when class weights
            # are skewed). If we see NaN, **do not** step the optimizer
            # (that would poison the LoRA weights).
            if not torch.isfinite(loss).item():
                if step_global == 1 or step_global % cfg.log_every == 0:
                    log(f"[train]   ! step {step+1} loss={loss.item():.4f} - "
                        f"NaN/Inf detected, skipping optimizer step")
                opt.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            loss_sum += float(loss.item())
            loss_count += 1
            step_global += 1
            resumed_this_run += 1

            # Progress: log_every steps AND always at step 1 of the
            # epoch (so we know training actually started). Skip logging
            # if the step was NaN (we already logged a warning above).
            should_log = (step + 1 == 1) or ((step + 1) % cfg.log_every == 0) \
                         or (step + 1 == n_steps_per_epoch)
            if should_log and loss_count > 0:
                avg = loss_sum / loss_count
                t_now = time.perf_counter()
                # Measure per-step time. Use the first step of the
                # epoch as a "warmup" because the first step pays
                # for kernel JIT, allocator init, etc.
                if t_warmup is None:
                    t_warmup = t_now
                    per_step_s = float("nan")
                    eta_s = float("nan")
                else:
                    elapsed_train = t_now - t_warmup
                    per_step_s = elapsed_train / max(1, (resumed_this_run - 1))
                    remaining = total_steps - step_global
                    eta_s = per_step_s * remaining
                log(
                    f"[train] epoch {epoch+1}/{cfg.epochs} "
                    f"step {step+1}/{n_steps_per_epoch} "
                    f"(global {step_global}/{total_steps}) "
                    f"loss={avg:.4f}  "
                    f"step_dt={per_step_s:.2f}s  "
                    f"epoch_elapsed={t_now-t_epoch:.1f}s  "
                    f"ETA={eta_s:.0f}s  "
                    f"total_elapsed={t_now-t0_total:.1f}s"
                )

            # Periodic partial save (--save-every N). Cheaper than
            # waiting for an epoch boundary if the user later kills
            # the process.
            if cfg.save_every > 0 and step_global > 0 \
                    and step_global % cfg.save_every == 0:
                log(f"[train]   -> periodic save: {cfg.partial_name} "
                    f"(step {step_global})")
                save_checkpoint(out_dir / cfg.partial_name, peft_model, head, cfg)

            if max_train_steps > 0 and resumed_this_run >= max_train_steps:
                log(f"[train] STEP LIMIT REACHED ({max_train_steps}) - "
                    f"stopping at epoch {epoch+1} step {step+1}")
                save_checkpoint(out_dir / cfg.checkpoint_name, peft_model, head, cfg)
                res.history.append({
                    "epoch": epoch,
                    "val_macro_f1": 0.0,
                    "val_accuracy": 0.0,
                    "report": {},
                    "confusion_matrix": [],
                    "note": "step_limit_reached",
                })
                summary = {
                    "config": cfg.__dict__,
                    "n_lora_params": n_lora_params,
                    "n_head_params": n_head_params,
                    "n_trainable_params": n_trainable,
                    "best_epoch": res.best_epoch,
                    "best_macro_f1": res.best_macro_f1,
                    "history": res.history,
                    "step_limit_reached": True,
                    "max_train_steps": max_train_steps,
                    "resume_from": getattr(cfg, "resume_from", None),
                    "skip_train_batches": int(getattr(cfg, "skip_train_batches", 0)),
                    "resume_global_step": int(getattr(cfg, "resume_global_step", 0)),
                }
                with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                return res

            # Wall-clock budget check.
            if budget_deadline is not None and time.perf_counter() > budget_deadline:
                log(f"[train] BUDGET EXCEEDED ({cfg.max_train_seconds}s) - "
                    f"stopping at epoch {epoch+1} step {step+1}")
                # Save what we have and break.
                save_checkpoint(out_dir / cfg.checkpoint_name, peft_model, head, cfg)
                res.history.append({
                    "epoch": epoch, "val_macro_f1": 0.0, "val_accuracy": 0.0,
                    "report": {}, "confusion_matrix": [],
                    "note": "budget_exceeded",
                })
                summary = {
                    "config": cfg.__dict__,
                    "n_lora_params": n_lora_params,
                    "n_head_params": n_head_params,
                    "n_trainable_params": n_trainable,
                    "best_epoch": res.best_epoch,
                    "best_macro_f1": res.best_macro_f1,
                    "history": res.history,
                    "budget_exceeded": True,
                }
                with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                return res

        # End of epoch: save immediately. Eval is usually too expensive
        # for CPU training, so Phase 2.2 runs it as a separate workflow step.
        if bool(getattr(cfg, "eval_after_epoch", False)):
            log(f"[train] epoch {epoch+1} train done; starting eval ...")
            t_eval = time.perf_counter()
            ev = evaluate(peft_model, head, val_dl, cfg.device)
            dt_eval = time.perf_counter() - t_eval
            macro_f1 = ev["report"]["macro avg"]["f1-score"]
            acc = ev["report"]["accuracy"]
            log(f"[train] epoch {epoch+1}: val Macro F1={macro_f1:.4f}  "
                f"acc={acc:.4f}  (eval in {dt_eval:.1f}s, "
                f"total_elapsed={time.perf_counter()-t0_total:.1f}s)")
            res.history.append({"epoch": epoch,
                                "val_macro_f1": macro_f1,
                                "val_accuracy": acc,
                                "confusion_matrix": ev["confusion_matrix"],
                                "report": ev["report"]})
        else:
            log(f"[train] epoch {epoch+1} train done; skipping eval "
                f"(eval_after_epoch=False)")
            macro_f1 = 0.0
            res.history.append({"epoch": epoch,
                                "val_macro_f1": None,
                                "val_accuracy": None,
                                "confusion_matrix": [],
                                "report": {},
                                "note": "eval_skipped"})

        # Always save the latest epoch's checkpoint so that even a
        # no-eval or sub-threshold run yields an artefact.
        log(f"[train] saving checkpoint {cfg.checkpoint_name} ...")
        save_checkpoint(out_dir / cfg.checkpoint_name, peft_model, head, cfg)
        if bool(getattr(cfg, "eval_after_epoch", False)):
            if macro_f1 > res.best_macro_f1:
                res.best_macro_f1 = macro_f1
                res.best_epoch = epoch
                patience_left = cfg.early_stop_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    log(f"[train] early stop at epoch {epoch+1} (best epoch {res.best_epoch+1})")
                    break

    phase("phase 6/6 收尾 + 写 train_summary.json")
    # Final summary
    summary = {
        "config": cfg.__dict__,
        "n_lora_params": n_lora_params,
        "n_head_params": n_head_params,
        "n_trainable_params": n_trainable,
        "best_epoch": res.best_epoch,
        "best_macro_f1": res.best_macro_f1,
        "history": res.history,
        "total_seconds": time.perf_counter() - t0_total,
        "resume_from": getattr(cfg, "resume_from", None),
        "skip_train_batches": int(getattr(cfg, "skip_train_batches", 0)),
        "resume_global_step": int(getattr(cfg, "resume_global_step", 0)),
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if bool(getattr(cfg, "eval_after_epoch", False)):
        log(f"[train] DONE  best Macro F1 = {res.best_macro_f1:.4f} at epoch {res.best_epoch+1}, "
            f"total time = {time.perf_counter()-t0_total:.1f}s")
    else:
        log(f"[train] DONE  eval skipped; checkpoint saved as {cfg.checkpoint_name}, "
            f"total time = {time.perf_counter()-t0_total:.1f}s")
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
    print(f"[train] saved {path} ({len(state)} tensors)", flush=True)


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
    "apply_lora_state",
    "compute_class_weights",
    "inject_lora",
]
