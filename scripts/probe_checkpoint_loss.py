"""Compute fixed-set classification loss for a LoRA checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.train import build_train_config, load_config  # noqa: E402
from mpid.data.dataset import MPIDJsonlDataset, collate  # noqa: E402
from mpid.heads.classification import ClassificationHead  # noqa: E402
from mpid.train.trainer import apply_lora_state, inject_lora, load_checkpoint  # noqa: E402
from mpid.adapters.vlm import VLMAdapter  # noqa: E402


@torch.inference_mode()
def compute_loss(args: argparse.Namespace) -> dict:
    raw = load_config(args.config)
    cfg = build_train_config(raw, None, args.config.resolve().parent)
    cfg.train_jsonl = str(args.jsonl.resolve())
    cfg.val_jsonl = str(args.jsonl.resolve())
    cfg.max_train_records = args.max_records
    cfg.max_val_records = args.max_records
    cfg.preload_dataset = False
    cfg.device = args.device or cfg.device

    adapter = VLMAdapter(
        backbone_name=cfg.backbone_name,
        dtype=cfg.dtype,
        quantization=cfg.quantization,
        device=cfg.device,
        gradient_checkpointing=False,
    )
    peft_model, _ = inject_lora(adapter.model, cfg)
    head = ClassificationHead(
        in_dim=adapter.hidden_size,
        num_classes=3,
    ).to(cfg.device)
    state = load_checkpoint(args.checkpoint, head)
    apply_lora_state(peft_model, state)
    peft_model.eval()
    head.eval()

    ds = MPIDJsonlDataset(
        args.jsonl,
        processor=adapter.processor,
        device=cfg.device,
        max_records=args.max_records,
        cache_size=max(args.max_records, 32),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    losses: list[float] = []
    for batch in dl:
        batch = {k: v.to(cfg.device) if torch.is_tensor(v) else v for k, v in batch.items()}
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
        loss = F.cross_entropy(logits, batch["label"])
        losses.append(float(loss.item()))

    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "jsonl": str(args.jsonl.resolve()),
        "records": len(losses),
        "avg_loss": sum(losses) / max(1, len(losses)),
        "losses": losses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=18)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compute_loss(args)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
