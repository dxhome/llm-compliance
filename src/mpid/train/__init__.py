"""Training loop (Phase 2 / T2.5)."""
from mpid.train.trainer import (
    TrainConfig,
    TrainResult,
    compute_class_weights,
    evaluate,
    inject_lora,
    load_checkpoint,
    save_checkpoint,
    train,
)

__all__ = [
    "TrainConfig",
    "TrainResult",
    "compute_class_weights",
    "evaluate",
    "inject_lora",
    "load_checkpoint",
    "save_checkpoint",
    "train",
]
