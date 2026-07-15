"""C4 早退机制（Phase 3 / T3.x）。

**设计思路（轻量版）**：
- C4 复用了 Phase 2 已经训练好的 3 分类 head；它**不引入新的模型**。
- 推理时：VLM + head 输出 3 个 softmax 概率。如果 ``P(clean) > θ``
  （默认 θ = 0.95），就**直接返回 ``"clean"``**，跳过后续可能需要的
  C5 规则 / C6 跨模态等更昂贵的判断。
- 早退的优势：clean 样本（占数据 ~80%）延迟从 ~2s 降到 < 50ms。
- 早退的风险：阈值过低 → 误判为 clean（漏报）；阈值过高 → 早退命中率
  太低（提速不明显）。**默认 0.95 是一个保守起点**。

**为什么不训一个独立的小早退模型？**
- 数据量小（间接注入只占 8%）→ 训小模型容易过拟合
- 复用现有 head 不增加参数量、零训练成本
- Phase 2 的 head 已经在 train set 上见过 clean 样本的特征，学到的
  ``P(clean)`` 概率足够准

**与 Phase 2.5 demo 的关系**：
- Demo 左栏 base VLM + 右栏 LoRA + head（C4 off）—— 这是 Phase 2 基线
- Phase 3 在 LoRA + head 之后再加一层 C4 → 右栏会更"快"

**与 Phase 4/5 的关系**：
- C4 早退 → clean 直接返回
- 非 clean → 走 C5 规则 → 命中 → 直接拦截
- C5 未命中 → 走 C6 跨模态 → 二次确认

详细对比与设计权衡参见 ``doc/reference.md § 2.A.5``（威胁模型的三层防御
对应）和 ``§ 2.E.6``（LoRA 不挂视觉侧的设计理由）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from mpid.heads.classification import (
    IDX2LABEL,
    LABEL2IDX,
    LABEL_ORDER,
    NUM_CLASSES,
    ClassificationHead,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EarlyExitConfig:
    """C4 早退配置。

    字段：
      - ``enabled``            : 总开关。关掉 = 退化为 Phase 2 基线行为
      - ``clean_threshold``    : P(clean) > 该值即早退。0.95 是保守起点
      - ``min_layer_to_exit``  : 预留字段。Phase 3 简化为 1（最后一层），
                                 后续可以做中间层退出（V2 优化）
    """
    enabled: bool = True
    clean_threshold: float = 0.95
    min_layer_to_exit: int = 1   # 1 = 仅最后一层；>1 = 中间层早退（V2）


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class EarlyExitResult:
    """单次早退判定的结果。

    字段：
      - ``label``              : 早退给出的 label（永远等于 ``"clean"``）
      - ``probs``              : 完整的 3 分类 softmax 概率（dict）
      - ``exited``             : True = 触发了早退；False = 没触发
      - ``latency_ms``         : 本次推理总耗时
      - ``p_clean``            : clean 类的概率
    """
    label: str
    probs: dict
    exited: bool
    latency_ms: float
    p_clean: float


# ---------------------------------------------------------------------------
# Stats (per-batch / per-run)
# ---------------------------------------------------------------------------

@dataclass
class EarlyExitStats:
    """早退运行统计。

    字段：
      - ``n_total``            : 处理的样本数
      - ``n_exited``           : 触发早退的样本数
      - ``n_clean_exited``     : 早退且 ground truth 也是 clean 的样本数（=正确早退）
      - ``n_clean_wrong_exit`` : 早退为 clean 但 ground truth 不是 clean 的样本数（=漏报）
      - ``latency_full_ms``    : 不早退（完整 VLM）路径的累计耗时
      - ``latency_exit_ms``    : 早退路径的累计耗时
    """
    n_total: int = 0
    n_exited: int = 0
    n_clean_exited: int = 0
    n_clean_wrong_exit: int = 0
    latency_full_ms: float = 0.0
    latency_exit_ms: float = 0.0
    # Per-class breakdown: how many ground-truth samples of each class
    # triggered an early exit.
    per_class_exits: dict = field(default_factory=lambda: {l: 0 for l in LABEL_ORDER})
    per_class_total: dict = field(default_factory=lambda: {l: 0 for l in LABEL_ORDER})

    def to_dict(self) -> dict:
        n = max(self.n_total, 1)
        return {
            "n_total": self.n_total,
            "n_exited": self.n_exited,
            "exit_rate": self.n_exited / n,
            "n_clean_wrong_exit": self.n_clean_wrong_exit,
            # Latency split (averages)
            "avg_latency_full_ms": self.latency_full_ms / max(1, self.n_total - self.n_exited),
            "avg_latency_exit_ms": self.latency_exit_ms / max(1, self.n_exited),
            # Per-class exit rates
            "per_class_exits": dict(self.per_class_exits),
            "per_class_total": dict(self.per_class_total),
            "per_class_exit_rate": {
                lbl: (self.per_class_exits[lbl] / max(1, self.per_class_total[lbl]))
                for lbl in LABEL_ORDER
            },
        }


# ---------------------------------------------------------------------------
# Core: should_early_exit?
# ---------------------------------------------------------------------------

def should_early_exit(
    probs: torch.Tensor,
    cfg: EarlyExitConfig,
) -> Optional[str]:
    """根据 softmax 概率判断是否要早退。

    Args:
        probs: shape ``(num_classes,)`` or ``(1, num_classes)`` 的 softmax 概率
        cfg:   EarlyExitConfig

    Returns:
        ``"clean"``  if  ``probs[LABEL2IDX["clean"]] > cfg.clean_threshold`` and ``cfg.enabled``
        ``None``     otherwise
    """
    if not cfg.enabled:
        return None
    if probs.dim() == 2:
        probs = probs[0]
    p_clean = float(probs[LABEL2IDX["clean"]].item())
    if p_clean > cfg.clean_threshold:
        return "clean"
    return None


# ---------------------------------------------------------------------------
# High-level: classify_with_early_exit
# ---------------------------------------------------------------------------

@torch.inference_mode()
def classify_with_early_exit(
    peft_model,
    head: ClassificationHead,
    processor,
    text: str,
    image,
    cfg: EarlyExitConfig,
    device: str,
    *,
    max_new_tokens: int = 0,
) -> EarlyExitResult:
    """带 C4 早退的分类流程。

    Args:
        peft_model : 已经 inject_lora 过的 VLM（来自 Phase 2 T2.5）
        head       : 训练好的 3 分类 head（来自 Phase 2 T2.3）
        processor  : VLM 对应的 processor
        text       : 用户文本
        image      : 图像（PIL.Image / path / None）
        cfg        : EarlyExitConfig
        device     : ``"cpu"`` / ``"mps"`` / ``"cuda"``
        max_new_tokens: 预留扩展。Phase 3 默认 0（不需要生成）

    Returns:
        EarlyExitResult

    **完整流程**：
        1. VLM forward → last_hidden (1, D)
        2. head(last_hidden) → logits (1, 3)
        3. softmax → probs (1, 3)
        4. should_early_exit(probs) → "clean" or None
        5. exit → 直接返回 "clean"
           no-exit → 用 head 预测的 label
    """
    t0 = time.perf_counter()

    # Preprocess
    if "<image>" not in text:
        text = "<image>" + text
    img = image if image is not None else _placeholder_image()
    encoded = processor(text=text, images=[img], return_tensors="pt")
    encoded = {k: v.to(device) if torch.is_tensor(v) else v
               for k, v in encoded.items()}

    # VLM forward
    outputs = peft_model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        pixel_values=encoded["pixel_values"],
        pixel_attention_mask=encoded.get("pixel_attention_mask"),
        output_hidden_states=True,
    )
    last_hidden = outputs.hidden_states[-1]   # (1, T, D)
    last_idx = encoded["attention_mask"].sum(dim=1) - 1
    b = torch.arange(last_hidden.size(0), device=last_hidden.device)
    pooled = last_hidden[b, last_idx]          # (1, D)

    # Head → logits → probs
    logits = head(pooled)                      # (1, 3)
    probs_t = F.softmax(logits, dim=-1)        # (1, 3)
    probs_list = probs_t[0].cpu().tolist()
    probs_dict = {LABEL_ORDER[i]: probs_list[i] for i in range(NUM_CLASSES)}

    # Decision
    early_label = should_early_exit(probs_t, cfg)
    if early_label is not None:
        label = early_label
        exited = True
    else:
        label = IDX2LABEL[int(probs_t.argmax(dim=-1).item())]
        exited = False

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return EarlyExitResult(
        label=label,
        probs=probs_dict,
        exited=exited,
        latency_ms=latency_ms,
        p_clean=probs_dict["clean"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _placeholder_image():
    """同 VLMAdapter._get_placeholder_image 的 512x512 浅灰占位图。

    避免跨包依赖（mpid.adapters.vlm 已经实现了，这里复制是为了不强制
    Phase 3 代码 import adapters 包）。
    """
    from mpid.adapters.vlm import _get_placeholder_image
    return _get_placeholder_image()


__all__ = [
    "EarlyExitConfig",
    "EarlyExitResult",
    "EarlyExitStats",
    "should_early_exit",
    "classify_with_early_exit",
]
