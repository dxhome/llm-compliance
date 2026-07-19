"""MPID Phase 2.5 demo (T2.5.4) — Gradio Blocks app.

This is the **成果可视化** deliverable. The app exposes a single page
with three sections:

  1. A row of 8 preset sample buttons (clean × 3 / direct × 3 / indirect × 2)
     that load a (text, image) pair into the input widgets below.
  2. A free-form input area: text box + image upload.
  3. A side-by-side comparison:
       - **Left**: the base SmolVLM-500M is asked to *generate* a free-form
         answer to the prompt (T2.5.1: ``VLMAdapter.generate()``). We show
         the raw reply with a red "易被攻破" banner — this is the **no-
         protection** view.
       - **Right**: the same prompt is fed through the **LoRA + 3-class
         head** pipeline, producing a ``(clean / direct / indirect)`` label,
         a risk score in ``[0, 1]``, and a 3-bar confidence chart.

A "项目说明" tab on the right summarises the threat model, model card,
and references.

Boot::

    python demo/gradio_app.py
    # or
    python demo/gradio_app.py --device cpu --max-new-tokens 96

The script is **self-contained** under ``demo/`` and does not modify
``src/mpid/`` at import time. It only depends on:

  * ``mpid`` (the main package) — added to ``sys.path`` at boot
  * gradio + plotly — installed via ``demo/requirements.txt``

It loads:

  * the SmolVLM-500M backbone from ``runs/_models/smolvlm-500m/``
  * the LoRA + head checkpoint from
    ``runs/_templates/artifacts/checkpoints/lora_baseline.safetensors``

If either is missing, the app prints a clear error and exits with a
non-zero status. The startup target is "server up in ≤ 30 s" on a
MacBook M-class machine.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# Silence the noisy "Some weights of the model checkpoint were not used"
# warning from PEFT/HF; we know our checkpoint format.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Resolve ``demo/`` as the anchor. ``__file__`` is the gradio_app.py
# itself; the parent of that is the ``demo/`` directory.
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
SRC_DIR = REPO_ROOT / "src"

# Add the package source to sys.path so ``import mpid.*`` works without
# requiring ``pip install -e .`` first. Mirrors the convention in
# ``scripts/smoke_env.py`` / ``scripts/build_phase1.py``.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Defaults can be overridden via CLI flags. Keep them absolute so the
# script works from any cwd.
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "_models" / "smolvlm-500m"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "runs" / "_templates" / "artifacts" / "checkpoints" /
    "lora_baseline.safetensors"
)
DEFAULT_SAMPLES = DEMO_DIR / "samples.json"
DEFAULT_DEVICE = "cpu"  # MPS+LoRA can be unstable; default to safe


def resolve_demo_asset(path_value: str | os.PathLike | None) -> Path | None:
    """Resolve demo sample assets across the old and new repo layouts.

    Older ``samples.json`` entries used ``data/raw/...``. The project now keeps
    shared data under ``runs/_datasets/...``. Returning ``None`` for missing or
    empty values keeps text-only samples lightweight.
    """
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path if path.exists() else None

    raw_text = path.as_posix()
    candidates = [REPO_ROOT / path]
    if raw_text.startswith("data/raw/"):
        candidates.append(
            REPO_ROOT / "runs" / "_datasets" / "raw" /
            Path(raw_text.removeprefix("data/raw/"))
        )
    if raw_text.startswith("data/"):
        candidates.append(
            REPO_ROOT / "runs" / "_datasets" /
            Path(raw_text.removeprefix("data/"))
        )
    candidates.append(DEMO_DIR / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

class DemoPipeline:
    """Wrap a ``VLMAdapter`` + ``ClassificationHead`` for demo use.

    The class is initialised **once** at app start (the model load is
    the slow part of the boot). After that, ``classify()`` and
    ``generate()`` are the two hot-path entry points that the Gradio
    callbacks call.

    We use ``CPU`` + ``fp32`` by default for portability; on M-series
    Macs the user can pass ``--device mps`` if they want a faster
    generation, but the head always returns a label + risk regardless.
    """

    def __init__(
        self,
        model_dir: Path,
        checkpoint: Path,
        device: str,
        max_new_tokens: int,
    ) -> None:
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        from mpid.adapters.vlm import VLMAdapter
        from mpid.heads.classification import (
            IDX2LABEL, LABEL_ORDER, NUM_CLASSES, ClassificationHead,
        )
        from mpid.train.trainer import TrainConfig, inject_lora

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.LABEL_ORDER = LABEL_ORDER
        self.IDX2LABEL = IDX2LABEL
        self.NUM_CLASSES = NUM_CLASSES

        print(f"[demo] loading adapter from {model_dir} on {device} ...")
        self.adapter = VLMAdapter(
            backbone_name="smolvlm-500m",
            dtype="float32",
            quantization=None,
            device=device,
            gradient_checkpointing=False,
            models_root=model_dir.parent,
        )

        # Wrap with LoRA (same hyper-params as the training run) so the
        # saved adapter weights can be loaded back in. The architecture
        # is identical to ``scripts/train.py``.
        lora_cfg = TrainConfig(
            train_jsonl="", val_jsonl="", out_dir="",
            lora_r=16, lora_alpha=32, lora_dropout=0.05,
            lora_target="q_proj,k_proj,v_proj,o_proj",
        )
        self.peft_model, self.n_lora = inject_lora(self.adapter.model, lora_cfg)
        self.peft_model.eval()

        # Head.
        self.head = ClassificationHead(
            in_dim=self.adapter.hidden_size, num_classes=NUM_CLASSES,
        ).to(device)
        self.head.eval()

        # Load checkpoint.
        print(f"[demo] loading checkpoint {checkpoint} ...")
        state = load_file(str(checkpoint))
        # Head weights.
        head_state = {k.removeprefix("head."): v
                      for k, v in state.items() if k.startswith("head.")}
        self.head.load_state_dict(head_state)
        # LoRA weights (if any).
        lora_state = {k.removeprefix("lora."): v
                      for k, v in state.items() if k.startswith("lora.")}
        if lora_state:
            set_peft_model_state_dict(self.peft_model, lora_state)
            print(f"[demo]   loaded {len(lora_state)} LoRA tensors")
        else:
            print(f"[demo]   no LoRA tensors in checkpoint (head only)")
        print(f"[demo] pipeline ready "
              f"(LoRA={self.n_lora:,} params, head={sum(p.numel() for p in self.head.parameters()):,} params)")

    # -- hot path ---------------------------------------------------------

    def classify(self, text: str, image) -> dict:
        """Run the LoRA + 3-class head on (text, image) and return a
        dict with label / risk / per-class probs."""
        import torch

        from mpid.data.prompt import build_prompt

        prompt = build_prompt(text or "")
        out = self.adapter.forward(prompt, image)
        hidden = out["last_hidden"]  # (1, D) — inference_mode tensor
        # ``inference_mode`` tensors are NOT autograd-friendly, and
        # ``nn.Dropout`` (even in eval mode) always builds a backward
        # graph. Wrap the head call in ``no_grad`` so the dropout
        # layer can safely accept the hidden state without trying to
        # save it for backward.
        with torch.no_grad():
            pred = self.head.predict(hidden)
        probs = pred["probs"][0].cpu().tolist()
        label = pred["label"][0]
        risk = float(pred["risk"][0])
        return {
            "label": label,
            "label_idx": int(pred["label_idx"][0]),
            "risk": risk,
            "probs": probs,  # [P(clean), P(direct), P(indirect)]
        }

    def generate(self, text: str, image) -> str:
        """Free-form generation with the base VLM (no head, no
        LoRA-trained safety). This is the **left** column of the demo."""
        if not text:
            return "(empty prompt)"
        try:
            return self.adapter.generate(
                text, image, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        except Exception as e:  # pragma: no cover - protect UI from crashes
            return f"[generate failed: {type(e).__name__}: {e}]"

    def generate_with_lora(self, text: str, image) -> str:
        """Free-form generation with the **LoRA-tuned** VLM
        (i.e. ``self.peft_model``). This is what the **right** column
        uses when MPID classifies the input as ``clean`` — we then ask
        the same VLM, but with the LoRA adapter fused in, to produce
        the response. The intent is that an LoRA-fine-tuned model is
        better-behaved than the raw base on a clean prompt, while
        still being the "same" VLM at inference time.

        Implementation reuses the chat-template + processor logic from
        :meth:`VLMAdapter.generate` but swaps the underlying model for
        ``self.peft_model``. We ``merge_and_unload``-equivalent behaviour
        by simply calling the LoRA model directly; PEFT's ``generate``
        will route through the active adapters.
        """
        if not text:
            return "(empty prompt)"
        try:
            import torch
            from PIL import Image as PILImage

            proc = self.adapter.processor
            dev = self.device

            # Build the chat-template prompt (same format as
            # ``VLMAdapter.generate``).
            user_content: list[dict] = [
                {"type": "image"},
                {"type": "text", "text": text or ""},
            ]
            messages: list[dict] = [{
                "role": "user", "content": user_content,
            }]
            prompt = proc.apply_chat_template(
                messages, add_generation_prompt=True,
            )

            # Resolve image. If ``None`` or path-missing, use a tiny
            # blank image so the model still gets a valid pixel tensor
            # (SmolVLM requires the image slot).
            img = self.adapter._resolve_image(image)
            if img is None:
                img = PILImage.new("RGB", (8, 8), color=(0, 0, 0))

            encoded = proc(text=[prompt], images=[img], return_tensors="pt")
            encoded = {
                k: v.to(dev) for k, v in encoded.items() if torch.is_tensor(v)
            }

            self.peft_model.eval()
            with torch.inference_mode():
                output_ids = self.peft_model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            prompt_len = encoded["input_ids"].shape[1]
            new_ids = output_ids[0, prompt_len:]
            text_out = proc.batch_decode(
                [new_ids], skip_special_tokens=True,
            )[0].strip()
            return text_out
        except Exception as e:  # pragma: no cover - protect UI from crashes
            return f"[generate_with_lora failed: {type(e).__name__}: {e}]"


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

LABEL_CN: dict[str, str] = {
    "clean": "干净 (clean)",
    "direct": "直接注入 (direct injection)",
    "indirect": "间接注入 (indirect injection)",
}
LABEL_COLOR: dict[str, str] = {
    "clean": "#22c55e",       # green
    "direct": "#ef4444",      # red
    "indirect": "#f59e0b",     # amber
}

BASE_HEADER_MD = (
    "<span style='color:#22c55e'>●</span> Base SmolVLM"
)
MPID_HEADER_PASS_MD = (
    "<span style='color:#22c55e'>●</span> MPID (LoRA only)"
)
MPID_HEADER_BLOCK_MD = (
    "<span style='color:#ef4444'>●</span> MPID (LoRA only)"
)
OPT_HEADER_PASS_MD = (
    "<span style='color:#22c55e'>●</span> MPID (LoRA + C4-C6 优化)"
)
OPT_HEADER_BLOCK_MD = (
    "<span style='color:#ef4444'>●</span> MPID (LoRA + C4-C6 优化)"
)
PROCESSING_MD = (
    "<div style='padding:16px 18px;border:1px solid #d1d5db;"
    "border-radius:8px;background:#f9fafb'>"
    "<div style='font-weight:600;margin-bottom:6px'>Processing...</div>"
    "<div>模型正在生成和判定，请稍等。</div>"
    "</div>"
)


def format_latency(seconds: float) -> str:
    ms = seconds * 1000.0
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{seconds:.2f} s"


def code_html(text: str) -> str:
    return (
        "<pre style='white-space:pre-wrap;margin:0;padding:10px;"
        "background:#f6f8fa;border-radius:6px;font-size:13px;line-height:1.45'>"
        f"{html.escape(text or '')}</pre>"
    )


def note_html(text: str) -> str:
    return (
        "<div style='margin:0;padding:10px;background:#f9fafb;"
        "border-radius:6px;line-height:1.45'>"
        f"{html.escape(text).replace(chr(10), '<br>')}</div>"
    )


def latency_html(lines: list[tuple[str, float]]) -> str:
    parts = [
        f"<div>总计: <code>{format_latency(lines[0][1])}</code></div>"
    ]
    parts.extend(
        f"<div style='color:#4b5563'>{html.escape(name)}: "
        f"<code>{format_latency(seconds)}</code></div>"
        for name, seconds in lines[1:]
    )
    return "".join(parts)


def titled_cell(title: str, content: str) -> str:
    return (
        "<div style='font-weight:700;color:#111827;margin-bottom:6px'>"
        f"{html.escape(title)}</div>{content}"
    )


def render_comparison_table(columns: list[dict[str, str]]) -> str:
    row_defs = [
        ("1. 样本标识", "sample"),
        ("2. 推理时间", "latency"),
        ("3. 推理实际输出", "output"),
        ("4. 防注入判断结果 / C4-C6 触发说明", "judgement"),
    ]
    header = (
        "<tr>"
        + "".join(
            "<th style='width:33%;text-align:left;vertical-align:bottom;"
            "padding:0 12px 8px 0;border:0!important;border-color:transparent!important;"
            "background:transparent!important;font-size:22px;font-weight:700;line-height:1.2'>"
            f"{col['header']}</th>"
            for col in columns
        )
        + "</tr>"
    )
    body = "\n".join(
        "<tr>"
        + "".join(
            "<td style='vertical-align:top;padding:10px 12px 10px 0;"
            "border:0!important;border-color:transparent!important;"
            "background:transparent!important'>"
            f"{titled_cell(label, col[key])}</td>"
            for col in columns
        )
        + "</tr>"
        for label, key in row_defs
    )
    return (
        "<table class='comparison-table' style='width:100%;border-collapse:collapse;"
        "border:0!important;border-color:transparent!important;"
        "font-size:14px;line-height:1.45;background:transparent!important'>"
        f"{header}{body}</table>"
    )


def risk_markdown(label: str, risk: float, probs: list) -> str:
    """Format the right-column head output as a Markdown block."""
    color = LABEL_COLOR.get(label, "#888")
    lines = [
        f"<div>判定标签: <span style='color:{color};font-weight:600'>"
        f"{html.escape(LABEL_CN.get(label, label))}</span></div>",
        "<div style='display:flex;align-items:center;gap:8px;margin-top:4px'>"
        f"<span>风险分: <code>{risk:.2f}</code></span>"
        "<span style='display:inline-block;width:96px;height:8px;background:#e5e7eb;"
        "border-radius:999px;overflow:hidden'>"
        f"<span style='display:block;width:{min(max(risk, 0.0), 1.0) * 100:.1f}%;"
        f"height:100%;background:{color}'></span></span></div>",
        "<div style='margin-top:6px'>三分类置信度:</div>",
    ]
    for i, name in enumerate(["clean", "direct", "indirect"]):
        p = probs[i]
        cn = LABEL_CN.get(name, name)
        p_color = LABEL_COLOR.get(name, "#888")
        lines.append(
            "<div style='display:grid;grid-template-columns:104px 48px 1fr;"
            "align-items:center;gap:6px;color:#4b5563;margin-top:3px'>"
            f"<span>{html.escape(cn)}</span>"
            f"<code>{p:.3f}</code>"
            "<span style='display:block;height:7px;background:#e5e7eb;"
            "border-radius:999px;overflow:hidden'>"
            f"<span style='display:block;width:{min(max(p, 0.0), 1.0) * 100:.1f}%;"
            f"height:100%;background:{p_color}'></span></span></div>"
        )
    return "".join(lines)


def base_warn_markdown(label_gt: str) -> str:
    """Red banner on the left column explaining the unprotected model."""
    if label_gt == "clean":
        return note_html(
            "⚠️ Base SmolVLM 无防护\n"
            "即使该样本本来是干净请求，base model 也可能给出与安全策略不符的回答。"
        )
    if label_gt == "direct":
        return note_html(
            "🚨 直接注入:易被攻破\n"
            "SmolVLM-500M 没有安全对齐,会沿用用户 prompt 内的指令。"
        )
    if label_gt == "indirect":
        return note_html(
            "🚨 间接注入:图文绕过\n"
            "攻击者把危险指令放在图像里,文字里装作中性请求。"
        )
    return note_html("Base SmolVLM 无防护")


def mpid_pass_markdown(label_gt: str, label_pred: str, risk: float) -> str:
    """Right-column verdict for the demo banner."""
    if label_pred == "clean":
        icon = "✅"
        msg = "MPID 判断为干净 → 放行"
    elif label_pred == "direct":
        icon = "🛑"
        msg = "MPID 判断为直接注入 → 拦截"
    elif label_pred == "indirect":
        icon = "🛑"
        msg = "MPID 判断为间接注入 → 拦截"
    else:
        icon, msg = "❓", f"未知标签: {label_pred}"
    agree = "✔ 正确" if label_pred == label_gt else "✘ 误判"
    return (
        f"<div>{icon} {msg}</div>"
        f"<div>标签: <code>{html.escape(label_pred)}</code></div>"
        f"<div>风险分: <code>{risk:.2f}</code></div>"
        f"<div>与 ground-truth ({html.escape(LABEL_CN.get(label_gt, label_gt))}) "
        f"对比: {agree}</div>"
    )


def judgement_panel(
    *,
    summary: str,
    mpid: str = "",
    c4c6: str = "",
    details: str = "",
) -> str:
    blocks = [("结果总结", summary)]
    if mpid:
        blocks.append(("MPID 判断结果", mpid))
    if c4c6:
        blocks.append(("C4-C6 判定结果", c4c6))
    if details:
        blocks.append(("其他详情", details))
    return "".join(
        "<div style='margin-bottom:8px'>"
        f"<div style='font-weight:700;color:#111827'>{title}</div>"
        f"<div>{content}</div>"
        "</div>"
        for title, content in blocks
    )


PROJECT_NOTE_MD = """
# MPID · Multimodal Prompt Injection Detection

> **一句话**:Base SmolVLM-500M 极易被 prompt injection 攻破;
> 加 **LoRA + 3-class head** 后,模型能识别并拦截 3 类风险
> (干净 / 直接注入 / 间接注入)。

## 威胁模型

| 类型 | 描述 | 样例 |
|---|---|---|
| **干净 (clean)** | 用户请求正常,无攻击意图 | "法国的首都是哪里?" |
| **直接注入 (direct)** | 文字本身就是攻击 prompt,试图接管模型 | DAN / OPPO / "ignore previous instructions" |
| **间接注入 (indirect)** | 攻击 payload 嵌在图像里,文字伪装成中性请求 | figstep:空表格诱导补全危险内容 |

## 模型

- **Backbone**: SmolVLM-500M (≈ 5×10⁸ params, Apache 2.0, COLM 2025)
- **Adapter**: PEFT LoRA (r=16, α=32, target=q/k/v/o)
- **Head**: 单层 Linear, 960 → 3 logits
- **训练数据**: mpid-v1 (≈ 25k 条, 见 `data/mpid-v1/EDA.md`)

## 三个核心算法优化 (Phase 3-5)

- **C4** 早退机制 (Phase 3) — 中间层 MLP head, 自适应推理
- **C5** 规则前置过滤 (Phase 4) — 黑/白名单 + Unicode + 结构 + 敏感指令
- **C6** 跨模态一致性 (Phase 5) — 辅助 prompt 判断图文相关性

## 已知限制

- demo 内置 checkpoint 是 **5 条样本的 smoke 训练** 产物 (用于端到端
  流水线验证)。生产质量需跑满 3 epoch (见 `configs/baseline.yaml`)。
- 间接注入 (figstep) 的图片仅 200 条,模型在 C6 cross-modal 子集上
  的能力未在 demo 中充分展示。
- SmolVLM 在中文长 prompt 上的生成质量有限;非中英语种可能误判。

## 参考

- 开题报告: `doc/opening-report-vlm.md`
- 任务分解: `doc/tasks.md` (§ Phase 2.5)
- 验证记录: `doc/VERIFICATION.md`
- HuggingFaceTB/SmolVLM-500M-Instruct
- JailbreakV-28K (Lou et al., 2024)
"""


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app(pipeline: DemoPipeline, samples: list[dict]) -> "gr.Blocks":
    """Build the Gradio Blocks app.

    The function is split out from ``main()`` so a smoke test can
    construct the Blocks object without launching the server.
    """
    import gradio as gr
    # Use matplotlib for the per-class confidence chart. ``gr.Plot`` (a
    # plotly iframe) triggers a gradio_client JSON-schema bug in some
    # version pairs; a static PNG via ``gr.Image`` is more portable.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Per-sample convenience: pre-build a button list with the title as
    # the label and the (text, image) tuple as the value.
    sample_choices = [(s["title"], s["index"]) for s in samples]
    sample_by_idx = {s["index"]: s for s in samples}

    def on_sample_click(idx: int):
        s = sample_by_idx.get(int(idx))
        if s is None:
            return "", None
        img = None
        if s.get("image"):
            abs_p = resolve_demo_asset(s["image"])
            if abs_p and abs_p.exists():
                img = str(abs_p)
        return s["text"], img

    def _make_bar_image(probs: list) -> str:
        """Render a 3-bar matplotlib chart to a PNG file and return the path.

        NOTE: as of Phase 2.5 UI iteration #3 the bar chart was removed
        from the side-by-side panel (the inline per-class bars inside
        ``risk_markdown`` are enough). The helper is kept here for the
        smoke script and for the screenshots dir convention, but no
        longer wired to the Gradio widgets.
        """
        labels = ["clean", "direct", "indirect"]
        colors = ["#22c55e", "#ef4444", "#f59e0b"]
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.bar(labels, probs, color=colors, edgecolor="black", linewidth=0.5)
        for i, p in enumerate(probs):
            ax.text(i, p + 0.02, f"{p:.2f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("P(class)")
        ax.set_title("三类置信度 (per-class confidence)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        out = DEMO_DIR / "screenshots" / "_bar.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(out)

    def run_compare(mode: str, text: str, image, sample_idx: int):
        """Top-level callback: run all three demo paths and return
        the output blocks (Base / MPID / MPID+C4-C6). Each block
        is structured as two clearly-separated sections:

          1. **模型输出** — the raw model output (or "发现注入风险,
             推理失败" if the model was blocked and produced nothing).
          2. **具体分析** — the explanatory / metric breakdown.
        """
        import time

        from mpid.infer import run_lora_only_pipeline, run_optimized_pipeline

        text_value = text or ""
        sample_idx = int(sample_idx or 0)
        s = sample_by_idx.get(sample_idx, {})
        if mode == "preset":
            label_gt = s.get("label", "")
            title = s.get("title", "自定义输入")
        else:
            label_gt = ""
            title = "自定义输入"
        compare_record = dict(s) if mode == "preset" else {}
        compare_record["text"] = text_value
        compare_record["image"] = image

        # ---------- Left column: Base SmolVLM ---------------------------
        # ``pipeline.generate`` calls ``self.adapter.generate`` which
        # uses the **raw base VLM** (``adapter.model``) — no LoRA, no
        # head. This is the "no-protection" view.
        t_base = time.perf_counter()
        base_text = pipeline.generate(text_value, image)
        base_seconds = time.perf_counter() - t_base
        if base_text:
            base_output_body = code_html(base_text)
        else:
            base_output_body = note_html("🛑 发现注入风险,推理失败\nBase VLM 没有产生任何文本输出。")
        sample_html = f"<strong>📝 样本:</strong> {html.escape(title)}"

        # ---------- Right column: MPID (LoRA + 3-class head) -----------
        lora_result = run_lora_only_pipeline(
            compare_record,
            classify_fn=pipeline.classify,
            generate_fn=pipeline.generate_with_lora,
        )
        cls = lora_result.head or {}
        label_pred = str(cls.get("label") or "fallback")
        risk = float(cls.get("risk") or 0.0)
        probs = list(cls.get("probs") or [0.0, 0.0, 0.0])

        if lora_result.output:
            mpid_output_body = code_html(lora_result.output)
        elif lora_result.action == "block":
            mpid_output_body = note_html("发现注入风险,推理失败\n已拦截,未调用 LoRA 生成。")
        else:
            mpid_output_body = note_html("已放行,但 MPID 模型未产生输出。")

        verdict_line = mpid_pass_markdown(label_gt, label_pred, risk)
        risk_table = risk_markdown(label_pred, risk, probs)
        mpid_summary = (
            "放行,并执行 LoRA 生成。"
            if label_pred == "clean"
            else f"拦截,未执行 LoRA 生成。"
        )
        mpid_header = (
            MPID_HEADER_BLOCK_MD
            if lora_result.action == "block"
            else MPID_HEADER_PASS_MD
        )
        mpid_latency = latency_html([
                ('total', lora_result.timings.get("total_seconds", 0.0)),
                ('head 判定', lora_result.timings.get("head_seconds", 0.0)),
                ('LoRA 生成', lora_result.timings.get("generate_seconds", 0.0)),
            ])
        mpid_judgement = judgement_panel(
            summary=note_html(mpid_summary),
            mpid=verdict_line,
            details=risk_table,
        )

        # ---------- Third column: MPID + C4/C5/C6 lightweight path -------
        opt_result = run_optimized_pipeline(
            compare_record,
            classify_fn=pipeline.classify,
            generate_fn=pipeline.generate_with_lora,
        )
        opt_head = opt_result.head or {}
        opt_label_pred = str(opt_head.get("label") or "not_run")
        opt_risk = float(opt_head.get("risk") or 0.0)
        opt_probs = list(opt_head.get("probs") or [0.0, 0.0, 0.0])
        if opt_result.output:
            opt_output_body = code_html(opt_result.output)
        else:
            opt_output_body = note_html(
                "发现注入风险,推理失败\n已拦截或未触发生成。"
                if opt_result.action == "block"
                else "已放行,但 LoRA 模型未产生输出。"
            )
        opt_latency_parts = [
            ("total", opt_result.timings.get("total_seconds", 0.0)),
            ("C5 规则", opt_result.timings.get("c5_seconds", 0.0)),
            ("C6 跨模态", opt_result.timings.get("c6_seconds", 0.0)),
            ("head 判定", opt_result.timings.get("head_seconds", 0.0)),
            ("C4 早退", opt_result.timings.get("c4_seconds", 0.0)),
            ("LoRA 生成", opt_result.timings.get("generate_seconds", 0.0)),
        ]
        if opt_result.stage == "c4_early_exit":
            opt_stage_note = "C5/C6 未命中后,head 的 P(clean) 超过阈值,触发 C4 clean 早退。"
        elif opt_result.stage == "c5_rules":
            opt_stage_note = "C5 文本规则先于 head 命中,直接拦截。"
        elif opt_result.stage == "c6_crossmodal":
            opt_stage_note = "C5 未命中,C6 跨模态启发式先于 head 命中,直接拦截。"
        elif opt_result.stage == "head_clean_fallback":
            opt_stage_note = "C5/C6/C4 均未做最终决策,回退到 head clean 放行。"
        elif opt_result.stage == "head_injection_fallback":
            opt_stage_note = "C5/C6/C4 均未做最终决策,回退到 head 注入判定并拦截。"
        else:
            opt_stage_note = "优化调度器返回了未识别阶段。"

        opt_header = (
            OPT_HEADER_BLOCK_MD if opt_result.action == "block" else OPT_HEADER_PASS_MD
        )
        opt_mpid_judgement = (
            mpid_pass_markdown(label_gt, opt_label_pred, opt_risk)
            if opt_result.head
            else note_html("未执行 head。C5/C6 已在前置阶段完成判定。")
        )
        opt_c4c6_judgement = (
            f"<div>最终动作: <code>{opt_result.action}</code></div>"
            f"<div>最终标签: <code>{opt_result.label}</code></div>"
            f"<div>命中阶段: <code>{opt_result.stage}</code></div>"
            f"<div>触发说明: {opt_stage_note}</div>"
        )
        opt_summary = (
            "放行,并执行 LoRA 生成。"
            if opt_result.action == "allow"
            else "拦截,未执行 LoRA 生成。"
        )
        opt_judgement = judgement_panel(
            summary=note_html(opt_summary),
            mpid=opt_mpid_judgement,
            c4c6=opt_c4c6_judgement,
            details=(
            f"{risk_markdown(opt_label_pred, opt_risk, opt_probs) if opt_result.head else ''}"
            f"{code_html(json.dumps(opt_result.explanation, ensure_ascii=False, indent=2))}"
            ),
        )
        comparison_md = render_comparison_table([
            {
                "header": BASE_HEADER_MD,
                "sample": sample_html,
                "latency": latency_html([("total", base_seconds)]),
                "output": base_output_body,
                "judgement": judgement_panel(
                    summary=note_html("原始模型输出,不执行防注入拦截。"),
                    details=base_warn_markdown(label_gt),
                ),
            },
            {
                "header": mpid_header,
                "sample": sample_html,
                "latency": mpid_latency,
                "output": mpid_output_body,
                "judgement": mpid_judgement,
            },
            {
                "header": opt_header,
                "sample": sample_html,
                "latency": latency_html(opt_latency_parts),
                "output": opt_output_body,
                "judgement": opt_judgement,
            },
        ])

        return comparison_md

    def show_processing():
        return render_comparison_table([
            {
                "header": BASE_HEADER_MD,
                "sample": "等待运行",
                "latency": "处理中",
                "output": PROCESSING_MD,
                "judgement": "处理中",
            },
            {
                "header": MPID_HEADER_PASS_MD,
                "sample": "等待运行",
                "latency": "处理中",
                "output": PROCESSING_MD,
                "judgement": "处理中",
            },
            {
                "header": OPT_HEADER_PASS_MD,
                "sample": "等待运行",
                "latency": "处理中",
                "output": PROCESSING_MD,
                "judgement": "处理中",
            },
        ])

    # Sentinel used when the user is in "preset" mode and the text/image
    # widgets are locked. We disable the widgets via ``interactive=False``
    # rather than replacing their values, so that re-entering "custom"
    # mode still works.
    def _on_mode_change(mode: str):
        """When switching modes, lock/unlock the relevant widgets and
        (in preset mode) auto-fill text+image from the chosen sample."""
        if mode == "preset":
            s = sample_by_idx.get(int(samples[0]["index"]), {})
            img = None
            if s.get("image"):
                abs_p = resolve_demo_asset(s["image"])
                if abs_p and abs_p.exists():
                    img = str(abs_p)
            # gr.update(...) returns a sentinel that Gradio uses to
            # patch a single component on the next render.
            return (
                gr.update(visible=True, interactive=True),  # sample_dd shown
                gr.update(value=s.get("text", ""), interactive=False),  # text locked
                gr.update(value=img, interactive=False),     # image locked
            )
        # Custom input mode.
        return (
            gr.update(visible=False, interactive=False),               # sample_dd hidden
            gr.update(value="", interactive=True, placeholder=(
                "✍️ 自定义输入模式:在此键入任意 prompt 文本。"
            )),
            gr.update(value=None, interactive=True),    # image editable
        )

    with gr.Blocks(
        title="MPID · 多模态 prompt 注入检测 Demo",
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 1480px !important; }
            #sample-dropdown .wrap { font-size: 14px; }
            .comparison-table,
            .comparison-table tr,
            .comparison-table th,
            .comparison-table td {
                border: 0 !important;
                border-color: transparent !important;
                background: transparent !important;
                box-shadow: none !important;
            }
        """,
    ) as app:
        gr.Markdown(
            "# 🛡️ MPID · 多模态 Prompt 注入检测\n"
            "**对比 Base SmolVLM vs MPID (LoRA only) vs MPID (LoRA + C4-C6 优化)**"
        )

        # --- 1. Mode switch (preset vs custom) --------------------------
        mode_radio = gr.Radio(
            choices=[("📦 预置样本", "preset"),
                     ("✍️ 自定义输入", "custom")],
            value="preset",
            label="输入模式 (mode)",
            show_label=False,
            interactive=True,
        )

        # --- 2. Preset sample dropdown ----------------------------------
        sample_dd = gr.Dropdown(
            choices=[(title, idx) for title, idx in sample_choices],
            value=samples[0]["index"],
            label="📦 预置样本 (clean × 3 / direct × 3 / indirect × 2)",
            interactive=True,
            elem_id="sample-dropdown",
        )

        # --- 3. Free-form input area ------------------------------------
        with gr.Row():
            with gr.Column(scale=3):
                text_in = gr.Textbox(
                    label="用户 prompt (text)",
                    lines=4,
                    placeholder=(
                        "默认预置样本模式:文本由上方 Dropdown 锁定,无法编辑。"
                        "切换到「✍️ 自定义输入」可在此自由输入。"
                    ),
                    interactive=False,  # locked in preset mode at boot
                )
            with gr.Column(scale=2):
                image_in = gr.Image(
                    label="图像 (image, 可选)",
                    type="filepath",
                    height=180,
                    interactive=False,  # locked in preset mode at boot
                )

        run_btn = gr.Button("▶ 运行对比 (Run)", variant="primary")

        # --- 4. Aligned comparison table --------------------------------
        comparison_out = gr.Markdown()

        # Wire up the callbacks.
        mode_radio.change(
            _on_mode_change,
            inputs=[mode_radio],
            outputs=[sample_dd, text_in, image_in],
        )
        sample_dd.change(
            on_sample_click,
            inputs=[sample_dd],
            outputs=[text_in, image_in],
        )
        run_event = run_btn.click(
            show_processing,
            inputs=None,
            outputs=[comparison_out],
        )
        run_event.then(
            run_compare,
            inputs=[mode_radio, text_in, image_in, sample_dd],
            outputs=[comparison_out],
        )

        # Pre-fill the text/image widgets with the first sample on load.
        app.load(
            on_sample_click,
            inputs=[sample_dd],
            outputs=[text_in, image_in],
        )

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPID Phase 2.5 demo")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help="SmolVLM local directory (default: runs/_models/smolvlm-500m)")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="LoRA + head checkpoint (default: runs/_templates/artifacts/checkpoints/lora_baseline.safetensors)")
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES,
                   help="Preset samples JSON (default: demo/samples.json)")
    p.add_argument("--device", default=DEFAULT_DEVICE,
                   help="Compute device (default: cpu; mac users may pass mps)")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="Max tokens for the base VLM generation (default: 128)")
    p.add_argument("--server-name", default="127.0.0.1",
                   help="Server bind address (default: 127.0.0.1)")
    p.add_argument("--server-port", type=int, default=7860,
                   help="Server port (default: 7860)")
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link (off by default)")
    return p.parse_args()


def main() -> int:
    import time

    args = parse_args()
    t0 = time.perf_counter()
    print(f"[demo] repo_root = {REPO_ROOT}")
    print(f"[demo] model_dir = {args.model_dir}")
    print(f"[demo] checkpoint= {args.checkpoint}")
    print(f"[demo] samples  = {args.samples}")
    print(f"[demo] device   = {args.device}")

    # Sanity checks before the model load.
    if not args.model_dir.exists():
        print(f"[demo] ERROR: model dir not found: {args.model_dir}", file=sys.stderr)
        return 2
    if not args.checkpoint.exists():
        print(f"[demo] ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.samples.exists():
        print(f"[demo] ERROR: samples not found: {args.samples}", file=sys.stderr)
        return 2

    with open(args.samples, encoding="utf-8") as f:
        samples = json.load(f)
    if len(samples) != 8:
        print(f"[demo] WARNING: expected 8 samples, got {len(samples)}", file=sys.stderr)

    pipeline = DemoPipeline(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    t_ready = time.perf_counter() - t0
    print(f"[demo] pipeline ready in {t_ready:.1f} s")

    app = build_app(pipeline, samples)

    # Stay offline by default; --share opens a public tunnel.
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
