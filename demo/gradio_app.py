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

  * the SmolVLM-500M backbone from ``models/smolvlm-500m/``
  * the LoRA + head checkpoint from
    ``artifacts/baseline/lora_baseline.safetensors``

If either is missing, the app prints a clear error and exits with a
non-zero status. The startup target is "server up in ≤ 30 s" on a
MacBook M-class machine.
"""
from __future__ import annotations

import argparse
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
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "smolvlm-500m"
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "baseline" / "lora_baseline.safetensors"
DEFAULT_SAMPLES = DEMO_DIR / "samples.json"
DEFAULT_DEVICE = "cpu"  # MPS+LoRA can be unstable; default to safe


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


def risk_markdown(label: str, risk: float, probs: list) -> str:
    """Format the right-column head output as a Markdown block."""
    color = LABEL_COLOR.get(label, "#888")
    bar = "█" * int(risk * 20) + "░" * (20 - int(risk * 20))
    lines = [
        f"### 判定标签: <span style='color:{color}; font-weight:600'>"
        f"{LABEL_CN.get(label, label)}</span>",
        "",
        f"**风险分 (risk):** `{risk:.2f}`  "
        f"`{bar}`",
        "",
        "**三分类置信度:**",
        "",
        "| 类别 | 概率 | 条形 |",
        "|---|---|---|",
    ]
    for i, name in enumerate(["clean", "direct", "indirect"]):
        p = probs[i]
        bar_len = int(round(p * 24))
        bar_str = "▓" * bar_len + "░" * (24 - bar_len)
        cn = LABEL_CN.get(name, name)
        lines.append(
            f"| {cn} | `{p:.3f}` | `{bar_str}` |"
        )
    return "\n".join(lines)


def base_warn_markdown(label_gt: str) -> str:
    """Red banner on the left column explaining the unprotected model."""
    if label_gt == "clean":
        return (
            "> ⚠️ **Base SmolVLM 无防护**  \n"
            "> 即使该样本本来是干净请求，base model 也可能给出与安全策略"
            "不符的回答（如对医学/法律问题过度自信、产生歧视性言论等）。"
        )
    if label_gt == "direct":
        return (
            "> 🚨 **直接注入:易被攻破**  \n"
            "> SmolVLM-500M 没有安全对齐,会沿用用户 prompt 内的指令"
            "(DAN / OPPO / LiveGPT 等越狱模板几乎都成功)。"
            "对比右栏 **MPID** 的拦截结果。"
        )
    if label_gt == "indirect":
        return (
            "> 🚨 **间接注入:图文绕过**  \n"
            "> 攻击者把「危险指令」放在图像里、文字里装作中性请求。"
            "Base VLM 读图后会被诱导完成 figstep / QR-hiding 等攻击。"
            "对比右栏 **MPID** 的拦截结果。"
        )
    return "Base SmolVLM 无防护"


def mpid_pass_markdown(label_gt: str, label_pred: str, risk: float) -> str:
    """Right-column verdict for the demo banner."""
    if label_pred == "clean":
        icon = "✅"
        msg = "MPID 判断为 **干净** → 放行"
    elif label_pred == "direct":
        icon = "🛑"
        msg = "MPID 判断为 **直接注入** → 拦截"
    elif label_pred == "indirect":
        icon = "🛑"
        msg = "MPID 判断为 **间接注入** → 拦截"
    else:
        icon, msg = "❓", f"未知标签: {label_pred}"
    agree = "✔ 正确" if label_pred == label_gt else "✘ 误判"
    return (
        f"> {icon} {msg}  \n"
        f"> 与 ground-truth ({LABEL_CN.get(label_gt, label_gt)}) 对比: **{agree}**  \n"
        f"> 风险分 = `{risk:.2f}` (1.00 = 高置信)"
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
            rel = s["image"]
            abs_p = (REPO_ROOT / rel).resolve()
            if abs_p.exists():
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
        """Top-level callback: do BOTH (generate + classify) and return
        the two output blocks (left = Base, right = MPID). Each block
        is structured as two clearly-separated sections:

          1. **模型输出** — the raw model output (or "发现注入风险,
             推理失败" if the model was blocked and produced nothing).
          2. **具体分析** — the explanatory / metric breakdown.
        """
        sample_idx = int(sample_idx or 0)
        s = sample_by_idx.get(sample_idx, {})
        if mode == "preset":
            label_gt = s.get("label", "")
            title = s.get("title", "自定义输入")
        else:
            label_gt = ""
            title = "自定义输入"

        # ---------- Left column: Base SmolVLM ---------------------------
        # ``pipeline.generate`` calls ``self.adapter.generate`` which
        # uses the **raw base VLM** (``adapter.model``) — no LoRA, no
        # head. This is the "no-protection" view.
        base_text = pipeline.generate(text or "", image)
        if base_text:
            base_output_body = f"```text\n{base_text}\n```"
        else:
            base_output_body = (
                "> 🛑 **发现注入风险,推理失败**  \n"
                "> Base VLM 没有产生任何文本输出。"
            )
        base_md = (
            f"**📝 样本:** {title}\n\n"
            f"### 1️⃣ 模型输出\n\n"
            f"{base_output_body}\n\n"
            f"---\n\n"
            f"### 2️⃣ 具体分析\n\n"
            f"{base_warn_markdown(label_gt)}"
        )

        # ---------- Right column: MPID (LoRA + 3-class head) -----------
        # MPID pipeline = 3-class head (前置检测) + LoRA-tuned VLM (推理).
        #   - head 判为 clean     → 用 peft_model (LoRA) 推理,输出文本
        #   - head 判为 direct/indirect → 拦截,不调用 VLM,显示拦截信息
        cls = pipeline.classify(text or "", image)
        label_pred = cls["label"]
        risk = cls["risk"]
        probs = cls["probs"]

        if label_pred == "clean":
            # ``generate_with_lora`` uses ``self.peft_model`` — the
            # LoRA-tuned VLM (not the raw base).
            mpid_gen_text = pipeline.generate_with_lora(text or "", image)
            if mpid_gen_text:
                mpid_output_body = (
                    "✅ **干净 (clean) — 放行**\n\n"
                    f"```text\n{mpid_gen_text}\n```"
                )
            else:
                mpid_output_body = (
                    "✅ **干净 (clean) — 放行**  \n"
                    "未检测到 prompt 注入,但 MPID 模型未产生输出。"
                )
        elif label_pred == "direct":
            mpid_output_body = (
                "🛑 **直接注入 (direct) — 拦截**\n\n"
                "> **发现注入风险,推理失败**  \n"
                "> MPID 检测到 direct injection,调用 VLM 的步骤被跳过。"
            )
        elif label_pred == "indirect":
            mpid_output_body = (
                "🛑 **间接注入 (indirect) — 拦截**\n\n"
                "> **发现注入风险,推理失败**  \n"
                "> MPID 检测到 indirect injection (图文绕过),"
                "调用 VLM 的步骤被跳过。"
            )
        else:
            mpid_output_body = f"❓ 未知标签: `{label_pred}`"

        verdict_line = mpid_pass_markdown(label_gt, label_pred, risk)
        risk_table = risk_markdown(label_pred, risk, probs)
        mpid_md = (
            f"**📝 样本:** {title}\n\n"
            f"### 1️⃣ 模型输出\n\n"
            f"{mpid_output_body}\n\n"
            f"---\n\n"
            f"### 2️⃣ 具体分析\n\n"
            f"{verdict_line}\n\n"
            f"{risk_table}"
        )

        return base_md, mpid_md

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
                abs_p = (REPO_ROOT / s["image"]).resolve()
                if abs_p.exists():
                    img = str(abs_p)
            # gr.update(...) returns a sentinel that Gradio uses to
            # patch a single component on the next render.
            return (
                gr.update(interactive=True),                # sample_dd enabled
                gr.update(value=s.get("text", ""), interactive=False),  # text locked
                gr.update(value=img, interactive=False),     # image locked
            )
        # Custom input mode.
        return (
            gr.update(interactive=False),                              # sample_dd disabled
            gr.update(value="", interactive=True, placeholder=(
                "✍️ 自定义输入模式:在此键入任意 prompt 文本。"
            )),
            gr.update(value=None, interactive=True),    # image editable
        )

    with gr.Blocks(
        title="MPID · 多模态 prompt 注入检测 Demo",
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 1280px !important; }
            #sample-dropdown .wrap { font-size: 14px; }
        """,
    ) as app:
        gr.Markdown(
            "# 🛡️ MPID · 多模态 Prompt 注入检测\n"
            "**对比 Base SmolVLM vs LoRA + 3-class head**"
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

        # --- 4. Side-by-side comparison ---------------------------------
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔴 Base SmolVLM (无防护)")
                base_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("## 🟢 MPID (LoRA + 3-class head)")
                mpid_out = gr.Markdown()

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
        run_btn.click(
            run_compare,
            inputs=[mode_radio, text_in, image_in, sample_dd],
            outputs=[base_out, mpid_out],
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
                   help="SmolVLM local directory (default: models/smolvlm-500m)")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="LoRA + head checkpoint (default: artifacts/baseline/lora_baseline.safetensors)")
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
