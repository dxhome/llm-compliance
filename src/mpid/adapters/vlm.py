"""VLM adapter wrapping SmolVLM-500M (Phase 2 / T2.1).

The adapter is a thin layer on top of ``transformers.AutoModelForVision2Seq``
that exposes a uniform ``forward(image, text) -> (logits, last_hidden)``
interface. It is used by:

  * the training loop (``src/mpid/train/trainer.py``), where the hidden
    state is fed into a 3-class head and the cross-entropy loss is
    computed against a label index;
  * the inference path (``scripts/infer.py``), where the adapter returns
    3-class logits and the caller applies ``argmax`` + softmax;
  * the offline package (``mpid_offline/``), where it is the only model
    dependency the runtime needs.

The adapter deliberately does **not** hold any business logic (no
prompt template, no label set, no head). Those are injected from
``mpid.data.prompt`` and ``mpid.heads.classification``.

Three orthogonal axes are configurable at construction time:

  * ``backbone_name``         — registry key (T2.2)
  * ``dtype``                 — ``float32`` / ``float16`` / ``bfloat16``
  * ``quantization``          — ``None`` / ``"4bit"`` / ``"8bit"`` (bnb)
  * ``device``                — ``mpid.device.get_device()`` output
  * ``gradient_checkpointing`` — bool, save memory at the cost of speed
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image

from mpid.device import get_device


# A 512x512 single-colour canvas used as a placeholder for text-only
# records. The model still receives the <image> token (mandatory for
# Idefics3) but the pixel content carries no signal — the prediction
# is effectively text-only.
_PLACEHOLDER_IMG: Optional[Image.Image] = None


def _get_placeholder_image() -> Image.Image:
    global _PLACEHOLDER_IMG
    if _PLACEHOLDER_IMG is None:
        _PLACEHOLDER_IMG = Image.new("RGB", (512, 512), (235, 235, 235))
    return _PLACEHOLDER_IMG


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class VLMAdapter:
    """Wrap a vision-language backbone for 3-class detection inference.

    Construction is heavy (loads the model + processor). After that,
    ``forward(image, text)`` is the only hot-path entry point.

    The class also caches the processor and exposes a ``num_layers`` /
    ``hidden_size`` view that downstream code (heads, early-exit, etc.)
    can read without knowing the backbone.
    """

    def __init__(
        self,
        backbone_name: str = "smolvlm-500m",
        *,
        dtype: str = "float16",
        quantization: Optional[str] = None,
        device: Optional[str] = None,
        gradient_checkpointing: bool = False,
        models_root: Optional[Union[str, Path]] = None,
    ) -> None:
        # Late import: torch + transformers are heavy; we want the
        # adapter import to be cheap for tooling / unit tests.
        from transformers import AutoModelForVision2Seq, AutoProcessor

        from mpid.backbones.registry import resolve_local_path

        self.backbone_name = backbone_name
        self.dtype = dtype
        self.quantization = quantization
        self.gradient_checkpointing = gradient_checkpointing
        self.device = device or get_device(prefer=None)

        # Resolve model path locally so we never hit the network.
        self.local_path = resolve_local_path(backbone_name, models_root=models_root)
        self.processor = AutoProcessor.from_pretrained(
            str(self.local_path), local_files_only=True
        )

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

        kwargs: dict = dict(
            torch_dtype=torch_dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        # Quantization is intentionally NOT applied on macOS (P0A-1 § 2:
        # bnb wheels lack a CUDA build; the mac wheel is CPU-only and
        # cannot quantize MPS tensors). The ``quantization`` arg is
        # accepted for API symmetry with the x86 + CUDA target.
        if quantization in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=(quantization == "4bit"),
                    load_in_8bit=(quantization == "8bit"),
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            except Exception as e:  # pragma: no cover - mac path
                raise RuntimeError(
                    f"quantization={quantization!r} requested but bitsandbytes "
                    f"is unavailable on this platform: {e}"
                ) from e

        self.model = AutoModelForVision2Seq.from_pretrained(
            str(self.local_path), **kwargs
        )
        # Move to target device (MPS / CUDA / CPU).
        if self.device != "cpu":
            self.model.to(self.device)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            # Idefics3's gradient checkpointing needs this flag or
            # the input embedding grads are silently dropped.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        self.model.eval()
        self._hidden_size: Optional[int] = None

    # -- introspection ----------------------------------------------------

    @property
    def hidden_size(self) -> int:
        if self._hidden_size is None:
            cfg = self.model.config
            # Idefics3 nests text config; for the language part the
            # hidden size is the canonical "text hidden size".
            self._hidden_size = (
                getattr(cfg, "hidden_size", None)
                or getattr(cfg.text_config, "hidden_size", None)
                or 960
            )
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        cfg = self.model.config
        return (
            getattr(cfg.text_config, "num_hidden_layers", None)
            or getattr(cfg, "num_hidden_layers", None)
            or 32
        )

    # -- I/O --------------------------------------------------------------

    def _resolve_image(self, image: Optional[Union[str, Path, Image.Image]]) -> Image.Image:
        if image is None:
            return _get_placeholder_image()
        if isinstance(image, Image.Image):
            return image
        return Image.open(image).convert("RGB")

    def preprocess(
        self,
        text: str,
        image: Optional[Union[str, Path, Image.Image]] = None,
    ) -> dict:
        """Run the processor and return a dict of tensors on the right device.

        We do **not** move ``pixel_values`` to the model's dtype here;
        the model does that internally. We only move to the device.
        """
        img = self._resolve_image(image)
        # The processor requires the <image> token in the text. The
        # prompt template (T2.4) is responsible for inserting it.
        if "<image>" not in text:
            text = "<image>" + text
        encoded = self.processor(
            text=text,
            images=[img],
            return_tensors="pt",
        )
        # Move all tensors to the target device. We keep float32 for
        # pixel_values (the vision encoder expects fp32) and let the
        # model cast as needed.
        out = {}
        for k, v in encoded.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    @torch.inference_mode()
    def forward(
        self,
        text: str,
        image: Optional[Union[str, Path, Image.Image]] = None,
    ) -> dict:
        """Run the model and return ``{"logits": (1, V), "last_hidden": (1, D)}``.

        ``logits`` is the LM head's logits at the **last input position**
        (the position whose hidden state we want to classify).
        ``last_hidden`` is the corresponding hidden state.
        """
        encoded = self.preprocess(text, image)
        outputs = self.model(**encoded, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]   # (1, T, D)
        # The last *non-pad* position is the position we want to classify.
        # For Phase 2 we use the last position of the input (T - 1) since
        # all inputs are right-padded to the same length per batch item.
        last_idx = encoded["attention_mask"].sum(dim=1) - 1
        b = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[b, last_idx]  # (1, D)
        # LM logits at the same position (for "answer token" alternative).
        lm_logits = outputs.logits[b, last_idx]  # (1, V)
        return {"last_hidden": pooled, "logits": lm_logits}

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        image: Optional[Union[str, Path, Image.Image]] = None,
        *,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Free-form text generation — used by the demo (T2.5.1).

        Unlike :meth:`forward`, this method lets the base VLM respond to
        the user prompt in a chat-like manner (no 3-class head, no
        LoRA-trained safety). It exists to power the **left column** of
        the Phase 2.5 demo, where the user sees that the unprotected
        base model will happily comply with prompt-injection attacks.

        The implementation:

          1. Builds a single-turn ``[user, image, text]`` message and
             applies the processor's chat template (matches the format
             used by SmolVLM-Instruct).
          2. Tokenises + pixel-encodes via the same processor.
          3. Calls ``model.generate(...)`` with greedy decoding by
             default (deterministic, repeatable in the demo).
          4. Decodes only the **new** tokens past the prompt length, so
             the response is just the model's reply (not the prompt).
        """
        img = self._resolve_image(image)
        # Build the chat-template prompt. The single image is referenced
        # first, then the user text — this is the format the model's
        # Idefics3 backbone was instruction-tuned on.
        user_content: list[dict] = []
        if image is not None:
            user_content.append({"type": "image"})
        else:
            # The chat template requires an image placeholder to keep
            # the visual-token slot; ``processor.apply_chat_template``
            # handles ``None`` by inserting a stub when no image is
            # given, but we add an explicit empty-image dict for
            # consistency with the single-image path.
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": text or ""})
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        # Tokenise + image-encode in one shot. The processor will
        # expand ``<image>`` into the visual-token sequence expected by
        # the Idefics3 backbone.
        encoded = self.processor(
            text=[prompt], images=[img], return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()
                   if torch.is_tensor(v)}
        # ``generate`` is incompatible with gradient-checkpointed /
        # peft-wrapped trainable graphs in some HF versions; we are in
        # inference mode here, but still keep the kwargs minimal.
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = 0.7
            gen_kwargs["top_p"] = 0.9
        output_ids = self.model.generate(**encoded, **gen_kwargs)
        # Strip the prompt prefix so we only return the newly generated
        # text. The input is batch size 1 so ``[0, input_len:]`` is the
        # right slice.
        prompt_len = encoded["input_ids"].shape[1]
        new_ids = output_ids[0, prompt_len:]
        text_out = self.processor.batch_decode(
            [new_ids], skip_special_tokens=True,
        )[0].strip()
        return text_out

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()


__all__ = ["VLMAdapter", "_get_placeholder_image"]
