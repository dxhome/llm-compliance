"""Offline package builder (Phase 2 / T2.11).

Bundles everything needed to run MPID inference on a target machine
that has only Python + the ``mpid`` package (and its pinned deps)
installed. The package directory contains:

  * ``models/<backbone>/``            — backbone weights
  * ``artifacts/checkpoints/lora_baseline.safetensors``
  * ``infer.py``                       — single-sample CLI entry
  * ``requirements.txt``               — pinned dependency list
  * ``CHECKSUMS.txt``                  — sha256 of every file
  * ``MANIFEST.json``                  — what was packaged and how

After packaging, the script writes ``package_offline.json`` with
the artefact sizes and a list of files. The companion smoke test
(:mod:`scripts.smoke_offline`) unpacks the package in a tempdir
and runs ``infer.py`` end-to-end without contacting the network.

Usage::

    python scripts/package_offline.py
    python scripts/package_offline.py --src models/smolvlm-500m \\
                                      --ckpt runs/my_run/artifacts/checkpoints/lora_final.safetensors \\
                                      --out runs/my_run/artifacts/package/mpid_offline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


PACKAGE_INFER = '''#!/usr/bin/env python3
"""Single-sample inference for the MPID offline package.

Reads a JSON line from stdin of the form ``{"text": "...", "image": null}``
and prints ``{"label": "clean|direct|indirect", "risk": 0.0-1.0}`` on stdout.

The script loads the backbone from ``models/<backbone>`` and the
head from ``artifacts/lora_baseline.safetensors`` — both relative
to the package root (the directory containing this script). It
makes **no network calls**.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the in-package mpid/ importable. The package is shipped with
# the source tree under ``src/`` so we add that to sys.path.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
os.environ.setdefault("MPID_OCR_MODELS_DIR", str(_HERE / "models" / "ocr"))

from mpid.adapters.vlm import VLMAdapter
from mpid.crossmodal import check_crossmodal, check_ocr_conflict, extract_ocr_text
from mpid.heads.classification import IDX2LABEL, NUM_CLASSES, ClassificationHead
from mpid.data.prompt import build_prompt
from mpid.early_exit import EarlyExitConfig, should_early_exit
from mpid.rules import scan_text
from mpid.train.trainer import inject_lora, load_checkpoint, apply_lora_state


# --- 1. Locate the artefacts inside the package ---------------------------
BACKBONE_DIR = _HERE / "models" / "smolvlm-500m"
META = json.loads((_HERE / "MANIFEST.json").read_text())
CHECKPOINT = _HERE / "artifacts" / META["checkpoint"]
LORA_R = int(META["lora_r"])
LORA_ALPHA = int(META["lora_alpha"])
LORA_TARGET = META["lora_target"]
CLEAN_THRESHOLD = float(META.get("clean_threshold", 0.95))
R0_INDIRECT_OFFSET = float(META.get("r0_indirect_logit_offset", 0.0))
IMAGE_OCR_DIRECT_PENALTY = float(META.get("image_ocr_direct_logit_penalty", 0.0))
ENABLE_MCR = bool(META.get("mcr", {}).get("enabled", False))


# --- 2. Stub config (matches the values that produced the checkpoint) ----
class _Cfg:
    backbone_name = "smolvlm-500m"
    dtype = "float32"
    device = "cpu"
    quantization = None
    lora_r = LORA_R
    lora_alpha = LORA_ALPHA
    lora_dropout = 0.0
    lora_target = LORA_TARGET


# --- 3. Load adapter + LoRA + head ----------------------------------------
adapter = VLMAdapter(
    backbone_name=_Cfg.backbone_name,
    dtype=_Cfg.dtype,
    device=_Cfg.device,
    models_root=_HERE / "models",
)
peft_model, _ = inject_lora(adapter.model, _Cfg())
head = ClassificationHead(in_dim=adapter.hidden_size,
                          num_classes=NUM_CLASSES).to(_Cfg.device)
state = load_checkpoint(CHECKPOINT, head)
apply_lora_state(peft_model, state)
peft_model.eval(); head.eval()


def predict(text: str, image=None, *, mcr_active: bool = False) -> dict:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    prompt_record = None
    if mcr_active:
        prompt_record = {
            "prompt_version": "trusted_boundary_v2",
            "content_role": "untrusted_image_ocr",
        }
    prompt = build_prompt(text, record=prompt_record)
    img = image if image is not None else Image.new("RGB", (512, 512), (235, 235, 235))
    enc = adapter.preprocess(prompt, img)
    with torch.inference_mode():
        out = peft_model(**enc, output_hidden_states=True)
    last_h = out.hidden_states[-1]
    last_idx = enc["attention_mask"].sum(dim=1) - 1
    b = torch.arange(last_h.size(0), device=last_h.device)
    pooled = last_h[b, last_idx]
    # F-3000-MCR-SBC: the frozen +0.55 indirect offset is global. The
    # -0.20 direct adjustment applies only when local OCR confirms that the
    # image carries text and MCR therefore activated the untrusted-image role.
    logits = head(pooled)
    logits[:, 2] += R0_INDIRECT_OFFSET
    if mcr_active:
        logits[:, 1] += IMAGE_OCR_DIRECT_PENALTY
    probs = F.softmax(logits, dim=-1)
    label_idx = probs.argmax(dim=-1)
    return {
        "label": IDX2LABEL[int(label_idx[0].item())],
        "risk": float(probs.max(dim=-1).values[0].item()),
        "probs": probs[0].detach().cpu().tolist(),
        "mcr_active": mcr_active,
    }


def _model_image(value):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.exists():
            from PIL import Image
            return Image.open(path).convert("RGB")
    return value


def optimized_predict(text: str, image=None) -> dict:
    import torch

    record = {"text": text, "image": image}
    c5 = scan_text(text)
    if c5.blocked:
        return {
            "label": c5.label,
            "risk": 1.0,
            "action": "block",
            "stage": "c5_rules",
            "explanation": c5.to_dict(),
        }

    ocr = extract_ocr_text(image)
    c6b = check_ocr_conflict(ocr)
    if c6b.suspicious:
        return {
            "label": c6b.label,
            "risk": 1.0,
            "action": "block",
            "stage": "c6b_lite_ocr",
            "explanation": c6b.to_dict(),
        }

    c6 = check_crossmodal(record)
    if c6.suspicious:
        return {
            "label": c6.label,
            "risk": 1.0,
            "action": "block",
            "stage": "c6_crossmodal",
            "explanation": c6.to_dict(),
        }

    # Runtime MCR is triggered solely by text extracted from image pixels.
    # It never consumes dataset OCR annotations, source fields, or labels.
    mcr_active = ENABLE_MCR and ocr.available and bool(ocr.text.strip())
    head = predict(text, _model_image(image), mcr_active=mcr_active)
    probs_t = torch.tensor(head["probs"], dtype=torch.float32)
    early = should_early_exit(
        probs_t,
        EarlyExitConfig(enabled=True, clean_threshold=CLEAN_THRESHOLD),
    )
    if early is not None:
        return {
            "label": "clean",
            "risk": head["risk"],
            "action": "allow",
            "stage": "c4_early_exit",
            "head": head,
        }
    if head["label"] == "clean":
        return {
            "label": "clean",
            "risk": head["risk"],
            "action": "allow",
            "stage": "head_clean_fallback",
            "head": head,
        }
    return {
        "label": head["label"],
        "risk": head["risk"],
        "action": "block",
        "stage": "head_injection_fallback",
        "head": head,
    }


def _run_payload(payload: dict) -> dict:
    return optimized_predict(payload.get("text", ""), payload.get("image"))


if __name__ == "__main__":
    # Accept one JSON object for interactive use and NDJSON for batch
    # evaluation. The model is loaded once in both modes.
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(_run_payload(json.loads(line)), ensure_ascii=False), flush=True)
'''


PACKAGE_REQUIREMENTS = """# Pinned dependency list for the MPID offline package.
# Generated by scripts/package_offline.py — keep in sync with the
# project's requirements.txt.
torch>=2.1
transformers>=4.45
peft>=0.11
safetensors>=0.4
Pillow>=9.0
PyYAML>=6.0
numpy>=1.24
scikit-learn>=1.3
rapidocr_onnxruntime==1.2.3
onnxruntime==1.28.0
opencv-python==5.0.0.93
pyclipper==1.4.0
Shapely==2.1.2
"""


PACKAGE_README = """# MPID Offline Package

This package runs the default protected pipeline:

`C5 rules -> C6B-lite local OCR -> C6A compatibility fallback -> MCR -> F-3000 LoRA -> R0/SBC -> C4 -> block/allow`

## Run one request

```powershell
@'{"text":"Please summarize the image.","image":"C:\\path\\to\\image.png"}'@ | python infer.py
```

`image` is optional. When local OCR detects visible text in an image, MCR marks
that image as untrusted content for the classifier. R0/SBC use the fixed policy
recorded in `MANIFEST.json`; dataset metadata, OCR annotations, and labels are
never used at runtime.

## Interactive demo

Run `python demo.py` for a terminal demo using the same protected pipeline.

## Verify the movable package

Run the bundled smoke without requiring the source repository:

```powershell
python smoke_offline.py --pkg . --stage-root ./offline_smoke_stage
```

## Offline prerequisites

Use the package on the same OS/Python architecture as the packaged runtime.
Install `requirements.txt` from a locally mirrored wheel repository before
disconnecting the target machine. Model and OCR inference perform no network
access; the package contains all required model weights under `models/`.

## Validation

Run `python infer.py` with the three inputs in `smoke_fixtures/` or execute the
repository's `scripts/smoke_offline.py --pkg <this package>` before delivery.

## Model limitation

Read `MANIFEST.json` for the exact checkpoint and release notes. Detection
quality depends on the bundled checkpoint; C4/C5/C6 layers do not repair a
collapsed base classifier.
"""


PACKAGE_DEMO = '''#!/usr/bin/env python3
"""Interactive terminal demo for the packaged protected pipeline."""
from __future__ import annotations

import json

from infer import optimized_predict


def main() -> None:
    print("MPID offline demo. Press Enter with empty text to exit.")
    while True:
        text = input("Text: ").strip()
        if not text:
            return
        image = input("Image path (optional): ").strip() or None
        result = optimized_predict(text, image)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPID offline package builder (T2.11)")
    p.add_argument("--backbone-dir", type=Path,
                   default=REPO_ROOT / "runs" / "_models" / "smolvlm-500m",
                   help="Local backbone directory to bundle")
    p.add_argument("--ckpt", type=Path,
                   default=REPO_ROOT / "runs" / "_templates" / "artifacts" / "checkpoints" / "lora_baseline.safetensors",
                   help="LoRA+head checkpoint to bundle")
    p.add_argument("--src", type=Path,
                   default=REPO_ROOT / "src",
                   help="mpid source tree to bundle (so the package can run without an install)")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "runs" / "_manual" / "artifacts" / "package" / "mpid_offline",
                   help="Output package directory")
    p.add_argument("--report", type=Path, default=None,
                   help="Path for package_offline.json (default: <out parent>/package_offline.json)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target", type=str,
                   default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--clean-threshold", type=float, default=0.95,
                   help="C4 threshold embedded in package inference")
    p.add_argument("--policy-name", type=str, default="F-3000-MCR-SBC",
                   help="Human-readable frozen inference policy name")
    p.add_argument("--r0-indirect-logit-offset", type=float, default=0.55,
                   help="Locked global indirect logit offset; do not tune in packaging")
    p.add_argument("--image-ocr-direct-logit-penalty", type=float, default=-0.20,
                   help="Locked direct logit adjustment when MCR activates")
    p.add_argument("--disable-mcr", action="store_true",
                   help="Disable runtime MCR (only for compatibility packages)")
    p.add_argument("--ocr-models-dir", type=Path, default=None,
                   help="RapidOCR ONNX model directory to bundle")
    p.add_argument("--smoke-image", type=Path, default=None,
                   help="Optional image copied as smoke_fixtures/crossmodal.png")
    p.add_argument("--model-note", type=str, default="",
                   help="Known model limitation recorded in MANIFEST.json")
    return p.parse_args()


def build(args: argparse.Namespace) -> dict:
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "models").mkdir()
    (out / "artifacts").mkdir()
    (out / "src").mkdir()

    # 1. Copy the backbone.
    dst_backbone = out / "models" / args.backbone_dir.name
    shutil.copytree(args.backbone_dir, dst_backbone)

    # 2. Copy the LoRA+head checkpoint.
    dst_ckpt = out / "artifacts" / args.ckpt.name
    shutil.copy2(args.ckpt, dst_ckpt)

    # 2b. Bundle OCR weights separately from the Python package so inference
    # can point to a known local path after the package is moved offline.
    if args.ocr_models_dir:
        if not args.ocr_models_dir.exists():
            raise FileNotFoundError(f"OCR models not found: {args.ocr_models_dir}")
        shutil.copytree(args.ocr_models_dir, out / "models" / "ocr")

    # 3. Copy the mpid source tree (read-only at runtime; we only
    #    need the layout to make ``import mpid`` work).
    src_dst = out / "src" / "mpid"
    shutil.copytree(args.src / "mpid", src_dst)
    # touch __init__.py's parent for namespace packages
    (out / "src" / "__init__.py").write_text("")

    # 4. Write the infer entry point.
    # The embedded entry points contain user-facing non-ASCII text; use an
    # explicit portable encoding instead of the Windows locale default.
    (out / "infer.py").write_text(PACKAGE_INFER, encoding="utf-8")
    (out / "infer.py").chmod(0o755)
    (out / "demo.py").write_text(PACKAGE_DEMO, encoding="utf-8")
    (out / "demo.py").chmod(0o755)
    # Keep the deployment smoke with the package so operators do not need a
    # checkout of this repository merely to verify the movable artifact.
    (out / "smoke_offline.py").write_text(
        (REPO_ROOT / "scripts" / "smoke_offline.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out / "smoke_offline.py").chmod(0o755)

    # 5. Write the requirements and manifest.
    (out / "requirements.txt").write_text(PACKAGE_REQUIREMENTS, encoding="utf-8")
    (out / "README.md").write_text(PACKAGE_README, encoding="utf-8")
    if args.smoke_image:
        if not args.smoke_image.exists():
            raise FileNotFoundError(f"Smoke image not found: {args.smoke_image}")
        fixture_dir = out / "smoke_fixtures"
        fixture_dir.mkdir()
        shutil.copy2(args.smoke_image, fixture_dir / "crossmodal.png")
    manifest = {
        "backbone":       args.backbone_dir.name,
        "checkpoint":     args.ckpt.name,
        "lora_r":         args.lora_r,
        "lora_alpha":     args.lora_alpha,
        "lora_target":    args.lora_target,
        "clean_threshold": args.clean_threshold,
        "policy_name": args.policy_name,
        "r0_indirect_logit_offset": args.r0_indirect_logit_offset,
        "image_ocr_direct_logit_penalty": args.image_ocr_direct_logit_penalty,
        "mcr": {
            "enabled": not args.disable_mcr,
            "activation": "local_ocr_nonempty",
            "prompt_version": "trusted_boundary_v2",
            "content_role": "untrusted_image_ocr",
            "runtime_evidence": "image_pixels_only",
        },
        "c6b_lite": {
            "enabled": bool(args.ocr_models_dir),
            "backend": "rapidocr_onnxruntime" if args.ocr_models_dir else None,
            "models_dir": "models/ocr" if args.ocr_models_dir else None,
            "runtime_evidence": "image_pixels_only",
        },
        "model_note": args.model_note,
        "python_min":     "3.10",
        "schema_version": "mpid-offline-v2",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 6. Checksums for every file (incl. backbone shards).
    files = sorted([p for p in out.rglob("*") if p.is_file()])
    lines = []
    for f in files:
        lines.append(f"{sha256_file(f)}  {f.relative_to(out)}")
    (out / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 7. Report.
    total_bytes = sum(p.stat().st_size for p in files)
    report = {
        "out_dir":            str(out),
        "files":              [str(f.relative_to(out)) for f in files],
        "total_size_bytes":   total_bytes,
        "total_size_mb":      round(total_bytes / (1024 * 1024), 2),
        "manifest":           manifest,
    }
    report_path = args.report or (out.parent / "package_offline.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    if not args.backbone_dir.exists():
        print(f"[package] backbone not found: {args.backbone_dir}", file=sys.stderr)
        return 1
    if not args.ckpt.exists():
        print(f"[package] checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1
    if args.ocr_models_dir and not args.ocr_models_dir.exists():
        print(f"[package] OCR models not found: {args.ocr_models_dir}", file=sys.stderr)
        return 1
    r = build(args)
    print(f"[package] wrote {r['out_dir']} ({r['total_size_mb']} MB, "
          f"{len(r['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
