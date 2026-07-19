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
import sys
from pathlib import Path

# Make the in-package mpid/ importable. The package is shipped with
# the source tree under ``src/`` so we add that to sys.path.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from mpid.adapters.vlm import VLMAdapter
from mpid.crossmodal import check_crossmodal
from mpid.heads.classification import NUM_CLASSES, ClassificationHead
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


def predict(text: str, image=None) -> dict:
    import torch
    from PIL import Image
    prompt = build_prompt(text)
    img = image if image is not None else Image.new("RGB", (512, 512), (235, 235, 235))
    enc = adapter.preprocess(prompt, img)
    with torch.inference_mode():
        out = peft_model(**enc, output_hidden_states=True)
    last_h = out.hidden_states[-1]
    last_idx = enc["attention_mask"].sum(dim=1) - 1
    b = torch.arange(last_h.size(0), device=last_h.device)
    pooled = last_h[b, last_idx]
    res = head.predict(pooled)
    return {
        "label": res["label"][0],
        "risk":  float(res["risk"][0].item()),
        "probs": res["probs"][0].detach().cpu().tolist(),
    }


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

    c6 = check_crossmodal(record)
    if c6.suspicious:
        return {
            "label": c6.label,
            "risk": 1.0,
            "action": "block",
            "stage": "c6_crossmodal",
            "explanation": c6.to_dict(),
        }

    head = predict(text, image)
    probs_t = torch.tensor(head["probs"], dtype=torch.float32)
    early = should_early_exit(
        probs_t,
        EarlyExitConfig(enabled=True, clean_threshold=0.95),
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


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())
    print(json.dumps(optimized_predict(payload.get("text", ""),
                                      payload.get("image")),
                     ensure_ascii=False))
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
"""


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

    # 3. Copy the mpid source tree (read-only at runtime; we only
    #    need the layout to make ``import mpid`` work).
    src_dst = out / "src" / "mpid"
    shutil.copytree(args.src / "mpid", src_dst)
    # touch __init__.py's parent for namespace packages
    (out / "src" / "__init__.py").write_text("")

    # 4. Write the infer entry point.
    (out / "infer.py").write_text(PACKAGE_INFER)
    (out / "infer.py").chmod(0o755)

    # 5. Write the requirements and manifest.
    (out / "requirements.txt").write_text(PACKAGE_REQUIREMENTS)
    manifest = {
        "backbone":       args.backbone_dir.name,
        "checkpoint":     args.ckpt.name,
        "lora_r":         args.lora_r,
        "lora_alpha":     args.lora_alpha,
        "lora_target":    args.lora_target,
        "python_min":     "3.10",
        "schema_version": "mpid-offline-v1",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    # 6. Checksums for every file (incl. backbone shards).
    files = sorted([p for p in out.rglob("*") if p.is_file()])
    lines = []
    for f in files:
        lines.append(f"{sha256_file(f)}  {f.relative_to(out)}")
    (out / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n")

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
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    args = parse_args()
    if not args.backbone_dir.exists():
        print(f"[package] backbone not found: {args.backbone_dir}", file=sys.stderr)
        return 1
    if not args.ckpt.exists():
        print(f"[package] checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1
    r = build(args)
    print(f"[package] wrote {r['out_dir']} ({r['total_size_mb']} MB, "
          f"{len(r['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
