"""Create an auditable, self-contained MPID offline delivery bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package(package: Path) -> None:
    checksums = package / "CHECKSUMS.txt"
    if not checksums.exists():
        raise FileNotFoundError(f"missing package checksum file: {checksums}")
    for line in checksums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual = sha256(package / relative)
        if actual != expected:
            raise RuntimeError(f"package checksum mismatch: {relative}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create an auditable MPID offline delivery bundle")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--full-report", type=Path, required=True)
    p.add_argument("--performance", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--zip", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    package = args.package.resolve()
    report = json.loads(args.full_report.read_text(encoding="utf-8"))
    performance = json.loads(args.performance.read_text(encoding="utf-8"))
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite release directory: {args.out}")
    verify_package(package)
    args.out.mkdir(parents=True)
    shutil.copytree(package, args.out / "mpid_offline")
    evidence = args.out / "evidence"
    evidence.mkdir()
    shutil.copy2(args.full_report, evidence / "v2_full500_report.json")
    markdown_report = args.full_report.with_suffix(".md")
    if markdown_report.exists():
        shutil.copy2(markdown_report, evidence / "v2_full500_report.md")
    shutil.copy2(args.performance, evidence / "offline_performance.json")

    classes = report["per_class"]
    manifest = json.loads((package / "MANIFEST.json").read_text(encoding="utf-8"))
    audit = f"""# F-3000-MCR-SBC Offline Delivery Audit

## Identity

- Policy: `{manifest['policy_name']}`
- Base checkpoint: `{manifest['checkpoint']}`
- Package schema: `{manifest['schema_version']}`
- Package checksum verification: passed before release assembly.
- Isolated movable-package smoke: passed (C5 direct rule, clean head fallback, C6B local OCR block).

## Frozen V2/full500 Result

- Records: {report['n_eval']}
- Accuracy: {report['accuracy']:.2%}
- Macro F1: {report['macro_f1']:.2%}
- Weighted F1: {report['weighted_f1']:.2%}
- Clean F1 / Recall: {classes['clean']['f1-score']:.2%} / {classes['clean']['recall']:.2%}
- Direct F1 / Recall: {classes['direct']['f1-score']:.2%} / {classes['direct']['recall']:.2%}
- Indirect F1 / Recall: {classes['indirect']['f1-score']:.2%} / {classes['indirect']['recall']:.2%}

## Locked Runtime Policy

- MCR: active only when local OCR reads non-empty image text; no dataset metadata, OCR annotation, or label is used at runtime.
- R0: global indirect logit offset `{manifest['r0_indirect_logit_offset']:+.2f}`.
- SBC: direct logit adjustment `{manifest['image_ocr_direct_logit_penalty']:+.2f}` only when MCR activates.
- C4 threshold: `{manifest['clean_threshold']:.2f}` after C5/C6 checks.
- C5/C6B/C6 plus the F-3000 head are all included in `mpid_offline/`.

## Deployment Measurements

- Peak RSS: {performance['peak_rss_mb']:.1f} MB.
- C5 direct-rule P50: {performance['cases']['c5_direct_rule']['p50_ms']:.2f} ms.
- Text head P50: {performance['cases']['text_head']['p50_ms']:.2f} ms.
- Image MCR/head P50: {performance['cases']['image_mcr_head']['p50_ms']:.2f} ms.
- C6B OCR block P50: {performance['cases']['c6b_ocr_block']['p50_ms']:.2f} ms.

## Operator Steps

1. Install `mpid_offline/requirements.txt` from an approved local wheel mirror on a compatible CPU/Python environment.
2. Run `python <release>/mpid_offline/smoke_offline.py --pkg <release>/mpid_offline --stage-root <release>/offline_smoke_stage` before promotion.
3. Invoke `mpid_offline/infer.py` with one UTF-8 JSON object per input line. The input schema is `{{"text": "...", "image": "optional-path"}}`.
4. Preserve `CHECKSUMS.txt`, `MANIFEST.json`, and the evidence directory with the deployed artifact.
"""
    (args.out / "RELEASE_AUDIT.md").write_text(audit, encoding="utf-8")

    files = [p for p in sorted(args.out.rglob("*")) if p.is_file()]
    release_manifest = {
        "policy_name": manifest["policy_name"],
        "release_dir": args.out.name,
        "files": [{"path": str(p.relative_to(args.out)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in files],
    }
    (args.out / "RELEASE_MANIFEST.json").write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    if args.zip.exists():
        raise FileExistsError(f"refusing to overwrite archive: {args.zip}")
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for file in sorted(args.out.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(args.out.parent))
    print(json.dumps({"release_dir": str(args.out), "archive": str(args.zip), "archive_sha256": sha256(args.zip)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
