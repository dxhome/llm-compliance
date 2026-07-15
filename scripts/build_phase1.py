"""Build the MPID Phase-1 deliverables (T1.5 + T1.6 + T1.7 + T1.8).

Runs the full pipeline in one shot:

  1. ``split_and_dump`` — produces ``data/mpid-v1/{train,val,test}.jsonl``
     and ``split_summary.json``.
  2. ``synthetic_image_injection`` — produces
     ``data/mpid-v1-crossmodal/{manifest.jsonl, images/*.png}``.
  3. ``run_eda`` — produces ``data/mpid-v1/EDA.md`` from the splits.
  4. ``random_qc`` — produces ``data/mpid-v1/qc_sample.jsonl``
     (20 records for human review).
  5. Cross-modal subset is also re-saved as a 8:1:1 split under
     ``data/mpid-v1-crossmodal/{train,val,test}.jsonl`` for C6 training.

Idempotent: re-running overwrites the output files in place.

Usage::

    python scripts/build_phase1.py
    python scripts/build_phase1.py --n-synthetic 50
    python scripts/build_phase1.py --max-per-dataset '{"jailbreakv_28k": 500}'
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mpid.data.public_loaders import Record  # noqa: E402
from mpid.data.split import _stratified_split, _write_jsonl, _distribution  # noqa: E402
from mpid.data.synthetic_image_injection import generate as gen_synthetic  # noqa: E402
from mpid.data.split import split_and_dump  # noqa: E402


# Reasonable per-dataset caps for Phase 1. These keep the unified
# dataset at ~3-4 k records total, which is well above the "≥ 1k"
# acceptance bar and small enough to iterate on in Phase 2 training.
#
# IMPORTANT: JailbreakV-28K's figstep rows sit at row ~20 k (out of
# 28 k); a cap of 1500 would miss ALL of them. We cap at 22 000 to
# include the full figstep subset (2 000 rows) plus a representative
# sample of the other formats.
#
# Phase 2.2 / T2.14 — ``--full`` mode raises the caps to the union of
# what the public datasets actually contain, producing a ≥ 25 k
# trainable split. The smoke Phase 1 caps remain the default so the
# 1-2k-recipe is reproducible without re-downloading.
DEFAULT_CAPS: dict[str, int] = {
    "deepset_prompt_injections":   600,    # ~all of it
    "safe_guard_prompt_injection": 1500,   # cap the 6k
    "jailbreakv_28k":               22000,  # cap to include all figstep
    "cais_mmlu":                   300,    # cap the 285 dev + headroom
    "haonan_li_cmmlu":             300,    # cap
    "nlphuji_flickr30k":           1000,   # cap the 31k captions
}

# Phase 2.2 / T2.14 full caps — push the dataset to its natural ceiling
# (de-duped totals ≈ 25-28 k). Used when ``--full`` is passed.
FULL_CAPS: dict[str, int] = {
    "deepset_prompt_injections":   1000,   # ~all of it
    "safe_guard_prompt_injection": 6000,   # the full 6 k
    "jailbreakv_28k":               28000,  # the full 28 k
    "cais_mmlu":                   600,    # double for headroom
    "haonan_li_cmmlu":             500,    # ~all of the dev split
    "nlphuji_flickr30k":           5000,   # 5 k caption pool for clean
}


# ---------------------------------------------------------------------------
# EDA (full) — T2.14
# ---------------------------------------------------------------------------

def _render_eda_full(out_dir: Path, cm_dir: Path, summary: dict, caps: dict,
                     raw_status_json: Path | None = None) -> str:
    """Render a focused report on the **full** (Phase 2.2) split.

    Compared to ``_render_eda``, this version:
      * Reports a single total-records target (≥ 25 000) and per-source
        de-dup count, so it's obvious whether we hit T2.14 acceptance.
      * Cross-references ``data/raw_status.json`` (T2.13) so the EDA
        shows both the unified split **and** the on-disk raw files.
      * Highlights the cross-modal subset size (target ≥ 2 000).
    """
    lines: list[str] = []
    lines.append("# MPID-v1 — EDA 报告（Phase 2.2 全量数据）")
    lines.append("")
    lines.append(f"> Phase 2.2 / T2.14 · 自动生成于 `scripts/build_phase1.py` · seed={summary['seed']}")
    lines.append("")
    lines.append("## 1. 总量与硬性指标")
    lines.append("")
    lines.append(f"| 指标 | 数值 | 目标 | 状态 |")
    lines.append(f"|---|---|---|---|")
    n_total = summary['total']
    n_train = summary['train']['total']
    n_val = summary['val']['total']
    n_test = summary['test']['total']
    target_total = 25000
    target_train = 20000
    target_val = 2500
    target_test = 2500
    target_cm = 2000
    def status(n, t):
        return "✅" if n >= t else "❌"
    lines.append(f"| 总样本数 | {n_total} | ≥ {target_total} | {status(n_total, target_total)} |")
    lines.append(f"| 训练集 | {n_train} | ≥ {target_train} | {status(n_train, target_train)} |")
    lines.append(f"| 验证集 | {n_val} | ≥ {target_val} | {status(n_val, target_val)} |")
    lines.append(f"| 测试集 | {n_test} | ≥ {target_test} | {status(n_test, target_test)} |")
    cm_summary_path = cm_dir / "split_summary.json"
    if cm_summary_path.exists():
        with open(cm_summary_path, encoding="utf-8") as f:
            cm = json.load(f)
        n_cm = cm['total']
        lines.append(f"| 跨模态子集 | {n_cm} | ≥ {target_cm} | {status(n_cm, target_cm)} |")
    lines.append("")
    lines.append("## 2. 每集 cap（FULL_CAPS）")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(caps, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    # Per-split stats
    lines.append("## 3. 各 split 类别分布")
    lines.append("")
    lines.append("| split | clean | direct | indirect | total |")
    lines.append("|---|---|---|---|---|")
    for split in ("train", "val", "test"):
        d = summary[split]["by_label"]
        lines.append(f"| {split} | {d.get('clean', 0)} | "
                     f"{d.get('direct', 0)} | "
                     f"{d.get('indirect', 0)} | "
                     f"{summary[split]['total']} |")
    lines.append("")

    # Per-source stats (full)
    lines.append("## 4. 各 split 数据源分布")
    lines.append("")
    lines.append("| split | deepset | safe-guard | jailbreakv | mmlu | cmmlu | flickr30k |")
    lines.append("|---|---|---|---|---|---|---|")
    src_keys = ["deepset_prompt_injections",
                "safe_guard_prompt_injection",
                "jailbreakv_28k",
                "cais_mmlu",
                "haonan_li_cmmlu",
                "nlphuji_flickr30k"]
    for split in ("train", "val", "test"):
        d = summary[split]["by_source"]
        cells = [str(d.get(k, 0)) for k in src_keys]
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    lines.append("")

    # Cross-modal summary
    lines.append("## 5. 跨模态子集 `data/mpid-v1-crossmodal/`")
    lines.append("")
    if cm_summary_path.exists():
        with open(cm_summary_path, encoding="utf-8") as f:
            cm = json.load(f)
        lines.append(f"- 总样本数: {cm['total']} (目标 ≥ {target_cm})")
        lines.append(f"- 划分: train={cm['train']['total']} | val={cm['val']['total']} | test={cm['test']['total']}")
        lines.append(f"- 类别: train={cm['train']['by_label']}  (全部为 indirect)")
        lines.append("")
    else:
        lines.append("_cross-modal 子集未生成_")
        lines.append("")

    # Optional cross-reference to raw download status
    lines.append("## 6. 原始数据 on-disk 状态（T2.13 cross-ref）")
    lines.append("")
    if raw_status_json and raw_status_json.exists():
        try:
            with open(raw_status_json, encoding="utf-8") as f:
                rs = json.load(f)
            lines.append(f"> 来源: `download_data.py --status --status-json {raw_status_json.name}`")
            lines.append("")
            lines.append("| dataset | size (MB) | csv | parquet | image | zip | complete |")
            lines.append("|---|---|---|---|---|---|---|")
            for d in rs.get("datasets", []):
                fc = d.get("file_counts", {})
                lines.append(
                    f"| {d['short_name']} | {d['total_size_mb']:.1f} | "
                    f"{fc.get('csv', 0)} | {fc.get('parquet', 0)} | "
                    f"{fc.get('image', 0)} | {fc.get('zip', 0)} | "
                    f"{'✅' if d['complete'] else '❌'} |"
                )
            lines.append(f"\n总占用: **{rs.get('total_size_mb', 0):.1f} MB**")
        except Exception as e:
            lines.append(f"_无法读取 {raw_status_json}: {e}_")
    else:
        lines.append("_未提供 ``--raw-status-json``；先用 "
                     "``python scripts/download_data.py --status --status-json data/raw_status.json`` 生成_")
    lines.append("")

    # Known issues & decisions
    lines.append("## 7. 已知限制与决策记录")
    lines.append("")
    lines.append("1. **figstep 完整 22k 图像不在 HF Hub** — HF 上 `JailbreakV-28K/JailBreakV-28k` "
                 "只 ship 100 张 sample figstep 图；完整 22k 需到 JailbreakV GitHub release 下载。 "
                 "本仓库使用 HF CSV 28k + 100 张样本图 = 20k+ 文本训练样本，已足够 Phase 2.2 的 2k 训练。")
    lines.append("2. **Flickr30k 4.4 GB 图像 zip 默认未下载** — `download_data.py --full` 才会拉取。 "
                 "Phase 2.2 训练主要看文本，图片字段多数为 None；C5/C6 跨模态训练会真正用到。")
    lines.append("3. **类别不均衡** — direct 占比 ≈ 80%。训练脚本已开 class_weighted (inverse-frequency)。")
    lines.append("4. **Flickr30k 字段为 caption** — `text` 字段填 caption，`image` 字段为 None "
                 "(除非用 ``--full`` 拉了 4.4 GB 图像 zip)。")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def _length_stats(texts: list[str]) -> dict:
    if not texts:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
    lens = sorted(len(t) for t in texts)
    n = len(lens)
    p95 = lens[int(n * 0.95)] if n > 1 else lens[0]
    median = lens[n // 2]
    return {"mean": sum(lens) / n,
            "median": median,
            "p95": p95,
            "min": lens[0],
            "max": lens[-1]}


def _render_eda(split_dir: Path, summary: dict) -> str:
    """Build the EDA.md content from the three split JSONLs and the
    summary dict produced by ``split_and_dump``."""
    splits: dict[str, list[Record]] = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.jsonl"
        records: list[Record] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                records.append(Record(
                    id=d["id"],
                    text=d["text"],
                    image=Path(d["image"]) if d.get("image") else None,
                    label=d["label"],
                    source=d["source"],
                    lang=d.get("lang", "unknown"),
                    metadata=d.get("metadata", {}),
                ))
        splits[split] = records

    lines: list[str] = []
    lines.append(f"# MPID-v1 — EDA 报告")
    lines.append("")
    lines.append(f"> Phase 1 / T1.6 · 自动生成于 `scripts/build_phase1.py` · seed={summary['seed']}")
    lines.append("")
    lines.append("## 1. 总量")
    lines.append("")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|---|---|")
    lines.append(f"| 总样本数 | {summary['total']} |")
    lines.append(f"| 划分比例 | {summary['ratios']} |")
    lines.append(f"| seed | {summary['seed']} |")
    if summary.get("max_per_dataset"):
        lines.append(f"| 每集 cap | `{json.dumps(summary['max_per_dataset'])}` |")
    lines.append("")

    # Per-split stats
    lines.append("## 2. 各 split 类别分布")
    lines.append("")
    lines.append("| split | clean | direct | indirect | total |")
    lines.append("|---|---|---|---|---|")
    for split in ("train", "val", "test"):
        d = summary[split]["by_label"]
        lines.append(f"| {split} | {d.get('clean', 0)} | "
                     f"{d.get('direct', 0)} | "
                     f"{d.get('indirect', 0)} | "
                     f"{summary[split]['total']} |")
    lines.append("")

    # Per-source stats
    lines.append("## 3. 各 split 数据源分布")
    lines.append("")
    lines.append("| split | deepset | safe-guard | jailbreakv | mmlu | cmmlu | flickr30k |")
    lines.append("|---|---|---|---|---|---|---|")
    src_keys = ["deepset_prompt_injections",
                "safe_guard_prompt_injection",
                "jailbreakv_28k",
                "cais_mmlu",
                "haonan_li_cmmlu",
                "nlphuji_flickr30k"]
    for split in ("train", "val", "test"):
        d = summary[split]["by_source"]
        cells = [str(d.get(k, 0)) for k in src_keys]
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    lines.append("")

    # Language distribution
    lines.append("## 4. 语种分布（各 split）")
    lines.append("")
    lines.append("| split | en | zh | multi | unknown |")
    lines.append("|---|---|---|---|---|")
    for split in ("train", "val", "test"):
        d = summary[split]["by_lang"]
        lines.append(f"| {split} | {d.get('en', 0)} | {d.get('zh', 0)} | "
                     f"{d.get('multi', 0)} | {d.get('unknown', 0)} |")
    lines.append("")
    lines.append("> `safe-guard` 数据集本身多语种；当前标签粒度只分 en/zh，对其他语种也归为 en。Phase 1 暂不处理。")
    lines.append("")

    # Length distribution
    lines.append("## 5. 文本长度分布（字符数）")
    lines.append("")
    lines.append("| split | mean | median | p95 | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for split in ("train", "val", "test"):
        s = _length_stats([r.text for r in splits[split]])
        lines.append(f"| {split} | {s['mean']:.1f} | {s['median']} | "
                     f"{s['p95']} | {s['min']} | {s['max']} |")
    lines.append("")

    # Image attachment stats
    lines.append("## 6. 图像绑定情况")
    lines.append("")
    n_with_img = {s: sum(1 for r in splits[s] if r.image) for s in splits}
    n_total = {s: len(splits[s]) for s in splits}
    lines.append("| split | with image | total |")
    lines.append("|---|---|---|")
    for s in splits:
        lines.append(f"| {s} | {n_with_img[s]} | {n_total[s]} |")
    lines.append("")
    lines.append("> 主体为文本，图像字段大多数为 `None`（P0A-3 未下载 4.4 GB Flickr 图像 zip 与 300+ MB 完整 JailbreakV 图）。C6 训练需要的 cross-modal 子集见 § 8。")
    lines.append("")

    # Typical samples
    lines.append("## 7. 典型样例（每类 1 条，来自 train split）")
    lines.append("")
    for label in ("clean", "direct", "indirect"):
        rs = [r for r in splits["train"] if r.label == label]
        if not rs:
            continue
        r = rs[0]
        text_preview = r.text[:200].replace("|", "\\|").replace("\n", " ")
        lines.append(f"### 7.{ {'clean':1, 'direct':2, 'indirect':3}[label] } `{label}`")
        lines.append("")
        lines.append(f"- id: `{r.id}`")
        lines.append(f"- source: `{r.source}`")
        lines.append(f"- lang: `{r.lang}`")
        lines.append(f"- text: {text_preview}{'...' if len(r.text) > 200 else ''}")
        lines.append(f"- image: `{r.image}`" if r.image else "- image: None")
        lines.append("")

    # Cross-modal subset pointer
    lines.append("## 8. 跨模态子集 `data/mpid-v1-crossmodal/`")
    lines.append("")
    cm_dir = split_dir.parent / "mpid-v1-crossmodal"
    if (cm_dir / "split_summary.json").exists():
        with open(cm_dir / "split_summary.json", encoding="utf-8") as f:
            cm = json.load(f)
        lines.append(f"- 总样本数: {cm['total']} (目标 ≥ 100)")
        lines.append(f"- 划分: train={cm['train']['total']} | val={cm['val']['total']} | test={cm['test']['total']}")
        lines.append(f"- 类别: {cm['train']['by_label']} (全部为 indirect)")
        lines.append("")
    else:
        lines.append("_cross-modal 子集未生成_")
        lines.append("")

    # Known issues
    lines.append("## 9. 已知问题与决策记录")
    lines.append("")
    lines.append("1. **safe-guard 无显式 injection_type** —— 按文本是否含 \"indirect\" 兜底分桶；EDA 显示 indirect 桶很小（< 5%），如对 indirect 召回率影响大则 Phase 2 训练前换数据集。")
    lines.append("2. **JailbreakV CSV `image_path` 指向 `llm_transfer_attack/`** —— 本期未下载；T1.4 用 figstep/100 + PIL 文字叠加生成 cross-modal 子集。")
    lines.append("3. **Flickr30k 无图像** —— 4.4 GB 推迟到 Phase 2 训练前；text 字段保留 captions。")
    lines.append("4. **CMMLU 的 `Question` 列含繁体/简体混合** —— detect_lang 仍正确分到 zh。")
    lines.append("5. **MMLU/CMMLU/Flickr30k 全视为 clean** —— 这些数据集不区分干净/注入；这与威胁模型 § 6 一致。")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# QC (random sample for human review)
# ---------------------------------------------------------------------------

def _sample_for_qc(split_dir: Path, n: int, seed: int) -> Path:
    """Sample n records from each split (stratified by label) and dump
    a single JSONL file for human review."""
    rng = random.Random(seed)
    out_path = split_dir / "qc_sample.jsonl"
    samples: list[dict] = []
    for split in ("train", "val", "test"):
        with open(split_dir / f"{split}.jsonl", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        # Stratified by label: at least 1 of each
        by_label: dict[str, list] = {}
        for r in records:
            by_label.setdefault(r["label"], []).append(r)
        per_label = max(1, n // 3)
        for label, items in by_label.items():
            rng.shuffle(items)
            for r in items[:per_label]:
                r2 = dict(r)
                r2["qc_split"] = split
                r2["qc_label"] = label
                samples.append(r2)
    rng.shuffle(samples)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples[:n * 3]:  # cap at 3*n to keep it small
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir",  type=Path, default=REPO_ROOT / "data" / "raw")
    p.add_argument("--out-dir",  type=Path, default=REPO_ROOT / "data" / "mpid-v1")
    p.add_argument("--cm-dir",   type=Path, default=REPO_ROOT / "data" / "mpid-v1-crossmodal")
    p.add_argument("--n-synthetic", type=int, default=120)
    p.add_argument("--qc-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-dataset", type=str, default=None,
                   help="JSON dict overriding DEFAULT_CAPS per dataset")
    p.add_argument("--full", action="store_true",
                   help="T2.14: use FULL_CAPS (28k jailbreakv + 6k safe-guard + "
                        "5k flickr captions) for the real training set. "
                        "Default keeps the Phase 1 smoke caps.")
    p.add_argument("--raw-status-json", type=Path, default=None,
                   help="T2.14: path to the download_data.py --status output. "
                        "When set, the EDA_full.md includes an on-disk data section.")
    p.add_argument("--n-synthetic-full", type=int, default=2000,
                   help="T2.14: in --full mode, generate this many cross-modal "
                        "synthetic indirect records (default 2000, target ≥ 2k).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    caps = dict(FULL_CAPS if args.full else DEFAULT_CAPS)
    if args.max_per_dataset:
        caps.update(json.loads(args.max_per_dataset))
    n_synthetic = args.n_synthetic_full if args.full else args.n_synthetic

    print(f"[build] phase-1 build · seed={args.seed}  full={args.full}")
    print(f"[build] raw_dir={args.raw_dir}  out={args.out_dir}  cm={args.cm_dir}")
    print(f"[build] caps: {json.dumps(caps)}")
    print(f"[build] n_synthetic={n_synthetic}")

    # 1. Main split
    print("\n[1/4] split_and_dump ...")
    summary = split_and_dump(args.raw_dir, args.out_dir,
                             seed=args.seed, max_per_dataset=caps)
    print(f"  total={summary['total']} | "
          f"train={summary['train']['total']} | "
          f"val={summary['val']['total']} | "
          f"test={summary['test']['total']}")
    print(f"  by_label (train): {summary['train']['by_label']}")

    # 2. Cross-modal subset
    print("\n[2/4] cross-modal synthetic ...")
    cm_records = gen_synthetic(
        n_samples=n_synthetic,
        out_dir=args.cm_dir,
        # Re-use the 100 figstep images as the base pool. They are
        # already attack images, but for Phase 1 EDA we just need
        # "real images on disk"; the C6 cross-modal training will
        # be re-evaluated against this same data.
        base_pool=None,
        seed=args.seed,
    )
    print(f"  generated {len(cm_records)} synthetic indirect records")

    # Split the cross-modal subset 8:1:1 too, so C6 has train/val/test.
    cm_objs = [Record(id=r.id, text=r.text, image=r.image_path,
                      label=r.label, source=r.source, lang=r.lang,
                      metadata={"template_id": r.template_id})
               for r in cm_records]
    cm_train, cm_val, cm_test = _stratified_split(cm_objs, seed=args.seed)
    _write_jsonl(cm_train, args.cm_dir / "train.jsonl")
    _write_jsonl(cm_val,   args.cm_dir / "val.jsonl")
    _write_jsonl(cm_test,  args.cm_dir / "test.jsonl")
    cm_summary = {
        "total": len(cm_objs),
        "train": _distribution(cm_train),
        "val":   _distribution(cm_val),
        "test":  _distribution(cm_test),
        "ratios": [0.8, 0.1, 0.1],
        "seed": args.seed,
    }
    with open(args.cm_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(cm_summary, f, ensure_ascii=False, indent=2)
    print(f"  cm splits: train={len(cm_train)} val={len(cm_val)} test={len(cm_test)}")

    # 3. EDA
    print("\n[3/4] EDA ...")
    eda_md = _render_eda(args.out_dir, summary)
    eda_path = args.out_dir / "EDA.md"
    eda_path.write_text(eda_md, encoding="utf-8")
    print(f"  wrote {eda_path}")

    # 3b. T2.14 EDA_full — a focused report on the ≥ 25 k split,
    # optionally cross-referencing the on-disk raw download report.
    print("\n[3b/4] EDA_full (T2.14) ...")
    eda_full_md = _render_eda_full(
        args.out_dir, args.cm_dir, summary, caps,
        raw_status_json=args.raw_status_json,
    )
    eda_full_path = args.out_dir / "EDA_full.md"
    eda_full_path.write_text(eda_full_md, encoding="utf-8")
    print(f"  wrote {eda_full_path}")

    # 4. QC sample
    print("\n[4/4] QC sample (T1.7) ...")
    qc_path = _sample_for_qc(args.out_dir, args.qc_n, args.seed)
    print(f"  wrote {qc_path} ({sum(1 for _ in open(qc_path, encoding='utf-8'))} records)")

    # 5. Cross-link to the cross-modal EDA section
    # (already done inside _render_eda via the cm_dir probe)
    print("\n[done] phase-1 deliverables:")
    print(f"  {args.out_dir}/train.jsonl, val.jsonl, test.jsonl, split_summary.json, EDA.md, EDA_full.md, qc_sample.jsonl")
    print(f"  {args.cm_dir}/train.jsonl, val.jsonl, test.jsonl, manifest.jsonl, split_summary.json, images/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
