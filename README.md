# MPID — Multimodal Prompt Injection Defense

> **面向离线场景的轻量级多模态提示注入防御系统**。
> 一套**backbone-agnostic** 的检测算法框架：同一套训练/评测/部署代码既能在本项目选用的
> SmolVLM-500M 上跑，也能平滑迁移到 7B / 13B 规模的视觉-语言模型继续研究。

[![phase](https://img.shields.io/badge/Phase%205-C4%2FC5%2FC6%20完成-green)](doc/VERIFICATION.md#phase-5--c6-跨模态自检)
[![macro-f1](https://img.shields.io/badge/Macro%20F1-85.5%25-brightgreen)](doc/reference.md#31-核心结论摘要)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

---

## 目录

1. [项目是什么](#1-项目是什么)
2. [最终目标长什么样](#2-最终目标长什么样)
3. [核心成果](#3-核心成果)
4. [从 0 跑通：7 步上手指南](#4-从-0-跑通7-步上手指南)
5. [项目结构](#5-项目结构)
6. [各 Phase 在哪、怎么跑](#6-各-phase-在哪怎么跑)
7. [重要声明](#7-重要声明)

---

## 1. 项目是什么

在 LLM / VLM 入口处构建一个**轻量、可打包、跨平台、零网络依赖**的"前置过滤"组件，
拦截**直接注入 / 间接注入 / 多模态注入**三类 prompt-injection 攻击。

| 维度 | 业界主流 | 本项目 |
|---|---|---|
| 部署形态 | 云端 API | 端侧 / 边缘独立部署 |
| 模型规模 | 数百 MB ~ 数十 GB | 默认 < 500 MB |
| 网络依赖 | 必需 | **无** |
| 数据流向 | 上传 prompt 至云端 | 数据完全在本地 |
| 适用场景 | 通用 SaaS 防护 | 隐私敏感、边缘、低成本合规 |

### 技术核心

- **Backbone**：[SmolVLM-500M](https://huggingface.co/HuggingFaceTB/SmolVLM-500M)（Apache 2.0，约 5 亿参数）
- **训练方法**：LoRA（PEFT 注入 q/k/v/o 投影）+ 3 类分类头（`clean / direct / indirect`）
- **三层防御链路（C4 / C5 / C6）**：
  - **C4 早退机制**——clean 高置信度样本提前放行，generation 次数 -45.6%
  - **C5 规则前置过滤**——文本规则拦截直接注入，direct recall 从 31% → 61%
  - **C6 跨模态一致性**——跨模态启发式拦截间接注入，indirect F1 从 0% → 100%
- **离线优先**：backbone + LoRA + tokenizer + 推理脚本打成一个 `mpid_offline/` 包，
  校验和覆盖 77 个文件，**零网络调用**

---

## 2. 最终目标长什么样

整个研究链条按 Phase 推进，最终交付一个**可分发的离线检测包 + 完整评测报告**。

```
Phase 0A        Phase 0         Phase 1         Phase 2         Phase 3
环境/模型/数据   项目脚手架     威胁模型         VLM 端到端基线   C4 早退
───────         ──────         ──────         ──────         ──────
(已完成)        (已完成)       (已完成)        (已完成)        (已完成)
  ↓               ↓               ↓               ↓               ↓
Phase 4         Phase 5         Phase 6         Phase 7
C5 规则前置     C6 跨模态        攻防评测体系     项目整理
───────         ──────         ──────         ──────
(已完成)        (已完成)        (待启动)        (待启动)
```

### 最终交付清单

| 文件 | 用途 | 状态 |
|---|---|---|
| `runs/*/artifacts/lora_baseline.safetensors` | MPID LoRA checkpoint | ✅ |
| `src/mpid/early_exit.py` | C4 早退机制 | ✅ |
| `src/mpid/rules/direct_rules.py` | C5 规则前置 | ✅ |
| `src/mpid/crossmodal/heuristics.py` | C6 跨模态启发式 | ✅ |
| `scripts/infer_c4_c5_c6.py` | 完整推理链路 | ✅ |
| `mpid_offline/` | 端侧离线包（含 infer.py、CHECKSUMS.txt） | ✅ |
| `report/technical_report.md` | Phase 6 完整技术报告 | ⏳ |

---

## 3. 核心成果

> 详见 [doc/reference.md § 3. 测试结果汇总](doc/reference.md#3-测试结果汇总)

### 推理效果（300 样本）

| 指标 | MPID LoRA | MPID LoRA + C4-C6优化 | Delta |
|---|---:|---:|---:|
| Macro F1 | 32.3% | **85.5%** | +53.2pp（+164.7% relative） |
| Accuracy | 42.7% | **86.0%** | +43.3pp |
| clean F1 | 55.4% | 82.2% | +26.8pp |
| direct F1 | 41.3% | 74.4% | +33.1pp |
| indirect F1 | 0.0% | **100.0%** | +100.0pp |

### 推理时间（300 样本）

| 指标 | MPID LoRA | MPID LoRA + C4-C6优化 | Delta |
|---|---:|---:|---:|
| 端到端总耗时 | 4945.4s | **2978.1s** | -39.8% |
| 平均耗时 / 样本 | 16.48s | **9.93s** | -39.7% |
| generation 次数 | 250 | **136** | -45.6% |

### C4 / C5 / C6 各自贡献

| 阶段 | 贡献 |
|---|---|
| **C5** | direct recall 从 31% → 61%，补齐 MPID classification head 的部分漏检 |
| **C6A** | indirect F1 从 0% → 100%，修复对 synthetic cross-modal indirect 的盲点 |
| **C4** | 主要起置信度决策与 generation gate 作用 |

---

## 4. 从 0 跑通：7 步上手指南

> **目标平台约定**（贯穿全文档）：
> - **mac** = macOS 12.5+ / Apple Silicon（MPS 可用，CUDA 不可用）
> - **x86** = Linux x86_64 / CPU-only（**无 CUDA、无 MPS**）

### 第 1 步 — 克隆 + Python 环境

```bash
git clone <your-fork-url> llm-compliance
cd llm-compliance
python3.11 -m venv .venv
source .venv/bin/activate
```

### 第 2 步 — 装依赖（按平台选一个）

```bash
# macOS（Apple Silicon，MPS）
pip install -r requirements.txt

# x86 / Linux CPU-only
pip install -r requirements-x86.txt

# macOS 13+ 想用 MLX 4-bit 路径
pip install -r requirements-mlx.txt
```

### 第 3 步 — 把 `mpid` 包安装成可编辑模式

```bash
pip install -e .
```

冒烟测试：

```bash
python scripts/smoke_env.py        # Phase 0A-1：import + 设备 + tensor + tokenizer
pytest tests/test_device.py -v     # Phase 0A-1：12 个设备抽象单测
```

### 第 4 步 — 下载 SmolVLM-500M + 4 个数据集

```bash
python scripts/download_models.py  # → models/smolvlm-500m/ （~970 MB）
python scripts/download_data.py    # → data/raw/ 4 个公开集（~43 MB）
```

冒烟：

```bash
python scripts/smoke_model.py      # Phase 0A-2：模型加载 + 最小推理
python scripts/smoke_data.py       # Phase 0A-3：5+ 样例/集 + 课题符合性
```

### 第 5 步 — 构数据集 + 跑 Phase 1 威胁模型

```bash
python scripts/build_phase1.py     # → data/mpid-v1/{train,val,test}.jsonl + EDA.md
# 跨模态子集 → data/mpid-v1-crossmodal/ （120 张间接注入攻击图）
```

### 第 6 步 — 跑 Phase 2 VLM 基线

```bash
# 训练（CPU smoke 5 records ≈ 7 min；x86 + 25k ≈ 30 h，CUDA 30 min）
python scripts/train.py --config configs/baseline.yaml --out-dir runs/baseline

# 评估（生成 report + 混淆矩阵）
python scripts/eval.py --max-records 30

# 量化离线指标
python scripts/measure_offline.py --samples 5

# 打包离线包
python scripts/package_offline.py

# 离线包 smoke
python scripts/smoke_offline.py
```

### 第 7 步 — 跑 Phase 3 / 4 / 5 优化

```bash
# C4 早退
python scripts/train_c4.py --config configs/c4_early_exit.yaml
python scripts/eval_c4.py

# C5 规则前置
python scripts/eval_c5.py --rules configs/rules.yaml

# C6 跨模态
python scripts/train_c6.py --config configs/c6_crossmodal.yaml
python scripts/eval_c6.py

# 完整推理链路（C4 + C5 + C6）
python scripts/infer_c4_c5_c6.py --checkpoint runs/balanced-600/artifacts/checkpoints/lora_baseline.safetensors

# 收尾：合并多模型结果 + 出报告
python scripts/aggregate_results.py    # → report/eval_*.json
python scripts/make_figures.py        # → report/figures/*.png
```

> 各命令的具体参数与产出都登记在 [`doc/tasks.md`](doc/tasks.md) 与
> [`doc/VERIFICATION.md`](doc/VERIFICATION.md) 对应 Phase 章节里。

---

## 5. 项目结构

```
llm-compliance/
├── README.md                # 本文件
├── pyproject.toml           # 包元数据 + 依赖（mac / x86 / mlx 三套）
├── requirements*.txt        # 三套依赖清单
│
├── doc/                     # 全部正式文档（不散落）
│   ├── opening-report-vlm.md    # 课题开题报告 v0.3
│   ├── reference.md            # 项目参考手册（核心概念 + Phase 详解 + 测试结果）
│   ├── tasks.md                 # 任务分解（按 Phase）
│   └── VERIFICATION.md          # 验证报告（实测基线、限制、决策）
│
├── src/mpid/                # 核心代码包（pip install -e . 后可 import）
│   ├── device.py            # 设备抽象（MPS / CUDA / CPU）
│   ├── adapters/vlm.py      # VLM 推理适配器（T2.1）
│   ├── backbones/registry.py    # Backbone 注册表（T2.2）
│   ├── heads/classification.py  # 3 类 head + 风险分（T2.3）
│   ├── early_exit.py        # C4 早退机制
│   ├── rules/direct_rules.py    # C5 规则前置
│   ├── crossmodal/heuristics.py # C6 跨模态启发式
│   ├── data/
│   │   ├── public_loaders.py        # 公开集统一 schema（T1.3）
│   │   ├── split.py                 # 8:1:1 划分（T1.5）
│   │   ├── synthetic_image_injection.py  # 跨模态合成（T1.4）
│   │   ├── dataset.py               # JSONL → Tensor
│   │   └── prompt.py                # 3 类指令 prompt 模板（T2.4）
│   └── train/trainer.py     # LoRA + 训练 + 评估回调（T2.5）
│
├── scripts/                 # CLI 入口
│   ├── download_models.py   # 下载 SmolVLM-500M
│   ├── download_data.py     # 下载 4 个公开集
│   ├── build_phase1.py      # 构数据集（Phase 1）
│   ├── smoke_*.py           # 各 Phase 冒烟
│   ├── train.py             # T2.7 训练入口
│   ├── eval.py              # T2.9 评估入口
│   ├── measure_offline.py   # T2.10 离线指标
│   ├── package_offline.py   # T2.11 离线打包
│   ├── smoke_offline.py     # T2.12 离线包 smoke
│   └── infer_c4_c5_c6.py    # 完整推理链路（Phase 3-5）
│
├── configs/                 # 训练 / 规则配置
│   └── baseline.yaml        # VLM 基线配置
│
├── tests/                   # 单元测试
│   └── test_device.py       # 设备抽象 12 用例
│
├── data/                    # 数据（gitignored）
│   ├── raw/                 # 原始公开集
│   ├── mpid-v1/             # 内部统一数据集（train/val/test + EDA）
│   └── mpid-v1-crossmodal/  # 跨模态子集
│
├── models/                  # 权重（gitignored）
│   └── smolvlm-500m/
│
├── runs/                    # 训练/评估产物（gitignored）
│   └── balanced-600/        # MPID LoRA checkpoint
│
├── mpid_offline/            # T2.11 离线分发包（gitignored）
│
└── report/                  # Phase 6 技术报告（gitignored）
    ├── technical_report.md
    └── figures/
```

---

## 6. 各 Phase 在哪、怎么跑

| Phase | 内容 | 入口脚本 | 验证 |
|---|---|---|---|
| **0A-1** | 环境搭建 + 设备抽象 | `scripts/smoke_env.py` | [§ 0A-1](doc/VERIFICATION.md#phase-0a-1--运行环境搭建) |
| **0A-2** | SmolVLM-500M 本地化 | `scripts/smoke_model.py` | [§ 0A-2](doc/VERIFICATION.md#phase-0a-2--smolvlm-500m-模型准备) |
| **0A-3** | 数据集准备 | `scripts/smoke_data.py` | [§ 0A-3](doc/VERIFICATION.md#phase-0a-3--训练--测试数据集准备) |
| **0** | 项目脚手架 | `python -c "from mpid.device import get_device"` | [§ Phase 0](doc/VERIFICATION.md#phase-0--脚手架) |
| **1** | 威胁模型 + 数据集 | `scripts/build_phase1.py` | [§ Phase 1](doc/VERIFICATION.md#phase-1--多模态注入威胁模型构建c1) |
| **2** | VLM 端到端基线 | `scripts/{train,eval,measure_offline,package_offline,smoke_offline}.py` | [§ Phase 2](doc/VERIFICATION.md#phase-2--vlm-端到端检测基线c2) |
| **3** | C4 早退 | `scripts/infer_c4_c5_c6.py` | ✅ 已完成 |
| **4** | C5 规则前置 | `scripts/infer_c4_c5_c6.py` | ✅ 已完成 |
| **5** | C6 跨模态 | `scripts/infer_c4_c5_c6.py` | ✅ 已完成 |
| **6** | 攻防基线评测 | `scripts/{aggregate_results,make_figures}.py` | 待启动 |
| **7** | 项目整理 | `Makefile` / `MODEL_CARD.md` | 待启动 |

### 当前状态（2026-07-20）

| Phase | 状态 | 备注 |
|---|---|---|
| 0A-1 / 0A-2 / 0A-3 | ✅ | 跨平台一致性已验证 |
| 0 | ✅ | `mpid` 包 + 3 CLI 占位 |
| 1 | ✅ | 25,646 条 × 3 类 + 120 张 cross-modal 图 |
| 2 | ✅ | MPID LoRA checkpoint（balanced-600） |
| 3 / 4 / 5 | ✅ | C4/C5/C6 三层防御链路已完成 |
| 6 | ⏳ | 依赖 3/4/5 产出 |
| 7 | ⏳ | 依赖 6 报告 |

**核心成果**：Macro F1 **85.5%**，端到端推理耗时 **-39.8%**，generation 次数 **-45.6%**

完整实测数字、跨平台差异、已知限制与决策见
[**doc/VERIFICATION.md**](doc/VERIFICATION.md)。

---

## 7. 重要声明

本项目**仅用于安全研究、合规审计与防御研究**：

- **不发布为对外可调用的在线服务或 API**
- **不构建主动攻击工具或越狱教程**
- **不替代**模型训练阶段的安全对齐，仅作前置过滤 / 审计用途
- 研究目标设备为普通 PC 与主流边缘设备，**不覆盖 MCU 级**极端嵌入式

代码许可见 [LICENSE](LICENSE)；课题伦理声明详见
[doc/opening-report-vlm.md](doc/opening-report-vlm.md) § 8 伦理。