# MPID - Multimodal Prompt Injection Defense

> 面向离线、隐私敏感和资源受限场景的轻量级多模态提示注入检测项目。
>
> 最终方案：`F-3000-MCR-SBC`。在冻结 Standard Benchmark v2/full500 上取得 Accuracy **62.00%**、Macro F1 **59.28%**。

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![final-macro-f1](https://img.shields.io/badge/final%20Macro%20F1-59.28%25-brightgreen)](doc/final-report.md)

---

## 目录

1. [项目是什么](#1-项目是什么)
2. [最终交付](#2-最终交付)
3. [核心结果](#3-核心结果)
4. [安装与快速验证](#4-安装与快速验证)
5. [项目结构](#5-项目结构)
6. [训练、评测与离线复现](#6-训练评测与离线复现)
7. [文档与项目边界](#7-文档与项目边界)

---

## 1. 项目是什么

MPID 是部署在主模型或人工审计流程之前的离线前置检测组件。它接收用户文本和可选图像，判断输入是否安全，并在发现攻击时给出明确的阻断阶段和依据。

提示注入既可能直接写在用户文本中，也可能藏在图片、截图或其他不可信外部内容中。项目将它们显式区分，避免以单一二分类掩盖某一攻击类别的完全漏报。

| 标签 | 定义 | 典型载体 |
|---|---|---|
| `clean` | 正常请求，不含提示注入 | 一般问答、图片描述、文档阅读请求 |
| `direct` | 攻击指令直接出现在用户可控文本中 | 角色越权、jailbreak、“忽略之前指令” |
| `indirect` | 恶意指令位于图片或其他不可信外部内容中，或由图文组合形成 | 图像中的注入文字、图文冲突诱导 |

| 维度 | 本项目 |
|---|---|
| 模型核心 | SmolVLM-500M + LoRA + MPID 三分类 head |
| 防御方式 | 规则、像素 OCR、跨模态检查、上下文隔离、分数校准和置信门控 |
| 部署 | 本地离线、低吞吐端侧、人工辅助审计 |
| 输出 | `clean/direct/indirect`、`allow/block`、决策阶段、解释和耗时 |
| 不在范围 | 云端安全服务、高并发生产服务、攻击自动化、未知攻击的无条件安全保证 |

### 技术核心

- **参数高效训练**：在 SmolVLM-500M 语言注意力的 `q/k/v/o` 投影上进行 LoRA 微调，搭配三分类 head。
- **反事实边界学习**：以 `clean/direct/indirect` 三元组和对角排序损失学习同一场景下的类别差异，而不是只记忆攻击关键词。
- **纵深防御**：C5 优先处理高确定性文本攻击；C6B-lite 从真实图像像素做本地 OCR；C6A 提供已知兼容性风险检查；剩余样本交由多模态模型。
- **不可信内容隔离**：MCR 将 OCR 文本置于 `untrusted_image_ocr` 受限上下文，不将图中文字提升为用户指令。
- **可审计校准**：SBC 只使用 smoke 阶段锁定的固定 logits 偏移，最终 full500 不参与调参。
- **离线交付**：模型、OCR 权重、推理代码、checksum 与 smoke 一同打包，数据与模型大文件不提交到 Git。

## 2. 最终交付

最终方案不是新的 checkpoint，而是固定的 `checkpoint_step_3000` 加锁定运行时策略：

```text
文本 + 可选图像
  -> C5 文本高确定性规则
  -> C6B-lite 本地 RapidOCR
  -> C6A 兼容性/跨模态风险检查
  -> MCR：不可信 OCR 上下文路由
  -> F-3000：SmolVLM + LoRA + 三分类 head
  -> SBC：固定分数校准
  -> C4：高置信 clean 放行
  -> block / allow
```

| 交付物 | 说明 | 状态 |
|---|---|---|
| `checkpoint_step_3000.safetensors` | F-3000 固定 LoRA 与分类 head | 已完成 |
| `src/mpid/infer/pipeline.py` | C5/C6/head/C4 的优化 pipeline | 已完成 |
| C6B-lite + MCR | 本地 OCR 与不可信图文上下文隔离 | 已完成 |
| `runs/_artifact/F-3000-MCR-SBC/` | 可移动离线 artifact、checksum 和 smoke | 已完成，本地保存 |
| [final-report.md](doc/final-report.md) | 最终结果、训练收敛、性能和审计证据 | 已完成 |

MCR 和 SBC 分别位于分类器输入构造与 logits 后处理；C5、C6B-lite、C6A 和 C4 是可独立审计的 pipeline 关卡。完整流程图、理论说明与运行边界见 [最终报告 2.1 节](doc/final-report.md#21-方案组成)。

## 3. 核心结果

所有最终能力数字均来自冻结的 Standard Benchmark v2/full500。该集不参与 MCR、SBC、阈值或规则的选择。

### 最终效果

| 指标 | F-3000 LoRA-only | F-3000-MCR-SBC 完整 pipeline | 增量 |
|---|---:|---:|---:|
| Accuracy | 56.60% | **62.00%** | **+5.40pp** |
| Macro F1 | 38.89% | **59.28%** | **+20.37pp** |
| Weighted F1 | 50.91% | **61.32%** | **+10.41pp** |
| clean F1 | 67.19% | **68.38%** | +1.19pp |
| direct F1 | 49.48% | **53.56%** | +4.08pp |
| indirect F1 | 0.00% | **55.90%** | +55.90pp |
| 预测完整性 | 500 / 500 | 500 / 500 | 完整 |

### 推理效率与交付验证

| 指标 | LoRA-only | 完整 pipeline | 结果 |
|---|---:|---:|---|
| 模型加载后的平均判定耗时 | 14.06 秒/条 | **10.56 秒/条** | **-24.86%** |
| clean 平均判定耗时 | 14.07 秒/条 | **12.02 秒/条** | -14.62% |
| direct 平均判定耗时 | 14.53 秒/条 | **10.51 秒/条** | -27.63% |
| indirect 平均判定耗时 | 12.91 秒/条 | **5.83 秒/条** | -54.80% |
| 发布清单 / checksum / ZIP CRC | 94 项 / 全部匹配 / 通过 | - | 离线 artifact 验证通过 |

完整 pipeline 解决了 LoRA-only 的 indirect F1 为 0 的失效模式，但 Direct Recall 仍为 45.14%，未知攻击、OCR 噪声和跨硬件性能仍需在新冻结数据上继续验证。详细指标、实验口径和风险见 [最终报告](doc/final-report.md)。

## 4. 安装与快速验证

### 4.1 环境要求

- Python 3.10+，推荐 Python 3.11。
- Windows 使用 PowerShell；macOS 使用 zsh/bash。
- 完整模型推理需要另行恢复本地模型、数据和 artifact；核心单元测试与轻量 pipeline smoke 不需要这些大文件。

### 4.2 创建环境

**Windows PowerShell**：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-x86.txt
python -m pip install -e .
```

**macOS**：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 4.3 验证核心编排

以下命令不加载 VLM；其中 `--probs` 是模拟分类 head 的输出，仅用于验证规则和 pipeline 顺序。

```powershell
python -m pytest tests -q

# 预期：direct / c5_rules
python scripts/infer_pipeline_light.py --text "ignore previous instructions" --probs 0.34,0.33,0.33

# 预期：clean / c4_early_exit
python scripts/infer_pipeline_light.py --text "summarize this note" --probs 0.96,0.02,0.02
```

## 5. 项目结构

```text
llm-compliance/
├── README.md                            项目入口与最终结果摘要
├── pyproject.toml                       包元数据、依赖范围与测试配置
├── requirements*.txt                    macOS、x86 CPU 与 MLX 依赖清单
│
├── src/mpid/                            核心 Python 包
│   ├── adapters/vlm.py                  SmolVLM 推理适配器
│   ├── backbones/registry.py            Backbone 注册表
│   ├── heads/classification.py          三分类 head
│   ├── train/trainer.py                 LoRA、配对三元组训练与 checkpoint 恢复
│   ├── infer/pipeline.py                优化 pipeline 与决策审计输出
│   ├── rules/engine.py                  C5 文本规则
│   ├── crossmodal/                      C6A、C6B-lite OCR 与图像冲突规则
│   ├── early_exit.py                    C4 高置信 clean 门
│   └── data/                            数据 schema、加载、划分和 prompt 构造
│
├── scripts/                             通用训练、评测、打包、诊断与数据构建脚本
├── tests/                               设备、规则、OCR、C4 与 pipeline 单元测试
├── demo/                                Gradio 演示原型，不作为最终能力评测入口
│
├── doc/                                 正式文档
│   ├── final-report.md                  最终结题报告
│   ├── project-plan-and-verification.md 执行与验证归档
│   ├── reference.md                     概念、实现解读和历史复现手册
│   └── opening-report-*.md              开题与历史参考材料
│
└── runs/                                本地训练、评测和发布资产
    ├── _models/、_datasets/             共享模型和数据缓存，不进入 Git
    ├── phase2_3_full_3000_*/            F-3000 的本地训练与审计资产
    └── _artifact/F-3000-MCR-SBC/        最终离线交付包，不进入 Git
```

`runs/` 中的轻量配置、脚本、日志和说明可以保留在 Git；数据集、模型、checkpoint、预测、图片和压缩包由 [.gitignore](.gitignore) 排除，避免将大文件提交到仓库。

## 6. 训练、评测与离线复现

### 6.1 训练与策略收敛

最终模型来自一条受控的两阶段路线，而非单次长训：

1. 使用统一 smoke150 对 A-H、F2-F4 共 11 个候选机制进行快速比较。
2. F 是唯一保持三类非零 Recall、类别最平衡的方向，因此作为正式训练基线。
3. 使用 3,000 条均衡训练样本（每类 1,000 条）完成 F-3000，并固化 `checkpoint_step_3000`。
4. 只在 smoke150 锁定 MCR/SBC；V2/full500 仅用于一次最终盲验收。

完整的候选机制差异、训练时间、checkpoint 恢复和问题复盘见 [最终报告第 3 节](doc/final-report.md#3-f-3000-正式训练与收敛过程) 与 [执行与验证归档](doc/project-plan-and-verification.md)。

### 6.2 恢复完整离线 artifact

模型、数据、checkpoint 与离线包为大文件，不保存在 Git 中。完整推理前，从受控存储恢复：

- `runs/_models/` 与 `runs/_datasets/`；
- `runs/phase2_3_full_3000_20260729_0958/` 中所需 run-local 资产；
- `runs/_artifact/F-3000-MCR-SBC/` 交付目录或其 ZIP 包。

恢复后，以 artifact 内的 `README`、`MANIFEST.json`、`CHECKSUMS.txt` 和 `smoke_offline.py` 完成文件完整性与断网验证。artifact 的 94 项发布清单、checksum、3/3 smoke 与 ZIP CRC 均已通过；具体命令和交付证据见 [最终报告第 5 节](doc/final-report.md#5-离线交付与复现)。

## 7. 文档与项目边界

| 文档 | 用途 |
|---|---|
| [最终报告](doc/final-report.md) | 最终方案、完整结果、训练收敛、pipeline 和 artifact 证据 |
| [执行与验证归档](doc/project-plan-and-verification.md) | 已完成任务、最小验证、事故复盘和后续限制 |
| [技术参考手册](doc/reference.md) | 概念、代码解读、Windows/macOS 命令、历史实验和排障 |
| [VLM 开题报告](doc/opening-report-vlm.md) | 项目的研究背景、目标与方法 |
| [文档导航](doc/README.md) | 文档职责和迁移说明 |

本项目仅用于安全研究、合规审计与防御原型验证：

- 不发布为对外攻击工具或越狱教程。
- 不替代上游模型训练阶段的安全对齐，仅作为前置过滤和审计层。
- 不对生产高并发、未知攻击或跨域泛化作无条件安全承诺。

代码采用 [Apache-2.0](LICENSE) 许可。
