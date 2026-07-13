# 任务分解 (Tasks)

> 与 [opening-report-vlm.md](opening-report-vlm.md) 严格对应。任务按 Phase 排序，标 `[P]` 优先级、`[D]` 依赖、`[T]` 预估时长。
> 文档版本：v2.0（对齐 vlm 开题报告 v0.3 · SmolVLM-500M 单一 VLM 端到端 · C4/C5/C6 三项算法优化）

---

## 任务总览

| 研究内容 | 对应 Phase |
|---|---|
| C1 多模态注入威胁模型构建 | Phase 1 |
| C2 VLM 端到端检测基线 | Phase 2 |
| C3 攻防基线评测体系 | Phase 6 |
| **C4 早退机制（核心算法优化）** | **Phase 3** |
| **C5 规则前置过滤 + VLM 精排（核心算法优化）** | **Phase 4** |
| **C6 跨模态语义一致性检测（核心算法优化）** | **Phase 5** |

---

## Phase 0A — 准备阶段（环境 / 模型 / 数据）

> 本阶段是写代码之前的"前置准备"——把工具链、基础模型、训练/测试数据全部打通并验证可用。
> 三个目标：① 环境跑通；② SmolVLM-500M 能加载并完成最小推理；③ 训练/测试数据已就位、可用、与课题要求一致。
> **P0A-1（环境）建议在 Phase 0 之前完成**；P0A-2（模型）/ P0A-3（数据）至少在 Phase 2 / Phase 1 启动前完成。

### P0A-1 运行环境搭建（适配 mac 与 x86）

- [ ] **TP1.1** 确认目标平台：`macOS (Apple Silicon, MPS)` 与 `x86 PC (CPU)`，记录参考机型与 OS 版本 `[P:high][D:-]`
- [ ] **TP1.2** 安装 Python 3.10+（推荐 3.11），在两个平台分别验证 `python --version` 输出 `[P:high][D:TP1.1]`
- [ ] **TP1.3** 创建项目 `venv`：`python -m venv .venv`，并把 `.venv/` 加入 `.gitignore` `[P:high][D:TP1.2]`
- [ ] **TP1.4** 写 `pyproject.toml` / `requirements.txt`，锁版本：`torch ≥ 2.1, transformers ≥ 4.45, peft ≥ 0.11, accelerate ≥ 0.34, bitsandbytes ≥ 0.43, datasets ≥ 2.20, scikit-learn, pillow` `[P:high][D:TP1.3]`
- [ ] **TP1.5** 安装依赖：`pip install -r requirements.txt`；在 mac 与 x86 两个平台分别跑一次，确认无 native 编译失败 `[P:high][D:TP1.4]`
- [ ] **TP1.6** 跨平台冒烟脚本 `scripts/smoke_env.py`：`import` 全部核心依赖 + 一行 `torch.randn(2,2).sum()` + 一行 SmolVLM-500M tokenizer 端到端小测 `[P:high][D:TP1.5]`
- [ ] **TP1.7** 设备选择冒烟：实现 `src/mpid/device.py` 暴露 `get_device(prefer)`，在两个平台各跑一次返回正确设备（M1 → mps，Intel → cpu） `[P:high][D:TP1.6]`
- [ ] **TP1.8** 4-bit 量化路径验证：在 mac 平台测试 `bitsandbytes` 4-bit 加载小模型是否成功；若失败准备 fallback 到 `mlx` 路径 `[P:high][D:TP1.6]`

**P0A-1 验收**：在 mac（MPS）与 x86（CPU）两个平台都跑通 `smoke_env.py`；`get_device()` 返回正确设备类型；4-bit 量化至少在一个平台能跑通最小测试。

### P0A-2 基础模型准备：SmolVLM-500M

- [ ] **TP2.1** 选型决策：`SmolVLM-500M`（约 5 亿参数，Apache 2.0，HuggingFace, COLM 2025） `[P:high][D:TP1.6]`
- [ ] **TP2.2** 模型下载脚本 `scripts/download_models.py`，把 SmolVLM-500M 本地化到 `models/smolvlm-500m/` 目录，**完全离线可加载** `[P:high][D:TP2.1]`
- [ ] **TP2.3** 模型加载冒烟 `scripts/smoke_model.py`：从 `models/smolvlm-500m` 加载 tokenizer + 模型，喂入 2-3 条 (image, text) 样例，输出 3 分类 logits shape 正确 `[P:high][D:TP2.2]`
- [ ] **TP2.4** 4-bit 量化加载验证：测试 `BitsAndBytesConfig(load_in_4bit=True)` 能成功加载 SmolVLM-500M `[P:high][D:TP2.3]`
- [ ] **TP2.5** 离线加载验证：拔网 / 关代理后再跑 `smoke_model.py`，确认**零网络依赖** `[P:high][D:TP2.4]`
- [ ] **TP2.6** 跨平台一致性：在 mac 与 x86 两个平台各跑一次 `smoke_model.py`，对比输出 logits shape 与数值范围是否一致 `[P:high][D:TP2.5]`

**P0A-2 验收**：SmolVLM-500M 在两个平台离线可加载并完成最小推理（3 分类 logits 输出 shape 正确）。

### P0A-3 训练 / 测试数据集准备

- [ ] **TP3.1** 注入样本（公开集）拉取：确定要拉取以下两个 `[P:high][D:TP1.6]`
  - `deepset/prompt-injections`（EN，~540 条，覆盖 direct / indirect 注入）
  - `xTRam1/safe-guard-prompt-injection`（多语种，~6k 条）
- [ ] **TP3.2** 多模态注入样本：`JailbreakV-28K`（EN/CN，28k 条，多模态越狱样本） `[P:medium][D:TP1.6]`
- [ ] **TP3.3** 干净负例：MMLU / CMMLU prompts（EN/CN，~2k）+ Flickr30k 图像描述（EN，~1k） `[P:medium][D:TP1.6]`
- [ ] **TP3.4** 数据下载脚本 `scripts/download_data.py` 把上述公开集落地到 `data/raw/` 目录（**只下载不修改**） `[P:high][D:TP3.1-TP3.3]`
- [ ] **TP3.5** 离线加载冒烟 `scripts/smoke_data.py`：
  - 每个公开集至少 5 条样本能完整读出
  - 字段（`text / prompt / image / label`）至少一个非空
  - 类别标签覆盖 `clean / direct / indirect` 三类
  - 多模态样本（JailbreakV-28K）能正确读出图像
- [ ] **TP3.6** 课题符合性自检清单：`[P:high][D:TP3.5]`
  - 是否包含中文样本？✅（CMMLU + JailbreakV-28K）
  - 是否包含图像样本？✅（Flickr30k 描述 + JailbreakV-28K）
  - 是否支持多语种？✅（SmolVLM 原生多语种 + safe-guard 多语）
  - 是否覆盖直接 / 间接注入？✅（deepset / safe-guard 含 direct / indirect）
  - 干净样本是否充足？✅（MMLU / CMMLU 远超 1k）

**P0A-3 验收**：4 类数据（注入公开集 / 多模态注入 / 干净负例）已落地；最小加载脚本可读出 5+ 条样本；课题符合性自检清单全部 ✅。

---

**Phase 0A 总验收**：
- 工具链在 mac MPS 与 x86 CPU 上都可跑；
- SmolVLM-500M 离线可加载并完成最小推理（3 分类）；
- 4-bit 量化路径在两个平台都验证通过；
- 训练 / 测试数据已就位、schema 校验通过、课题符合性清单全部 ✅。

---

## Phase 0 — 脚手架

- [ ] **T0.1** 创建项目目录结构 `src/mpid/`、`scripts/`、`configs/`、`data/`、`models/`、`artifacts/` `[P:high][D:-]`
- [ ] **T0.2** 实现 `src/mpid/device.py` 设备抽象，暴露 `get_device(prefer)` 函数；单元测试 M1 / Intel 两种环境各跑通 `[P:high][D:T0.1]`
- [ ] **T0.3** 写 `src/mpid/__init__.py` 与 `scripts/train.py` / `scripts/eval.py` / `scripts/infer.py` 入口占位（仅 echo） `[P:medium][D:T0.1]`
- [ ] **T0.4** 创建 venv 并跑通 `import mpid` smoke test `[P:high][D:T0.2]`

**Phase 0 验收**：在 mac MPS 与 x86 CPU 上分别能 `python -c "from mpid.device import get_device; print(get_device())"` 打印正确设备。

---

## Phase 1 — 多模态注入威胁模型构建（对应 C1）

- [ ] **T1.1** 形式化定义威胁模型：直接注入 / 间接注入 / 多模态注入三类 `[P:high][D:T0.1]`
- [ ] **T1.2** 编写 `docs/threat_model.md`：攻击分类、典型样例、形式化定义、攻击者能力假设 `[P:high][D:T1.1]`
- [ ] **T1.3** 编写 `src/mpid/data/public_loaders.py`，把 P0A-3 拉取的公开集统一为内部 schema `(text, image, label)` `[P:high][D:T1.2]`
- [ ] **T1.4** 编写 `src/mpid/data/synthetic_image_injection.py`（可选）：输入干净图 + 攻击模板库，输出注入图像 + 标注 JSONL `[P:medium][D:T1.3]`
- [ ] **T1.5** 实现 `src/mpid/data/split.py`，8:1:1 划分 + 类别均衡检查 `[P:high][D:T1.3]`
- [ ] **T1.6** EDA 报告 `data/mpid-v1/EDA.md`：类别分布、语种分布、长度分布、典型样例 `[P:medium][D:T1.5]`
- [ ] **T1.7** 数据质量自检：随机抽 20 条人工核对标签 `[P:high][D:T1.5]`
- [ ] **T1.8** 自构造 `data/mpid-v1-crossmodal/` 子集（100+ 条，C6 用） `[P:high][D:T1.4]`

**Phase 1 验收**：`data/mpid-v1/` 下有 `train.jsonl` / `val.jsonl` / `test.jsonl`，总样本 ≥ 1k，含 cross-modal 增强子集。

---

## Phase 2 — VLM 端到端检测基线（对应 C2）

- [ ] **T2.1** 封装 VLM 推理抽象 `src/mpid/adapters/vlm.py`：暴露 `forward(image, text) -> logits` `[P:high][D:T0.4]`
- [ ] **T2.2** 实现 `Backbone` 注册表 `src/mpid/backbones/registry.py`：注册 `smolvlm-500m`，支持 4-bit 量化配置项 `[P:high][D:T0.3]`
- [ ] **T2.3** 实现 `DetectorHead` 抽象 `src/mpid/heads/classification.py`：3 分类（clean / direct / indirect）+ 风险分（0~1） `[P:high][D:-]`
- [ ] **T2.4** 实现 `src/mpid/data/prompt.py`：构造 §3.3.3 的 3 分类 prompt 模板 `[P:high][D:T2.3]`
- [ ] **T2.5** 实现 `src/mpid/train/trainer.py`：LoRA 注入 + 训练循环 + 评估回调 + 早停逻辑 `[P:high][D:T2.2-T2.4]`
  - 支持 `device` 参数
  - 支持 `gradient_checkpointing`
  - 支持 `quantization_config`（4-bit）
- [ ] **T2.6** 写 `configs/baseline.yaml` 训练配置（LoRA r=16, alpha=32, epochs=3, lr=2e-4） `[P:medium][D:T2.5]`
- [ ] **T2.7** 在 MPS 上跑 3 epoch 训练，保存 `artifacts/lora_baseline.safetensors` `[P:high][D:T2.6]`
- [ ] **T2.8** 在 x86 CPU 上跑同样配置，验证跨平台一致（F1 差异 < 2%） `[P:high][D:T2.7]`
- [ ] **T2.9** 编写 `scripts/eval.py`，输出 `report_baseline.json` + 混淆矩阵 `[P:high][D:T2.7]`
- [ ] **T2.10** 编写 `scripts/measure_offline.py` 量化离线部署指标：
  - 模型权重大小（MB）
  - 冷启动加载到推理就绪时间（秒）
  - 单样本推理延迟 P50 / P95（ms）
  - 单样本推理峰值内存（MB）
  - 推理过程网络流量（应为 0）
- [ ] **T2.11** 编写 `scripts/package_offline.py`：把 backbone 权重 + LoRA + tokenizer + 推理脚本 + 依赖清单打包到 `mpid_offline/` 目录 `[P:high][D:T2.10]`
  - 校验：目录内**无 `.git/` 引用、无网络初始化代码**
- [ ] **T2.12** 离线包冒烟：解包 → 在目标机器跑通 `python mpid_offline/infer.py` 单样本推理 `[P:high][D:T2.11]`

**Phase 2 验收**：VLM 端到端基线在 test 集 Macro F1 ≥ 0.80，误报率 ≤ 5%；跨平台一致；离线包可独立分发与运行；离线特性指标全部量化记录。

---

## Phase 3 — C4 早退机制（对应 §3.6.1）

> C4 是**纯加速方向**优化。基础设施可与 Phase 2 共用（共用 backbone + head），建议在 Phase 2 完成后启动。
> 对应开题报告 §3.6.1。

- [ ] **T3.1** 实现 `src/mpid/heads/early_exit.py`：`EarlyExitHead` 类（2 层 MLP + 3 分类头） `[P:high][D:T2.2]`
- [ ] **T3.2** 在 `SmolVLMBackbone` 包装类中暴露中间层 hidden state 钩子 `[P:high][D:T3.1]`
- [ ] **T3.3** 扩展 `trainer.py`：增加 `early_exit_loss`（联合损失） + 记录 `exit_rate` 指标 `[P:high][D:T3.1]`
- [ ] **T3.4** 实现自适应推理 `src/mpid/infer/early_exit.py`：连续 2 层置信度 > θ 即返回 `[P:high][D:T3.3]`
- [ ] **T3.5** 写 `configs/c4_early_exit.yaml`（中间层位置、loss 权重、阈值 θ） `[P:medium][D:T3.3]`
- [ ] **T3.6** 训练 + 保存 `artifacts/c4_early_exit.safetensors` + EarlyExit 头权重 `[P:high][D:T3.5]`
- [ ] **T3.7** 跑 `scripts/eval_c4.py`：latency P50/P95、exit rate by class、F1 变化、干净/注入/OOD 集延迟分布 `[P:high][D:T3.6]`

**Phase 3 验收**：Macro F1 下降 ≤ 1%，平均推理速度提升 ≥ 1.5x；中间层退出率分布合理。

---

## Phase 4 — C5 规则前置过滤 + VLM 精排（对应 §3.6.2）

> C5 是**安全 + 加速方向**优化，是工业界 defense-in-depth 防御范式在 VLM 检测中的落地。
> 对应开题报告 §3.6.2。

- [ ] **T4.1** 实现 `src/mpid/rules/engine.py`：规则引擎接口，支持 4 类规则（关键词 / Unicode / 结构 / 敏感指令） `[P:high][D:T2.5]`
- [ ] **T4.2** 实现 4 类规则的具体匹配函数：
  - `src/mpid/rules/keyword.py`：黑名单 + 白名单关键词（中英文）
  - `src/mpid/rules/unicode.py`：零宽字符、RTL 覆盖、同形字符
  - `src/mpid/rules/structure.py`：超长 base64、隐藏 URL、IP / 邮箱
  - `src/mpid/rules/sensitive.py`：`/system`、`assistant:`、`<<SUDO>>` 等
- [ ] **T4.3** 实现 `src/mpid/rules/registry.py`：规则文件可热加载（YAML/JSON） `[P:high][D:T4.1]`
- [ ] **T4.4** 实现 `src/mpid/infer/refine.py`：规则引擎 → 触发策略（黑名单/白名单/可疑信号/无信号）→ VLM 精排 完整流程 `[P:high][D:T4.1-T4.3]`
- [ ] **T4.5** 实现可解释输出：规则命中时输出"命中规则名 + 命中片段" `[P:high][D:T4.4]`
- [ ] **T4.6** 写规则库 `configs/rules.yaml`：黑/白名单关键词 + Unicode / 结构 / 敏感指令 规则条目（≥ 20 条） `[P:medium][D:T4.2-T4.3]`
- [ ] **T4.7** 评估规则库在 clean 集 / injection 集上的命中率与误报率，**确保 clean 集 FPR 上升 ≤ 1%** `[P:high][D:T4.6]`
- [ ] **T4.8** 写 `scripts/eval_c5.py`：规则命中率、规则拦截样本的 Recall、VLM 兜底样本的 F1、干净集 FPR、平均延迟 `[P:high][D:T4.7]`

**Phase 4 验收**：规则库能热加载；黑名单 100% 拦截已知威胁；clean 集 FPR 上升 ≤ 1%；平均延迟下降 40-60%；规则命中可解释输出完整。

---

## Phase 5 — C6 跨模态语义一致性检测（对应 §3.6.3）

> C6 是**纯范围方向**优化，扩展对间接注入的检测能力。
> 对应开题报告 §3.6.3。

- [ ] **T5.1** 实现辅助 prompt 模板 `src/mpid/data/prompt_c6.py`："图像中的内容是否与用户的问题/请求相关？回答 yes 或 no" `[P:high][D:T2.4]`
- [ ] **T5.2** 实现 `src/mpid/heads/dual_output.py`：主输出（3 分类）+ 辅助输出（yes/no 二分类）双输出融合 `[P:high][D:T5.1]`
- [ ] **T5.3** 扩展 `trainer.py`：增加辅助 prompt 的 loss 权重与训练流程 `[P:high][D:T5.2]`
- [ ] **T5.4** 实现 `src/mpid/rules/crossmodal.py`：触发规则——辅助输出 = "no" AND prompt 含敏感词 → 强制判 `indirect_injection` `[P:high][D:T5.2]`
- [ ] **T5.5** 维护 `configs/sensitive_words.yaml`（中英文敏感词列表）："退款"、"取消"、"同意"、"system" 等 `[P:medium][D:T5.4]`
- [ ] **T5.6** 训练数据准备：在 cross-modal 子集上让辅助 prompt 学会判断"图文是否相关" `[P:high][D:T5.3,T1.8]`
- [ ] **T5.7** 写 `configs/c6_crossmodal.yaml` + 训练 + 保存 `artifacts/c6_crossmodal.safetensors` `[P:high][D:T5.6]`
- [ ] **T5.8** 跑 `scripts/eval_c6.py`：cross-modal 子集 Recall（目标 ≥ 80%）、干净集 FPR（**核心约束：不上升**）、Macro F1、辅助 prompt 一致性 `[P:high][D:T5.7]`

**Phase 5 验收**：cross-modal 子集 Recall ≥ 80%；clean 集 FPR 不上升；Macro F1 提升 2-3%。

---

## Phase 6 — 攻防基线评测体系（对应 C3）

- [ ] **T6.1** 实现 `Keyword` 基线（regex 关键词匹配） `[P:medium][D:-]`
- [ ] **T6.2** 跑 `meta-llama/PromptGuard-86M` zero-shot 作为对比（需考虑网络，离线环境下提前下载本地） `[P:medium][D:-]`
- [ ] **T6.3** 实现 `src/mpid/eval/aggregate.py`：合并 baseline / C4 / C5 / C6 多模型结果，按子集切片 `[P:high][D:T2.9,T3.7,T4.8,T5.8]`
- [ ] **T6.4** 输出 `report/figures/` 下混淆矩阵、F1 柱图、按语种/类型切片热力图 `[P:medium][D:T6.3]`
- [ ] **T6.5** 撰写 `report/technical_report.md` 主体
  - 章节：摘要 / 背景 / 威胁模型 / 方法 / 数据 / 实验 / 消融 / 局限 / 伦理
- [ ] **T6.6** 在 CPU 上重跑验证报告数字可复现 `[P:high][D:T6.5]`

**Phase 6 验收**：`report/technical_report.md` 完稿，含 4 张以上图表；与 PromptGuard-86M、关键词基线、规则前置基线对比报告完整。

---

## Phase 7 — 项目整理

- [ ] **T7.1** 完善 `README.md`：项目说明、快速开始、复现命令 `[P:high][D:T6.5]`
- [ ] **T7.2** 写 `data/mpid-v1/README.md`：数据来源、统计、license 说明 `[P:medium][D:T1.5]`
- [ ] **T7.3** 整理 `scripts/`：train / eval / infer / c4 / c5 / c6 入口，CLI 完整 `[P:high][D:T6.5]`
- [ ] **T7.4** 加 `Makefile` 或 `tox.ini` 提供 `make train / make eval` 入口 `[P:low][D:T7.3]`
- [ ] **T7.5** 写 Model Card：`MODEL_CARD.md` 描述能力、局限、风险 `[P:high][D:T6.5]`
- [ ] **T7.6** 最后一次跨平台冒烟（mac + x86） `[P:high][D:T7.1-T7.5]`

**Phase 7 验收**：从克隆到跑通 `make eval` 在两个平台均 ≤ 10 分钟。

---

## 关键依赖图

```
【Phase 0A — 准备阶段】
TP1.1 ── TP1.2 ── TP1.3 ── TP1.4 ── TP1.5 ── TP1.6 ── TP1.7 ── TP1.8
   │                                          │
TP2.1 ── TP2.2 ── TP2.3 ── TP2.4 ── TP2.5 ── TP2.6
   │                                          │
TP3.1 ── TP3.2 ── TP3.3 ── TP3.4 ── TP3.5 ── TP3.6
                                              │
【Phase 0 — 脚手架】                            │
T0.1 ── T0.2 ── T0.4                          │
   └─ T0.3                                    │
                                              │
【Phase 1 — 威胁模型 C1】                       │
T1.1 ── T1.2 ── T1.3 ── T1.5 ── T1.6          │
                              └─ T1.7          │
T1.4 ── T1.8                                  │
                                              │
【Phase 2 — VLM 端到端基线 C2】                  │
T2.1 ── T2.2 ─┐                                │
T2.3 ───┤      ├─ T2.5 ── T2.6 ── T2.7 ── T2.8
T2.4 ───┘                       └─ T2.9       │
                                └─ T2.10 ── T2.11 ── T2.12
                                                       │
【Phase 3 — C4 早退】                                  │
T2.2 ── T3.1 ── T3.3 ── T3.5 ── T3.6 ── T3.7         │
        └─ T3.2                                       │
                └─ T3.4                              │
                                                       │
【Phase 4 — C5 规则前置】                              │
T4.1 ── T4.2 ── T4.3 ── T4.4 ── T4.6 ── T4.7 ── T4.8 │
                          └─ T4.5                     │
                                                       │
【Phase 5 — C6 跨模态】                                │
T2.4 ── T5.1 ── T5.2 ── T5.3 ── T5.6 ── T5.7 ── T5.8 │
                       └─ T5.4 ── T5.5                │
              T1.8 ─────┘                              │
                                                       │
【Phase 6 — 攻防基线 C3】                              │
T6.1 / T6.2 (独立) ── T6.3 ── T6.4 ── T6.5 ── T6.6    │
                  T2.9 + T3.7 + T4.8 + T5.8 ──┘      │
                                                       │
【Phase 7 — 项目整理】                                  │
T6.5 ── T7.1 ── T7.4                                  │
       T7.2                                           │
       T7.3 ── T7.6                                   │
       T7.5
```

> **C4 / C5 / C6 三条链相对独立**：
> - C4 依赖 T2.2（VLM backbone 注册）
> - C5 依赖 T2.5（trainer 可扩展）+ T2.11（离线包打包）
> - C6 依赖 T2.4（prompt 模板）+ T1.8（cross-modal 子集）
> 可任选一条/两条/全部；建议**先做 C5（多层防御，最具实用价值）→ C4（加速）→ C6（范围）**。

---

## 风险缓冲

- 任一 Phase 超期 1 天：先收尾当前任务，跳过次要子项（如美化、扩展数据）
- MPS 不可用：直接在 CPU 上跑（总时间 ×3-5）
- 关键路径阻塞（T2.5 trainer）：先做最小可跑版本，特性延后
- **C4 / C5 / C6 进度选择**：
  - 时间紧（< 1 天）→ 只做 C5 规则库配置（不写新代码）
  - 时间适中（1-3 天）→ C5 + C4（多层防御 + 加速）
  - 时间宽裕（3-5 天）→ C4 + C5 + C6 全开
  - 任意优化导致 Macro F1 下降 > 1% → **放弃该优化并记录到 known issues**，不影响主交付

---

## 不在范围

- 在线服务部署
- 实时拦截系统
- 对抗样本主动生成
- 第三方大模型 API 微调
- 7B+ 规模 VLM 迁移
- 早退机制在更大规模 VLM（如 LLaVA-7B）上的应用
- 通用跨模态矛盾检测基准（仅做 prompt injection 子集）
- 零宽字符检测的鲁棒性对抗（Unicode normalization 绕过）
- 多 VLM 家族横向对比
