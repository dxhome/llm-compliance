# 任务分解 (Tasks)

> 与 [opening-report-vlm.md](opening-report-vlm.md) 严格对应。任务按 Phase 排序，标 `[P]` 优先级、`[D]` 依赖、`[T]` 预估时长。
> 文档版本：v2.7（校准 Phase 2.2 balanced 600，并合并增强微调任务到 Phase 2.3）
>
> **v2.7 变更**：
> - 校准 **Phase 2.2**：实际执行是 balanced 600 训练，使用 clean / direct / indirect 均衡的 600 条训练样本，不再描述为 2000+ 全量训练
> - 合并今天新增的增强微调训练要求，统一纳入 **Phase 2.3 — 高 F1 增强微调**
> - Phase 2.3 统一覆盖数据集契约、增强训练、训练期 compare、断点续训、best checkpoint 选择、package 和 offline smoke
>
> **v2.6 变更**：
> - 优化 **P0A-3 训练 / 测试数据集准备**：将 core + synthesis 合并为默认基础数据，manual-review 作为候选保留
> - 新增可直接下载的 prompt injection / jailbreak / indirect-email 数据源候选，用于支撑 Phase 2.3 的高 F1 微调目标
> - 明确 `WildJailbreak` 等需要接受使用条款或人工审查的数据集不进入 P0A-3 自动下载范围
>
> **v2.5 变更**：
> - 新增 **Phase 2.3 — 高 F1 微调改进**，目标是通过重新设计数据、训练、验证和 checkpoint 选择，使 clean / direct / indirect 三个分类的目标类 F1 均达到 **> 0.70**
> - Phase 2.3 基于 `runs/phase2_2_balanced_600_20260718_1955` 的 balanced 600 结果复盘：clean F1=0.985、direct F1=0.473、indirect F1=0.000
> - Phase 2.3 暂只登记任务计划，不立即实施；实施前需要单独确认 run id、训练预算和是否引入 OCR / C6 轻量信号
>
> **v2.4 变更**：
> - 顶层只保留 `runs/` 作为本地执行目录；旧 `configs/`、`data/`、`models/`、`artifacts/`、`logs/` 不再作为执行入口
> - 具体 run 使用带时间后缀的 `runs/<run_id>/`，run-local launcher 放入 `runs/<run_id>/scripts/`
> - 顶层 `scripts/` 只保留通用脚本和 `scripts/run_phase2_workflow.ps1` 模板入口
>
> **v2.3 变更**：
> - **Phase 2 拆分为 2.1（smoke 训练）和 2.2（balanced 600 训练）两个子阶段**
>   - **Phase 2.1** = 用 `max_train_records=5` 跑通管线、**不验证指标**（避免误用 5 条训出的 head 跑 C4/C5/C6）
>   - **Phase 2.2** = 使用 clean / direct / indirect 均衡的 600 条样本完成端到端 balanced 训练，产出 `lora_full.safetensors` / 离线包，并作为 Phase 2.3 增强微调前的实验基线
> - Phase 2.2 新增 9 个任务（T2.13-T2.21）：基础数据下载 → balanced 600 split → balanced 训练 → compare 复盘 → 离线包升级
> - 任务总览表 C2 对应 Phase 更新为 "Phase 2.1（smoke 训练）+ Phase 2.2（balanced 600 训练）"
>
> **v2.2 变更**：
> - **Phase 3 重新设计为"轻量级阈值判定"**（原设计的中间层 MLP + 联合训练太重，改为复用 Phase 2 head + 阈值判定）
> - T3.1-T3.7 中：T3.1, T3.2, T3.4-T3.7 已完成 ✅；T3.3 中间层早退作为 V2 扩展保留（低优先级）
> - 新增的产物文件：`src/mpid/early_exit.py` + `tests/test_early_exit.py`（13 个单测通过）+ `scripts/eval.py --early-exit` 模式
> - 端到端验证：`python scripts/eval.py --early-exit --max-records 5` 输出 3 个文件（json/md/jsonl）
>
> **v2.1 变更**：
> - **Phase 5 拆分为 5A 和 5B 两个子阶段**——5A 是轻量版（规则 + CLIP 相似度，无需 LoRA 微调），5B 是完整版（训练辅助输出头，需要 LoRA 微调）
> - 必须**先完成 5A 再做 5B**：5B 在 5A 之上叠加 LoRA 训练
> - 原 T5.1-T5.8 重命名为 T5B.1-T5B.8

---

## 当前运行目录约定

所有本地执行输出统一放在 `runs/` 下。

- 顶层 `configs/`、`data/`、`models/`、`artifacts/`、`logs/` 已废弃，不应再创建。
- 一次独立端到端执行对应一个带时间后缀的目录，例如 `runs/phase2_2_balanced_600_20260718_1955/`。
- 每个 run 目录拥有自己的 `configs/`、可选 run-local `data/`、`artifacts/`、`logs/`、`scripts/`、`execution_plan.json`、`execution_plan.md`、`execution_log.md`、`status.json`。
- 共享本地资产放在 `runs/_datasets/`、`runs/_models/`、`runs/_templates/`、`runs/_manual/`。
- `runs/` 整体被 git 忽略；fresh clone 后不要假设 run-local 配置或 checkpoint 已存在。
- 顶层 `scripts/` 只放通用脚本。具体 run 的 PowerShell launcher 放在 `runs/<run_id>/scripts/`；通用 launcher 模板是 `scripts/run_phase2_workflow.ps1`。

Phase 2.2 推荐入口：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\runs\<run_id>\scripts\launch.ps1
```

## 任务总览

| 研究内容 | 对应 Phase |
|---|---|
| C1 多模态注入威胁模型构建 | Phase 1 |
| C2 VLM 端到端检测基线 | **Phase 2.1（smoke 训练）+ Phase 2.2（balanced 600 训练）+ Phase 2.3（高 F1 增强微调）** |
| C3 攻防基线评测体系 | Phase 6 |
| **C4 早退机制（核心算法优化）** | **Phase 3**（依赖 Phase 2.2 balanced 600 基线模型；Phase 2.3 完成后切换到高 F1 checkpoint） |
| **C5 规则前置过滤 + VLM 精排（核心算法优化）** | **Phase 4**（可先用 Phase 2.2 balanced 600 基线，Phase 2.3 完成后切换） |
| **C6 跨模态语义一致性检测（核心算法优化）** | **Phase 5A + 5B**（可先用 Phase 2.2 balanced 600 基线，Phase 2.3 完成后切换） |

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
- [ ] **TP2.2** 模型下载脚本 `scripts/download_models.py`，把 SmolVLM-500M 本地化到 `runs/_models/smolvlm-500m/` 目录，**完全离线可加载** `[P:high][D:TP2.1]`
- [ ] **TP2.3** 模型加载冒烟 `scripts/smoke_model.py`：从 `runs/_models/smolvlm-500m` 加载 tokenizer + 模型，喂入 2-3 条 (image, text) 样例，输出 3 分类 logits shape 正确 `[P:high][D:TP2.2]`
- [ ] **TP2.4** 4-bit 量化加载验证：测试 `BitsAndBytesConfig(load_in_4bit=True)` 能成功加载 SmolVLM-500M `[P:high][D:TP2.3]`
- [ ] **TP2.5** 离线加载验证：拔网 / 关代理后再跑 `smoke_model.py`，确认**零网络依赖** `[P:high][D:TP2.4]`
- [ ] **TP2.6** 跨平台一致性：在 mac 与 x86 两个平台各跑一次 `smoke_model.py`，对比输出 logits shape 与数值范围是否一致 `[P:high][D:TP2.5]`

**P0A-2 验收**：SmolVLM-500M 在两个平台离线可加载并完成最小推理（3 分类 logits 输出 shape 正确）。

### P0A-3 训练 / 测试数据集准备

- [ ] **TP3.1** 默认基础注入公开集拉取：优先选择可直接下载、无需人工申请或 gated approval 的数据源 `[P:high][D:TP1.6]`
  - `deepset/prompt-injections`（EN，prompt injection binary；用于 direct / clean 初始覆盖）
  - `xTRam1/safe-guard-prompt-injection`（多语种 prompt injection；用于 direct / clean 多语覆盖；下载前记录 dataset card / license 状态）
  - `Lakera/gandalf_ignore_instructions`（prompt leaking / ignore-instruction；用于 direct 与 prompt extraction）
  - `Lakera/mosscap_prompt_injection`（较大规模 prompt injection；用于 direct、hard negative 和模板多样性）
  - `microsoft/llmail-inject-challenge`（email / tool-use 场景 indirect prompt injection；用于 indirect-text，再可渲染成 indirect-image）
  - `cyberec/Prompt-injection-dataset` 或同源 `NeurAlchemy` prompt-injection 数据（用于 direct / indirect / obfuscation 的二次标注候选）
  - `hlyn-labs/prompt-injection-judge-deberta-dataset`（大规模 judge 训练数据；用于 binary injection 预筛和 hard negative）
- [ ] **TP3.2** 默认基础多模态注入样本准备 `[P:high][D:TP1.6]`
  - `JailbreakV-28K`（EN/CN，文本 jailbreak + FigStep 图像注入；用于 direct 和 image-based indirect）
  - 将 TP3.1 中的 indirect-text / direct-text 攻击样本渲染为截图、邮件、网页、表格、便签、水印、低对比度文字等图片，生成 `indirect-image` 合成集
  - 为每条 indirect-image 配 benign user prompt，例如“请总结图片内容 / 图片里有什么文字 / describe this image”，避免模型只依赖用户文本判断
- [ ] **TP3.3** 干净负例与 hard negative 准备 `[P:medium][D:TP1.6]`
  - MMLU / CMMLU prompts（EN/CN，通用问答 clean）
  - Flickr30k 图像描述与可用图像（EN，图文 clean；优先补足“带图 clean”）
  - 从 prompt injection 数据集中筛选含 `ignore / system / policy / security / execute` 等词但语义安全的 hard negative，降低误报
  - 为 clean/direct/indirect 都准备 text-only 与 image+text 对照样本，避免模型把“有图”误学为 indirect
- [ ] **TP3.4** 数据下载脚本 `scripts/download_data.py` 把默认基础数据落地到 `runs/_datasets/raw/` 目录（**下载原始数据；合成 indirect-image 只写入派生目录，不改原始数据**） `[P:high][D:TP3.1-TP3.3]`
  - 支持 `--tier basic`、`--dataset <name>` 和断点续传；默认等价于 `--tier basic`
  - 每个数据集保存 dataset card 摘要、下载时间、来源 URL、license / terms 字段快照和 sha256
  - 默认跳过 gated / 需人工接受条款 / license 不清晰的数据集，并在 manifest 中记录为 `manual_review`
- [ ] **TP3.5** 离线加载冒烟 `scripts/smoke_data.py`：
  - 每个公开集至少 5 条样本能完整读出
  - 字段（`text / prompt / image / label`）至少一个非空
  - 类别标签覆盖 `clean / direct / indirect` 三类
  - 多模态样本（JailbreakV-28K + 合成 indirect-image）能正确读出图像
  - 数据集 manifest 能输出每个 source 的用途：`clean` / `direct` / `indirect-text` / `indirect-image` / `hard-negative`
- [ ] **TP3.6** 课题符合性自检清单：`[P:high][D:TP3.5]`
  - 是否包含中文样本？✅（CMMLU + JailbreakV-28K）
  - 是否包含图像样本？✅（Flickr30k / JailbreakV-28K / 合成 indirect-image）
  - 是否支持多语种？✅（SmolVLM 原生多语种 + safe-guard / Lakera / JailbreakV 等多来源）
  - 是否覆盖直接注入？✅（deepset / safe-guard / Lakera / cyberec / JailbreakV）
  - 是否覆盖间接注入文本场景？✅（LLMail-Inject / BIPIA 类场景可作为后续 manual-review 扩展；默认基础数据中先使用 LLMail-Inject）
  - 是否覆盖间接注入图像场景？✅（JailbreakV FigStep + 基于 indirect-text 渲染的合成图像）
  - 干净样本是否充足？✅（MMLU / CMMLU / Flickr30k / hard negative）

**P0A-3 manual-review 候选（默认不自动下载）**：
- `allenai/wildjailbreak`：数据质量高，适合补 direct 和 hard negative；但 Hugging Face 页面通常需要接受 Responsible Use / 条款后访问，不进入“直接下载无门槛”默认范围
- `microsoft/BIPIA`：非常适合 indirect prompt injection benchmark，但需单独确认数据源、license、任务格式和是否可直接自动化下载
- `TensorTrust`：适合 prompt hijacking / extraction，但需单独确认当前可下载位置、license 和字段映射
- 其他未明确 license / terms 的数据集：只记录为候选，不进入 P0A-3 默认基础数据

**P0A-3 验收**：默认基础数据源已落地；最小加载脚本可读出每个 source 5+ 条样本；manifest 记录来源 URL、license / terms 快照、sha256、用途标签；基础数据覆盖 clean / direct / indirect-text / indirect-image / hard-negative；课题符合性自检清单全部 ✅。

---

**Phase 0A 总验收**：
- 工具链在 mac MPS 与 x86 CPU 上都可跑；
- SmolVLM-500M 离线可加载并完成最小推理（3 分类）；
- 4-bit 量化路径在两个平台都验证通过；
- 训练 / 测试数据已就位、schema 校验通过、课题符合性清单全部 ✅。

---

## Phase 0 — 脚手架

- [ ] **T0.1** 创建项目目录结构 `src/mpid/`、`scripts/`、`tests/`，本地执行目录统一使用 `runs/` `[P:high][D:-]`
- [ ] **T0.2** 实现 `src/mpid/device.py` 设备抽象，暴露 `get_device(prefer)` 函数；单元测试 M1 / Intel 两种环境各跑通 `[P:high][D:T0.1]`
- [ ] **T0.3** 写 `src/mpid/__init__.py` 与 `scripts/train.py` / `scripts/eval.py` / `scripts/infer.py` 入口占位（仅 echo） `[P:medium][D:T0.1]`
- [ ] **T0.4** 创建 venv 并跑通 `import mpid` smoke test `[P:high][D:T0.2]`

**Phase 0 验收**：在 mac MPS 与 x86 CPU 上分别能 `python -c "from mpid.device import get_device; print(get_device())"` 打印正确设备。

---

## Phase 1 — 多模态注入威胁模型构建（对应 C1）

- [ ] **T1.1** 形式化定义威胁模型：直接注入 / 间接注入 / 多模态注入三类 `[P:high][D:T0.1]`
- [ ] **T1.2** 威胁模型内容已并入 [`reference.md` § 2.1](reference.md)（攻击分类、典型样例、形式化定义、攻击者能力假设）`[P:high][D:T1.1]`
- [ ] **T1.3** 编写 `src/mpid/data/public_loaders.py`，把 P0A-3 拉取的公开集统一为内部 schema `(text, image, label)` `[P:high][D:T1.2]`
- [ ] **T1.4** 编写 `src/mpid/data/synthetic_image_injection.py`（可选）：输入干净图 + 攻击模板库，输出注入图像 + 标注 JSONL `[P:medium][D:T1.3]`
- [ ] **T1.5** 实现 `src/mpid/data/split.py`，8:1:1 划分 + 类别均衡检查 `[P:high][D:T1.3]`
- [ ] **T1.6** EDA 报告 `runs/_datasets/mpid-v1/EDA.md`：类别分布、语种分布、长度分布、典型样例 `[P:medium][D:T1.5]`
- [ ] **T1.7** 数据质量自检：随机抽 20 条人工核对标签 `[P:high][D:T1.5]`
- [ ] **T1.8** 自构造 `runs/_datasets/mpid-v1-crossmodal/` 子集（100+ 条，C6 用） `[P:high][D:T1.4]`

**Phase 1 验收**：`runs/_datasets/mpid-v1/` 下有 `train.jsonl` / `val.jsonl` / `test.jsonl`，总样本 ≥ 1k，含 cross-modal 增强子集。

---

## Phase 2.1 — VLM 端到端检测基线 · smoke 训练（对应 C2）

- [ ] **T2.1** 封装 VLM 推理抽象 `src/mpid/adapters/vlm.py`：暴露 `forward(image, text) -> logits` `[P:high][D:T0.4]`
- [ ] **T2.2** 实现 `Backbone` 注册表 `src/mpid/backbones/registry.py`：注册 `smolvlm-500m`，支持 4-bit 量化配置项 `[P:high][D:T0.3]`
- [ ] **T2.3** 实现 `DetectorHead` 抽象 `src/mpid/heads/classification.py`：3 分类（clean / direct / indirect）+ 风险分（0~1） `[P:high][D:-]`
- [ ] **T2.4** 实现 `src/mpid/data/prompt.py`：构造 §3.3.3 的 3 分类 prompt 模板 `[P:high][D:T2.3]`
- [ ] **T2.5** 实现 `src/mpid/train/trainer.py`：LoRA 注入 + 训练循环 + 评估回调 + 早停逻辑 `[P:high][D:T2.2-T2.4]`
  - 支持 `device` 参数
  - 支持 `gradient_checkpointing`
  - 支持 `quantization_config`（4-bit）
- [ ] **T2.6** 写 `runs/_templates/configs/baseline.yaml` 训练配置（LoRA r=16, alpha=32, epochs=3, lr=2e-4） `[P:medium][D:T2.5]`
- [ ] **T2.7** 在 MPS 上跑 3 epoch 训练，保存 `runs/<run_id>/artifacts/checkpoints/lora_baseline.safetensors` `[P:high][D:T2.6]`
- [ ] **T2.8** 在 x86 CPU 上跑同样配置，验证跨平台一致（F1 差异 < 2%） `[P:high][D:T2.7]`
- [ ] **T2.9** 编写 `scripts/eval.py`，输出 `report_baseline.json` + 混淆矩阵 `[P:high][D:T2.7]`
- [ ] **T2.10** 编写 `scripts/measure_offline.py` 量化离线部署指标：
  - 模型权重大小（MB）
  - 冷启动加载到推理就绪时间（秒）
  - 单样本推理延迟 P50 / P95（ms）
  - 单样本推理峰值内存（MB）
  - 推理过程网络流量（应为 0）
- [ ] **T2.11** 编写 `scripts/package_offline.py`：把 backbone 权重 + LoRA + tokenizer + 推理脚本 + 依赖清单打包到 `runs/<run_id>/artifacts/package/mpid_offline/` 目录 `[P:high][D:T2.10]`
  - 校验：目录内**无 `.git/` 引用、无网络初始化代码**
- [ ] **T2.12** 离线包冒烟：解包 → 在目标机器跑通 `python mpid_offline/infer.py` 单样本推理 `[P:high][D:T2.11]`

**Phase 2.1 验收**（**仅验证端到端管线，不验证防御能力**）：
- Smoke 训练能跑通：`python scripts/train.py --max-train-records 5` 跑完一轮 ≤ 15 分钟
- Smoke 评估能跑通：`python scripts/eval.py` 输出 report + 混淆矩阵
- 离线包能独立运行：解包 → `python mpid_offline/infer.py "test"` 有输出
- **不验收** Macro F1 ≥ 0.80（5 条训不出来）
- **不验收** 误报率 ≤ 5%（同上）

> ⚠️ **重要**：Phase 2.1 完成后**不要**用 `lora_baseline.safetensors` 跑 C4/C5/C6 评估——5 条训出的 head 没有真实能力。Phase 2.2 的 balanced 600 模型只能作为增强微调前的实验基线；真正高质量底座见 Phase 2.3。

---

## Phase 2.2 — balanced 600 均衡训练（对应 C2 · 续）

> **balanced 600 训练** = 使用 clean / direct / indirect 三类均衡的 600 条训练样本，跑通从数据抽样、LoRA + 3-class head 训练、compare、package 到 offline smoke 的完整端到端流程。
> 与 Phase 2.1 的 smoke 训练相比，Phase 2.2 训出的 `lora_full.safetensors` 是一个真实训练过的实验基线，但结果已证明还不足以作为最终高质量底座；后续 Phase 2.3 需要在更强数据集和训练设计上继续增强。
>
> **与 Phase 2.1 的关键差异**：
>
> | 维度 | Phase 2.1（smoke） | Phase 2.2（balanced 600） |
> |---|---|---|
> | 训练样本数 | 5 条 | 600 条，clean / direct / indirect 各 200 |
> | 数据集 | smoke 子集 | 从基础数据中均衡抽样，重点验证端到端训练流程 |
> | 训练时长 | ≤ 15 min | 约数小时到十余小时，取决于 CPU / MPS 环境 |
> | 目标 | 验证管线 | 验证 balanced 训练闭环并形成 Phase 2.3 的复盘基线 |
> | 产物用途 | CI / 回归 | 增强微调前的实验基线，不作为最终高 F1 底座 |
> | 离线包 | `lora_baseline.safetensors`（占位） | `lora_full.safetensors`（生产） |
>
> **为什么是 balanced 600？**：
> - 三类各 200 条，避免 direct / indirect 被大规模 clean 或 direct 原始分布淹没
> - 规模足够验证训练、compare、package、offline smoke 的端到端执行
> - 成本低于增强训练，适合先发现路径、日志、恢复、报告和离线包问题
> - 结果复盘显示：clean 表现好，但 direct 仍弱、indirect 基本未学到，因此后续统一进入 Phase 2.3 增强微调

### T2.13-T2.15 数据准备

- [ ] **T2.13** 扩展 `scripts/download_data.py`：下载 Phase 2.2 balanced 600 所需基础数据 `[P:high][D:T1.4]`
  - 覆盖 clean / direct / indirect 三类候选来源
  - 支持后续从基础数据中进行均衡抽样
  - 下载到 `runs/_datasets/raw/`
  - 进度条 + 断点续传 + sha256 校验
- [ ] **T2.14** 重新跑数据构建脚本生成 balanced 600 训练集和独立 compare 数据集 `[P:high][D:T2.13]`
  - 训练集 600 条：clean / direct / indirect 各 200 条
  - compare 使用 clean-only / direct-only / indirect-only 三组独立数据，尽量避免与训练样本重叠
  - 输出数据分布说明（类别 / 来源 / 是否带图 / 模板类型）
  - 保留 cross-modal 增强子集作为后续 C6 和 Phase 2.3 的候选数据
- [ ] **T2.15** 写 `runs/_templates/configs/full.yaml` balanced 600 训练配置 `[P:high][D:T2.14]`
  - 复用 `runs/_templates/configs/baseline.yaml` 的 LoRA 超参（r=16, alpha=32）
  - 训练规模：`max_train_records=600`，clean / direct / indirect 均衡
  - 训练轮数：以 run 配置为准，优先确保端到端流程可稳定完成
  - 学习率：2e-4（LoRA 标准），warmup_ratio=0.1
  - 保留训练日志、checkpoint、compare、package 和 offline smoke 配置

### T2.16-T2.19 训练与跨平台验证

- [ ] **T2.16** 跑 balanced 600 训练，保存 `runs/<run_id>/artifacts/checkpoints/lora_full.safetensors` `[P:high][D:T2.15]`
  - 输出 `runs/<run_id>/artifacts/checkpoints/lora_full.safetensors`
  - 输出 `runs/<run_id>/artifacts/checkpoints/train_summary_full.json`（loss 曲线、val F1、训练时长）
- [ ] **T2.17** 评估 balanced 600 模型：`python scripts/eval.py --ckpt runs/<run_id>/artifacts/checkpoints/lora_full.safetensors` `[P:high][D:T2.16]`
  - 输出模型绝对效果报告、混淆矩阵、预测分布
  - 至少覆盖 clean-only / direct-only / indirect-only 三组独立 compare
  - 记录三类目标类 Precision / Recall / F1，为 Phase 2.3 复盘提供基线
- [ ] **T2.18** 对比 baseline 与 balanced 600 模型 `[P:high][D:T2.17]`
  - 输出 baseline vs balanced 600 的分类指标差异
  - 同时保留新模型绝对效果报告和 baseline 对比报告
  - 记录已知结果：clean 较好，direct 有提升但仍不足，indirect 未达预期
- [ ] **T2.19** 复盘 balanced 600 局限并形成 Phase 2.3 输入 `[P:high][D:T2.17]`
  - 分析训练数据来源、标签分布、是否带图、文本长度、模板类型和预测混淆
  - 明确未达预期的类别和原因假设
  - 输出 Phase 2.3 数据与训练改造建议

### T2.20-T2.21 离线包更新

- [ ] **T2.20** 重新打包离线包，使用 `lora_full.safetensors` `[P:high][D:T2.19]`
  - 跑 `python scripts/package_offline.py --ckpt runs/<run_id>/artifacts/checkpoints/lora_full.safetensors --out runs/<run_id>/artifacts/package/mpid_offline/`
  - 校验目录内**无 `.git/` 引用、无网络初始化代码**（同 T2.11）
  - 输出 `package_offline_full.json`（含 lora 权重 sha256）
- [ ] **T2.21** 离线包冒烟测试 `[P:high][D:T2.20]`
  - 解包 `runs/<run_id>/artifacts/package/mpid_offline/` 到临时目录
  - 跑 `python runs/<run_id>/artifacts/package/mpid_offline/infer.py` 单样本推理（5 条预置样本）
  - 跑 `python scripts/measure_offline.py --package runs/<run_id>/artifacts/package/mpid_offline` 量化离线指标
  - 确认推理过程**零网络流量**

**Phase 2.2 验收**：
- balanced 600 训练数据已生成，clean / direct / indirect 各 200 条
- 训练完成：`runs/<run_id>/artifacts/checkpoints/lora_full.safetensors` 存在，`train_summary_full.json` 记录训练时长和 loss
- clean-only / direct-only / indirect-only 三组 compare 均完成
- 输出 balanced 600 模型绝对效果报告和 baseline 对比报告
- 明确记录未达预期的类别与原因，作为 Phase 2.3 增强微调输入
- 离线包 `runs/<run_id>/artifacts/package/mpid_offline/` 解包后能单样本推理，零网络流量

**Phase 2.2 关键产物**：
- `runs/<run_id>/artifacts/checkpoints/lora_full.safetensors` — balanced 600 LoRA + head 权重（增强微调前实验基线）
- `runs/<run_id>/artifacts/checkpoints/train_summary_full.json` — 训练时长 / loss / val F1
- `runs/<run_id>/artifacts/compare/` — clean-only / direct-only / indirect-only compare 报告
- `runs/<run_id>/artifacts/compare/*baseline*` — baseline 对比报告
- `runs/<run_id>/artifacts/reports/` — balanced 600 结果复盘与 Phase 2.3 改进输入
- `runs/<run_id>/artifacts/package/mpid_offline/` — 基于 `lora_full` 的离线包

**下游依赖**：Phase 2.2 的 `lora_full.safetensors` 只作为 balanced 600 实验基线和 Phase 2.3 复盘输入；Phase 3（C4 早退）/ Phase 4（C5 规则前置）/ Phase 5A + 5B（C6 跨模态一致性）在 Phase 2.3 完成后，应优先切换到 Phase 2.3 高 F1 checkpoint 作为新的模型底座。

---

## Phase 2.3 — 高 F1 微调改进（对应 C2 · 质量提升）

> **目标**：在 Phase 2.2 balanced 600 结果基础上，重新设计微调数据、训练方式和验证方式，产出一个真正可作为后续 C4/C5/C6 基线的高质量 checkpoint。硬性验收目标：clean / direct / indirect 三个独立验证集上的目标类 F1 均 **> 0.70**。
>
> **背景复盘**：`runs/phase2_2_balanced_600_20260718_1955` 的 600-step 模型完成了端到端流程，但分类效果不达预期。三组 100 条独立验证集结果为：clean F1=0.985，direct F1=0.473，indirect F1=0.000。主要问题是 direct 大量漏判为 clean，indirect 完全未学出稳定边界。
>
> **阶段边界**：本阶段仍属于 Phase 2 的模型底座改进，不依赖 Phase 3/4/5 的规则或级联兜底来“补分”。Phase 3/4/5 可以后续叠加，但 Phase 2.3 的验收必须先证明 VLM + LoRA + head 本身达到可接受分类能力。
>
> **前置约束**：Phase 2.3 在启动训练前，必须先定义并冻结一套合适的训练 / 验证数据集契约，包括来源清单、标签映射、每类样本量、text-only 与 image+text 比例、OCR / cross-modal / C6B 兼容字段、去重规则、train/val/test 隔离规则、验证切片和验收指标。未形成数据集 manifest 前，不允许直接启动训练。

### T2.22-T2.25 数据集改造

- [x] **T2.22** 建立 Phase 2.3 数据审计报告：复盘 balanced 600 的 train/eval 来源、标签分布、是否带图、文本长度、语言、模板类型和预测混淆；输出 `phase2_3_data_audit.md`，明确 clean / direct / indirect 未达预期的可验证原因假设 `[P:high][D:T2.21][T:0.2-0.5h 实测约 10min]`
- [x] **T2.23** 定义并冻结 Phase 2.3 数据集契约：明确 train / val / test / compare eval 的来源、标签映射、每类样本量、source/template/lang/has_image 分层比例、去重规则和验收指标；必须包含后续 C6B 可复用字段：`ocr_text`、`ocr_confidence`、`text_image_consistency_label`、`cross_modal_attack_type`、`has_instruction_override`、`hard_negative_type`；输出 `phase2_3_dataset_manifest.json` 与人工可读说明 `[P:high][D:T2.22][T:1-2h]`
- [x] **T2.24** 重建训练集抽样策略：每类至少 1000 条；clean / direct / indirect 都要包含多来源、多语言、长短文本；indirect 必须覆盖 text-only indirect、OCR/image indirect、email/tool-use indirect 和非模板化改写；加入带图 clean、含 OCR 但无攻击、图文不相关但无攻击等 hard negative，避免模型把“有图/OCR/不相关”误学为 indirect；抽样必须严格遵守 T2.23 数据集契约。已在 `runs/phase2_3_full_2000_20260727_1211/data/train.jsonl` 生成 3000 条训练样本，clean/direct/indirect 各 1000 条 `[P:high][D:T2.23][T:3-8h 实测约几十分钟]`
- [x] **T2.25** 加强并冻结验证集：冻结两套互斥 compare eval set，quick set 为 clean-only / direct-only / indirect-only 各 50 条，full set 为三类各 200 条；大幅引入 `synthetic_image_injection`、figstep、截图文字、水印、表格/list/聊天截图、OCR 改写、跨语言和 hard negative；验证分布必须覆盖训练分布但不与训练样本重叠。已生成 `compare_quick_*`、`compare_full_*`、`resume_smoke.jsonl` 与 `phase2_3_dataset_hashes.json`，并确认与训练集无 `dedup_key/source_record_id` 交叉 `[P:high][D:T2.23][T:1-3h 实测约几十分钟]`

### T2.26-T2.29 训练设计改造

- [x] **T2.26** 设计并冻结 Phase 2.3 完整训练配置：将原训练配置模板与 run 决策冻结合并处理，输出 `runs/_templates/configs/phase2_3_high_f1.yaml`、run-local 配置副本和 `phase2_3_run_decisions.md`；必须一次性确认 run id、随机种子、训练步数、首轮 sweep 组合数量、是否启用两阶段分类、是否启用扩展可训练模块、训练集 manifest、quick/full compare set 路径、checkpoint/log/断点续训策略、best checkpoint 选择标准；未完成本项前不启动 resume smoke 或正式训练。已冻结首轮单组配置：2000 steps、LoRA r=32/alpha=64/dropout=0.10、lr=1e-4、每 5 step log、每 50 step checkpoint、10% quick compare、50%/100% full compare、best_by_min_class_f1 选优；readiness 预检通过 `[P:high][D:T2.25][T:1.5-3h]`
- [ ] **T2.27** 增加训练期 checkpoint 评估：根据配置中的 `train_steps`，每完成 10% steps 在 quick set 上跑目标类 F1，50% / 100% / best candidate 在 full set 上跑完整 compare，保存 `best_by_min_class_f1` checkpoint，而不是只保存最后一步 `[P:high][D:T2.26][T:1-3h 开发；每次 quick compare 约 1.5-2h CPU；每次 full compare 约 3-4h CPU]`
- [ ] **T2.28** 评估是否扩展可训练模块：除语言 attention 的 `q_proj,k_proj,v_proj,o_proj` 外，调研 vision projector / multimodal connector / 更高 rank LoRA 是否能改善 indirect；必须先用小规模 smoke/sanity run 记录显存/CPU 时间、loss 稳定性和 clean FPR 风险，不默认启用 `[P:medium][D:T2.26][T:1-2h]`
- [ ] **T2.29** 评估两阶段分类方案：先做 clean vs attack，再做 direct vs indirect；如三分类 argmax 继续偏 clean，则将两阶段方案作为 Phase 2.3 的备选训练目标；若启用，compare 报告必须同时输出三分类口径和两阶段口径 `[P:medium][D:T2.27][T:0.5-1h]`

### T2.30-T2.32 验证与报告改造

- [ ] **T2.30** 校验三组独立验证集：确认 quick/full 两套 clean-only / direct-only / indirect-only 均符合 T2.23 数据集契约，按 source/template/lang/has_image/ocr_present/cross_modal_attack_type/hard_negative_type 输出切片分布，并在训练前冻结 sha256 `[P:high][D:T2.25][T:0.5-1h]`
- [ ] **T2.31** 扩展 compare 报告：同时输出新模型绝对效果、baseline 对比、目标类 P/R/F1、混淆矩阵、预测分布、按 source/template/lang/has_image/ocr_present/cross_modal_attack_type/hard_negative_type 切片指标和误判样本清单；报告必须标注使用 quick set 还是 full set `[P:high][D:T2.30][T:1-3h]`
- [ ] **T2.32** 跑 Phase 2.3 端到端训练与最终验收：resume smoke → 正式训练 → 训练期 compare → best checkpoint 选择 → 三组 full compare → package → offline smoke；最终报告必须说明三个目标类 F1 是否均 > 0.70，以及未达标类别的下一步原因定位 `[P:high][D:T2.31,T2.36,T2.37][T:resume smoke ≤1h；训练约 16-30h CPU；compare 合计约 15-24h CPU；package/smoke 约 1h]`

### T2.33-T2.39 增强训练执行控制

- [ ] **T2.33** 明确训练期 compare 节奏：每完成 10% training steps 触发一次 quick compare；如果 `train_steps` 不是 10 的倍数，使用向上取整 step 边界，最终 100% 必跑 `[P:high][D:T2.26,T2.30][T:0.5-1h]`
- [ ] **T2.34** 明确训练期 compare 样本规模：quick compare 固定使用 clean / direct / indirect 各 50 条样本（共 150 条）；关键 checkpoint（50% / 100% / best candidate）追加 clean / direct / indirect 各 200 条 full compare（共 600 条） `[P:high][D:T2.30][T:0.5h]`
- [ ] **T2.35** 保留训练日志和 checkpoint 频率：每 5 step 打一次训练 log，每 50 step 保存一次 checkpoint；compare 报告必须标注当前 step、样本规模、目标类 P/R/F1、loss 摘要和预测分布 `[P:high][D:T2.26,T2.31][T:0.5-1h]`
- [ ] **T2.36** 完成第 4 步：补齐 checkpoint / resume 能力并通过 resume smoke 验收。先改造或封装训练 launcher，使正式长训练前具备以下能力：每 50 step 保存 `checkpoint_step_<step>.safetensors`，同步维护 `latest.safetensors`，恢复时自动按 `latest.safetensors` → 最新 `checkpoint_step_*.safetensors` → 兼容旧 `partial_name` 的顺序发现 checkpoint，恢复 LoRA/head 权重、逻辑 step、`resume_global_step`、日志追加模式和 compare 调度状态；再用 `configs/resume_smoke.yaml` 与 `data/resume_smoke.jsonl` 做短步数验证，第一段跑到 10 step 并生成 checkpoint，第二段从 checkpoint 恢复再跑 5 step，确认 step 不回退、日志不覆盖、checkpoint 可发现、compare 调度不重复或漏跑；输出 `phase2_3_resume_smoke_report.md/json`，目标总耗时 ≤ 1h。未通过本项前不启动正式长训练 `[P:high][D:T2.26][T:≤1h]`
- [ ] **T2.37** 明确 best checkpoint 选择标准：训练期不只保留最后一步，必须输出 `best_by_min_class_f1`，优先选择 clean / direct / indirect 三类目标 F1 的最小值最高的 checkpoint；若 min F1 并列，再按 macro F1、indirect F1、clean FPR 依次打破平局 `[P:high][D:T2.27,T2.31][T:0.5-1h]`
- [ ] **T2.38** 补充阶段时间评估：数据审计 0.2-0.5h；数据契约冻结 1-2h；数据构造 3-8h；验证集冻结 1-3h；训练配置 1-2h；resume smoke ≤1h；训练主流程 16-30h；训练期 compare 15-24h；最终验收 2-4h；整体约 40-75h，不含人工复核返工 `[P:high][D:T2.22-T2.37][T:0.5h]`
- [ ] **T2.39** 输出增强微调执行报告：报告需要覆盖训练耗时、compare 耗时、sweep 组合与选择理由、各 checkpoint 指标走势、最终 best checkpoint 指标、C6B 兼容数据字段覆盖情况、是否达到所有分类目标类 F1 > 0.70，以及未达标类别的下一步定位 `[P:high][D:T2.32,T2.37][T:1-3h]`

### T2.40-T2.42 执行前决策冻结

- [x] **T2.40** 并入 T2.26，不再作为独立执行项：Phase 2.3 run 配置冻结、训练参数选择和执行决策统一在 T2.26 中完成；`phase2_3_run_decisions.md` 仍作为 T2.26 的输出之一 `[P:high][D:T2.26]`
- [x] **T2.41** 冻结数据源纳入规则：确认哪些 downloaded basic 数据直接进入训练，哪些只进入 validation/compare，哪些保留 manual review；明确 license/terms 状态、重复样本处理和 label 冲突处理策略 `[P:high][D:T2.23][T:1-2h]`
- [ ] **T2.42** 冻结成功/失败处理策略：若任一分类 full-set F1 ≤ 0.70，必须先输出失败分析和下一轮数据/超参改动建议；若 clean FPR 明显上升，必须优先回滚到更保守 checkpoint 或增加 hard negative，不允许只追求 indirect F1 `[P:high][D:T2.31,T2.37][T:0.5-1h]`

**Phase 2.3 验收**：
- 训练前已冻结数据集契约和 manifest，且训练 / 验证 / compare 全部引用同一份契约
- 已冻结 `phase2_3_run_decisions.md`，明确 run id、随机种子、sweep 组合、训练步数和是否启用备选训练方案
- clean-only 目标类 F1 > 0.70
- direct-only 目标类 F1 > 0.70
- indirect-only 目标类 F1 > 0.70
- 三组验证集均输出绝对效果报告和 baseline 对比报告
- indirect 验证集中不能出现目标类 recall = 0 的失败模式
- 训练期 compare 样本规模明确：快速集 3×100，完整集 3×200
- 中断后可从最新 checkpoint 继续训练
- 训练过程保持每 5 step log、每 50 step checkpoint
- 正式长训练前 resume smoke 验证通过：中断恢复后 step/log/checkpoint/compare 调度均连续，且耗时 ≤ 1h
- 最终 checkpoint 按 `best_by_min_class_f1` 选择，而不是默认最后一步
- 报告说明 C6B 兼容字段覆盖情况，后续 C6B 可复用 Phase 2.3 数据资产
- 通过 package 和 offline smoke，产出可离线运行的 Phase 2.3 checkpoint 包

**Phase 2.3 暂缓事项**：
- 暂不启动新训练 run
- 暂不修改 Phase 3/4/5 的实现
- 暂不依赖 C5/C6 规则兜底来满足 Phase 2.3 F1 验收

---

## Phase 2.5 — 成果可视化 Demo（独立交付）

> 本节是**独立的展示目标**，不依赖 Phase 3/4/5/6/7。可以最早在 Phase 2（T2.7）产出 `lora_baseline.safetensors` 后启动。
> 目标：**最直观地**展示本项目"Base VLM 易被注入攻击" → "+ LoRA + 3 类 head 后能识别并拦截"的对比。
> 决策记录：方案 C（Gradio 网页）作为唯一交付方案，见 [doc/reference.md](reference.md) § 3.x。
>
> **目录约定**：所有 demo 相关内容（脚本、依赖、样本、文档、截图）统一放在项目根的 **`demo/`** 目录下，**不**进 `scripts/` / `doc/` / `data/` / `report/` 等核心目录。`demo/` 是端到端自洽的子项目，依赖核心 `src/mpid/` 但本身不修改核心代码。

- [ ] **T2.5.1** 在 `src/mpid/adapters/vlm.py` 的 `VLMAdapter` 上加 `generate(text, image, max_new_tokens)` 方法，供 demo 让 base VLM **按用户 prompt 自由生成**（不被 head 拦截的状态）`[P:high][D:T2.1]`
- [ ] **T2.5.2** 写 `demo/samples.json`：从 `runs/_datasets/mpid-v1/test.jsonl` 选 8 条预置样本（clean ×3 / direct ×3 / indirect ×2），含 text、image path、label 元数据 `[P:high][D:T1.5]`
- [ ] **T2.5.3** 写 `demo/requirements.txt`（gradio + plotly；与主 `requirements.txt` 分离；可通过 `pip install -r demo/requirements.txt` 安装）`[P:high][D:-]`
- [ ] **T2.5.4** 写 `demo/gradio_app.py`：Gradio Blocks 应用，UI 布局包括：
  - 上方：8 个预置样本按钮（点选即填入 text + image）
  - 中部：text 输入框 + 图像上传
  - 下方：左右两栏对比
    - 左：**Base SmolVLM** — `generate()` 出的自由生成文本 + "易被攻破"红色提示
    - 右：**LoRA + 3-class head** — label（clean/direct/indirect）+ risk 进度条 + 三类置信度条
  - 右侧"项目说明"标签页：威胁模型、模型卡、参考文献
  `[P:high][D:T2.5.1-T2.5.3]`
- [ ] **T2.5.5** 写 `demo/README.md`：启动命令、UI 截图、8 条预置样本的预期对比结果、依赖安装 `[P:medium][D:T2.5.4]`
- [ ] **T2.5.6** 端到端冒烟：起 gradio server → 跑 8 条预置样本 → 截图保存到 `demo/screenshots/`（base 被攻破 vs LoRA 拦截）`[P:high][D:T2.5.4]`
- [ ] **T2.5.7** 在 `doc/VERIFICATION.md` 加 § Phase 2.5 节：UI 布局说明 + 8 条对比的预期 + 实际结果 + 已知限制（**此处属于核心文档，不是 demo 内容**，保留在 `doc/`）`[P:high][D:T2.5.6]`
- [ ] **T2.5.8** 在 `README.md` 加 § "在线体验" 段：贴 1-2 张 `demo/screenshots/` 截图链接 + 启动命令 `[P:medium][D:T2.5.5]`

**Phase 2.5 验收**：
- `python demo/gradio_app.py` 起 server ≤ 30 s
- 8 条预置样本全部能跑通，base 侧输出有效生成文本、LoRA 侧输出合法 label+risk
- 至少 1 张对比截图（base 被攻破 / LoRA 拦截）保存到 `demo/screenshots/`
- `demo/README.md` 含启动 + 预期 + 限制三节
- **`demo/` 目录结构自洽**（含 `gradio_app.py` / `samples.json` / `requirements.txt` / `README.md` / `screenshots/`），不依赖项目根目录外的相对路径

---

## Phase 3 — C4 早退机制（对应 §3.6.1）

> C4 是**纯加速方向**优化。基础设施可与 Phase 2 共用（共用 backbone + head），**不引入新模型、不需要 LoRA 微调**。
> 对应开题报告 §3.6.1。
>
> **本 Phase 设计哲学**：复用 Phase 2 已经训练好的 3 分类 head，加一层"高置信度 clean 快速放行"判定器。
> clean 样本（占 80%）→ 走 C4 早退 → 跳过 C5/C6 的更复杂判断 → **延迟大幅降低**
> 非 clean 样本 → 走完整 C5 → C6 链 → 完整判断

- [x] **T3.1** 实现 `src/mpid/early_exit.py`：`EarlyExitConfig` / `should_early_exit()` / `classify_with_early_exit()` / `EarlyExitStats` `[P:high][D:T2.2]` ✅
- [x] **T3.2** 实现单测 `tests/test_early_exit.py`：13 个 pure-Python 用例（边界条件 + 统计累计 + 构造） `[P:high][D:T3.1]` ✅
- [ ] **T3.3** （V2 扩展，**本次实现未做**）中间层早退：暴露 Idefics3 中间层 hidden states，连续 2 层置信度 > θ 才退出 `[P:low][D:T3.1]`
- [x] **T3.4** 实现 `src/mpid/early_exit.py` 完整推理接口 `classify_with_early_exit()`（V1 简化版：最后一层 + 阈值） `[P:high][D:T3.1]` ✅
- [x] **T3.5** `scripts/infer.py` 接入 `--early-exit` / `--clean-threshold` flag（CLI 占位） `[P:high][D:T3.1]` ✅
- [x] **T3.6** `scripts/eval.py` 接入 `--early-exit` / `--clean-threshold` / `--simulate-c5-c6-ms` 选项（T3.7 内部） `[P:high][D:T3.1]` ✅
- [x] **T3.7** 跑 `scripts/eval.py --early-exit --max-records 5` 验证端到端：3 个产物（`early_exit_compare.{json,md}` + `early_exit_per_sample.jsonl`）齐全 `[P:high][D:T3.6]` ✅

**Phase 3 验收**：
- 单测 13/13 PASS
- 端到端跑通 + 3 个产物文件存在
- F1 退化 ≤ 0.02（实测：smoke 训练下 delta = +0.0000）
- 无 clean 漏报（实测：n_clean_wrong_exit = 0）
- 节省 ≥ 10%（**真实训练后才有意义**，smoke 训练下 exit_rate = 0% 是符合预期的）

**Phase 3 状态**：✅ V1 简化版已完成。V2 中间层早退作为低优先级扩展保留。

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
- [ ] **T4.6** 写规则库 `runs/<run_id>/configs/rules.yaml`：黑/白名单关键词 + Unicode / 结构 / 敏感指令 规则条目（≥ 20 条） `[P:medium][D:T4.2-T4.3]`
- [ ] **T4.7** 评估规则库在 clean 集 / injection 集上的命中率与误报率，**确保 clean 集 FPR 上升 ≤ 1%** `[P:high][D:T4.6]`
- [ ] **T4.8** 写 `scripts/eval_c5.py`：规则命中率、规则拦截样本的 Recall、VLM 兜底样本的 F1、干净集 FPR、平均延迟 `[P:high][D:T4.7]`

**Phase 4 验收**：规则库能热加载；黑名单 100% 拦截已知威胁；clean 集 FPR 上升 ≤ 1%；平均延迟下降 40-60%；规则命中可解释输出完整。

---

## Phase 5 — C6 跨模态语义一致性检测（对应 §3.6.3）

> C6 是**纯范围方向**优化，扩展对间接注入（图像中嵌入攻击文字）的检测能力。
> 对应开题报告 §3.6.3。
>
> **本 Phase 拆分为两个子阶段**：
>
> | 子阶段 | 思路 | 是否需要 LoRA | 工作量 | 关系 |
> |---|---|---|---|---|
> | **5A 轻量版** | 规则 + CLIP/SigLIP 相似度 + OCR 关键词冲突 | ❌ 不需要 | 🟢 ~200-300 行 | 独立可交付 |
> | **5B 完整版** | 训练 VLM 辅助输出头（yes/no 二分类） + 5A 规则 | ✅ 需要 | 🟡 ~400-500 行 + 训练 | 依赖 5A |
>
> **执行顺序**：
> 1. **先 5A**（轻量版）：零 LoRA，立即可跑通，立即提升 cross-modal 子集 Recall
> 2. **后 5B**（完整版）：在 5A 之上叠加 LoRA 微调，让 VLM 学会"图像中有什么文字" + "图文是否相关"
> 3. 5B 的对比基线 = 5A，所以 5A 评估结果要先有，5B 才能跑 `--compare` 证明"LoRA 又带来多少提升"
>
> 详见 [reference.md § 2.E.6 / § 3.3](reference.md) 对"为什么 LoRA 故意只挂语言侧"的设计权衡。

---

### Phase 5A — C6 轻量版（规则 + CLIP 相似度，无 LoRA）

> **核心思路**：用现成的预训练模型（SigLIP 提图像特征、PaddleOCR 提图像文字、CLIP 算相似度）+ 关键词冲突规则，**完全不需要 LoRA 微调**。
> **何时完成**：Phase 4 验收通过后启动（共用相同的"多层防御"模式）。

- [ ] **T5A.1** 在 `pyproject.toml` / `requirements.txt` 加入新依赖：`sentence-transformers`（含 CLIP）、`paddleocr` 或 `pytesseract`（OCR 二选一） `[P:high][D:T2.4]`
- [ ] **T5A.2** 实现 `src/mpid/crossmodal/siglip_embed.py`：用本地 SigLIP 提取图像 embedding（与 SmolVLM-500M 视觉侧共享权重） `[P:high][D:T2.2]`
- [ ] **T5A.3** 实现 `src/mpid/crossmodal/ocr_extract.py`：从图像中提取文字，返回 `(text, bbox, conf)` 列表 `[P:high][D:T5A.1]`
- [ ] **T5A.4** 实现 `src/mpid/crossmodal/clip_similarity.py`：用 CLIP 计算 `(text, image)` 相似度分数（0~1） `[P:high][D:T5A.1]`
- [ ] **T5A.5** 实现 `src/mpid/crossmodal/conflict_rules.py`：跨模态冲突检测规则——
  - **规则 1**：OCR 文字 ∩ 注入关键词库 ≠ ∅ → 标记 `has_ocr_injection`
  - **规则 2**：CLIP(text, image) < 0.3 AND OCR 非空 → 标记 `semantic_mismatch`
  - **规则 3**：`has_ocr_injection AND semantic_mismatch` → 强制判 `indirect`
  - 输出：`CrossModalSignals(has_ocr_injection, semantic_mismatch, ocr_text, clip_score)`
  `[P:high][D:T5A.2-T5A.4]`
- [ ] **T5A.6** 维护 `runs/<run_id>/configs/crossmodal_rules.yaml`：中英文注入关键词库（≥ 50 条，"忽略指令" / "忽略以上" / "disregard" / "system prompt" / "DAN" 等） `[P:medium][D:T5A.5]`
- [ ] **T5A.7** 接入 `scripts/infer.py` 完整流程：VLM 输出 → 若 `clean` 走 Phase 3 早退路径；若 `direct/indirect` 走 C5 规则路径；若 C5 命中 `indirect` 候选 → 走 C6 crossmodal 路径做二次确认 `[P:high][D:T5A.5]`
- [ ] **T5A.8** 写 `scripts/eval_c6a.py`：cross-modal 子集 Recall、干净集 FPR（**核心约束：不上升**）、Macro F1、CLIP 相似度分布、OCR 召回率 `[P:high][D:T5A.7]`
- [ ] **T5A.9** 跑 `scripts/eval.py --compare --checkpoint runs/<run_id>/artifacts/checkpoints/c6a_xxx.safetensors`（与 Phase 4 对比，量化 5A 带来的提升） `[P:high][D:T5A.8]`

**Phase 5A 验收**：
- cross-modal 子集 Recall ≥ 60%（轻量版基线，5B 应更高）
- clean 集 FPR 不上升
- Macro F1 提升 ≥ 1-2%
- 完全不需要 LoRA 微调，0 训练成本
- 与 Phase 4 相比的对比报告 `c6a_vs_phase4_comparison.md` 完整

---

### Phase 5B — C6 完整版（训练辅助输出头，需要 LoRA）

> **核心思路**：在 5A 之上，让 VLM 学会"图像中有什么文字" + "图文是否相关"——给 VLM 加一个辅助输出头（yes/no 二分类），并在 cross-modal 子集上做 LoRA 微调。
> **何时完成**：Phase 5A 验收通过后启动。
> **前置条件**：5A 评估结果先有（5B 的对比基线）。

- [ ] **T5B.1** 实现辅助 prompt 模板 `src/mpid/data/prompt_c6.py`："图像中的内容是否与用户的问题/请求相关？回答 yes 或 no" `[P:high][D:T2.4]`
- [ ] **T5B.2** 实现 `src/mpid/heads/dual_output.py`：主输出（3 分类）+ 辅助输出（yes/no 二分类）双输出融合 `[P:high][D:T5B.1]`
- [ ] **T5B.3** 扩展 `src/mpid/train/trainer.py`：增加辅助 prompt 的 loss 权重与训练流程 `aux_loss_weight=0.3` `[P:high][D:T5B.2]`
- [ ] **T5B.4** 完善 `src/mpid/crossmodal/conflict_rules.py`：将 T5A.5 的规则升级——"辅助输出 = no" 也作为强信号叠加到 indirect 判定 `[P:high][D:T5B.2]`
- [ ] **T5B.5** 维护 `runs/<run_id>/configs/sensitive_words.yaml`（中英文敏感词列表）："退款"、"取消"、"同意"、"system" 等 `[P:medium][D:T5B.4]`
- [ ] **T5B.6** 训练数据准备：在 cross-modal 子集上让辅助 prompt 学会判断"图文是否相关"，为每条 cross-modal 样本打辅助标签 `(text_relevant_to_image: yes/no)` `[P:high][D:T5B.3,T1.8]`
- [ ] **T5B.7** 写 `runs/<run_id>/configs/c6_crossmodal.yaml` + 在 cross-modal 子集上重训 LoRA（**比 Phase 2 多 1 个 epoch + 加 aux_loss**） + 保存 `runs/<run_id>/artifacts/checkpoints/c6_crossmodal.safetensors` `[P:high][D:T5B.6]`
- [ ] **T5B.8** 写 `scripts/eval_c6b.py`：cross-modal 子集 Recall（目标 ≥ 80%）、干净集 FPR（**核心约束：不上升**）、Macro F1、辅助 prompt 一致性 `[P:high][D:T5B.7]`
- [ ] **T5B.9** 跑 `scripts/eval.py --compare --checkpoint runs/<run_id>/artifacts/checkpoints/c6_crossmodal.safetensors`（与 5A 对比，量化 5B 相对 5A 的提升） `[P:high][D:T5B.8]`

**Phase 5B 验收**：
- cross-modal 子集 Recall ≥ 80%（比 5A 提升 ≥ 20%）
- clean 集 FPR 不上升
- Macro F1 提升 2-3%
- 辅助 prompt 一致性（与人工标注的 yes/no 一致率）≥ 75%
- 5B vs 5A 对比报告 `c6b_vs_c6a_comparison.md` 完整（含 LoRA 训练带来的提升量化）

---

## Phase 6 — 攻防基线评测体系（对应 C3）

- [ ] **T6.1** 实现 `Keyword` 基线（regex 关键词匹配） `[P:medium][D:-]`
- [ ] **T6.2** 跑 `meta-llama/PromptGuard-86M` zero-shot 作为对比（需考虑网络，离线环境下提前下载本地） `[P:medium][D:-]`
- [ ] **T6.3** 实现 `src/mpid/eval/aggregate.py`：合并 baseline / C4 / C5 / C6A / C6B 多模型结果，按子集切片 `[P:high][D:T2.9,T3.7,T4.8,T5A.8,T5B.8]`
- [ ] **T6.4** 输出 `report/figures/` 下混淆矩阵、F1 柱图、按语种/类型切片热力图 `[P:medium][D:T6.3]`
- [ ] **T6.5** 撰写 `report/technical_report.md` 主体
  - 章节：摘要 / 背景 / 威胁模型 / 方法 / 数据 / 实验 / 消融 / 局限 / 伦理
- [ ] **T6.6** 在 CPU 上重跑验证报告数字可复现 `[P:high][D:T6.5]`

**Phase 6 验收**：`report/technical_report.md` 完稿，含 4 张以上图表；与 PromptGuard-86M、关键词基线、规则前置基线对比报告完整。

---

## Phase 7 — 项目整理

- [ ] **T7.1** 完善 `README.md`：项目说明、快速开始、复现命令 `[P:high][D:T6.5]`
- [ ] **T7.2** 写 `runs/_datasets/mpid-v1/README.md`：数据来源、统计、license 说明 `[P:medium][D:T1.5]`
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
【Phase 2.5 — 成果可视化 Demo（独立）】            │
T2.1 ── T2.5.1                                │
T1.5 ── T2.5.2 ─┐                              │
                ├─ T2.5.4 ── T2.5.5 ── T2.5.6 ── T2.5.7
                │             └─ T2.5.8     │
T2.7 ───────────┘                              │
                                                       │
【Phase 3 — C4 早退（轻量级版本）】                    │
T2.2 ── T3.1 ── T3.2 (单测)                  │
              └─ T3.4 (完整接口)               │
                   ├─ T3.5 (infer CLI)        │
                   └─ T3.6 (eval CLI)         │
                        └─ T3.7 (端到端) ✅     │
                                                       │
【Phase 4 — C5 规则前置】                              │
T4.1 ── T4.2 ── T4.3 ── T4.4 ── T4.6 ── T4.7 ── T4.8 │
                          └─ T4.5                     │
                                                       │
【Phase 5A — C6 轻量版（无 LoRA）】                        │
T2.2 ── T5A.2 ─┐                                              │
T5A.1 ── T5A.3 ┼─ T5A.5 ── T5A.6 ── T5A.7 ── T5A.8 ── T5A.9  │
T5A.1 ── T5A.4 ┘                                              │
                                                              │
【Phase 5B — C6 完整版（需要 LoRA，依赖 5A）】                 │
T2.4 ── T5B.1 ── T5B.2 ── T5B.3 ── T5B.6 ── T5B.7 ── T5B.8   │
                         └─ T5B.4 ── T5B.5                    │
T1.8 ───────────┘ ──── T5B.6                                  │
5A 评估结果 ─── T5B.9                                         │
                                                              │
【Phase 6 — 攻防基线 C3】                                │
T6.1 / T6.2 (独立) ── T6.3 ── T6.4 ── T6.5 ── T6.6    │
                  T2.9 + T3.7 + T4.8 + T5A.8 + T5B.8 ──┘      │
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
> - **C6 拆分为 5A + 5B**：
>   - 5A 依赖 T2.2（视觉侧 SigLIP）+ T2.4（prompt 模板）
>   - 5B 依赖 5A 评估结果 + T1.8（cross-modal 子集）+ 5A 评估结果作为 baseline
> 可任选一条/两条/全部；建议**先做 C5（多层防御，最具实用价值）→ C4（加速）→ 5A（轻量版跨模态）→ 5B（完整版跨模态，需 LoRA）**。

---

## 风险缓冲

- 任一 Phase 超期 1 天：先收尾当前任务，跳过次要子项（如美化、扩展数据）
- MPS 不可用：直接在 CPU 上跑（总时间 ×3-5）
- 关键路径阻塞（T2.5 trainer）：先做最小可跑版本，特性延后
- **C4 / C5 / C6 / 5A / 5B 进度选择**：
  - 时间紧（< 1 天）→ 只做 C5 规则库配置（不写新代码）
  - 时间适中（1-3 天）→ C5 + C4（多层防御 + 加速）
  - 时间宽裕（3-5 天）→ C4 + C5 + 5A（轻量版跨模态，无需 LoRA）
  - 时间充裕（5+ 天）→ 全部完成含 5B（完整版跨模态 + LoRA 微调）
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
