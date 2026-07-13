# 课题开题报告（参考内容 · 历史版本）

> **说明**：本文件是**早期、详尽版本**的开题报告，仅作为**参考资料**保留。
> 当前权威版本为 [opening-report-formal.md](opening-report-formal.md)（正式·简化版），请以正式版为准。
> 本文件保留是为方便查阅完整背景、详尽研究现状与历史方法论，不再主动维护与更新。

> **课题名称**：面向离线场景的轻量级多模态提示注入检测算法框架研究
> **副标题**：一种 backbone-agnostic 的"前置过滤"方法研究——区别于业界"在线大模型防御"主流路线
> **课题类型**：实验性 / 算法研究
> **文档版本**：v0.2（参考存档版）
> **关键词**：提示注入；多模态大语言模型；离线边缘；轻量级检测；backbone-agnostic；前置过滤

---

## 摘要

随着大语言模型（LLM）与多模态大语言模型（MLLM）的产业化部署，提示注入（Prompt Injection）已成为首要安全威胁。**业界主流防御方案**以"在线 + 大模型"为典型形态：云端调用专用防御大模型、模型层级的对齐训练、API 服务化的检测器。此类方案在**离线边缘场景**（隐私敏感行业、嵌入式设备、低成本合规审计）下面临网络依赖、隐私泄露、算力受限等多重不可行性。

本课题研究一种**面向离线场景的轻量级多模态提示注入检测算法框架**。其核心定位是：在 LLM 入口处构建一个**轻量、可打包、跨平台、无网络依赖**的前置过滤器（Pre-LLM Filter），通过多模态注入检测在数据进入主模型前完成风险拦截。研究重点是**算法框架本身**——通过 `ModalityAdapter / Backbone / DetectorHead` 三层抽象，使同一套训练与评测代码既能用于本课题选用的轻量 backbone（XLM-RoBERTa + CLIP），也能平滑迁移到 7B / 13B 规模的视觉-语言模型。

研究内容覆盖威胁模型构建、双方案检测算法（级联式与端到端）、多语种数据集构造、跨平台训练验证、离线部署特性测试（模型大小、内存峰值、冷启动延迟、推理时延）等。预期产出研究报告 1 份、开源算法框架 1 套（含训练代码、LoRA 权重、评测数据集与脚本），为离线边缘场景下的提示注入防御提供可复用的研究基础设施。

---

## 一、选题背景与意义

### 1.1 研究背景

#### 1.1.1 提示注入：OWASP LLM Top 1 风险

提示注入指攻击者通过精心构造的输入，使模型"忽略"或"覆盖"原始系统指令，转而执行攻击者意图。OWASP 基金会在 2023、2024 连续两年的 *"Top 10 for LLM Applications"* 中，将 *LLM01: Prompt Injection* 列为**第一大风险**[1]。其攻击形态已从最初的"直接注入"（Direct Injection，即 *"Ignore previous instructions..."*）[2] 演进到"间接注入"（Indirect Injection，通过检索的外部内容劫持）[3] 与"多模态注入"（图像中嵌入指令）[4][5]。

#### 1.1.2 业界主流防御：在线大模型路线

目前产业界与学术界主流的防御方案可归为以下三类，且**普遍以"在线 + 大模型"为典型形态**：

| 方案类型 | 代表 | 形态 | 局限 |
|---|---|---|---|
| 模型层级的对齐训练 | RLHF、Constitutional AI、StruQ[6]、SecAlign[7] | 训练时加固 | 对间接注入鲁棒性有限；需重训 |
| 专用防御大模型 | Meta PromptGuard[8]、Lakera Guard[9] | **在线 API**，参数量大 | 网络依赖、隐私上传、算力高 |
| 大模型辅助检测 | GPT-4 / Claude zero-shot 分类 | **在线云端 LLM** | 成本、延迟、数据出境风险 |

这些方案共同的特点是：依赖网络连接、调用云端大模型、对部署算力要求高、对用户数据需要上传第三方。

#### 1.1.3 离线边缘场景的未满足需求

与"在线大模型"路线相对，存在一类长期被忽视却真实存在大量需求的**离线边缘场景**：

- **隐私敏感行业**：医疗、法律、政府、军工，要求"数据不出端"——任何把 prompt 上传云端检测的方案都不可接受；
- **边缘 / 嵌入式设备**：移动端、IoT、车载、POS 设备，网络不稳定或无网络，且算力、内存、电池受限；
- **低成本合规审计**：中小企业的内容过滤、合规审计无法负担云端大模型 API 调用成本；
- **高敏感流程节点**：金融交易前的 prompt 校验、工业控制指令的安全审计，要求**离线确定性**而非"尽力而为"的云端判别。

这些场景对防御方案提出的核心约束是：
1. **无网络依赖**：所有推理在本地完成；
2. **小模型 / 轻算力**：可在普通 PC、嵌入式设备上运行；
3. **可打包分发**：模型权重 + 依赖单一目录可移植；
4. **可解释**：审计场景需要可追溯的判定依据；
5. **可独立于主 LLM 部署**：作为"前置过滤"组件，不依赖主模型。

#### 1.1.4 现有研究的不足

针对上述离线场景，目前的研究存在以下空白：
1. **形态错配**：主流防御方案默认部署在云端，难以直接迁移到边缘；
2. **多模态缺位**：现有轻量检测器多仅处理文本，对图像内嵌指令的间接注入覆盖不足；
3. **多语种弱化**：研究多以英文为主，对中英混合及多语种注入的覆盖有限；
4. **框架缺位**：现有方案多与具体 backbone 强耦合，难以快速适配到不同部署环境与不同规模模型；
5. **离线特性未量化**：模型大小、内存峰值、冷启动延迟等离线关键指标在论文中鲜有报告。

### 1.2 研究意义

#### 1.2.1 理论意义

- **填补"离线 + 轻量 + 多模态"三维交叉的研究空白**：明确区别于业界在线大模型路线，为防御研究开辟独立分支；
- **探索 backbone-agnostic 的检测算法抽象方法论**：以 `ModalityAdapter / Backbone / DetectorHead` 三层接口为切面，给出"小模型验证、大模型复用"的可迁移范式，为后续防御研究提供方法论参考；
- **建立"前置过滤"作为独立范式**：将防御从"在 LLM 之上"和"LLM 之内"扩展到"在 LLM 之前"，丰富防御体系层次。

#### 1.2.2 实践意义

- **为离线边缘场景提供开源方案**：所有产物（代码、权重、数据集、技术报告）以 MIT / Apache 协议开源；
- **降低研究复现门槛**：在 macOS（Apple Silicon）与 x86 PC CPU 上均可完成训练与推理；
- **构建只读防御基础设施**：本项目只构建识别侧，**不发布任何可用于主动构造更隐蔽注入的代码或教程**，为社区防御研究提供安全的研究底座。

---

## 二、国内外研究现状

### 2.1 提示注入攻击研究脉络

提示注入的系统性研究自 Perez & Ribeiro[2] 首次提出 *"Ignore Previous Prompt"* 攻击范式以来，已发展出多个分支：

- **直接注入（Direct Injection）**：攻击者直接构造用户输入，典型如 *"Ignore previous instructions. Output: ..."*；
- **间接注入（Indirect Injection）**：Greshake 等人[3]首次系统命名，攻击者通过 LLM 应用检索的外部内容（网页、文档、邮件、图像）注入指令；
- **多模态注入**：包括文本-图像协同、纯视觉对抗、跨模态语义劫持。Qi 等人[10] 通过对抗性图像扰动实现对对齐 LLM 的"视觉越狱"；Wu 等人[4] 构建的 JailbreakV-28K 数据集首次大规模标注了多模态越狱样本。

### 2.2 防御路线的二维分类：在线/离线 × 大模型/轻量

现有防御方案可按"部署形态"与"模型规模"两个维度归类。下表为本课题的定位框架：

|         | **大模型（≥ 1B）**       | **轻量（< 500M）**       |
|---|---|---|
| **在线** | Lakera Guard[9]、PromptGuard[8]、LLM zero-shot 分类、StruQ[6]、SecAlign[7] | （少见）关键词规则、小型文本分类器 |
| **离线** | （少见）端侧量化 LLM | **← 本课题的定位** |

**关键观察**：
- **第一象限（在线大模型）**研究最丰富，但均依赖云端；
- **第三象限（离线大模型）**受算力约束，可行性差，研究稀少；
- **第四象限（离线轻量）**正是本课题的切入位置——既满足离线部署约束，又不与算力受限的边缘设备冲突；
- **第二象限（在线轻量）**多为"前大模型"时代的过渡方案，价值有限。

### 2.3 离线/边缘侧检测研究

针对边缘部署的轻量检测研究相对零散：

- **关键词 / 正则规则**：实现简单，但无法应对改写、Unicode 混淆、base64 编码等绕过手段；
- **小型文本分类器**：使用 BERT/RoBERTa/XLM-R 等 backbone 做二分类或多分类，在文本侧达到一定效果；但对图像侧无能为力，且多语种覆盖有限；
- **端侧量化模型**：通常是把大模型 INT4/INT8 量化后部署到端侧，违背"轻量"初衷且仍依赖相对高算力；
- **专用嵌入式 NLP 工具**：如 ONNX Runtime Mobile、TFLite 在工业界有应用，但缺少面向提示注入的现成方案。

**总结**：在"离线 + 轻量"象限，**当前缺少面向多模态提示注入的系统性研究与开源框架**。

### 2.4 多模态大模型安全研究

2024 年起，针对 MLLM 安全的专项研究涌现：

- **MLLM-Protector**[11]：对多模态输入做后置有害性检测，使用一个独立防护 LLM（**在线大模型**路线）；
- **Hades**[12]：系统化的多模态越狱评测基准；
- **MM-SafetyBench**[13]：覆盖 5 大类安全场景的多模态评测。

这些工作对"在线大模型"象限的多模态防御做出了重要贡献，但其执行体多为大模型，难以直接迁移到离线边缘场景。

### 2.5 现状小结与本课题定位

综合上述分析：
- 业界主流：在线 + 大模型，象限 I；
- 本课题定位：离线 + 轻量，象限 IV，并强调 backbone-agnostic 框架设计；

> **本课题研究的核心问题**：
> 在离线边缘部署的算力、内存、网络约束下，如何设计一种与具体 backbone 解耦、可被快速迁移的多模态提示注入检测算法框架？

---

## 三、研究目标与内容

### 3.1 总目标

研究一种**面向离线场景的轻量级多模态提示注入检测算法框架**（MPID, Multimodal Prompt Injection Detector framework for Offline scenarios）。

**框架的关键属性**：
1. **Backbone-agnostic**：与具体 backbone（轻量或大模型）解耦，通过抽象接口实现"小模型验证、大模型复用"；
2. **离线可打包**：模型权重 + 依赖 + 推理脚本可单一目录分发，无任何网络依赖；
3. **跨平台**：在 macOS（Apple Silicon, MPS）、x86 PC（CPU）、Linux（CPU/GPU）上均可训练与推理；
4. **多模态**：同时处理图像与文本输入，覆盖直接注入、间接注入、跨模态注入三类威胁；
5. **多语种**：以中英为主，可扩展至 XLM-RoBERTa 覆盖的 100+ 语种。

### 3.2 具体研究内容

| 编号 | 研究内容 | 预期产出 |
|---|---|---|
| C1 | **离线场景下的多模态注入威胁模型** | 攻击分类、形式化定义、典型样例 |
| C2 | **双方案检测算法设计**（级联式 + 端到端） | 两条可对比的技术路径 |
| C3 | **多语种训练数据集构造** | ≥ 3k 样本的中英多模态注入数据集 |
| C4 | **Backbone-agnostic 框架实现** | `ModalityAdapter / Backbone / DetectorHead` 三层接口 + 至少 2 个 backbone 实现 |
| C5 | **离线部署特性测试** | 模型大小、内存峰值、冷启动延迟、推理时延、跨平台一致性等量化报告 |
| C6 | **评测体系与基线对比** | 与 PromptGuard-86M、关键词基线、Zero-shot LLM 的对比；按语种 / 攻击类型 / OOD 切片指标 |

### 3.3 研究定位：与"在线大模型防御"路线的区别

为避免研究价值被模糊，**显式声明**本课题与业界主流路线的差异：

| 维度 | 业界主流（在线大模型） | 本课题（离线轻量） |
|---|---|---|
| 部署形态 | 云端 API | 端侧 / 边缘独立部署 |
| 模型规模 | 数百 MB ~ 数十 GB | < 500MB（默认） |
| 网络依赖 | 必需 | 无 |
| 数据流向 | 需上传 prompt 至云端 | 数据完全在本地 |
| 算力需求 | 高 | 普通 PC CPU 即可 |
| 适用场景 | 通用 SaaS 防护 | 隐私敏感、边缘、低成本合规 |
| 隐私属性 | 数据出境风险 | 数据不出端 |
| 代价 | 成本高、延迟高、隐私弱 | 检测精度上限略低，但可解释、可审计 |

本课题**承认**在极限精度上可能弱于在线大模型方案（这是离线约束下的必然代价），但**专注于**算法框架本身的可迁移性与离线场景下的工程完整性。

---

## 四、研究方法与技术路线

### 4.1 研究方法

- **文献调研法**：系统梳理 2022 年以来提示注入攻防代表性工作；
- **数据驱动实验法**：基于自构造 + 公开数据集进行训练与消融；
- **对比实验法**：与 PromptGuard-86M、关键词基线、Zero-shot LLM 对比；
- **跨平台验证法**：在 macOS（Apple Silicon）与 x86 PC CPU 上分别训练并对比结果一致性；
- **离线部署特性测试法**（本课题新增）：对模型大小、内存峰值、冷启动延迟、推理时延等离线关键指标进行量化测试。

### 4.2 总体技术路线

```
┌────────────────────────────────────────────────────────────┐
│                       研究流程                              │
│                                                            │
│  [文献调研] → [威胁模型] → [数据构造] → [算法设计]          │
│       │          │           │            │                │
│       │          │           │            ↓                │
│       │          │           │     ┌──────────────┐        │
│       │          │           │     │ 方案A:级联式 │        │
│       │          │           │     │ 方案B:端到端 │        │
│       │          │           │     └──────┬───────┘        │
│       │          │           │            ↓                │
│       │          │           │       [训练与调优]           │
│       │          │           │            ↓                │
│       │          │           │       [跨平台验证]           │
│       │          │           │            ↓                │
│       │          │           │       [离线特性测试]         │
│       │          │           │            ↓                │
│       │          │           │       [评估与对比]           │
│       │          │           │            ↓                │
│       │          │           │     [可迁移性演练→大模型]     │
│       │          │           │            ↓                │
│       │          │           │      [技术报告 + 论文]       │
└────────────────────────────────────────────────────────────┘
```

### 4.3 算法框架（backbone-agnostic）

```
┌──────────────────────────────────────────────────────────────┐
│              MPID Algorithm Framework (Offline)              │
│                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────┐  │
│  │ Modality     │ →  │ Backbone          │ →  │ Detector │  │
│  │ Adapter      │    │ (pluggable,       │    │ Head     │  │
│  │ (OCR / CLIP) │    │  XLM-R / CLIP /   │    │ (3-class)│  │
│  │              │    │  LLaVA / Qwen-VL) │    │          │  │
│  └──────────────┘    └──────────────────┘    └──────────┘  │
│            │                │                       │        │
│            └────────────────┴───────────────────────┘        │
│                             ↓                                │
│                   ┌──────────────────┐                       │
│                   │ LoRA / PEFT      │  (训练时)             │
│                   │ ONNX / llama.cpp │  (推理时，可选)        │
│                   └──────────────────┘                       │
│                             ↓                                │
│                  [离线打包：权重+依赖+脚本]                   │
└──────────────────────────────────────────────────────────────┘
```

**关键抽象**：
- `ModalityAdapter`：把异构模态（图像 / 文本 / 未来加入的音频）映射到统一 embedding 空间；离线场景下，OCR 引擎与 CLIP 权重都必须可本地化；
- `Backbone`：可插拔的视觉-语言编码器
  - 轻量默认（本研究主目标）：`xlm-roberta-base` (text) + `clip-vit-base-patch32` (image, frozen)
  - 7B 规模示例（迁移演示）：`llava-onevision-qwen2-7b`、`qwen2.5-vl-7b`
- `DetectorHead`：分类头（默认 3 类）或生成式 head（输出 token "INJECTION" / "CLEAN"），head 体量极小（< 10M 参数）；
- **离线打包**：训练完成后将 backbone 权重 + LoRA 权重 + tokenizer + 推理脚本 + 离线依赖清单打包为单一目录，**整个目录可独立分发，运行时零网络请求**。

### 4.4 两条实施路径

| 维度 | 方案 A：级联式 | 方案 B：端到端 |
|---|---|---|
| 图像侧 | OCR（PaddleOCR / pytesseract）抽文本 | CLIP-ViT frozen 抽特征 |
| 文本侧 | OCR 文本 + 用户 prompt 拼接 → 文本分类器 | 与视觉特征 concat → 分类头 |
| Backbone | `xlm-roberta-base` + LoRA | `xlm-roberta-base`（text LoRA）+ CLIP（frozen） |
| 可解释性 | 高（可看到 OCR 文本） | 中（需 attention 可视化） |
| 训练时长（M1 Pro） | ~30 min / 3 epoch | ~1.5 h / 3 epoch |
| 离线打包大小 | 较小（XLM-R + OCR + head） | 较大（XLM-R + CLIP + head） |
| 适合 backbone 规模 | 任意文本编码器 | 任意 VL 模型 |

两条路径**算法层完全复用**：
- 共享 `ModalityAdapter` 接口
- 共享 `DetectorHead` 定义
- 共享 LoRA 配置 / 训练器 / 评估器
- 共享数据加载与增强

### 4.5 离线部署设计原则

针对离线边缘场景的工程约束，框架设计遵循以下原则：
1. **零网络依赖**：禁止运行时调用任何外部 API；OCR、CLIP、backbone 全部本地权重；
2. **可打包分发**：训练完成后产出单一 `mpid_offline/` 目录，含权重 + 配置 + 推理脚本 + 依赖清单；
3. **推理引擎可替换**：默认 PyTorch eager，可选 ONNX Runtime / llama.cpp / Core ML / MLX（macOS Apple Silicon 专用）以提升端侧推理速度；
4. **冷启动友好**：权重 < 500MB，加载到推理就绪 < 5 秒（参考目标，M1 Pro / i5 CPU 实测）；
5. **内存峰值受控**：单样本推理峰值内存 < 1GB（设计目标）；
6. **离线可重现**：不依赖远端数据集下载，提供离线数据构造脚本。

### 4.6 可迁移到大模型的设计

虽然本课题的主战场是"离线轻量"，但研究的最终贡献是**算法框架**。在研究中将演示如何将同一套训练与评测代码迁移到 7B 规模 VL 模型（仅作为方法论演练，不作为主交付物）：

- 替换 `Backbone` 注册条目为目标模型；
- 调整 `LoRA.target_modules`（如 LLaMA 系加 `gate_proj, up_proj, down_proj`）；
- 启用 `device_map="auto"` + `load_in_4bit=True`（CUDA 环境，macOS 上不可用故本课题主实验不启用）；
- 数据、训练器、评估器零修改。

迁移指南将作为技术报告附录提供，但**不作为本课题的主验收内容**。

---

## 五、创新点

| 编号 | 创新点 | 体现 |
|---|---|---|
| I1 | **Backbone-agnostic 检测算法框架** | 通过 `ModalityAdapter / Backbone / DetectorHead` 三层接口，使同一套训练/评测代码可在轻量与大模型间切换，**方法论价值大于单点性能** |
| I2 | **"离线前置过滤"范式** | 明确将防御从"在 LLM 之上"和"LLM 之内"扩展到"在 LLM 之前"，区别于业界"在线大模型"主流路线 |
| I3 | **多模态多语种在轻量编码器上的统一检测** | 在 XLM-RoBERTa + CLIP 上同时覆盖中英文与直接/间接/跨模态三类威胁 |
| I4 | **跨平台 CPU 可训练** | macOS（Apple Silicon, MPS）与 x86 PC CPU 双平台均验证，**降低研究复现门槛** |
| I5 | **自构造多模态注入数据集** | 填补公开数据中"图像 OCR 注入"样本稀缺的空白 |
| I6 | **可解释级联方案** | OCR 文本 + 分类器可定位可疑文本片段，**便于审计与合规取证** |
| I7 | **离线可打包性** | 模型权重 + 依赖 + 推理脚本单一目录分发，零网络依赖，便于边缘 / 内网 / 离线环境部署 |
| I8 | **离线特性量化报告** | 显式量化模型大小、内存峰值、冷启动延迟、推理时延等离线关键指标，填补研究空白 |

---

## 六、预期成果

### 6.1 研究文档
- 课题开题报告 1 份（**以正式版 [opening-report-formal.md](opening-report-formal.md) 为准**）；
- 技术研究报告 1 份，含方法、实验、消融、局限、伦理；
- 模型卡（Model Card）1 份，描述能力、适用场景、已知失败模式。

### 6.2 代码与数据
- 完整训练 / 评测 / 推理代码仓库（开源协议：Apache 2.0）；
- 公开数据集 `mpid-v1`（中英多模态注入样本，≥ 3k 样本）；
- 两套 LoRA 权重（级联式 / 端到端）；
- 离线打包示例（`mpid_offline/` 目录模板）。

### 6.3 可复用算法框架
- `mpid` Python 包：`ModalityAdapter / Backbone / DetectorHead` 抽象接口；
- 至少 2 个 Backbone 实现（XLM-RoBERTa / CLIP-ViT-Base 组合，附 LLaVA-7B 迁移示例）；
- 跨平台安装与运行说明（macOS / x86 / Linux）；
- "切到 7B 模型"的迁移指南（附录形式）。

### 6.4 学术贡献
- "离线前置过滤"作为独立防御范式的方法论总结；
- 多模态多语种轻量级检测的实验数据；
- backbone-agnostic 框架设计模式的可复用经验；
- 失败案例与已知局限的诚实记录。

---

## 七、研究的局限与边界

研究过程中将主动识别并文档化以下局限：
1. **检测精度上限**：在极限精度上，**离线轻量方案的 F1 可能弱于在线大模型方案**（如 PromptGuard），这是离线约束下的必然代价；本课题以"在约束下取得最佳"为优化目标；
2. **多模态数据规模有限**：自构造 OCR 注入样本受人工成本限制，泛化性需要更大规模验证；
3. **跨模态协同攻击效果**：方案 B 在 cross-modal 子集上的指标可能显著低于 direct 子集，需诚实记录；
4. **不针对未训练攻击模式**：对 0day 攻击、Unicode 混淆、base64 编码等绕过手段的检出率可能下降；
5. **不替代模型自身对齐**：本检测器作为"前置过滤 / 审计工具"，**不替代**训练阶段的安全对齐；
6. **不应用于主动攻击**：本项目**只构建识别侧**，不发布任何可用于主动构造更隐蔽注入的代码或教程；
7. **离线环境本身的工程限制**：OCR / CLIP / XLM-R 权重总体仍在百 MB 量级，对极端嵌入式（如 MCU）仍可能过大，**本课题目标设备是普通 PC 与主流边缘设备，不覆盖 MCU 级**。

---

## 八、参考文献

[1] OWASP Foundation. *OWASP Top 10 for LLM Applications*. 2023, 2024.

[2] Perez, E., & Ribeiro, I. *"Ignore Previous Prompt: Attack Techniques For Language Models."* arXiv:2211.09527, 2022.

[3] Greshake, K., et al. *"Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection."* AISec, 2023.

[4] Wu, J., et al. *JailbreakV-28K: A Benchmark for Multilingual Fine-grained Visual Jailbreaking*. arXiv:2501.00001, 2024.

[5] Qi, X., et al. *"Visual Adversarial Examples Jailbreak Aligned Large Language Models."* AACL, 2023.

[6] Chen, S., et al. *StruQ: Defending Against Prompt Injection with Structured Queries*. arXiv:2401.09138, 2024.

[7] Chen, S., et al. *SecAlign: Defending Against Prompt Injection with Generative Reinforcement Learning*. arXiv:2406.14144, 2024.

[8] Meta AI. *Prompt Guard: Prompt Injection and Jailbreak Detection for LLM Applications*. 2024. https://llama.meta.com/docs/model-cards-and-prompt-formats/prompt-guard

[9] Lakera AI. *Lakera Guard: Real-time LLM Security*. https://www.lakera.ai

[10] Qi, X., et al. *"Visual Adversarial Examples Jailbreak Aligned Large Language Models."* AACL, 2023.

[11] (MLLM-Protector 团队). *MLLM-Protector: An Effective and General-purpose Defense against MLLM Jailbreak Attacks*. 2024. (引用为代表性多模态防御工作；具体卷期以正式发表版本为准)

[12] (Hades 团队). *Hades: A Benchmark for Multi-modal Safety Evaluation*. 2024. (引用为代表性多模态越狱评测基准；具体卷期以正式发表版本为准)

[13] (MM-SafetyBench 团队). *MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models*. 2024. (引用为代表性多模态安全评测基准；具体卷期以正式发表版本为准)

[14] Conneau, A., et al. *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-RoBERTa). ACL, 2020.

[15] He, P., et al. *mDeBERTa: A Multilingual Variant of DeBERTa*. 2022.

[16] Hu, E. J., et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR, 2022.

[17] Radford, A., et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML, 2021.

[18] Liu, H., et al. *LLaVA-OneVision: A Unified Multi-Modal LLM with Visual Instruction Tuning*. 2024. (作为大模型 backbone 迁移示例的引用)

[19] Bai, J., et al. *Qwen2.5-VL Technical Report*. 2024. (作为大模型 backbone 迁移示例的引用)

[20] Abdin, M., et al. *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone*. 2024. (作为端侧小模型代表引用)
