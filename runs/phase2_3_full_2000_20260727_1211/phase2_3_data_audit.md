# Phase 2.3 数据审计报告

## 范围

- 来源 run：`runs/phase2_2_balanced_600_20260718_1955`
- 目的：复盘 Phase 2.2 balanced 600 模型为什么没有达到 clean / direct / indirect 三分类目标，并为 Phase 2.3 增强微调提供明确输入。
- 本审计对 Phase 2.2 run 只读；没有启动训练，也没有重建数据。

## 运行概览

- 工作流状态：已完成。
- 最终 checkpoint：`artifacts/checkpoints/lora_balanced_600.safetensors`。
- 训练数据：600 条，按 label 均衡。
- 训练标签：clean 200，direct 200，indirect 200。
- 训练设备和精度：CPU，float32。
- LoRA：r=16，alpha=32，dropout=0.05，目标模块为 `q_proj,k_proj,v_proj,o_proj`。
- 训练设置：1 epoch，batch size 1，lr=2e-4，class weighted，每 5 step 打 log，每 50 step 保存。
- 训练中验证：跳过，`eval_after_epoch=false`。
- 总训练耗时：17,354 秒，约 4.82 小时。
- loss 趋势：step 300 附近约 0.72，step 600 降到 0.484，说明训练目标被优化了，但没有学到可泛化的 indirect 边界。

## 训练数据审计

### 标签与来源分布

| label | 数量 | 来源分布 |
|---|---:|---|
| clean | 200 | safe_guard_prompt_injection 50；deepset_prompt_injections 40；nlphuji_flickr30k 50；haonan_li_cmmlu 30；cais_mmlu 30 |
| direct | 200 | jailbreakv_28k 100；deepset_prompt_injections 40；safe_guard_prompt_injection 60 |
| indirect | 200 | jailbreakv_28k 200 |

### 格式、图像与文本长度特征

| label | 主要格式 | 图像情况 | 文本长度 |
|---|---|---|---|
| clean | none | 200 条无图 | median 150.5，mean 256.4，max 4064 |
| direct | Template / none | 200 条无图 | median 292.0，mean 531.3，max 4734 |
| indirect | 仅 figstep | 200 条有图 | min = median = mean = max = 179 |

### 训练数据主要风险

- indirect 训练数据全部来自 `jailbreakv_28k` figstep，模型只见过一种非常窄的 indirect 模式。
- 所有 indirect 训练样本都有图，而 clean/direct 全部无图，容易让模型学到“某种固定图文模式 = indirect”的捷径，而不是真正学习语义型 indirect injection。
- indirect 文本长度恒定为 179，这是另一个明显捷径信号，也不符合真实 OCR/image injection 的分布。
- direct 数据虽然混合了几个来源，但仍偏窄：100 条 JailbreakV 加 100 条公开 prompt-injection 二分类数据。
- clean 有多个来源，但没有 image-backed clean 或 OCR hard-negative，模型没有学到“有图/OCR 也可能是 benign”。

## 验证集审计

### 验证集来源分布

| 验证集 | 数量 | 来源分布 | 图像情况 |
|---|---:|---|---|
| clean-only | 100 | nlphuji_flickr30k 34；safe_guard_prompt_injection 35；deepset_prompt_injections 7；haonan_li_cmmlu 12；cais_mmlu 12 | 100 条无图 |
| direct-only | 100 | jailbreakv_28k 85；safe_guard_prompt_injection 9；deepset_prompt_injections 6 | 100 条无图 |
| indirect-only | 100 | synthetic_image_injection 93；jailbreakv_28k 7 | 100 条有图 |

### 验证集长度与格式

| 验证集 | 主要格式 | 语言分布 | 文本长度 |
|---|---|---|---|
| clean-only | none | en 88；zh 12 | median 231.5，mean 251.2，max 2066 |
| direct-only | Template 75；none 15；Persuade 10 | en 56；zh 44 | median 537.0，mean 809.4，max 1939 |
| indirect-only | synthetic/no format 93；figstep 7 | en 79；zh 21 | median 36.0，mean 41.7，max 179 |

### 训练/验证分布错位

- indirect train：100% JailbreakV figstep。
- indirect eval：93% synthetic_image_injection，只有 7% JailbreakV figstep。
- indirect train 文本：固定长度 179。
- indirect eval 文本：median 36，主要是短 OCR 风格文本。
- clean train/eval 都是无图，因此当前 clean 指标不能证明 image/OCR 场景下的 clean FPR 安全。
- direct eval 比 direct train 更长，且 85% 来自 JailbreakV，增加了长 direct attack 被当成 benign instruction 的风险。

## 对比评估结果

### 新模型绝对效果

| 验证集 | 目标标签 | 目标召回率 | 目标精确率 | 目标 F1 | 主要漏判模式 |
|---|---|---:|---:|---:|---|
| clean-only | clean | 0.970 | 1.000 | 0.985 | 3 条 clean 被预测为 direct |
| direct-only | direct | 0.310 | 1.000 | 0.473 | 69 条 direct 被预测为 clean |
| indirect-only | indirect | 0.000 | 0.000 | 0.000 | 84 条 indirect 被预测为 clean；16 条被预测为 direct；0 条被预测为 indirect |

### 与基线的差异

- clean 相比 baseline 有提升，已经比较强。
- direct 相比 baseline 有明显提升，但 recall 仍只有 0.31。
- indirect 相比 smoke/baseline 反而退化：baseline 检出 8/100，balanced 600 检出 0/100。

## 失败模式分析

### Clean

- 当前 clean 结果强，目标 F1 为 0.985。
- 3 条 false alarm 说明模型对少量 clean 有 direct 过敏感。
- 由于 clean 训练/验证都无图，当前 clean 分数不能证明 clean image/OCR FPR 可控。

### Direct

- direct precision 是 1.0，因为模型只有在非常确信时才预测 direct。
- direct recall 弱，只有 0.31，69% 漏到 clean。
- 可能原因：
  - direct 训练数据量不足，覆盖不了 JailbreakV 和公开 prompt-injection 的多样性。
  - direct 样本长度和风格差异很大，600 条总训练规模不足以学习稳健边界。
  - 没有训练期验证和 checkpoint 选择，最终 checkpoint 不是按 direct F1 选择。
  - 模型整体偏保守：保住了 clean precision，但牺牲了 attack recall。

### Indirect

- indirect 在 100 条 eval 上完全失败：recall 0.0，F1 0.0。
- 大多数 indirect 漏到 clean，而不是 direct，说明模型对 eval 分布下的 indirect 边界几乎没有学习到。
- 可能原因：
  - 来源错位严重：train indirect = JailbreakV figstep；eval indirect = mostly synthetic_image_injection。
  - prompt/text 错位严重：train indirect 是固定 figstep prompt；eval indirect 是短 OCR 风格文本。
  - 缺少 clean image/OCR hard negative，模型无法学习稳健的跨模态区分。
  - 训练目标没有包含 OCR 文本、图文一致性或 C6B 兼容辅助标签。
  - 只训练了语言侧 q/k/v/o LoRA；如果 indirect 需要图文交互，单纯语言侧 LoRA 可能不足。

## Phase 2.3 的已验证根因假设

1. 数据量过小：每类 200 条不足以覆盖 direct/indirect 多样性。
2. indirect 数据过窄：训练 indirect 只用了 JailbreakV figstep。
3. indirect 训练/验证错位严重：synthetic_image_injection 主导 eval，但没有进入 train。
4. 缺少 hard negative：没有 image-backed clean、benign OCR、视觉无关但安全的样本。
5. 缺少 C6B-ready 字段：Phase 2.2 数据没有 OCR text、OCR confidence、图文一致性、indirect subtype 等字段。
6. 训练没有基于验证指标选择 checkpoint：`eval_after_epoch=false`，最终 checkpoint 不是 metric-driven。
7. 模型对不确定 attack 偏向 clean，这有利于 clean precision，但压低 direct/indirect recall。

## Phase 2.3 数据要求

### 训练数据

- 每类至少 1000 条。
- indirect 必须包含：
  - JailbreakV figstep。
  - synthetic_image_injection。
  - OCR/image instruction override 样本。
  - email/tool-use indirect 样本。
  - 短 OCR、长 OCR、多语言 OCR 文本。
  - 非模板化改写和布局变体。
- clean 必须包含：
  - text-only clean。
  - image-backed clean。
  - 有 OCR 但 benign。
  - 图文不相关但 benign。
  - 长 benign instruction 和短 benign caption。
- direct 必须包含：
  - JailbreakV templates。
  - 公开 prompt-injection 语料。
  - 短 direct attack。
  - 长 roleplay / policy override / DAN / dual-response attack。
  - 多语言 direct attack。

### Manifest 字段要求

Phase 2.3 记录在适用时应支持这些字段：

- `source`
- `label`
- `lang`
- `has_image`
- `template`
- `attack_family`
- `indirect_subtype`
- `ocr_text`
- `ocr_confidence`
- `ocr_present`
- `text_image_consistency_label`
- `cross_modal_attack_type`
- `has_instruction_override`
- `hard_negative_type`
- `dedup_key`
- `split`

### 验证集要求

- 冻结 quick eval：clean/direct/indirect 各 50 条。
- 冻结 full eval：clean/direct/indirect 各 200 条。
- quick 和 full 必须互斥，且都不能与 train 重叠。
- full indirect 必须包含 figstep、synthetic_image_injection、OCR-style 和 hard cross-modal variants。
- eval 报告必须包含 source/template/lang/has_image/ocr_present/cross_modal_attack_type/hard_negative_type 切片。

## 训练与 checkpoint 选择要求

- 每完成 10% steps 做一次训练期 compare。
- 10% checkpoint 使用 quick compare。
- 50%、100%、best-candidate checkpoint 使用 full compare。
- checkpoint 选择使用 `best_by_min_class_f1`，不能默认最后一步。
- 如果 min class F1 并列，按 macro F1、indirect F1、clean FPR 依次打破平局。
- 正式长训练前必须跑 resume smoke。
- 保留每 5 step log、每 50 step checkpoint。
- clean FPR 是硬约束；不能通过过度拦截 clean 来换取 indirect 提升。

## Step 2 执行决策

- 已决定：使用最新完整 basic 数据池，而不是只使用 Phase 2.2 数据。
- 已决定：LLMail True 用作 indirect email/tool-use，False 用作 clean email false positive，Unclear 暂时排除。
- 已决定：JailbreakV figstep 用于 indirect，Template/Persuade/Logic 用于 direct，SD/SD_typo/typo 暂作 manual review。
- 已决定：Hlyn 可有限使用，但需要先 spot check 标签语义。
- 已决定：Cyberec 必须先做 canonical variant 或 group-level dedup。
- 待后续 T2.40 冻结：首轮 sweep 组合数、是否启用两阶段分类、是否启用扩展可训练模块。

## 完整原始数据集可用性检查

前面的审计主要基于 Phase 2.2 已完成 run 的产物：balanced 600 train、val、三组 label-only eval、训练日志和 compare 报告。这足以解释 balanced 600 失败，但还不足以评估最新扩展后的 raw/basic 数据池。

本节补充使用当前本地 `runs/_datasets/raw` 和 `runs/_datasets/raw_status_basic.json` 的完整可用性检查。11 个默认 basic 数据集都已存在并标记 complete。结构化快照已写入本 run 的 `artifacts/raw_data_availability.json`。

### 可用原始数据快照

| source | 本地行数 / 文件 | Phase 2.3 可能用途 |
|---|---:|---|
| `deepset_prompt_injections` | 662 行；label 0=399，label 1=263 | clean/direct seed data |
| `safe_guard_prompt_injection` | 10,296 行；label 0=7,150，label 1=3,146 | clean/direct seed data |
| `jailbreakv_28k` | primary 28,000 行；Template 18,336；figstep 2,000；SD/SD_typo/typo 6,000；Persuade 1,368；Logic 296 | direct、indirect figstep、多模态变体 |
| `cais_mmlu` | parquet 扫描检测 570 行 | clean seed data |
| `haonan_li_cmmlu` | zip 内 11,917 行；dev 335，test 11,582 | 中文 clean seed data |
| `nlphuji_flickr30k` | caption 31,014 行；train 29,000，val 1,014，test 1,000 | clean caption；若图片可用，可作 image-backed clean |
| `lakera_gandalf_ignore_instructions` | 1,000 行 | direct instruction-ignore 候选；需要 label mapping |
| `lakera_mosscap_prompt_injection` | 278,945 行 | 大规模 direct/hard prompt-injection pool；需要 level mapping |
| `microsoft_llmail_inject_challenge` | labelled phase1 160,741；labelled phase2 37,303；raw submissions 461,640；FP emails 203 | indirect email/tool-use injection、clean email false-positive |
| `cyberec_prompt_injection_dataset` | parquet 变体合计 32,867 行；label 1=16,553，label 0=16,314 | clean/direct/adversarial categories；需要跨变体去重 |
| `hlyn_prompt_injection_judge_deberta` | 399,741 行；label 0=203,067，label 1=196,674 | 大规模 clean/direct classifier-style pool；需验证 label 语义 |

### 对 Phase 2.3 的含义

- 扩展后的 raw pool 足以支持每类至少 1000 条的计划。
- Phase 2.2 没有使用多个现在可用的高价值来源：Lakera Gandalf、Lakera Mosscap、Microsoft LLMail、Cyberec、Hlyn。
- indirect 最关键的新来源是 Microsoft LLMail，因为它提供 realistic email/tool-use prompt injection，这是 balanced 600 完全缺失的模式。
- direct 最关键的新来源是 Lakera Mosscap、Cyberec、Hlyn、JailbreakV Template/Persuade/Logic，以及已有 deepset/safe_guard。
- clean/hard-negative 可通过 MMLU/CMMLU/Flickr30k、LLMail false-positive email、benign OCR/image 派生样本来增强。
- 若干数据源需要在训练前明确映射或去重：
  - Lakera Gandalf quick scan 中没有简单二分类 label。
  - Lakera Mosscap 需要 level-to-label mapping。
  - Cyberec 似乎包含重叠 parquet 变体，需要选择 canonical variant 或强制去重。
  - Hlyn 需要先验证 label 语义再大规模使用。
  - LLMail `Unclear` 在策略冻结前不得进入 train。

### 更新后的根因结论

balanced 600 失败不是因为可用原始数据不足，而是 Phase 2.2 采样设计过窄，尤其 indirect 只用了非常有限的子分布。最新 raw/basic 数据池已经包含足够的 direct、indirect-email/tool-use 和 hard-negative 材料，可以支撑更强的 Phase 2.3 数据集。下一步重点是按已经冻结的数据契约实现 source inclusion、label mapping、dedup 和 split isolation。
