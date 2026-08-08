# MPID 离线多模态提示注入检测结题报告

> **报告版本**：v2.0
>
> **完成日期**：2026-08-08
>
> **最终方案**：`F-3000-MCR-SBC`
>
> **最终效果口径**：冻结 `MPID Standard Benchmark v2/full500` 的完整离线 pipeline 验收

---

## 摘要

本项目面向离线、隐私敏感和资源受限场景，构建轻量级多模态提示注入检测方案。任务将输入分为正常输入（`clean`）、直接提示注入（`direct`）和间接/多模态提示注入（`indirect`）三类。基础检测器为 SmolVLM-500M 加载 LoRA 与 MPID 三分类 head；在其上组合 C4/C5/C6 防线、本地 OCR 和可审计的推理校准，形成最终离线交付方案 **F-3000-MCR-SBC**。

正式训练以策略 F 为基线，使用 3,000 条均衡训练样本完成至 `checkpoint_step_3000`。原始 head 在正式验收中出现 Indirect 类塌缩，因此项目保持 checkpoint 不变，仅在冻结的 smoke150 上选择并锁定运行时补强：R0 全局 Indirect logit offset `+0.55`、本地 OCR 驱动的多模态上下文路由（MCR）以及图像/OCR 分组的 Direct logit 校准（SBC `-0.20`）。full500 不参与上述参数选择。

最终完整离线 pipeline 在冻结 V2/full500 上取得 Accuracy **62.00%**、Macro F1 **59.28%**、Weighted F1 **61.32%**；clean/direct/indirect F1 分别为 **68.38% / 53.56% / 55.90%**。三类 Recall 均非零，满足预设的 Direct F1 >= 35% 和 Macro F1 >= 45% 门槛。最终方案已形成可移动离线 artifact、完整 checksum、离线 smoke、性能证据和发布审计。该结论只适用于锁定的模型、推理策略和 Benchmark v2，不构成对未知攻击或生产高并发环境的无条件安全承诺。

**关键词**：提示注入；离线部署；多模态安全；LoRA；OCR；纵深防御

---

## 1. 研究目标与范围

### 1.1 目标

项目研究的问题是：在不依赖云端 API 的条件下，如何对用户文本和可选图像进行三分类提示注入检测，并在检测效果、离线可部署性、可解释性和资源消耗之间取得可审计的平衡。

交付目标如下：

1. 形成可离线运行的 `clean` / `direct` / `indirect` 三分类检测器。
2. 使用 LoRA 降低轻量 VLM 的微调与迭代成本。
3. 将规则、OCR 与模型判断组合为可解释的分层防线。
4. 建立训练、checkpoint、评测、预测、打包和发布审计闭环。
5. 以冻结 Benchmark v2 的独立 full500 结果作为最终能力结论。

### 1.2 范围与边界

| 维度 | 本项目范围 |
|---|---|
| 部署环境 | 离线 PC、低吞吐端侧和人工辅助审计场景 |
| 输入 | 文本和可选图像 |
| 安全任务 | `clean` / `direct` / `indirect` 三分类 |
| 模型 | SmolVLM-500M + LoRA + MPID classification head |
| 运行时防线 | C4 高置信 clean 放行、C5 规则、C6 本地 OCR / 跨模态检查、MCR、R0、SBC |
| 不在范围内 | 云端安全服务、7B 以上模型横评、攻击自动化、生产级安全承诺、高并发实时服务 |

`direct` 指攻击指令直接出现在用户文本中；`indirect` 指恶意指令位于图像或其他不可信外部内容中，或由图文组合形成。三分类而非二分类的原因是两类攻击依赖不同证据和防线，单一 Accuracy 无法揭示某一攻击类完全漏报的问题。

---

## 2. 最终方案：F-3000-MCR-SBC

### 2.1 方案组成

最终方案不是新的训练 checkpoint，而是固定的 `checkpoint_step_3000` 与锁定运行时策略组成的离线 pipeline：

```text
C5 rules
  -> C6B-lite local OCR
  -> C6A compatibility check
  -> MCR
  -> F-3000 LoRA + MPID head
  -> R0 / SBC calibration
  -> C4 clean gate
  -> block / allow
```

| 组件 | 作用 | 锁定规则 |
|---|---|---|
| F-3000 head | 处理未被显式规则覆盖的三分类边界 | `checkpoint_step_3000.safetensors`，不再改动权重 |
| C5 | 对高确定性直接注入模式给出可解释短路 | 命中即 `direct`；未命中只表示继续检测，不表示安全 |
| C6B-lite | 从图像像素做本地 OCR 并识别明确风险文本 | 使用随包 RapidOCR 权重；不读取 benchmark OCR 标注 |
| C6A | 兼容性与跨模态风险检查 | 位于 OCR 之后、模型之前 |
| MCR | 将本地 OCR 非空的图像标记为不可信图像文本上下文 | 仅由运行时像素 OCR 是否非空触发，使用 `untrusted_image_ocr` 结构化角色 |
| R0 | 恢复 Indirect 类分数边界 | 全局 Indirect logit offset 固定为 `+0.55` |
| SBC | 减少 MCR 场景下 Direct 偏置 | 仅在 MCR 激活时施加 Direct logit `-0.20` |
| C4 | 对高置信 clean 做保守放行 | 阈值 `0.95`，且位于 C5/C6 之后，不绕过已知风险证据 |

### 2.2 设计原则

1. **规则优先，语义兜底。** C5 与 C6B-lite 处理高确定性证据，剩余难例交给 VLM head。
2. **不可信外部内容隔离。** MCR 将图像 OCR 作为不可信内容的角色边界，而不是把 benchmark 中的 OCR 标注文字注入模型。
3. **校准与训练分离。** R0、MCR、SBC 都是推理策略，不改变 LoRA、optimizer、训练数据或 checkpoint。
4. **先选策略，再做盲验收。** R0 与 SBC 均在 smoke150 锁定，full500 只用于一次最终验收，不能再用于调参。
5. **保留决策证据。** 输出可追溯至规则命中、OCR 分支、MCR 激活、head fallback 或校准阶段。

---

## 3. F-3000 正式训练与收敛过程

### 3.1 主要训练版本的演进

MPID 的训练路线不是单次长训直接得到最终方案，而是先建立轻量 LoRA 基线和离线防线，再通过长期训练、统一 smoke 筛选和冻结 V2/full500 验收收敛至 F-3000。下表先给出主要版本的全局脉络，再在后续小节展开 Full-3000 的具体过程。

表中的“迭代训练时间”是日志可提取的训练墙钟时间，未计入人工分析、数据构造、评测和打包；Full-3000 的约 124 小时包含 A-H、F2-F4 等策略筛选与 F 正式训练，不能把它理解为单次 3,000-step 训练耗时。

| 主要版本 | 训练与迭代投入 | 正式 Benchmark | 已记录的正式结果 | 阶段结论 |
|---|---:|---|---|---|
| Balanced-600 | 600 step，约 4.8 小时 | V1/full，300 条 | LoRA-only：Accuracy 34.3%、Macro F1 22.5%；加 C4-C6：45.3% / 40.7% | 首次证明分层防线可提升弱攻击类，但整体仍未达到可靠三分类 |
| Full-2000 | 2,000 step，约 21.8 小时 | V1/full，300 条 | LoRA-only：35.3% / 21.7%；加 C4-C6：40.3% / 29.4% | 验证长任务恢复与离线链路；基础 head 偏向 Direct，clean/Indirect 不足 |
| F-3000 | 策略筛选与正式训练合计约 124 小时 | V2/full，500 条 | LoRA-only：56.60% / 38.89%；最终 F-3000-MCR-SBC：62.00% / 59.28% | 唯一通过预设三分类门槛的最终方案 |

两套 V1 结果与 V2/full 的类别分布、样本数不同，不能横向比较绝对分数；本表只说明工程与方法的演进路径。最终能力结论只以最后一行的冻结 V2/full500 完整离线 pipeline 为准。

### 3.2 Full-3000：从策略筛选到 F 基线

Full-3000 的第一阶段是在冻结 V2/smoke150 上，以相同评测脚本和 `batch-size=1` 比较 A-H、F2、F3、F4 等训练策略。多数候选出现明显类别偏置：A/B/C/H 偏向 clean，D/E 偏向 indirect，G 虽能发现大部分 indirect 但丢失 direct。F 的绝对 smoke 分数未触发自动放行门槛，但它是类别表现最可控的候选，因此经受控审查后进入正式训练。

| 候选 | Macro F1 | 主要现象 | 处理结论 |
|---|---:|---|---|
| F | 40.72% | 三类均有非零 Recall，整体较平衡 | 选为正式训练基线 |
| F2 | 40.13% | 调整 direct margin 无收益 | 不采用 |
| F3 | 40.77% | Macro F1 仅小幅波动，Direct 不变 | 不构成实质改善 |
| F4 | 38.01% | Direct 改善但 Indirect 明显退化 | 不采用 |
| 其余 A-H | 22.02%-28.67% | 单类偏置或类别塌缩 | 淘汰 |

完整横评指标和每个历史方案的细节见 [reference.md](/C:/work/llm-compliance/doc/reference.md:3557) 第三章；这些 smoke 结果只用于选型，不是最终能力结论。

### 3.3 Full-3000：正式训练、恢复与 checkpoint 固化

第二阶段使用 F 的冻结配置进行正式训练。训练先经过数值稳定预检和阶段 checkpoint 验证，再通过保存的 optimizer/RNG 状态恢复至 step 3000。

| 项目 | 配置 |
|---|---|
| 训练数据 | 3,000 条均衡样本，clean/direct/indirect 各 1,000 条 |
| 边界样本 | 240 组可信边界三元组 |
| 骨干 | SmolVLM-500M |
| LoRA | `r=32`，`alpha=64`，`dropout=0.10` |
| 批处理 | `batch_size=3`，`paired_batch_mode=true` |
| 直接注入边界 | `direct_margin=0.35` |
| 保存策略 | 每 50 step 保存 checkpoint、optimizer 与 RNG sidecar |
| 训练完成 | `checkpoint_step_3000`，退出码 0，无 NaN、Inf 或 traceback |

`checkpoint_step_3000`、对应恢复 state、训练日志和原始预测都被保留。此后不再通过训练改动 LoRA 权重；后续工作只在固定 checkpoint 上评估运行时策略，以保证最终方案的权重来源清晰、可回退且可审计。

### 3.4 Full-3000：固定权重后的诊断与运行时补强

第三阶段首先对原始 F-3000 head 做 full500 诊断，再仅以“不训练、不修改权重”的方式寻找补强策略。

| 尝试 | 是否改变训练权重 | 处理与结论 |
|---|---|---|
| 原始 step3000 head | 否 | full500 的 Indirect F1 为 0，存在类别塌缩，不能作为最终方案 |
| R0 | 否 | 在 smoke 锁定 Indirect offset `+0.55`；仅部分恢复 Indirect，仍不够 |
| step2250/3000 logits ensemble | 否 | 未改善三类平衡，不采用 |
| 条件 OCR rescue / OCR 文本注入 | 否 | smoke 未达到基线或带来误报，不采用 |
| MCR | 否 | 将本地 OCR 非空图像置于 `untrusted_image_ocr` 角色，恢复图像/OCR 边界 |
| SBC | 否 | 在 MCR 场景固定 Direct logit `-0.20`，与 R0、MCR 组成最终策略 |

原始 step3000 head 的 V2/full500 结果为 Accuracy 56.60%、Macro F1 38.89%、clean/direct/indirect F1 为 67.19% / 49.48% / 0.00%。这说明 loss 收敛和 checkpoint 成功保存不等同于安全三分类达到要求，也说明最终提升主要来自“固定 F-3000 head + 明确的运行时证据与校准”这一组合，而不是重新训练出另一套 LoRA 权重。

---

## 4. 数据、评测协议与审计边界

### 4.1 冻结 Benchmark v2

| 数据集 | 样本数 | 类别分布（clean/direct/indirect） | 用途 |
|---|---:|---:|---|
| V2/smoke | 150 | 75 / 53 / 22 | 训练策略与推理参数选择、失败诊断 |
| V2/full | 500 | 250 / 175 / 75 | 最终一次盲验收 |

Benchmark 通过 `dedup_key`、来源、记录标识、归一化文本指纹和 manifest/checksum 与训练数据隔离。最终评测只把 `text` 和可选 `image` 传给离线包；gold 标签、benchmark 元数据和 OCR 注释均保留在包外。所有 500 条预测均已生成，日志无 NaN、Inf 或 traceback。

### 4.2 指标与放行门槛

Macro F1 是主指标，因为它能惩罚仅预测多数类或遗漏某个攻击类的情况。报告同时给出 Accuracy、Weighted F1 和各类 Precision/Recall/F1。

预先声明的放行门槛：

1. 三类 Recall 均大于 0；
2. Direct F1 >= 35%；
3. Macro F1 >= 45%。

---

## 5. 最终 V2/full500 结果

### 5.1 完整离线 pipeline 指标

| 指标 | F-3000-MCR-SBC 结果 |
|---|---:|
| Accuracy | **62.00%** |
| Macro F1 | **59.28%** |
| Weighted F1 | **61.32%** |
| clean Precision / Recall / F1 | 63.27% / 74.40% / 68.38% |
| direct Precision / Recall / F1 | 65.83% / 45.14% / 53.56% |
| indirect Precision / Recall / F1 | 52.33% / 60.00% / 55.90% |
| 预测数（clean/direct/indirect） | 294 / 120 / 86 |
| C5 命中 / C6B-lite 命中 / head fallback | 35 / 40 / 425 |
| MCR 激活 | 40 |

所有预设门槛均已通过。完整预测、报告和运行日志位于 [offline_f_3000_mcr_sbc_v2_full500](/C:/work/llm-compliance/runs/phase2_3_full_3000_20260729_0958/artifacts/offline_f_3000_mcr_sbc_v2_full500)。

### 5.2 结果解读

#### 5.2.1 MPID LoRA-only 与完整优化链路的同基准对照

两行均使用相同的 `checkpoint_step_3000`、相同的冻结 V2/full500 输入和同一三分类任务定义。`MPID LoRA-only` 只运行 LoRA + MPID head；最终方案在该 head 外叠加 C4/C5/C6、MCR、R0 和 SBC。因此，该表衡量的是固定 F-3000 权重下完整离线防线的端到端增量，而不是新 checkpoint 带来的增量。

| 指标 | MPID LoRA-only | F-3000-MCR-SBC（LoRA + 全部优化） | 增量 |
|---|---:|---:|---:|
| Accuracy | 56.60% | **62.00%** | **+5.40pp** |
| Macro F1 | 38.89% | **59.28%** | **+20.37pp** |
| Weighted F1 | 50.91% | **61.32%** | **+10.41pp** |
| clean Precision / Recall / F1 | 55.64% / 84.80% / 67.19% | 63.27% / 74.40% / 68.38% | F1 **+1.19pp** |
| direct Precision / Recall / F1 | 63.39% / 40.57% / 49.48% | 65.83% / 45.14% / 53.56% | F1 **+4.08pp** |
| indirect Precision / Recall / F1 | 0.00% / 0.00% / 0.00% | 52.33% / 60.00% / 55.90% | F1 **+55.90pp** |
| 预测完整性 | 500 / 500 | 500 / 500 | 完整 |

原始 LoRA-only 报告位于 [full_step_3000](/C:/work/llm-compliance/runs/phase2_3_full_3000_20260729_0958/artifacts/formal_f_benchmark_v2/full_step_3000)，最终完整链路报告位于 [offline_f_3000_mcr_sbc_v2_full500](/C:/work/llm-compliance/runs/phase2_3_full_3000_20260729_0958/artifacts/offline_f_3000_mcr_sbc_v2_full500)。两次均已产出完整 `predictions.jsonl`；最终方案的 500 条预测未出现 NaN、Inf 或 traceback。

#### 5.2.2 结果解读

1. **三类攻击不再被单类优势掩盖。** Indirect Recall 60.00%、Indirect F1 55.90%，解决了原始 F-3000 head 的 Indirect F1 为 0 的失效模式。
2. **直接注入仍是优先改进项。** Direct Recall 为 45.14%，175 条 Direct 中仍有 96 条没有被最终判为 Direct；后续应按规则漏报、改写攻击、Unicode 混淆和语义边界分桶分析。
3. **分层组合有可测增益。** 仅含 MCR+SBC 的 head full500 验收 Macro F1 为 56.80%；完整 C5/C6B-lite 链路达到 59.28%，增加 2.48 个百分点。
4. **OCR 路径是可审计的运行时证据。** C6B-lite 基于本地 OCR 像素证据短路 40 条样本，另有 40 条 OCR 非空图像激活 MCR；两种路径均不使用 benchmark 标注字段。

---

## 6. 离线交付、性能与复现

### 6.1 最终 artifact

推荐交付目录：[runs/_artifact/F-3000-MCR-SBC](/C:/work/llm-compliance/runs/_artifact/F-3000-MCR-SBC)。它包含：

1. 固定的 `checkpoint_step_3000.safetensors`、SmolVLM-500M 与 RapidOCR 权重；
2. 含 MCR/R0/SBC 和 C4/C5/C6 的离线推理代码；
3. `MANIFEST.json`、`CHECKSUMS.txt`、发布清单和发布审计；
4. 包内 `smoke_offline.py`、图像 smoke fixture、full500 报告和性能证据。

可移动压缩包：[F-3000-MCR-SBC.zip](/C:/work/llm-compliance/runs/_artifact/F-3000-MCR-SBC.zip)。

| 验证项 | 结果 |
|---|---|
| 发布文件清单 | 94 个文件全部匹配 |
| 包内 checksum | 全部匹配 |
| 离线 smoke | 3/3 通过：C5 direct、clean head fallback、C6B OCR block |
| ZIP CRC | 通过 |
| ZIP SHA-256 | `95c80b7cf4a5a96c9b4779b1aebd9b29f3eca943fbbbbf9ef5a2800cc41c8349` |

部署前应在兼容的 CPU/Python 环境从受控本地 wheel 源安装 `requirements.txt`，再执行：

```powershell
python mpid_offline/smoke_offline.py --pkg mpid_offline --stage-root offline_smoke_stage
```

单条 NDJSON 输入格式为：

```json
{"text":"待检测文本","image":"可选本地图片路径"}
```

### 6.2 性能边界

| 路径 | P50 或峰值 |
|---|---:|
| C5 直接规则短路 | 6.81 ms |
| C6B OCR 拦截 | 203.34 ms |
| 文本 VLM head | 9,201.95 ms |
| 图像 MCR/head | 8,671.17 ms |
| 常驻峰值 RSS | 3,004.0 MB |

该方案适合离线批处理、人工辅助审计和低吞吐端侧防护；当前性能不支持高并发实时服务的声明。

---

## 7. 结论、限制与后续建议

### 7.1 最终结论

`F-3000-MCR-SBC` 是本项目的最终推荐方案。它以已完成训练的 `checkpoint_step_3000` 为基础，将规则、像素 OCR、多模态上下文隔离和锁定分数校准按固定顺序组合，在冻结 V2/full500 上达到 Accuracy 62.00%、Macro F1 59.28%、Direct F1 53.56%、Indirect F1 55.90%。结果通过预设三分类门槛，并已完成可移动离线 artifact 的完整性、断网 smoke 和 ZIP CRC 验证。

### 7.2 适用边界

1. 本结论只适用于当前 checkpoint、R0 `+0.55`、SBC `-0.20`、C4-C6 版本和冻结 Benchmark v2。
2. R0/MCR/SBC 是推理策略，不应被表述为新的微调模型或新的 LoRA 权重。
3. 不能在当前 full500 上继续搜索阈值、offset 或规则；任何后续变体须使用新的、未参与选择的冻结评测集验收。
4. Direct 漏报、OCR 噪声、未知攻击形式、低质量图像和跨平台运行差异仍是主要风险。

### 7.3 后续建议

1. 基于 Direct 漏报建立困难样本桶，重点补充改写攻击、Unicode 混淆、引用型攻击和角色伪装。
2. 以新的独立验证集做逐层消融：LoRA-only、+R0、+MCR、+SBC、+C5/C6 与完整 pipeline，并按图像/OCR、语言和攻击来源报告子集指标；不得在当前 full500 上继续搜索这些参数。
3. 如需进一步通过训练改善模型，应单独提出新的训练方案、冻结新的验证集，并保留当前 `checkpoint_step_3000` 作为可回退基线。
4. 针对生产部署，增加吞吐、内存上限、OCR 失败回退、模型完整性和人工复核流程的系统级验收。

---

## 附录：证据与历史记录原则

- 最终效果结论只引用冻结 V2/full500 的完整离线 pipeline 结果。
- V2/smoke 只用于策略选择和回归诊断，不作为最终能力排名。
- 早期 V1、run-local smoke、组件测试和离线包 smoke 只证明工程可运行性或用于问题定位。
- A-H、F2、F3、F4 的完整训练配置、逐类指标、日志和预测产物保留在 `runs/phase2_3_full_3000_20260729_0958/artifacts/`，汇总说明见 [reference.md](/C:/work/llm-compliance/doc/reference.md:3557)。
