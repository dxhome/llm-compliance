# Phase 2.3 Full 3000 执行计划

## 任务定义

- 任务名：`Phase 2.3 Full 3000`
- Run 目录：`runs/phase2_3_full_3000_20260729_0958`
- 初始化：加载 Full 2000 的 LoRA 权重
- 不继承：旧分类 head、optimizer state、global step
- 训练量：3,000 个新的优化 step，而不是从 global 2000 简单续到 global 3000
- 目标：在混合验证集上实现 clean/direct/indirect 三类 F1 均大于 0.70，并保持 direct 能力

这样可以保留 direct 的底层表征，同时重建已明显塌缩的三分类决策边界。

## 执行阶段

### 1. 创建 run 与冻结基线

预计时间：15-30 分钟

预期：建立新的独立目录、配置、日志、artifacts 和执行记录；复制或引用 Full 2000 checkpoint，并记录 hash。

验收：旧 run 保持只读，新 run 可独立复现；明确 `full_2000` 是本轮初始化基线和回滚点。

### 2. 修正训练输入语义

预计时间：2-4 小时

预期：让模型真正看见“指令来自哪里”，而不是把全部内容平铺为 `User text`。

训练 prompt 统一为三种显式结构：

- clean：可信用户请求或正常文档/图片内容
- direct：当前用户请求中包含的越权指令
- indirect：不可信邮件、网页、检索文档或图片 OCR 中的指令，以及用户对该内容的正常请求

验收：同一句 override 指令放在用户请求、邮件正文、图片 OCR 时，输入结构明确不同；不把数据集 source、人工标签等泄漏字段直接交给模型。

### 3. 重建训练与验证数据集

预计时间：3-6 小时

预期：保持每类 1,000 条、合计 3,000 条，使 3,000 step 能完整覆盖一轮高质量样本。

数据重点：

- clean：增加 benign OCR、正常邮件、包含安全术语但非攻击的文本、正常引用“ignore previous instructions”的讨论内容。
- direct：保留并扩展当前高质量直接注入样本，尤其保留现有表现最好的攻击家族。
- indirect：按 OCR 图片、邮件/工具上下文、RAG/网页文档三类均衡构成；增加中英混合、改写、字体、版式、对比度和上下文变化。
- 配对样本：同一攻击文本分别作为用户指令、邮件正文、文档片段、图片 OCR 出现，仅改变可信边界。

验收：每类 1,000 条；训练与验证无重复；各类、来源、语言、图片/OCR、间接攻击子类型分布均有中文审计报告。

### 4. 重构评测方式

预计时间：2-4 小时

预期：消除“单标签集合上的 F1 实际近似召回率”的盲点。

评测产物：

- `compare_full_all`：600 条混合集，clean/direct/indirect 各 200 条，输出真实 3×3 混淆矩阵、每类 Precision/Recall/F1、Macro F1。
- 三个单类切片：保留，用于观察 clean FPR、direct recall、indirect recall。
- indirect 专项切片：OCR 图片、邮件/工具上下文、RAG/文档、中文、英文。
- direct 保留指标：混合集 direct F1 与 direct recall。

验收：报告可同时回答“模型绝对效果”“相对 Full 2000 的变化”“哪个 indirect 子类仍是瓶颈”。

### 5. 实现训练保护与选择策略

预计时间：2-4 小时

预期：训练过程不再只看平均 loss，而是按业务指标选择模型。

策略：

- 从 Full 2000 加载 LoRA；新建分类 head；新建 optimizer。
- 前 300-500 step 冻结 LoRA，仅校准新 head。
- 后续解冻 LoRA，以较低学习率继续训练；head 使用独立且略高的学习率。
- 使用严格均衡的类别轮转或等效采样，避免长窗口偏向某一类。
- 每 50 step 保存 checkpoint；每 5 step 打日志。
- 每 200-300 step 在小型混合 holdout 上评测；按 `min(clean_f1, direct_f1, indirect_f1)` 选择最佳 checkpoint。

验收：训练日志包含各类 F1、预测分布、混淆矩阵摘要和 early-stop/回滚判断，不仅有 loss。

### 6. 训练前 smoke 与消融验证

预计时间：1-2 小时

预期：在长训练前验证结构化 prompt、新 head、checkpoint 保存、加载和联合评测链路。

验收：

- 100-150 step 小规模训练后，预测分布不再接近“全部 direct”。
- clean、direct、indirect 在混合 smoke 集均有非零召回。
- direct recall 不低于 0.90；若明显下降，先调整数据配比或训练策略，不进入正式训练。

### 7. 正式 3,000-step 训练

预计时间：34-42 小时

预期：先恢复多类边界，再在保留 direct 的前提下提升 clean 与 indirect。

推荐 checkpoint 评测点：

- step 300：混合 quick compare，验证新 head 没有立即塌缩。
- step 750：混合 quick compare，确认 clean/indirect 开始改善。
- step 1,500：full compare。
- step 2,250：混合 quick compare。
- step 3,000：full compare 与最终切片报告。

说明：按上一轮实际约 39 秒/step 估算，纯训练约 32-33 小时；评测与磁盘开销预留后为 34-42 小时。若 compare 与训练争用 CPU，总耗时可能更长。

### 8. 最终决策、打包与 smoke

预计时间：4-6 小时

预期：只将满足目标的最佳 checkpoint 打包。

发布门槛：

- 混合集 clean F1 > 0.70，且 clean FPR 可控。
- 混合集 direct F1 > 0.85，direct recall >= 0.95。
- 混合集 indirect F1 > 0.70。
- OCR indirect 与邮件/文档 indirect 均不得出现接近 0 的子类失效。
- 相比 Full 2000，direct 不发生明显退化。

验收：输出中文最终报告、Full 2000 对比报告、最终 checkpoint、离线 package 与 offline smoke 结果。

## 关键风险与处理

- 不能把当前 direct-only 的 `0.995 F1` 当作真实 direct F1 门槛，因为该分数没有计入 clean/indirect 的误报。新计划将以混合集 direct F1 为准。
- 只增加训练步数风险很高：当前 loss 已接近随机三分类基线且预测分布塌缩，必须先改输入结构、数据对照组和评测。
- 如果结构化输入后 OCR indirect 仍明显落后，再单独做图文连接层/vision projector 的小规模消融；不在首轮就解冻完整视觉编码器，以保护 direct 能力与训练稳定性。

## 本 run 的执行约束

- Full 2000 checkpoint 仅作为 LoRA 初始化基线，不继承旧 head、optimizer state 或 global step。
- 本轮正式训练的 3,000 steps 是新的优化步数。
- 所有本轮日志、checkpoint、compare、package 和报告必须写入本 run 目录。
- 正式训练前必须完成 smoke 验收；未通过 smoke 不得进入长训练。

## Smoke 多策略执行计划（正式训练前置）

### 目标与原则

- 目标：在正式 3,000-step 训练前，选择一套能保留 direct、恢复 indirect 且不发生类别坍缩的训练策略。
- 约束：未通过本节全部门槛，不得启动正式训练；所有产物写入本 run 目录。
- 统一初始化：仅加载 Full 2000 的 LoRA，新建分类 head、optimizer 和 global step。

### 阶段 S1：统一数据与审计

- `smoke_train_300`：clean/direct/indirect 各 100 条；paired counterfactual 至少 120 条；direct replay 至少 30 条。
- `smoke_mixed_90`：三类各 30 条；`smoke_quick_150`：三类各 50 条。
- indirect 覆盖 OCR、邮件/工具、网页/RAG；clean 覆盖正常任务和带攻击关键词的安全引用。
- 使用来源记录、文本去重键和 paired group 三重隔离，输出中文审计报告。

### 阶段 S2：策略实现

| 策略 | head LR | LoRA LR | 训练 | 额外约束 |
|---|---:|---:|---|---|
| A 稳定基线 | 5e-5 | 2e-5 | head 100 step + 联合 100 step | 无 |
| B 边界强化 | 3e-5 | 1e-5 | head 100 step + 联合 100 step | direct/indirect 权重 1.25 |
| C direct 保留 | 3e-5 | 1e-5 | head 100 step + 联合 100 step | direct replay teacher 蒸馏 |

- 所有策略类别轮换或等效均衡采样；每 5 step 日志、每 50 step checkpoint。

### 阶段 S3：三策略 200-step 筛选

- 每个策略在 step 100 和 step 200 运行 `smoke_mixed_90`；step 200 额外运行 direct replay。
- 淘汰：任意类别 recall=0、单类预测占比>70%、direct recall<0.80、最小类别 F1<0.25、NaN 或持续不稳定。
- 选择：最大化三类 F1 最小值；direct recall 为硬约束；B/C 差异<0.05 时优先 B，A 达标时优先 A。

### 阶段 S4：最佳策略 400-step 验证

- head 校准 100 step，联合训练 300 step；在 step 100/200/400 运行 mixed smoke。
- step 400 额外运行 quick 150、direct replay 和 indirect 子类切片。
- 门槛：clean F1>=0.45，direct F1>=0.60，direct recall>=0.85，indirect F1>=0.45，macro F1>=0.50；不得发生类别坍缩。

### 阶段 S5：恢复验证与启动确认

- 从最佳策略 step 200 恢复 20-30 step，验证 LoRA 不重复冻结、训练状态恢复且 loss 连续。
- 输出中文 smoke 总结和正式训练确认清单。仅当全部门槛通过并获得用户明确确认时启动正式训练。

### 时间估算

- 数据与策略实现：1.5-3 小时；三策略筛选：9-11.5 小时；最佳策略验证：6-7 小时；恢复与报告：1-1.5 小时；合计约 18-23 小时。
# Strategy H smoke addendum (2026-07-31)

- H is a smoke-only F-D hybrid: reduced F diagonal ranking plus conservative D
  generic margin and triplet-balance loss.
- It writes exclusively to `artifacts/smoke_h`; it must not start formal 3000-step
  training or write into `artifacts/checkpoints`.
- The pair-suite120 gate is all recalls >= 0.35, direct F1 >= 0.35, macro F1 >=
  0.45. Any zero recall stops the candidate before mixed150.
