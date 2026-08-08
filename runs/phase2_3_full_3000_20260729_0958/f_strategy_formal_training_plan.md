# F 策略正式 3000-step 训练计划（V2-only，待批准）

## 0. 决策与固定边界

- 本次所有模型验证只使用 `MPID Standard Benchmark v2`：开发诊断使用互斥的 `smoke_all_150.jsonl`，正式阶段验收使用 `full_all_500.jsonl`。不使用 V1，也不使用 run-local `compare_*` 作为模型效果结论。
- F 是当前最稳定的平衡候选，但 smoke 为 Accuracy 46.67%、Macro F1 40.72%、Direct F1 33.73%，未达到原预设 45% / 35% 门槛。因此本轮定义为**用户批准后的受控正式实验**，不得表述为 smoke 已放行。
- 本计划仅供审阅。用户明确批准前，禁止启动 3000-step 训练，禁止写入 `artifacts/checkpoints/`。

## 1. 冻结训练策略与数据事实

### F 策略参数

- 初始化：仅加载 Full-2000 `checkpoint_step_2000.safetensors` 的 LoRA；新建分类 head、optimizer 和 global step。
- 模型：CPU float32、gradient checkpointing、LoRA `r=32/alpha=64/dropout=0.1`，目标 `q_proj/k_proj/v_proj/o_proj`。
- 优化：`batch_size=3`、`paired_batch_mode=true`、LoRA LR `5e-6`、head LR `2e-5`、无 class weights、`logit_margin=0.25`、`direct_margin=0.35`、`triplet_diagonal_weight=0.8`、`triplet_diagonal_margin=0.4`。

### 正式训练集事实

- 输入固定为 `data/train.jsonl`：3,000 条，clean/direct/indirect 各 1,000 条。
- 其中有 240 个完整可信边界三元组（720 条）；其余 2,280 条为非配对记录。
- 配对 sampler 每个 epoch 先输出 240 批完整三元组，再输出 760 批余下记录；3,000 step 等于 3 个 1,000-step epoch。因此第一阶段的 1-240 step 是最强的边界训练窗口，后续每个 epoch 开头会重复这一窗口。

## 2. 逐步执行清单

| 步骤 | 要完成的工作 | 验收产物与门禁 | 预计时间 |
| --- | --- | --- | --- |
| P0 | 冻结 V2 full/smoke：校验 `manifest.json`、`checksums.sha256`，物化带绝对图片路径的 150/500 条输入副本 | 完整性报告、150/500 条计数、图片路径可读 | 0.5-1 小时 |
| P1 | 审计 `train.jsonl` 与两个 V2 集：标签分布、240 个完整三元组、dedup key、`source_record_id`、规范化文本均无重合 | 中文数据审计；任一泄漏为阻断 | 0.5-1 小时 |
| P2 | 新建 `train_f_formal.yaml`，只能采用上述 F 参数；输出位置、checkpoint 命名、恢复参数全部写入配置 | 配置 diff、路径和参数审计 | 0.25-0.5 小时 |
| P3 | 10-step 预检，使用正式配置/数据但只写 `artifacts/formal_f_preflight/` | 一个完整三元组 batch、有限 loss、可读恢复状态、无 NaN/Inf；不写正式 checkpoint | 0.75-1.5 小时 |
| P4 | 根据 P3 的单进程真实 step 时间更新 ETA，并由用户确认启动长训练 | 时间审计和启动确认 | 0.1 小时 |
| T1 | 正式训练 step 1-300；每 50 step 保存 checkpoint | step 300 在 V2/smoke 150 诊断：三类 recall 均大于 0、单类预测占比不超过 70% | 13-16 小时 |
| T2 | 正式训练 step 301-750 | step 750 V2/smoke 150 诊断；若较 step 300 任一类 F1 连续恶化则停止 | 19-24 小时 |
| T3 | 正式训练 step 751-1,500 | step 1,500 在 V2/full 500 首次正式评测；Direct F1 不低于 F smoke 基线 33.73%，无类塌缩 | 33-41 小时 |
| T4 | 正式训练 step 1,501-2,250 | step 2,250 V2/smoke 150 诊断；连续两次 Macro F1 或任一类 F1 恶化则回退 | 32-40 小时 |
| R0 | 无训练推理校准：基于 `checkpoint_step_2250` 的已保存 logits 扫描 Indirect 单标量偏置 | 只按预测分布锁定偏置，不修改权重、标签或 checkpoint；随后在 V2/smoke150 和 V2/full500 复核 | 0.5-1 小时 |
| R1 | 仅当 R0 失败时，从 `checkpoint_step_2250` 派生 Indirect 恢复分支 | 原 checkpoint 只读保留；重置 optimizer，降低 Direct 单边 margin，增加 Indirect target margin 与温和 class cost；先 100-150 step smoke | 5-8 小时 |
| T5 | 正式训练 step 2,251-3,000 | step 3,000 V2/full 500 最终评测、80 条图像/OCR 子集、中文审计报告 | 34-42 小时 |
| P5 | 对最佳 checkpoint 复跑 V2/full 500，核验预测数、混淆矩阵、hash、离线包与 smoke | 最终报告、预测、混淆矩阵、包验收 | 2-4 小时 |

时间按 F2 单进程训练约 150-180 秒/step 估算，已包含阶段性 V2 评测的合理缓冲。实际总时长预计 **134-170 小时（约 5.5-7 天）**；P3 的实测速度是最终排期依据。训练与评测必须串行，避免 CPU 争用造成评测漂移和更长墙钟时间。

## 3. 每个阶段的 V2 评测规则

- V2/smoke：仅做开发诊断，固定 150 条、`--batch-size 1`、输出完整 predictions、报告和混淆矩阵；不作为最终效果结论。
- V2/full：正式结论，固定 500 条、`--batch-size 1`、输出完整 predictions、报告和混淆矩阵；同时报告 80 条图像/OCR 子集。
- 最终 checkpoint 的选择指标为 `min(clean_f1, direct_f1, indirect_f1)`，而不是 Accuracy 或单类最高分。

## 4. 停止、回退与放行

- 立即停止：NaN/Inf、traceback、checkpoint 不可读、配对 batch 异常、任一类 recall 为 0、单类预测占比超过 70%。
- 阶段停止：V2/smoke 连续两次出现 Macro F1 或任一类 F1 下降；回退到最近一个健康 checkpoint，并在审计报告中说明原因。
- 最终放行仅基于 V2/full：三类 recall 均非零、Direct F1 >= 35%、Macro F1 >= 45%、图像/OCR 子集不发生相对最佳中期 checkpoint 的明显退化。
- 未达到最终门槛：只保留实验记录和 checkpoint，不标记为正式可用模型，也不进入发布/打包流程。

## 4.1 T4 实际阻断与恢复分支（2026-08-06）

- `checkpoint_step_2250` 保持只读，不进入 T5。其 V2/smoke150 的 Indirect Recall/F1 均为 `0.00%`；V2/full500 的 Indirect Recall/F1 为 `2.67%/4.44%`。虽然 Clean/Direct 和总体指标提升，但 Indirect 类塌缩，未满足前序安全预期。
- R0 为**无训练、无权重修改**的探索性推理校准：只使用已落盘的 `log_probs`，对 Indirect 施加单一加性 logit offset。偏置的选择不得读取逐样本 gold 标签，只以已公开的 V2 类别先验（50%/35%/15%）约束预测分布，再锁定为一个固定值。
- R0 的正式复核必须使用锁定偏置重新汇总 V2/smoke150 与 V2/full500。通过条件：三类 recall 均非零、Indirect Recall 不低于 step1500 full500 的 `6.67%`、Direct F1 不低于 step1500 full500 的 `34.88%`、单类预测占比不超过 70%。R0 结果仅作为探索性推理策略记录，不能替代未校准 checkpoint 的训练结论。
- R0 已于 2026-08-06 完成：以 full500 的公开 15% Indirect 类别先验（未读取逐样本 gold）锁定 `indirect_logit_offset=+0.55`。锁定后 full500 为 Accuracy `50.60%`、Macro F1 `39.62%`、Clean/Direct/Indirect F1=`64.92%/42.03%/11.92%`、Indirect Recall `12.00%`；smoke150 为 Accuracy `52.00%`、Macro F1 `40.82%`、Clean/Direct/Indirect F1=`66.67%/43.04%/12.77%`、Indirect Recall `13.64%`。两套 V2 复核均满足 R0 护栏，R1 不启动。
- 用户于 2026-08-06 选择 R0 路径：`+0.55` 固化为后续 checkpoint 的唯一 Indirect 推理校准策略，T5 获准从 `checkpoint_step_2250` 继续。原始未校准预测必须同时保留；所有后续 V2 报告同时给出未校准与 R0 校准指标，最终模型明确标注为“R0 校准推理模型”。R1 不启动。
- T5 期间不得重新搜索或调整 offset；step3000 仅应用已锁定的 `+0.55`，并以三类校准 recall 均非零、校准 Direct F1 >= `34.88%`、校准 Indirect Recall >= `6.67%` 作为继续进入 P5 的必要条件。
- 若 R0 不通过，才启动 R1。R1 从 `checkpoint_step_2250.safetensors` 加载模型权重但**不恢复 optimizer state**，新分支单独输出至 `artifacts/f_recovery_r1/`，不得覆盖 `artifacts/checkpoints/` 中任何正式 checkpoint。R1 参数候选：`direct_margin=0.15`、Indirect target margin `0.50`、`class_weights=[1.0,0.9,1.35]`、`triplet_diagonal_weight=1.0`。
- R1 先运行 100 step 后做 V2/smoke150；仅当三类 recall 均非零、Indirect Recall >= `6.67%` 且 Direct F1 >= `34.88%` 时，才允许扩展至 300 step 与 V2/full500。否则停止分支并回到 checkpoint 选择与数据诊断，不启动 T5。

## 5. 预期目录与交付

- 预检：`artifacts/formal_f_preflight/`，不写正式 checkpoint。
- 正式训练：仅在用户批准后写入 `artifacts/checkpoints/`；每 50 step 产生可恢复 checkpoint。
- V2 产物：`artifacts/formal_f_benchmark_v2/{smoke_step_300,smoke_step_750,full_step_1500,smoke_step_2250,full_step_2250,full_step_3000,best_full_500}/`；R0 仅写入 `artifacts/formal_f_benchmark_v2/r0_indirect_offset_2250/`，R1 仅写入 `artifacts/f_recovery_r1/`。
- 最终交付：V2 完整性报告、训练/配置 hash、所有 checkpoint manifest、500 条 predictions、三类报告、80 条图像/OCR 子集报告、中文审计结论与 Reference 第三章更新。
