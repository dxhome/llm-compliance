# F3/F4 完整 Smoke 执行计划

## 目标与边界

- F3：以 F 为基线，只将 `class_weights` 设为 `[1.0, 1.25, 1.0]`，增强 Direct 的交叉熵梯度。
- F4：以 F 为基线，使用 30 个泄漏审计通过的 Direct replay 三元组与 90 个原始训练三元组构成 120 个完整三元组；每个 replay 样本均有同载荷 clean/direct/indirect 对照，强化可信边界而不回灌 Benchmark v2 样本。
- 两项均从 Full-2000 step-2000 初始化，训练 120 steps、`batch_size=3`、`paired_batch_mode=true`，只写 `artifacts/smoke_f3` 或 `artifacts/smoke_f4`。
- 禁止启动正式 3000-step 训练；不得创建或写入 `artifacts/checkpoints`。

## 阶段

1. 构建并验证 F4 数据：360 条、120 个完整三元组、每类 120 条，且与冻结 Benchmark v2 smoke 的 dedup key 和规范化文本均无重合。
2. 并行训练 F3/F4；检查 step 40/80/120 checkpoint、loss、NaN/Inf 与异常日志。
3. 只有对应策略的 step-120 checkpoint 完整且训练健康后，串行执行冻结 Benchmark v2 smoke 150（`--batch-size 1`）。
4. 核验 150 条 predictions，比较 F/F2/F3/F4 的 Accuracy、Macro/Weighted F1、三类 Precision/Recall/F1 与图像/OCR 子集，并生成中文审计结论。

## 晋级门槛

- 训练：step-120 checkpoint 存在；无 NaN/Inf/traceback；三元组 batch 正常。
- 评测：三类 recall 均大于零，Direct F1 至少 0.35，Macro F1 至少 0.45；同时不以明显牺牲 clean/indirect 为代价。
- 任一条件不满足，仅记录 smoke 结论，不得启动正式训练。
