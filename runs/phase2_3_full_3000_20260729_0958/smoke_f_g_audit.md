# Phase 2.3 F/G Smoke 审计结论

## 执行状态

- F 和 G 都从 Full2000 的 LoRA 初始化，完成了计划内 120/120 步 smoke 训练。
- 两路训练均未出现 NaN、Inf、异常退出或 checkpoint 缺失；`checkpoint_step_120.safetensors` 及训练状态文件均已保留在各自的 `artifacts/smoke_f_g/{f,g}/checkpoints` 目录。
- 正式训练 checkpoint 目录 `artifacts/checkpoints` 保持为空，未启动正式 3000-step 训练。

## 策略 F

- 配置特点：flat 分类概率、三元组对角排序约束。
- pair-suite120 的第 1 个分层 chunk（50 条，clean/direct/indirect 支持数 17/17/16）结果：
  - clean：recall 0.000，F1 0.000。
  - direct：recall 0.412，F1 0.368。
  - indirect：recall 0.500，F1 0.356。
  - macro F1：0.241。
- 判定：clean recall 为 0，触发硬性淘汰条件；未继续运行 mixed150。

## 策略 G

- 配置特点：hierarchical direct/non-direct 概率分解。
- pair-suite120 的第 1 个分层 chunk（50 条，clean/direct/indirect 支持数 17/17/16）结果：
  - clean：recall 0.176，F1 0.200。
  - direct：recall 0.118，F1 0.133。
  - indirect：recall 0.000，F1 0.000。
  - macro F1：0.111。
- 判定：indirect recall 为 0，触发硬性淘汰条件；未继续运行 mixed150。

## 门槛与后续

- 目标门槛为三类 recall 均不低于 0.35、direct F1 不低于 0.35、macro F1 不低于 0.45。
- F 和 G 均未达到门槛，当前没有可进入 mixed150 或正式训练的候选。
- 结论：保持正式训练未启动，后续应基于 F 的 direct/indirect 边界能力和 G 的概率分解失败案例重新设计数据配比与损失约束后，再进入新的 smoke 验证。
