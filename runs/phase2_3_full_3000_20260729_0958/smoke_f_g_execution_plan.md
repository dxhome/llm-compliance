# Smoke F/G 并行执行计划

## 目标

在不启动正式 3,000-step 训练的前提下，同时验证两种互补机制能否消除 direct 类归零并保持 clean/indirect 的改善。

## 策略 F

- 三分类 flat head。
- 每个完整 clean/direct/indirect triplet 使用对角排序损失：每一类样本在自身类别 logit 上需高于同组另两条样本。
- 保留低强度通用 margin 与 direct margin，移除会只约束 batch 平均分布的 coverage loss。

## 策略 G

- 同一三输出 head 以层级概率解释：先判 direct/non-direct，再判 clean/indirect。
- 不复用 Full2000 的全分布 KL，也不使用 D/E 的 coverage loss。

## 执行与门槛

- F/G 都从 Full2000 step 2000 的 LoRA 初始化，head 与优化器重新初始化；每套训练 120 步、每 40 步保存 smoke checkpoint。
- 两套训练并行；评测在训练完成后串行执行，使用固定的 pair-suite 120 与 mixed 150。
- 任一训练出现 NaN/Inf、无 checkpoint、或任一类 recall 为 0 时淘汰。
- 120 步候选需满足三类 recall >= 0.35、direct F1 >= 0.35、macro F1 >= 0.45，才进入后续 smoke；绝不进入正式训练。
