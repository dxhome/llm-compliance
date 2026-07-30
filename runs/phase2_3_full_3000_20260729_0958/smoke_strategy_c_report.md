# 策略 C 筛选结论

## 结论

策略 C 在第 100 步混合评测后淘汰。训练在评测完成时已推进至第 145 步；该 checkpoint 仅保留为诊断记录，不作为候选模型。

## 第 100 步混合评测（90 条，三类各 30 条）

- 宏平均 F1：0.2342，较 Full 2000 基线提升 0.0717。
- clean：Precision 0.3291，Recall 0.8667，F1 0.4771。
- direct：Precision 0.4000，Recall 0.0667，F1 0.1143。
- indirect：Precision 0.3333，Recall 0.0667，F1 0.1111。

## 淘汰依据

- direct recall 0.0667，低于 0.80 的硬门槛。
- indirect F1 0.1111，低于 0.25 的最低类别 F1 门槛。
- 预测分布严重偏向 clean：90 条中预测为 clean 79 条，direct 6 条，indirect 5 条。

## 诊断

策略 C 的 direct replay teacher 蒸馏没有改变第 100 步的主要决策边界：结果与策略 B 近似，只是 indirect F1 从 0.1081 微升至 0.1111。蒸馏训练 loss 在 40-145 步从约 1.76 降至约 1.48，数值上可以收敛，但目标类别召回仍接近零。因此当前蒸馏强度、仅对 direct replay 施加的约束以及 100 步 head warm-up 组合，不足以避免 clean 偏置。
