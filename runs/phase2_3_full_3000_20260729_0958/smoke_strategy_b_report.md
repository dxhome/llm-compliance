# 策略 B 筛选结论

## 结论

策略 B 在第 100 步混合评测后淘汰。训练在评测完成时已推进至第 150 步；第 150 步 smoke checkpoint 仅保留为诊断记录，不作为候选模型。

## 第 100 步混合评测（90 条，三类各 30 条）

- 宏平均 F1：0.2346，较 Full 2000 基线提升 0.0722。
- clean：Precision 0.3333，Recall 0.8667，F1 0.4815。
- direct：Precision 0.4000，Recall 0.0667，F1 0.1143。
- indirect：Precision 0.2857，Recall 0.0667，F1 0.1081。

## 淘汰依据

- direct recall 0.0667，低于 0.80 的硬门槛。
- indirect F1 0.1081，低于 0.25 的最低类别 F1 门槛。
- 预测分布严重偏向 clean：90 条中预测为 clean 78 条，direct 6 条，indirect 6 条。

## 诊断

类别加权使模型从 Full 2000 的 direct 单类塌缩转为 clean 偏置，但未建立 direct 与 indirect 的有效边界。训练损失在 100-150 步间从约 1.40 降至约 1.31，说明优化过程数值稳定；然而 loss 的下降没有转化为目标类别召回，因此继续训练不具备成本效益。
