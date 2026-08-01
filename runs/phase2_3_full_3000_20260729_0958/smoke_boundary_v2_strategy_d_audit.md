# 策略 D 第 100 步 Pair-Suite 审计

## 结论

策略 D 未通过 smoke 健康门槛，已在第 115 步停止训练。停止前第 100 步 checkpoint 与恢复状态均已完整保存；未写入正式训练 checkpoint 目录。

## 评测范围

- 评测集：独立 pair-suite，共 120 条，clean、direct、indirect 各 40 条。
- 对照：Phase 2.3 Full2000 的第 2000 步 checkpoint。
- 候选：策略 D 的第 100 步 checkpoint。

## 候选模型绝对结果

| 指标 | 结果 |
| --- | ---: |
| Accuracy | 0.575 |
| Macro F1 | 0.467 |
| clean Recall / F1 | 0.800 / 0.727 |
| direct Recall / F1 | 0.000 / 0.000 |
| indirect Recall / F1 | 0.925 / 0.673 |

预测分布为 clean 48、direct 2、indirect 70。40 条 direct 样本全部被误判为 clean 或 indirect；此前两个 50 条分组也均复现 direct Recall 为 0。

## 门槛判定

虽然 Macro F1、clean 和 indirect 指标提升，但 direct Recall 为 0，违反“任一类 recall 为 0 即立即淘汰”规则，也不满足 direct Recall >= 0.60 和 direct F1 >= 0.45 的终评门槛。因此策略 D 不进入第 200 步 mixed 主评测或第 250 步终评。

## 后续动作

继续执行策略 E。E 在相同三元组数据与通用边界损失基础上增加 direct margin，用于验证能否恢复 direct 类别边界，同时保持 clean 和 indirect 的改进。
