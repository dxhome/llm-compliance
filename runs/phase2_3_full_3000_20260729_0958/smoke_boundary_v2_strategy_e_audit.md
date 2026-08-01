# 策略 E 第 150 步 Pair-Suite 审计

## 结论

策略 E 未通过 V2-S3 健康门槛，训练已在第 165 步安全停止。第 150 步 checkpoint 及训练状态完整保留；正式训练 checkpoint 目录未写入任何文件。

## 评测范围

- 候选 checkpoint：策略 E `checkpoint_step_150.safetensors`。
- 评测集：独立 pair-suite，共 120 条，clean、direct、indirect 各 40 条。
- 评测按 50 / 50 / 20 三组顺序完成，并在汇总前确认全部 120 条均已落盘。

## 汇总结果

| 指标 | 结果 |
| --- | ---: |
| Accuracy | 0.650 |
| Macro F1 | 0.521 |
| clean Recall / F1 | 1.000 / 0.755 |
| direct Recall / F1 | 0.000 / 0.000 |
| indirect Recall / F1 | 0.950 / 0.809 |

混淆矩阵显示：40 条 clean 全部预测正确；40 条 direct 中 24 条误判为 clean、16 条误判为 indirect，没有一条预测为 direct；40 条 indirect 中 38 条预测正确、2 条误判为 clean。策略 E 的预测分布为 clean 66 条、direct 0 条、indirect 54 条。

## 门槛判定

虽然 clean、indirect 和 macro F1 均优于策略 D 的第 100 步结果，但 direct recall 为 0，触发执行计划中“任一类 recall 为 0 即立即淘汰”的规则，同时不满足终评所需的 direct recall >= 0.60、direct F1 >= 0.45 与任一类 recall >= 0.35。

因此，策略 E 不进入第 200 步 mixed 150 主评测、第 250 步终评或后续 400 步恢复验证。
