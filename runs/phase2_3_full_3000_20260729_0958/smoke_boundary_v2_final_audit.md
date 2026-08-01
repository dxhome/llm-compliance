# Smoke 边界重设计 v2 最终审计

## 决策

策略 D 与策略 E 均未通过 V2-S3 的 pair-suite 健康门槛。本轮 smoke 在策略筛选阶段结束；不启动正式 3,000-step 训练。

## 对比

| 策略 | checkpoint | Accuracy | Macro F1 | clean Recall / F1 | direct Recall / F1 | indirect Recall / F1 | 判定 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| D | step 100 | 0.575 | 0.467 | 0.800 / 0.727 | 0.000 / 0.000 | 0.925 / 0.673 | 淘汰 |
| E | step 150 | 0.650 | 0.521 | 1.000 / 0.755 | 0.000 / 0.000 | 0.950 / 0.809 | 淘汰 |

## 主要发现

两种策略都能改善 clean 与 indirect，但都完全丢失 direct 类召回。E 的更强 direct margin 没有修复该问题，反而在 120 条评测中完全不输出 direct。该现象说明当前 margin 与三元组覆盖损失尚未把“同一 payload 的 direct 边界”学成可泛化的判别条件，不能把 clean/indirect 的提升视为正式训练可接受的信号。

## 后续约束

- 保留 D 的 step 100 与 E 的 step 150 checkpoint，仅用于离线误差分析。
- 不执行 E 的 mixed 150、250 步终评或恢复验证；不创建、续写或恢复正式训练 checkpoint。
- 下一轮 smoke 应先针对 direct 的 24 条 clean 偏置与 16 条 indirect 偏置做错误分层，再修改数据构成和目标函数；新的候选策略必须先在独立 direct 校准集达到非零召回，才进入完整 pair-suite。
