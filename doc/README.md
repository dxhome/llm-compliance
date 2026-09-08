# 文档导航

> **项目状态**：已完成
>
> **最终方案**：`F-3000-MCR-SBC`
>
> **最终验收**：冻结 Standard Benchmark v2/full500，Accuracy 62.00%，Macro F1 59.28%。

本目录按“结论、立项、复现、归档”组织。模型、数据集、checkpoint、评测预测和离线 artifact 是本地大文件，均不进入 Git；文档中的 `runs/` 路径用于在已恢复本地资产的环境中追溯证据。

## 推荐阅读

1. [final-report.md](final-report.md)：最终方案、量化结果、训练收敛、pipeline、artifact 与适用边界。
2. [project-plan-and-verification.md](project-plan-and-verification.md)：项目完成项、验收证据、最小验证命令、事故复盘和后续限制。
3. [reference.md](reference.md)：威胁模型、LoRA、代码解读、平台命令、历史实验记录和排障细节。
4. [opening-report-vlm.md](opening-report-vlm.md)：当前 VLM 路线的开题报告。

## 文档职责

| 文档 | 状态 | 读者与用途 |
|---|---|---|
| [final-report.md](final-report.md) | 当前权威结论 | 答辩、交付、结果引用 |
| [project-plan-and-verification.md](project-plan-and-verification.md) | 当前执行归档 | 复现、审计、项目回顾 |
| [reference.md](reference.md) | 当前技术手册 | 开发、排障、理解实现与历史决策 |
| [opening-report-vlm.md](opening-report-vlm.md) | 保留 | VLM 路线开题材料 |
| [opening-report-formal.md](opening-report-formal.md) | 保留 | 级联方案的对照开题材料 |
| [opening-report-reference.md](opening-report-reference.md) | 历史存档 | 早期背景和方法论参考 |

## 已迁移文档

- 原 `tasks.md` 与 `VERIFICATION.md` 已合并为 [project-plan-and-verification.md](project-plan-and-verification.md)，避免计划、验收和最终状态互相矛盾。
- 原 `phase2_workflow_ops.md` 已删除；Phase 2.2 的历史操作说明和当前复现入口已归入 [reference.md](reference.md) 与执行验证归档。

## 本地资产约定

- `runs/<run_id>/`：一次训练或评测的本地执行资产，包括配置、脚本、日志和不进入 Git 的数据/模型/产物。
- `runs/_artifact/F-3000-MCR-SBC/`：最终可移动离线交付包的本地位置。
- `runs/_models/` 与 `runs/_datasets/`：共享模型和数据缓存。
- 需要完整推理时，从受控存储恢复上述资产；仅验证代码编排时可直接运行测试与轻量 pipeline smoke。具体命令见 [project-plan-and-verification.md](project-plan-and-verification.md#3-最小复现与验证)。
