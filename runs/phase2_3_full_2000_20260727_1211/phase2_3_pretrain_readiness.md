# Phase 2.3 正式训练前 readiness

## 当前结论

- 状态：已具备启动正式 2000-step 增强微调训练的前置条件；尚未启动正式训练。
- 下一步：由用户确认后，执行 `python scripts/run_phase2_3_training_workflow.py --run-dir runs/phase2_3_full_2000_20260727_1211 --execute`。
- 防误启动：workflow 默认只 dry-run，必须显式传 `--execute` 才会训练。

## 已完成校验

- 数据集：`train.jsonl` 3000 条，clean/direct/indirect 各 1000 条。
- quick compare：clean/direct/indirect 各 50 条，共 150 条。
- full compare：clean/direct/indirect 各 200 条，共 600 条。
- JSONL 校验：所有 Phase 2.3 JSONL 文件均可逐行解析；训练集中已清理 `U+2028/U+2029` 等特殊行分隔符。
- 数据 hash：已重新生成 `data/phase2_3_dataset_hashes.json`。
- baseline checkpoint：已确认 Phase 2.2 balanced 600 checkpoint 存在。
- resume smoke：已通过基于固定 probe loss 的恢复验收；恢复模型权重、optimizer state、RNG state 与确定性数据顺序。

## 正式训练编排

- 总步数：2000 optimizer steps。
- 训练方式：单个连续训练进程跑到 2000 step；不为了 compare 停止训练再 resume。
- 训练日志：每 5 step 打印一次。
- checkpoint：每 50 step 保存一次 `checkpoint_step_<step>.safetensors`，并维护 `latest.safetensors`。
- 中断恢复：从 `latest.safetensors` 恢复，按已完成 step 设置 `resume_global_step` 和 `skip_train_batches`。
- 计划文件：`phase2_3_training_workflow_plan.json`。
- checkpoint 观察点：500 / 1000 / 1500 / 2000。

## compare 与报告

- quick compare 共 2 次，在 500 / 1500 step 执行，和 full compare 错开。
- quick compare 每类 50 条，分别输出 clean/direct/indirect 三份报告。
- 50% 和 100% checkpoint 追加 full compare。
- full compare 每类 200 条，按 50 条为一组执行，保留分组日志。
- 每次 compare 都输出 baseline 与新模型对比，同时生成 Phase 2.3 中文汇总报告。
- 汇总报告包含目标类 P/R/F1、baseline 指标、F1 delta、预测分布、切片指标和误判样本清单。

## best checkpoint 选择

- 选择标准：`best_by_min_class_f1`。
- 主排序：`min(clean_f1, direct_f1, indirect_f1)` 越高越好。
- 平局排序：macro target F1、indirect F1、clean FPR proxy、step。
- 输出文件：`artifacts/compare/phase2_3_checkpoint_scorecard.json`。
- best checkpoint 名称：`artifacts/checkpoints/lora_phase2_3_best_by_min_class_f1.safetensors`。

## package 与 offline smoke

- workflow 已接入最终 package 和 offline smoke。
- package 只在 best checkpoint 的 clean/direct/indirect full-set 目标类 F1 全部大于 0.70 后执行。
- package checkpoint：`artifacts/checkpoints/lora_phase2_3_best_by_min_class_f1.safetensors`。
- package 输出：`artifacts/package/mpid_offline`。
- package manifest 会显式写入 Phase 2.3 LoRA 参数：r=32、alpha=64、target=`q_proj,k_proj,v_proj,o_proj`。
- offline smoke 输出日志：`logs/phase2_3_offline_smoke_best.log`。

## 成功与失败处理

- 成功标准：clean/direct/indirect 三类 full-set 目标类 F1 均大于 0.70。
- 如果任一类 F1 ≤ 0.70：先停止进入 package，输出失败分析，定位数据覆盖、标签噪声、模型欠拟合/过拟合、clean FPR 与 indirect 漏检。
- 如果 clean FPR proxy 明显上升：优先回滚到更保守 checkpoint，或增加 hard negative 后再训练，不允许只追 indirect F1。
- 如果 loss 持续下降但 F1 未达标：优先从 best/最新 checkpoint 续训到 3000 steps，而不是直接换方案。
- 如果 direct/indirect 仍大量混淆为 clean：再评估两阶段分类或扩展可训练模块。

## 时间评估

- 正式训练：约 16-30 小时 CPU。
- quick compare：2 次，每次三类合计约 1.5-2 小时 CPU，合计约 3-4 小时。
- full compare：50% 和 100% 各三类，每轮约 3-4 小时 CPU。
- best candidate full compare：如果 best 不在 50%/100%，额外约 3-4 小时 CPU。
- package + offline smoke：约 1 小时。
- 整体剩余耗时：约 26-47 小时，主要取决于训练和 full compare 实际速度。
