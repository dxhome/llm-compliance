# Phase 2.3 执行计划：phase2_3_full_2000_20260727_1211

- 目标：执行增强版 Phase 2.3 微调，构建 clean / direct / indirect 目标类 F1 均大于 0.70 的 high-F1 模型底座。
- 基线 run：`runs/phase2_2_balanced_600_20260718_1955`
- 当前已完成步骤：T2.22 数据审计、T2.23 数据集契约、T2.24 训练集抽样策略、T2.25 quick/full compare 验证集构建。
- 当前下一步：T2.36 断点续训 smoke。原第 3 步和第 4 步已合并为 T2.26，并已完成。
- 断点续训 smoke 顺序：保持在 T2.26 之后执行，因为 smoke 要验证的正是冻结后的 checkpoint、log、resume 和 compare 调度配置。
- 修订后整体估时：约 39-73h，主要耗时仍在 CPU 正式训练和 compare。
- 文档语言约定：本 run 后续人工可读 Markdown 文档统一使用中文；JSON 字段名保持英文以兼容脚本。

## 计划阶段

| 阶段 | 任务 | 状态 |
|---|---|---|
| 数据审计 | T2.22 | 已完成；包含 Phase 2.2 失败复盘和完整 raw/basic 数据可用性检查 |
| 数据集契约 | T2.23 | 已完成；使用最新完整 raw/basic 数据池 |
| 训练/验证集构建 | T2.24-T2.25 | 已完成；训练集 3000 条，quick compare 150 条，full compare 600 条 |
| 训练配置与执行决策冻结 | T2.26，合并原 T2.40 | 已完成；配置和决策已冻结，readiness 预检通过 |
| Checkpoint / resume 能力补齐与 smoke 验收 | T2.36，第 4 步 | 待执行；目标不超过 1h |
| 正式训练与训练期 compare | T2.27、T2.32-T2.35、T2.37 | 待执行；约 31-54h，主要为运行耗时 |
| 打包与最终报告 | T2.39 | 待执行；报告约 1-3h，package/offline smoke 约 1h |

## 已冻结数据资产

- `data/train.jsonl`：训练集 3000 条，clean/direct/indirect 各 1000 条。
- `data/compare_quick_clean.jsonl`、`data/compare_quick_direct.jsonl`、`data/compare_quick_indirect.jsonl`：训练期 quick compare，每类 50 条。
- `data/compare_full_clean.jsonl`、`data/compare_full_direct.jsonl`、`data/compare_full_indirect.jsonl`：关键 checkpoint 和最终验收 full compare，每类 200 条。
- `data/resume_smoke.jsonl`：断点续训 smoke，每类 6 条。
- `data/phase2_3_dataset_hashes.json`：上述数据文件 sha256。
- `phase2_3_dataset_build_summary.md`：中文构建摘要和关键校验结果。

## T2.26 需要冻结的内容

- 训练步数、随机种子、首轮 sweep 组合数量。
- LoRA rank/alpha/dropout、学习率、等效 batch size、class weight 策略。
- 是否启用两阶段分类；是否启用扩展可训练模块。
- 每 5 step 训练日志、每 50 step checkpoint、每 10% steps quick compare。
- 50% / 100% / best candidate full compare 规则。
- 中断恢复策略：最新 checkpoint 发现、step 续接、日志追加、compare 调度不重复。
- best checkpoint 选择标准：优先使用 `min(clean_f1, direct_f1, indirect_f1)` 最大。

## T2.26 已冻结结果

- 首轮只跑 1 组配置，不做 sweep。
- 训练步数为 2000 optimizer steps；如 underfit 且 loss 仍下降，再从 checkpoint 续训到 3000 steps。
- LoRA 使用 `r=32, alpha=64, dropout=0.10`，target 为 `q_proj,k_proj,v_proj,o_proj`。
- 学习率为 `1.0e-4`，batch size 为 1，class weight 开启。
- quick compare 每类 50 条，每 10% steps 执行一次。
- full compare 每类 200 条，在 50% / 100% / best candidate 执行。
- 暂不启用两阶段分类和扩展可训练模块；如三分类继续偏 clean，再进入下一轮设计。
- `configs/train.yaml` 与 `configs/resume_smoke.yaml` 的 readiness 预检均已通过。

## 顺序说明

- resume smoke 不建议放在 T2.26 前面，因为它不是单纯“跑通脚本”，而是验证冻结后的训练配置是否真的支持中断恢复。
- 如果先 smoke、后改配置，smoke 结论会失效，后面仍然需要重跑。
- 因此当前顺序保持为：T2.26 冻结完整配置 → T2.36 resume smoke → 正式训练。

## 第 4 步执行计划：Checkpoint / Resume 能力补齐与验收

### 目标

- 正式长训练中断后，可以从最近 checkpoint 安全恢复，而不是只能依赖覆盖式 `partial_name`。
- 每 50 step 产出可追踪的 step checkpoint，并维护一个明确的 `latest.safetensors`。
- 恢复后 step、日志、checkpoint、compare 调度都连续，不回退、不覆盖、不重复、不漏跑。
- 第 4 步通过前，不启动正式 2000-step 长训练。

### 需要补齐的能力

- 保存策略：每 50 step 保存 `checkpoint_step_<step>.safetensors`。
- latest 策略：每次保存 step checkpoint 后，同步更新 `latest.safetensors`。
- 恢复发现顺序：优先 `latest.safetensors`，其次最新 `checkpoint_step_*.safetensors`，最后兼容旧的 `lora_phase2_3_latest.safetensors`。
- 恢复参数：自动设置 `resume_from`、`resume_global_step`、必要时设置 `skip_train_batches`。
- 日志策略：恢复训练时追加日志，不覆盖已有日志。
- compare 调度状态：记录已完成 compare checkpoint，恢复后不重复跑已完成 compare，也不漏掉后续 compare。

### Smoke 验证做法

- 使用 `configs/resume_smoke.yaml` 和 `data/resume_smoke.jsonl`。
- 第一段训练跑 10 step，要求生成 `checkpoint_step_10.safetensors` 和 `latest.safetensors`。
- 模拟中断后，从最新 checkpoint 恢复。
- 第二段恢复训练再跑 5 step，逻辑 step 应从 10 继续到 15。
- 验证日志为追加模式，且能看到恢复信息、step 续接信息和 checkpoint 发现信息。
- 验证 checkpoint 目录中保留 step checkpoint、latest checkpoint 和 smoke final checkpoint。
- 如加入 compare dry-run 或 mini compare，必须验证调度状态不重复、不漏跑。

### 验收产物

- `artifacts/resume_smoke/checkpoint_step_10.safetensors`
- `artifacts/resume_smoke/latest.safetensors`
- `artifacts/resume_smoke/lora_phase2_3_resume_smoke.safetensors`
- `logs/resume_smoke_first_leg.log`
- `logs/resume_smoke_resume_leg.log`
- `phase2_3_resume_smoke_report.md`
- `artifacts/resume_smoke/phase2_3_resume_smoke_report.json`

### 通过标准

- readiness 仍然通过。
- 第一段 10 step 成功完成并保存 checkpoint。
- 第二段能自动发现最新 checkpoint 并从逻辑 step 10 继续。
- 恢复后至少继续 5 step，最终逻辑 step 达到 15。
- 日志不覆盖，checkpoint 不丢失，恢复路径可复现。
- 总耗时不超过 1 小时；如果超过，需要先定位慢点，不能直接进入长训练。
