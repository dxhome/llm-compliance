# Phase 2.3 训练配置与执行决策冻结

## 结论

- 本轮执行使用单组增强配置，不做首轮 sweep。
- 首轮训练目标为 2000 optimizer steps，训练数据池为 3000 条均衡增强样本。
- LoRA 使用 `r=32, alpha=64, dropout=0.10`，target modules 保持 `q_proj,k_proj,v_proj,o_proj`。
- 学习率使用 `1.0e-4`，比 Phase 2.2 的 `2.0e-4` 更保守，用于降低 3000 条增强数据和更高 LoRA rank 下的 loss 抖动风险。
- 训练日志保持每 5 step 打印一次；checkpoint 目标策略为每 50 step 保存一次 step checkpoint，并维护 `latest.safetensors`。
- 每完成 10% steps 跑一次 quick compare；50% / 100% / best candidate 跑 full compare。
- best checkpoint 选择标准为 `min(clean_f1, direct_f1, indirect_f1)` 最大，避免只优化 macro F1 或 clean 类。
- 正式长训练前必须先完成 resume smoke，目标总耗时不超过 1 小时。

## 冻结文件

- 模板配置：`runs/_templates/configs/phase2_3_high_f1.yaml`
- 本 run 训练配置：`configs/train.yaml`
- 本 run resume smoke 配置：`configs/resume_smoke.yaml`
- 数据集 hash：`data/phase2_3_dataset_hashes.json`
- 数据构建摘要：`phase2_3_dataset_build_summary.md`

## 训练参数

| 项目 | 决策 |
|---|---|
| backbone | `smolvlm-500m` |
| device / dtype | `cpu` / `float32` |
| train records | 3000 |
| first-run train steps | 2000 |
| batch size | 1 |
| learning rate | `1.0e-4` |
| weight decay | 0 |
| class weighted | true |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.10 |
| LoRA target | `q_proj,k_proj,v_proj,o_proj` |
| log every | 5 steps |
| save every | 50 steps |
| eval after epoch | false |
| seed | 42 |

## Compare 策略

- quick compare 使用 clean / direct / indirect 各 50 条，共 150 条。
- quick compare 触发点：10%、20%、30%、40%、50%、60%、70%、80%、90%、100%。
- 以 2000 steps 计算，quick compare step 为 200、400、600、800、1000、1200、1400、1600、1800、2000。
- full compare 使用 clean / direct / indirect 各 200 条，共 600 条。
- full compare 触发点：1000 steps、2000 steps、best candidate。
- compare 日志每 5 条样本打印一次进度。
- compare 报告必须包含新模型绝对效果、Phase 2.2 baseline 对比、目标类 P/R/F1、混淆矩阵、预测分布和关键切片指标。

## Checkpoint 与恢复策略

- 目标策略是每 50 step 保存 `checkpoint_step_<step>.safetensors`，并同步维护 `latest.safetensors`。
- 当前 trainer 只支持周期性覆盖 `partial_name`，还没有原生 step checkpoint 和 latest 别名。
- 因此正式长训练前，第 4 步必须先补齐并验证 checkpoint / resume 能力，确保 `checkpoint_step_*.safetensors`、`latest.safetensors`、`resume_global_step` 和日志追加行为可用。
- 恢复顺序固定为：优先 `latest.safetensors`，其次最新的 `checkpoint_step_*.safetensors`，最后才使用兼容旧字段的 `lora_phase2_3_latest.safetensors`。
- resume smoke 需要验证中断后 step 不回退、日志不覆盖、checkpoint 可发现、compare 调度不重复或漏跑。

## 暂不启用的选项

- 暂不做首轮 sweep：CPU 成本高，先用单组增强配置拿到完整训练期曲线；若 50%/100% compare 显示 underfit，再从 checkpoint 续训或开第二组配置。
- 暂不启用两阶段分类：先验证增强数据和更强 LoRA 是否已经解决 direct/indirect；如果三分类仍明显偏 clean，再把两阶段分类作为下一轮训练目标。
- 暂不启用扩展可训练模块：vision projector / multimodal connector 可能改善 indirect，但也会增加训练不稳定和 clean FPR 风险；需要先做小规模 sanity，再决定是否纳入后续实验。

## 升级条件

- 如果 1000-step full compare 显示 indirect recall 仍明显偏低，但 loss 正常下降，则优先继续训练到 2000 steps。
- 如果 2000-step full compare 显示三类 F1 未全部超过 0.70，但 loss 仍在下降且 clean FPR 可控，则从 best/latest checkpoint 续训到 3000 steps。
- 如果 direct 或 indirect 出现明显混淆且 loss 不再下降，则暂停长训，分析误判样本后再调整数据比例或启用两阶段分类。
- 如果 clean FPR 明显升高，则优先增加 clean OCR hard negative 和 email FP 样本，而不是继续增大 LoRA 或训练步数。

## 第 4 步顺序确认

- 第 4 步不是简单跑通 smoke，而是先补齐 checkpoint / resume 能力，再用 smoke 验证。
- 原因：resume smoke 要验证的是冻结后的真实 checkpoint、log、resume 和 compare 调度配置；如果只用旧的覆盖式 partial checkpoint 跑通 smoke，不能证明长训练可安全恢复。
- 第 4 步通过前，不启动正式长训练。
