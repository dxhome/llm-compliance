# F 策略正式训练审计

## 2026-08-02 P3 预检结论

- V2 完整性输入已通过：smoke 150（75/53/22，30 图像/OCR）与 full 500（250/175/75，80 图像/OCR）。
- P3 使用正式 F 配置完成 10 step 预检，配对三元组 batch 已启用；loss 为有限值，step 1/5/10 分别为 3.3014、2.8230、2.8667。
- `checkpoint_step_10.safetensors`、对应 `.state.pt`、`latest` 及 partial 文件均存在且可读取；恢复状态的 `step_global=10`。
- 未发现 NaN、Inf 或 traceback；`artifacts/checkpoints/` 在 T1 启动前为空，没有并行 Python 训练进程。

## 当前推进

- 按正式计划启动 T1：step 1-300，50 step 保存一次，仅写入 `artifacts/checkpoints/`。
- 达到 step 300 后，先验证 checkpoint/state，再串行运行 V2 smoke-150 评测；在得到三类指标前不进入 T2。

## 2026-08-08 T5 完成与 V2/full500 结论

- T5 已从 `checkpoint_step_2250` 正常续训至 `checkpoint_step_3000`；最终 checkpoint 与 state 文件完整，训练退出码为 `0`，未见 NaN、Inf 或 traceback。训练末尾滚动 loss 为 `0.9396`，数值有限。
- 冻结 V2/full500 原始评测已串行完成：500/500 predictions 完整，评测退出码为 `0`。原始 Accuracy/Macro F1/Weighted F1 为 `56.60%/38.89%/50.91%`；Clean/Direct/Indirect F1 为 `67.19%/49.48%/0.00%`，Indirect Recall 为 `0.00%`，原始输出仍发生 Indirect 类塌缩。
- 按既定策略，仅对已落盘的原始 `log_probs` 应用已锁定的 `indirect_logit_offset=+0.55`；未重新搜索偏置，校准产物写入 `artifacts/formal_f_benchmark_v2/r0_indirect_offset_3000/`。校准 Accuracy/Macro F1/Weighted F1 为 `55.00%/41.23%/51.49%`；Clean/Direct/Indirect F1 为 `66.56%/48.20%/8.93%`，三类 Recall 为 `81.20%/38.29%/6.67%`，预测数为 `360/103/37`。
- R0 的阶段进入条件通过：三类 Recall 非零，Direct F1 高于 `34.88%`，Indirect Recall 与 `6.67%` 基线持平，单类预测占比不超过 70%。但最终放行失败：校准 Macro F1 `41.23%` 低于 `45.00%` 门槛；同时以 `min(clean_f1, direct_f1, indirect_f1)` 选最优 checkpoint 时，step3000 的 `8.93%` 低于 step2250 R0 的 `11.92%`。
- 因最终评测门槛未通过，不启动 P5 复跑、发布或打包流程；保留所有原始与 R0 校准产物以及正式 checkpoint，后续应以 step2250 R0 作为当前最佳已验证状态，并单独讨论恢复 Indirect 的后续训练策略。

## 2026-08-08 MCR + SBC 无训练恢复结论

- MCR（Multimodal Context Routing，多模态上下文路由）只对冻结 V2 中缺少路由字段的图像/OCR记录启用正式训练期的结构化 `untrusted_image_ocr` prompt，保留原始图像和原始用户请求，不修改 benchmark、LoRA、optimizer 或 checkpoint。
- SBC（Scoped Bias Calibration，分组分数校准）仅在 smoke150 上锁定图像/OCR样本的 Direct logit penalty `-0.20`；既定 R0 Indirect offset `+0.55` 保持不变。full500 未参与参数选择。
- V2/full500 盲验收：Accuracy `60.20%`，Macro F1 `56.80%`，Weighted F1 `59.00%`；Clean/Direct/Indirect F1=`67.99%/48.20%/54.22%`，Recall=`75.60%/38.29%/60.00%`。图像/OCR子集恢复 `40/40` Indirect。
- 该推理策略满足最终三类 recall 非零、Direct F1 >= `35%`、Macro F1 >= `45%` 的门槛，是当前最佳已验证候选；其结果必须标注为“F-3000-MCR-SBC 推理策略”，不能误表述为新的训练 checkpoint。

## 2026-08-02 T1 启动阻塞

- 已进行了两次有界启动尝试，均在 PowerShell `Start-Process` 创建子进程前失败：当前会话环境同时包含 `Path` 与 `PATH`，触发 `Item has already been added. Key in dictionary: 'Path'`。
- 两次均未创建 Python 训练进程，`artifacts/checkpoints/` 仍为空，`formal_f_train.log` 未产生训练输出；没有污染正式训练或 V2 基准。
- 按正式训练计划的停止规则，不继续重复启动。保留健康的 P3 预检 checkpoint/state，待修复宿主 PowerShell 环境变量冲突后，再从 T1 step 1 启动。

## 2026-08-02 T1 运行观察

- 后续由外部启动器成功创建正式训练；日志确认使用 `train_f_formal.yaml`、V2 smoke-150 验证输入、`batch_size=3` 和 `paired_batch_mode=true`，正式输出目录为 `artifacts/checkpoints/`。
- 已完成 global step 10，loss 记录为 step 1 `3.3014`、step 5 `2.8230`、step 10 `2.8667`，均为有限值；未发现 NaN、Inf 或 traceback。
- 训练脚本每 5 step 记录一次，当前实测单 step 约 150 秒；观察时训练主进程持续消耗 CPU 和约 8.6 GB 工作集，未见停滞。尚未达到 step 50，因此正式 checkpoint 数量为 0 属正常状态。
- 预计约 12.5-15 小时达到 T1 step 300；届时必须先检查 checkpoint/state，再串行运行 V2 smoke-150。
