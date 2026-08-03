# F 策略正式训练审计

## 2026-08-02 P3 预检结论

- V2 完整性输入已通过：smoke 150（75/53/22，30 图像/OCR）与 full 500（250/175/75，80 图像/OCR）。
- P3 使用正式 F 配置完成 10 step 预检，配对三元组 batch 已启用；loss 为有限值，step 1/5/10 分别为 3.3014、2.8230、2.8667。
- `checkpoint_step_10.safetensors`、对应 `.state.pt`、`latest` 及 partial 文件均存在且可读取；恢复状态的 `step_global=10`。
- 未发现 NaN、Inf 或 traceback；`artifacts/checkpoints/` 在 T1 启动前为空，没有并行 Python 训练进程。

## 当前推进

- 按正式计划启动 T1：step 1-300，50 step 保存一次，仅写入 `artifacts/checkpoints/`。
- 达到 step 300 后，先验证 checkpoint/state，再串行运行 V2 smoke-150 评测；在得到三类指标前不进入 T2。

## 2026-08-02 T1 启动阻塞

- 已进行了两次有界启动尝试，均在 PowerShell `Start-Process` 创建子进程前失败：当前会话环境同时包含 `Path` 与 `PATH`，触发 `Item has already been added. Key in dictionary: 'Path'`。
- 两次均未创建 Python 训练进程，`artifacts/checkpoints/` 仍为空，`formal_f_train.log` 未产生训练输出；没有污染正式训练或 V2 基准。
- 按正式训练计划的停止规则，不继续重复启动。保留健康的 P3 预检 checkpoint/state，待修复宿主 PowerShell 环境变量冲突后，再从 T1 step 1 启动。

## 2026-08-02 T1 运行观察

- 后续由外部启动器成功创建正式训练；日志确认使用 `train_f_formal.yaml`、V2 smoke-150 验证输入、`batch_size=3` 和 `paired_batch_mode=true`，正式输出目录为 `artifacts/checkpoints/`。
- 已完成 global step 10，loss 记录为 step 1 `3.3014`、step 5 `2.8230`、step 10 `2.8667`，均为有限值；未发现 NaN、Inf 或 traceback。
- 训练脚本每 5 step 记录一次，当前实测单 step 约 150 秒；观察时训练主进程持续消耗 CPU 和约 8.6 GB 工作集，未见停滞。尚未达到 step 50，因此正式 checkpoint 数量为 0 属正常状态。
- 预计约 12.5-15 小时达到 T1 step 300；届时必须先检查 checkpoint/state，再串行运行 V2 smoke-150。
