# Smoke 边界重设计 v2 执行计划

## 背景

上一轮 A/B/C 在第 100 步评测时，LoRA 刚刚解冻，实际只验证了新分类 head 的短暂校准。原 smoke 同时使用 batch size 1 和普通交叉熵，不能把同一 payload 在可信用户请求、直接请求和非可信外部内容中应得到不同标签的关系作为一个训练单位学习。

本计划只启动 D 和 E 两套有结构差异的策略。未通过本计划的 smoke 门槛，不进入正式 3,000-step 训练。

## V2-S1：数据与审计

预计 20-35 分钟。

- 生成 `smoke_boundary_train_900`：240 个完整 clean/direct/indirect 反事实三元组（720 条）加每类 60 条真实补充样本（180 条），总计每类 300 条。
- 生成独立的 `smoke_boundary_mixed_150`（每类 50 条）和与其去重的 `smoke_boundary_pair_suite_120`（40 个完整三元组）。
- 验收：训练集严格 300/300/300；每个训练三元组含同一 `pair_id` 的三类记录；评测与训练按 dedup key 隔离。

## V2-S2：训练机制

预计 30-60 分钟。

- 新增三元组 batch sampler：每个 batch 固定为一个 clean/direct/indirect 三元组，batch size 为 3。
- 主损失为三分类交叉熵加多类 logit margin；每个样本的正确类别 logit 必须超过最高错误类别固定边际。另加入三元组预测覆盖损失，惩罚三元组同时预测为同一类别的塌缩。
- 策略 D 使用通用边界 margin；策略 E 在 D 基础上对 direct 样本增加更强的 direct margin，不使用只对 direct 样本做 KL 的 teacher 蒸馏。
- head 仅 warm-up 25 步，之后立即联合训练 LoRA；主评测放在第 200 步，确保 LoRA 已联合训练 175 步，并与每 50 步 checkpoint 周期对齐。
- 验收：配置加载、三元组 batch、边界损失、checkpoint 均通过单步 dry-run。

## V2-S3：D/E 筛选

预计每套 10-13 小时，串行执行以避免 CPU 与内存竞争。该估计基于三元组 batch 的单步 dry-run 实测约 164 秒；第 200 步主评测约在 9-10 小时后触发。

- 每套训练 250 步；每 5 步日志、每 50 步 smoke checkpoint。
- 第 100 步运行轻量 pair-suite 健康检查；第 200 步运行 mixed 150 主评测；第 250 步仅对通过主门槛的策略运行 mixed 150 和 pair-suite 终评。所有评测点均与 50-step checkpoint 周期对齐，保证评测可复现。
- 任何一类 recall 为 0、单类预测占比超过 70%、出现 NaN/Inf 或训练停滞时立即淘汰。

## V2-S4：策略选择与后续

预计 30-60 分钟。

- Smoke 终评门槛：clean F1 >= 0.45、direct recall >= 0.60、direct F1 >= 0.45、indirect F1 >= 0.35、macro F1 >= 0.45、pair-suite 三类平均准确率 >= 0.50，且任一类 recall 不低于 0.35。
- 通过者按最小类别 F1、pair-suite 准确率、direct recall 依次选择；然后才进入 400-step 验证与恢复验证。
- 若 D/E 都失败，输出中文诊断报告并停在 smoke，不启动正式训练。
