# Phase 2.3 A-H 标准化 Benchmark v2 Smoke 横向验证计划

## 目标与边界

在完全相同的 `benchmarks/mpid_standard_v2/smoke/smoke_all_150.jsonl` 上，重新
评测策略 A-H 的既有 smoke checkpoint，消除历史评测集不同造成的横向不可比问题。

- Benchmark v2 smoke 固定为 150 条：clean=75、direct=53、indirect=22；含 30 个图像/OCR
  样本，且与 V1、历史训练数据及本 run 均已隔离。
- 本轮只评测现有 checkpoint，不重训、不改策略、不生成任何正式训练 checkpoint。
- `artifacts/checkpoints` 必须始终为空；所有产物仅写入
  `artifacts/standard_benchmark_v2_a_h/`。
- benchmark 原文件只读。若现有数据加载器不能解析相对图片路径，只在产物目录生成
  逐字段等价、图片路径绝对化的运行副本，并记录原始 SHA-256 和记录数。

## S0：冻结与预检

1. 验证 `smoke_all_150.jsonl`、manifest 和 30 张图片的 SHA-256；核对标签数
   clean/direct/indirect=75/53/22。
2. 对 benchmark 做 3 条 smoke dry-run（文字 clean、文字 direct、含图 indirect），确认：
   图片实际加载、OCR 样本没有降级为占位图、标签和路径不变。
3. 为 A-H 锁定各自最高编号且模型/状态文件齐全的 checkpoint，并读取 checkpoint 内
   `__classification_mode__`。不以事后选取较优中间 checkpoint 取代最终 checkpoint。
4. 生成 `checkpoint_manifest.json`，包含策略、绝对 checkpoint 路径、step、SHA-256、
   分类模式、原始配置路径和开始前 formal checkpoint 数量（应为 0）。

## S1：统一单模型评测

每个策略均使用相同配置：CPU、batch size=1、完整 150 条 benchmark、固定顺序、无
`--max-records`、无分块早停。评测加载 checkpoint 保存的分类模式，尤其保证 G 以
hierarchical 模式运行。

每项输出：

- 总体 accuracy、macro F1、weighted F1；三类 precision/recall/F1 与混淆矩阵。
- 30 条图像/OCR 子集的相同指标，以及文本-only 120 条子集指标。
- 每条预测、gold、预测类别、置信度/类别分数、是否含图、是否 OCR、source、lang；用于
  后续错误审计与可复算排名。
- 运行日志、耗时、异常信息和 checkpoint 哈希。

## S2：两路并行队列

为避免两模型同时占满 CPU 导致每项明显变慢，最多并发 2 个评测进程；同一波完成并且
产物完整后再启动下一波。

| 波次 | 并行任务 | 前置条件 |
| --- | --- | --- |
| 1 | A + B | S0 全部通过 |
| 2 | C + D | 波次 1 两份报告与每条预测完整 |
| 3 | E + F | 波次 2 两份报告与每条预测完整 |
| 4 | G + H | 波次 3 两份报告与每条预测完整 |

单个评测仅在进程异常、NaN/Inf、缺 checkpoint、图片解析失败或结果记录数非 150 时停止；
停止该项并保留其他健康任务。不得因为某类 recall 为 0 而截断该 benchmark 评测，必须
保留 150 条完整结果以保证横向可比。

## S3：统一汇总与审计

1. 仅纳入记录数=150、标签计数正确、checkpoint 哈希已锁定的方案。
2. 主排序：macro F1；并列依次比较 direct F1、indirect recall、clean F1、accuracy。
3. 同时报告 operational-prevalence 加权指标（weighted F1）但不以它替代三类安全门槛。
4. 门槛：三类 recall 均大于 0；direct F1 >= 0.35；macro F1 >= 0.45。该门槛只用于
   候选筛选，不会触发正式训练。
5. 输出 `standard_benchmark_v2_a_h_summary.md`、`summary.json`、中文审计报告和
   A-H 统一对比表；明确标注 benchmark v2 smoke 仅用于回归筛选，full 500 留作最终报告。

## 预计耗时与完成条件

- S0：约 10-20 分钟。
- 单模型完整评测预计约 30-40 分钟；两路并行四个波次约 3-5 小时，取决于图像样本推理速度。
- 完成条件：A-H 每项都有 150 条完整、可复算结果及统一汇总；formal checkpoint 数量仍为 0。
- 本计划完成后只给出候选排序，不启动正式 3000-step 训练；是否进入后续训练仍需用户明确确认。
