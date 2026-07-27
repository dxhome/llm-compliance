# Phase 2.3 执行日志：phase2_3_full_2000_20260727_1211

- [2026-07-27 12:11] 创建 Phase 2.3 run 目录。
- [2026-07-27 12:11] 初始化 run 内目录：`configs/`、`data/`、`artifacts/`、`logs/`、`scripts/`。
- [2026-07-27 12:11] 完成 T2.22 数据审计，并将审计报告放到 run 根目录。
- [2026-07-27 12:11] 在 `phase2_3_data_audit.md` 中补充完整 raw/basic 数据集可用性检查。
- [2026-07-27 12:11] 将结构化原始数据可用性快照写入 `artifacts/raw_data_availability.json`。
- [2026-07-27 12:20] 基于最新完整 `runs/_datasets/raw` basic 数据池完成 T2.23 数据集契约。
- [2026-07-27 12:20] 在 `phase2_3_dataset_contract.md` 与 `data/phase2_3_dataset_manifest.json` 中冻结 T2.41 数据源纳入、标签映射、人工复核和 split 隔离规则。
- [2026-07-27 12:30] 将本 run 已生成 Markdown 文档统一改为中文；JSON 文件保持机器可读字段不变。
- [2026-07-27 12:35] 用户确认 quick compare 调整为每类 50 条；LLMail/JailbreakV/Hlyn/Cyberec/source cap 等第 2 步关键决策已确认。
- [2026-07-27 12:58] 完成 T2.24-T2.25 数据集构建：训练集 3000 条，每类 1000；quick compare 每类 50；full compare 每类 200；resume smoke 每类 6。
- [2026-07-27 12:58] 生成并冻结 `data/phase2_3_dataset_hashes.json`，确认 train / compare_quick / compare_full / resume_smoke 之间无 `dedup_key` 或 `source_record_id` 交叉。
- [2026-07-27 12:58] 生成中文构建摘要 `phase2_3_dataset_build_summary.md`，记录来源分布、C6B 兼容字段覆盖和 Hlyn 标签抽样判断。
- [2026-07-27 14:21] 完成 T2.26 训练配置与执行决策冻结：生成模板配置、本 run 训练配置、resume smoke 配置和中文决策说明。
- [2026-07-27 14:21] 冻结首轮训练为单组强配置：2000 steps、LoRA r=32/alpha=64/dropout=0.10、lr=1e-4、每 5 step log、每 50 step checkpoint、10% quick compare、50%/100% full compare、`best_by_min_class_f1` 选优。
- [2026-07-27 14:21] `scripts/check_phase2_readiness.py` 对 `configs/train.yaml` 与 `configs/resume_smoke.yaml` 预检均通过；下一步进入 T2.36 resume smoke。
