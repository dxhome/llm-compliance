# 文档目录

本目录汇集本项目的全部正式文档。

## 当前运行目录约定

本项目的本地执行状态统一收敛到顶层 `runs/` 目录。旧的顶层本地目录 `configs/`、`data/`、`models/`、`artifacts/`、`logs/` 不应再新建或继续作为执行入口。

- `runs/<run_id>/` 表示一次独立端到端执行，包含该次执行自己的 `configs/`、`data/`、`artifacts/`、`logs/`、`scripts/`、`execution_plan.*` 和 `execution_log.md`。
- 具体 run 目录必须带时间后缀，例如 `runs/phase2_2_balanced_600_20260718_1955/`。
- `runs/_datasets/`、`runs/_models/`、`runs/_templates/`、`runs/_manual/` 是共享的本地数据、模型、模板和手动输出目录。
- 整个 `runs/` 已加入 `.gitignore`；其中的 run-local 配置、数据、checkpoint、日志、离线包和 launcher 都是本地执行资产，不进入 git。
- 顶层 `scripts/` 只保留通用脚本；具体执行的 PowerShell launcher 放在 `runs/<run_id>/scripts/`，通用模板入口为 `scripts/run_phase2_workflow.ps1`。

## 文档清单

| 文档 | 用途 | 读者 |
|---|---|---|
| [opening-report-vlm.md](opening-report-vlm.md) | **课题开题报告（独立版 v0.3·轻量级 VLM 专用）**：基于 **SmolVLM-500M** 单一 VLM 的离线多模态提示注入检测；**算法优化为核心研究内容**（**C4 早退机制** / **C5 规则前置过滤** / **C6 跨模态自检**）；强调离线 AI 趋势、defense-in-depth 多层防御与独立自洽 | 评审专家、导师、答辩组（开题提交）|
| [reference.md](reference.md) | **项目参考手册（FAQ 速查 v4.0）**：Phase 0A/0/1/2 详解、**核心概念速查（2.A 威胁模型 / 2.B 数据集构造 / 2.C EDA / 2.D Macro F1）**、框架 vs 能力辨析、7 步端到端校验、Phase 2.2 可复制执行流程、Windows/macOS 分流命令、常见坑、术语表、速查卡片 | 项目作者（开发/执行时快速回看） |
| [tasks.md](tasks.md) | **任务分解**：与 VLM 开题报告（v0.3）严格对应；分阶段任务（含 Phase 0A 准备阶段、Phase 3/4/5 三项算法优化 C4/C5/C6、Phase 6 攻防基线评测、Phase 7 项目整理），依赖图、风险缓冲 | 项目执行者 |
| [opening-report-formal.md](opening-report-formal.md) | **课题开题报告（级联方案·v1.2）**：OCR + XLM-RoBERTa + CLIP 的级联式检测路线，作为 **VLM 方案的对照参考** | 备查、对照 |
| [opening-report-reference.md](opening-report-reference.md) | **课题开题报告（参考存档·历史版本）**：早期详尽版，**不维护，仅供查阅完整背景与方法论** | 备查 |
| [VERIFICATION.md](VERIFICATION.md) | **验收报告**：各 Phase 验收清单与执行记录 | 项目审计 |
| [README.md](README.md) | 本文件：文档目录与版本说明 | 所有读者 |

> **威胁模型**：原独立文件 `threat_model.md`（C1 产出）已合并到 [reference.md](reference.md) § 2.1。

## 文档定位

- **`opening-report-vlm.md`（v0.3）**：当前**权威**开题报告，面向"轻量级 VLM 端到端 + 算法优化"路线，所有任务实施以本文件为准。
- **`reference.md`（v4.0）**：执行期间的**速查手册**——Phase 2 细节、框架/能力区分、Macro F1 解读、Phase 2.2 可复制执行流程、Windows/macOS 分流命令、常见坑等都集中在此，不用每次去翻长篇文档。
- **`opening-report-formal.md`（v1.2）**：级联式方案的开题报告，作为**VLM 方案的对照参考**，保留但不再作为主交付物。
- **`opening-report-reference.md`（v0.2）**：历史参考存档，不维护。
- **`tasks.md`（v2.0）**：与 `opening-report-vlm.md` v0.3 严格对应的执行计划。

> **权威源**：`opening-report-vlm.md` → `tasks.md`。
> **执行速查**：`reference.md`（开发/调试时随时翻）。

## 阅读顺序建议

- **开题提交 / 答辩汇报**：用 `opening-report-vlm.md`；
- **开发执行 / 调试 bug**：用 `reference.md`（看 Phase 2 详解、常见坑、Macro F1 解读）；
- **项目执行 / 任务分配**：用 `tasks.md`；
- **查阅历史背景与级联方案对照**：用 `opening-report-reference.md` / `opening-report-formal.md`；
- **了解本目录结构**：用 `README.md`（本文件）。

## 文档版本

| 文档 | 版本 | 最后更新 |
|---|---|---|
| opening-report-vlm.md | v0.3（独立自洽·轻量级 VLM 专用） | 2026-07-13 |
| reference.md | v4.1（FAQ 速查·统一 runs 执行目录结构） | 2026-07-18 |
| tasks.md | v2.0（对齐 vlm 开题报告 v0.3） | 2026-07-13 |
| opening-report-formal.md | v1.2（级联方案·对照参考） | 2026-07-13 |
| opening-report-reference.md | v0.2（参考存档） | 2026-07-13 |
| VERIFICATION.md | v1.0 | 2026-07-13 |
| README.md | v1.3 | 2026-07-18 |
