# 文档目录

本目录汇集本项目的全部正式文档。

## 文档清单

| 文档 | 用途 | 读者 |
|---|---|---|
| [opening-report-formal.md](opening-report-formal.md) | **课题开题报告（正式版）**：背景、现状、目标、方法（含 §3.6 算法优化三方案）、创新点、预期成果、局限、参考文献 | 评审专家、导师、答辩组（开题提交）|
| [opening-report-vlm.md](opening-report-vlm.md) | **课题开题报告（独立版 v0.3·轻量级 VLM 专用）**：基于 **SmolVLM-500M** 单一 VLM 的离线多模态提示注入检测；**算法优化为核心研究内容**（**C4 早退机制** / **C5 规则前置过滤** / **C6 跨模态自检**）；强调离线 AI 趋势、defense-in-depth 多层防御与独立自洽 | 探索"轻量级 VLM 端到端 + 算法优化"方向的读者 |
| [opening-report-reference.md](opening-report-reference.md) | **课题开题报告（参考存档·历史版本）**：早期详尽版，**不维护，仅供查阅完整背景与方法论** | 备查 |
| [tasks.md](tasks.md) | **任务分解**：与 VLM 开题报告（v0.3）严格对应；分阶段任务（含 Phase 0A 准备阶段、Phase 3/4/5 三项算法优化 C4/C5/C6、Phase 6 攻防基线评测、Phase 7 项目整理），依赖图、风险缓冲 | 项目执行者 |
| [README.md](README.md) | 本文件：文档目录与版本说明 | 所有读者 |

## 文档定位

- **`opening-report-formal.md`（v1.2）**：当前**权威**开题报告，正式格式，范围与作者当前水平匹配，含 §3.6 三项算法优化（早退机制 / 零宽字符检测 / 跨模态矛盾检测）。**所有任务实施以本文件为准**。
- **`opening-report-reference.md`（v0.2）**：早期详尽版本（v0.2），包含双方案（级联式 + 端到端）、backbone-agnostic 框架设计、7B 迁移示例等更丰富的方法论。**仅作为参考存档，不作为主交付物，不再主动维护**。
- **`tasks.md`（v1.0）**：根据 `opening-report-formal.md` 的 C1–C8 研究内容拆解的执行计划，删除端到端方案与 7B 迁移相关任务。

> **权威源**：`opening-report-formal.md` → `tasks.md`。
> **可选参考**：`opening-report-reference.md`（历史背景）。

## 阅读顺序建议

- **开题提交 / 答辩汇报**：用 `opening-report-formal.md`；
- **查阅历史背景与详尽方法论**：用 `opening-report-reference.md`；
- **项目执行 / 任务分配**：用 `tasks.md`；
- **了解本目录结构**：用 `README.md`（本文件）。

## 文档版本

| 文档 | 版本 | 最后更新 |
|---|---|---|
| opening-report-formal.md | v1.2 | 2026-07-13 |
| opening-report-vlm.md | v0.3（独立自洽·轻量级 VLM 专用） | 2026-07-13 |
| opening-report-reference.md | v0.2（参考存档） | 2026-07-13 |
| tasks.md | v2.0（对齐 vlm 开题报告 v0.3） | 2026-07-13 |
| README.md | v1.0 | 2026-07-13 |
