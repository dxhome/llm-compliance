# 项目执行与验证归档

> **状态**：已完成
>
> **最后整理**：2026-09-08
>
> **用途**：合并原 `tasks.md` 的执行计划与 `VERIFICATION.md` 的验收记录。本文保留项目做了什么、如何验证、如何最小复现以及哪些边界仍然存在；最终能力结论以 [final-report.md](final-report.md) 为准，机制与代码级细节以 [reference.md](reference.md) 为准。

---

## 1. 交付概览

项目已完成一个面向离线场景的多模态提示注入检测 pipeline。最终推荐方案为 **F-3000-MCR-SBC**：以 SmolVLM-500M、LoRA 和 MPID 三分类头为模型核心，组合 C5 文本规则、C6B-lite 本地 OCR、C6A 兼容性检查、MCR 不可信上下文路由、SBC 分数校准和 C4 高置信 clean 放行。

在冻结的 Standard Benchmark v2/full500 上，完整 pipeline 取得 Accuracy **62.00%**、Macro F1 **59.28%**、Weighted F1 **61.32%**；`clean/direct/indirect` F1 分别为 **68.38% / 53.56% / 55.90%**。三类 Recall 均非零，并满足项目预设的 Direct F1 >= 35% 和 Macro F1 >= 45% 门槛。

| 交付物 | 状态 | 证据或位置 |
|---|---|---|
| 离线三分类模型与运行时策略 | 已完成 | `checkpoint_step_3000` + F-3000-MCR-SBC 固定策略 |
| C4/C5/C6 与优化 pipeline | 已完成 | `src/mpid/infer/pipeline.py` 及对应单元测试 |
| 本地 OCR 与上下文隔离 | 已完成 | C6B-lite + MCR；不使用 benchmark OCR 标注 |
| 训练、smoke、盲测与审计材料 | 已完成 | `runs/phase2_3_full_3000_20260729_0958/`（本地运行资产） |
| 可移动离线 artifact | 已完成 | `runs/_artifact/F-3000-MCR-SBC/`（不进入 Git） |
| 最终研究报告 | 已完成 | [final-report.md](final-report.md) |

## 2. 执行路线与验收结论

| 阶段 | 完成内容 | 验收结论 |
|---|---|---|
| 环境与基础设施 | Python 包、设备抽象、离线模型/数据准备和跨平台脚本 | 已验证 macOS 与 Windows CPU 的可运行性；正式长训转至 x86 Windows |
| 数据与威胁模型 | `clean/direct/indirect` 三分类 schema、公开数据集整合、图像注入样本与 EDA | 已完成；早期类别失衡问题已记录并在后续训练中修正 |
| Phase 2.2 | Full-500、Balanced-600 等 LoRA 训练与评测 | 已完成；确认数据分布会显著影响 indirect 边界 |
| Full-3000 策略筛选 | A-H、F2-F4 共 11 个 smoke 候选统一横评 | 已完成；F 是唯一保留三类非零 Recall 且最平衡的机制方向 |
| F 正式训练 | 3,000 条均衡样本、成组三元组 batch、可恢复 checkpoint | 已完成并固化 `checkpoint_step_3000` |
| 最终运行时策略 | C5/C6A/C6B-lite/MCR/SBC/C4 组合与阈值锁定 | 已完成；仅使用 smoke 选型，V2/full500 作为最终盲验收 |
| 离线交付验证 | checksum、断网 smoke、ZIP CRC、性能测量 | 已完成；artifact 可移动验证通过 |

## 3. 最小复现与验证

### 3.1 环境

项目要求 Python 3.10+。先按所在平台创建并激活虚拟环境，然后安装项目和依赖。

**Windows PowerShell**：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-x86.txt
python -m pip install -e .
```

**macOS**：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

模型、数据集、checkpoint 与离线包均是本地大文件，刻意不进入 Git。需要完整模型推理时，请从受控存储恢复 `runs/_models/`、相关 run 的数据/配置和 `runs/_artifact/F-3000-MCR-SBC/`；不要将它们重新提交到仓库。

### 3.2 不依赖模型资产的快速验证

下列命令可在未恢复数据集与 checkpoint 时验证核心规则和 pipeline 编排。

```powershell
python -m pytest tests -q
python scripts/infer_pipeline_light.py --text "ignore previous instructions" --probs 0.34,0.33,0.33
python scripts/infer_pipeline_light.py --text "summarize this note" --probs 0.96,0.02,0.02
```

第一条命令运行单元测试；第二条应走 C5 并返回 `direct / c5_rules`；第三条模拟高置信 clean head，预期返回 `clean / c4_early_exit`。这些 smoke 使用模拟概率，不代表模型能力评测。

### 3.3 完整交付验证

恢复离线 artifact 后，以其中随包的 README、`CHECKSUMS.txt` 和 smoke 脚本为准完成完整性与断网验证。最终评测数字、文件清单和性能口径见 [final-report.md](final-report.md) 的第 4、5 节；训练参数和运行流程见 [reference.md](reference.md) 的 Full-3000 与离线交付章节。

## 4. 关键决策与问题复盘

| 事项 | 影响 | 最终处理 |
|---|---|---|
| macOS 内存与训练速度受限 | 初始训练过慢，无法承担长时间全量实验 | 保留 macOS 用于环境/smoke，转至 x86 Windows CPU 完成长训；脚本和文档按平台分流 |
| 早期 Full-500 数据不均衡 | 约 70% 为 direct，clean 和 indirect 过少，模型对 indirect 判别弱 | 构造均衡数据并重新训练；最终 Full-3000 使用每类 1,000 条训练样本 |
| 训练中计算机重启 | Full-600 训练无法从中断处恢复，造成重复训练成本 | 在 Full-3000 增加 checkpoint、optimizer 和 RNG sidecar 保存与恢复流程 |
| 单纯类别权重或蒸馏不足 | A-C smoke 出现 clean 偏置；D/E 又丢失 direct | 改为反事实三元组和对角排序，并用 A-H、F2-F4 统一 smoke 筛选 |
| 外部图中文字可能被当作指令 | 单纯把 OCR 文本拼入 prompt 会模糊权限边界 | 使用本地 OCR + MCR 的 `untrusted_image_ocr` 受限角色，不使用 benchmark 标注 |
| MCR 后仍有类别偏置 | 直接类分数偏高，indirect 边界不足 | 在 smoke 中锁定 SBC：Indirect logit `+0.55`，MCR 激活时 Direct logit `-0.20`，并禁止在 full500 上再调参 |

## 5. 验收边界与后续工作

- 最终结论仅适用于锁定的 checkpoint、运行时策略与 Standard Benchmark v2/full500，不能推断为对未知攻击的无条件防护能力。
- 本项目是离线防御研究原型，不包括云端安全服务、高并发生产部署、攻击自动化或更大模型横评。
- C5/C6 的未命中不代表安全；只有全部前置关卡通过且最终判定为 clean 时才放行。
- 进一步工作应优先扩展真实图像注入、多语言与分布外测试集，并在新的独立测试集上重新验证校准策略。

## 6. 文档职责

| 文档 | 职责 |
|---|---|
| [final-report.md](final-report.md) | 最终方案、量化结果、结论与交付证据 |
| [opening-report-vlm.md](opening-report-vlm.md) | 当前 VLM 路线的开题材料 |
| [opening-report-formal.md](opening-report-formal.md) | 级联方案开题对照材料 |
| [opening-report-reference.md](opening-report-reference.md) | 历史开题参考存档 |
| [reference.md](reference.md) | 概念、代码解读、平台命令、历史实验与复现细节 |
| 本文 | 已完成任务、验收结论、最小验证与遗留边界 |
