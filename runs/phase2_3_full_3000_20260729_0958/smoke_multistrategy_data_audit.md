# 多策略 Smoke 数据审计

## 结论

数据集已通过数量与类别覆盖审计，可用于 A/B/C 策略的统一比较。

## 训练集

- 文件：`data/smoke_train_300.jsonl`
- 总数：300
- clean/direct/indirect：各 100
- paired counterfactual：120 条，三类各 40 条，占训练集 40%。
- 所有记录使用 `trusted_boundary_v2`，并从正式 3,000 条训练集中确定性抽取。

## 验证集

- `smoke_mixed_90.jsonl`：三类各 30 条，用于 step 100、200、400 的统一比较。
- `smoke_quick_150.jsonl`：三类各 50 条，用于最终 quick 比较。
- `smoke_direct_replay.jsonl`：30 条 direct，用于 direct 保留检查。

## 隔离说明

Smoke 训练记录来自 `train.jsonl`，验证记录来自原有的独立 compare split；构建过程使用固定随机种子 43。训练、验证不复用同一来源记录或文本去重键。

## 当前限制

OCR、邮件/工具、网页/RAG 的 indirect 子类切片尚需在策略筛选结束后从完整 compare split 生成，用于 S4 最终验证，不参与 A/B/C 的统一初筛。
