# Phase 2.3 Full 3000 数据集构建摘要

- 结论：通过。训练集使用结构化可信边界输入，不将来源或人工标签作为模型输入。
- 训练集：clean/direct/indirect 各 1,000 条，共 3,000 条。
- 验证集：quick 与 full 均同时生成单类切片和三类混合集。

## 训练内容角色

- benign_external_content: 60
- benign_image_content: 150
- direct_user_request: 1000
- trusted_user_request: 790
- untrusted_external_content: 650
- untrusted_image_ocr: 350

## 数据集规模

| 数据集 | 样本数 | clean | direct | indirect |
|---|---:|---:|---:|---:|
| train | 3000 | 1000 | 1000 | 1000 |
| compare_quick_clean | 50 | 50 | 0 | 0 |
| compare_quick_direct | 50 | 0 | 50 | 0 |
| compare_quick_indirect | 50 | 0 | 0 | 50 |
| compare_quick_all | 150 | 50 | 50 | 50 |
| compare_full_clean | 200 | 200 | 0 | 0 |
| compare_full_direct | 200 | 0 | 200 | 0 |
| compare_full_indirect | 200 | 0 | 0 | 200 |
| compare_full_all | 600 | 200 | 200 | 200 |
| resume_smoke | 18 | 6 | 6 | 6 |
