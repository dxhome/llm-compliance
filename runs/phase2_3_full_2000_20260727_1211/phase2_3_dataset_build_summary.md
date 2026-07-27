# Phase 2.3 数据集构建摘要

- 运行目录：`C:\work\llm-compliance\runs\phase2_3_full_2000_20260727_1211`
- 原始数据目录：`C:\work\llm-compliance\runs\_datasets\raw`
- 随机种子：42
- 构建结论：通过：训练/验证数据集已按合同生成，来源上限、数量、去重与 quick compare 规模均满足当前计划。

## 生成结果

| 数据集 | 样本数 | clean | direct | indirect | 图像样本 | OCR样本 |
|---|---:|---:|---:|---:|---:|---:|
| train | 3000 | 1000 | 1000 | 1000 | 570 | 500 |
| compare_quick_clean | 50 | 50 | 0 | 0 | 15 | 15 |
| compare_quick_direct | 50 | 0 | 50 | 0 | 0 | 0 |
| compare_quick_indirect | 50 | 0 | 0 | 50 | 28 | 23 |
| compare_quick_all | 150 | 50 | 50 | 50 | 43 | 38 |
| compare_full_clean | 200 | 200 | 0 | 0 | 50 | 50 |
| compare_full_direct | 200 | 0 | 200 | 0 | 0 | 0 |
| compare_full_indirect | 200 | 0 | 0 | 200 | 105 | 85 |
| compare_full_all | 600 | 200 | 200 | 200 | 155 | 135 |
| resume_smoke | 18 | 6 | 6 | 6 | 5 | 4 |

## 训练集来源分布

| 标签 | 来源 | 样本数 | 占该标签比例 |
|---|---|---:|---:|
| clean | cais_mmlu | 70 | 7.0% |
| clean | cyberec_prompt_injection_dataset | 160 | 16.0% |
| clean | deepset_prompt_injections | 80 | 8.0% |
| clean | haonan_li_cmmlu | 100 | 10.0% |
| clean | hlyn_prompt_injection_judge_deberta | 120 | 12.0% |
| clean | microsoft_llmail_inject_challenge | 70 | 7.0% |
| clean | nlphuji_flickr30k | 100 | 10.0% |
| clean | safe_guard_prompt_injection | 150 | 15.0% |
| clean | synthetic_hard_negative | 150 | 15.0% |
| direct | cyberec_prompt_injection_dataset | 180 | 18.0% |
| direct | deepset_prompt_injections | 80 | 8.0% |
| direct | hlyn_prompt_injection_judge_deberta | 120 | 12.0% |
| direct | jailbreakv_28k | 250 | 25.0% |
| direct | lakera_gandalf_ignore_instructions | 80 | 8.0% |
| direct | lakera_mosscap_prompt_injection | 170 | 17.0% |
| direct | safe_guard_prompt_injection | 120 | 12.0% |
| indirect | cyberec_prompt_injection_dataset | 230 | 23.0% |
| indirect | jailbreakv_28k | 70 | 7.0% |
| indirect | microsoft_llmail_inject_challenge | 350 | 35.0% |
| indirect | synthetic_image_injection | 350 | 35.0% |

## 关键校验

- 每个训练标签样本数：clean=1000，direct=1000，indirect=1000。
- compare_quick 已按用户确认调整为每类 50 条，总计 150 条。
- compare_full 每类 200 条，总计 600 条。
- train / compare_quick / compare_full / resume_smoke 之间没有 dedup_key 或 source_record_id 交叉。
- Hlyn 抽样判断：抽样显示 label=1 明显包含注入/越权指令，label=0 为普通或安全相关请求；按合同进行有限纳入。
- 已冻结数据文件 sha256，记录在 `data/phase2_3_dataset_hashes.json`。

## 注意事项

- 本步骤只构建数据集，不启动训练。
- quick compare 只用于训练过程趋势判断；最终验收仍以 full compare 和 best_by_min_class_f1 为准。
- JSON 字段名保持英文，便于后续脚本兼容；人工说明文档使用中文。
