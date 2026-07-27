# Phase 2.3 数据集契约

## 范围

- Run：`phase2_3_full_2000_20260727_1211`
- 基线审计报告：`phase2_3_data_audit.md`
- 机器可读 manifest：`data/phase2_3_dataset_manifest.json`
- 原始数据可用性快照：`artifacts/raw_data_availability.json`
- 状态：契约已冻结，可进入实现；尚未构建 train/eval 记录。

## 核心决策

Phase 2.3 必须使用 `runs/_datasets/raw` 下最新完整 raw/basic 数据池，不能只沿用 Phase 2.2 balanced 600 的数据。Phase 2.2 的失败主要来自采样过窄，尤其是 indirect 采样过窄，不是因为原始数据不足。

默认训练目标至少 3000 条记录：

| split | clean | direct | indirect | total |
|---|---:|---:|---:|---:|
| train | >=1000 | >=1000 | >=1000 | >=3000 |
| compare_quick | 50 | 50 | 50 | 150 |
| compare_full | 200 | 200 | 200 | 600 |
| resume_smoke | 6 | 6 | 6 | 18 |

quick compare 和 full compare 必须互斥，并且都不能与 train 重叠。

用户已确认：quick compare 每类 50 条即可；quick 仅用于训练期趋势观察和异常发现，最终验收与 `best_by_min_class_f1` 仍以 full compare 为准。

## 数据源纳入规则

| source | 决策 | 标签映射 | 主要用途 |
|---|---|---|---|
| `deepset_prompt_injections` | 纳入 | 0 -> clean，1 -> direct | clean/direct |
| `safe_guard_prompt_injection` | 纳入 | 0 -> clean，1 -> direct | clean/direct |
| `jailbreakv_28k` | 按 format 纳入 | Template/Persuade/Logic -> direct；figstep -> indirect；SD/SD_typo/typo -> manual review | direct 和 figstep indirect |
| `microsoft_llmail_inject_challenge` | 纳入 | attack_attempt=True -> indirect；False -> clean；Unclear -> excluded | email/tool-use indirect、clean email false positive |
| `cyberec_prompt_injection_dataset` | 去重后纳入 | label 0 -> clean；label 1 默认 direct；RAG/agent 类别可作为 indirect 候选 | direct、clean hard negative、indirect-text 候选 |
| `hlyn_prompt_injection_judge_deberta` | 抽样 spot check 后有限纳入 | 0 -> clean，1 -> direct | clean/direct 扩容 |
| `lakera_gandalf_ignore_instructions` | 有限纳入 | 全部作为 direct | instruction-ignore direct |
| `lakera_mosscap_prompt_injection` | 有限纳入 | Level 1-8 全部作为 direct | hard direct |
| `nlphuji_flickr30k` | 纳入 | caption 作为 clean | clean caption、image clean 候选 |
| `cais_mmlu` | 有限纳入 | question 作为 clean | 英文 clean |
| `haonan_li_cmmlu` | 有限纳入 | question 作为 clean | 中文 clean |
| `synthetic_image_injection` | 生成并纳入 | 生成攻击样本作为 indirect | OCR/image indirect |
| `synthetic_hard_negative` | 生成并纳入 | 生成 benign OCR/image 样本作为 clean | clean FPR 防护 |

## 标签组成目标

Clean 应覆盖 text-only clean、image/OCR hard negative、benign email false positive 和中文 clean 记录。

Direct 应覆盖 JailbreakV template、公开二分类注入数据、Lakera/Cyberec hard direct 和多语言攻击。

Indirect 必须覆盖 figstep、synthetic OCR/image injection、LLMail email/tool-use injection，以及少量 indirect-text/RAG/agent 切片。

除非在 `phase2_3_run_decisions.md` 中单独批准，否则任何单一 source 在同一 label 的 train 中占比不应超过 35%。

## 标准化记录字段

每条标准化记录必须包含：

- `id`
- `text`
- `label`
- `source`
- `split`
- `dedup_key`

每条标准化记录还应携带以下 C6B 兼容字段；不适用时允许为空，但 key 需要存在：

- `image`
- `lang`
- `template`
- `attack_family`
- `indirect_subtype`
- `has_image`
- `ocr_text`
- `ocr_confidence`
- `ocr_present`
- `text_image_consistency_label`
- `cross_modal_attack_type`
- `has_instruction_override`
- `hard_negative_type`
- `source_record_id`
- `source_license_status`
- `metadata`

## 去重与 split 隔离

- 基础 `dedup_key` 使用 `sha256(normalized_text + '|' + normalized_image_id + '|' + source_family)`。
- 标准化规则包括 Unicode NFKC、空白折叠、仅用于去重的小写化、本地绝对路径剥离、JSON caption 字符串归一化。
- 同一个 `dedup_key` 不能跨 train、quick compare、full compare。
- 同一个 `source_record_id` 不能跨 train、quick compare、full compare。
- 近重复候选必须进入同一个 split。
- manual review 和 excluded 记录不得进入 train。

## 明确排除与人工复核

- LLMail `Unclear` 记录在明确策略前排除。
- JailbreakV `SD`、`SD_typo`、`typo` 在 loader 验证图文语义前进入人工复核，不直接入 train。
- WildJailbreak、BIPIA、TensorTrust 不在默认 basic 下载池中，保留为 manual review。
- Hlyn 需要先 spot check 标签语义，再有限抽样。
- Cyberec 必须先确定 canonical variant 或按 group 去重后再采样。

## Step 3 待办

- 按本契约实现 loaders。
- 生成足够的 synthetic image injection 和 synthetic hard-negative 样本来满足 source quota。
- 对 Hlyn 标签语义做 spot check。
- 决定 Cyberec 的 canonical variant 或 group-level dedup 策略。
- 在数据构建 summary 中冻结最终 per-source quota。
