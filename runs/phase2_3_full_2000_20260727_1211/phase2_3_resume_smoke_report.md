# Phase 2.3 第 4 步 Resume Smoke 报告

- 运行目录：`C:\work\llm-compliance\runs\phase2_3_full_2000_20260727_1211`
- 总耗时：403.1 秒
- 总结论：通过

## 执行过程

- 第一段：跑 10 step，returncode=0，耗时 265.1 秒。
- checkpoint 发现：latest.safetensors -> `C:\work\llm-compliance\runs\phase2_3_full_2000_20260727_1211\artifacts\resume_smoke\latest.safetensors`
- 第二段：从 checkpoint 恢复后再跑 5 step，returncode=0，耗时 138.0 秒。

## 验收检查

- first_leg_returncode_zero：通过
- checkpoint_step_10_exists：通过
- latest_exists：通过
- checkpoint_discovered：通过
- resume_leg_returncode_zero：通过
- resume_started_after_step_10：通过
- resume_reached_step_15：通过
- final_checkpoint_exists：通过
- optimizer_state_sidecar_exists：通过
- optimizer_state_loaded：通过
- probe_step10_returncode_zero：通过
- probe_latest_step10_returncode_zero：通过
- probe_resume_final_returncode_zero：通过
- probe_latest_matches_step10：通过
- probe_final_loss_not_higher_than_step10：通过
- logs_are_separate：通过
- total_under_1h：通过

## Loss 证据

- 训练日志窗口 loss：断点=1.9748，恢复段=2.1579。
- 说明：训练窗口 loss 来自不同 batch，仅作为观察值；严格恢复验收使用固定 probe loss。
- 固定 probe loss：checkpoint_step_10=2.141803388380342。
- 固定 probe loss：latest@step10=2.141803388380342。
- 固定 probe loss：resume_final=2.103751758320464。
- 结论：step10 checkpoint 与 latest checkpoint 在固定集合上完全一致；恢复后 final checkpoint 的固定集合 loss 更低。

## 关键证据

- 第一段 global step：[1, 5, 10]
- 第二段 global step：[15]
- 第一段日志：`C:\work\llm-compliance\runs\phase2_3_full_2000_20260727_1211\logs\resume_smoke_first_leg.log`
- 第二段日志：`C:\work\llm-compliance\runs\phase2_3_full_2000_20260727_1211\logs\resume_smoke_resume_leg.log`

## 下一步

- 可以继续进入训练期 compare / launcher 调度实现；正式长训练前仍需保证该 launcher 使用同一套 checkpoint/resume 机制。
