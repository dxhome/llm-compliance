@echo off
cd /d C:\work\llm-compliance
start "F4 smoke" /b "C:\work\llm-compliance\.venv\Scripts\python.exe" -X utf8 -u scripts\train.py --config runs\phase2_3_full_3000_20260729_0958\configs\smoke_strategy_f4.yaml > runs\phase2_3_full_3000_20260729_0958\logs\smoke_strategy_f4_train.log 2> runs\phase2_3_full_3000_20260729_0958\logs\smoke_strategy_f4_train.err.log
