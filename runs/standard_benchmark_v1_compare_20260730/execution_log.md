
- 2026-07-30 16:18:00: Attempt 1 stopped at LoRA-only 45/300 with launcher exit code -1 and no traceback. Archived to balanced_600_attempt1_partial. Retrying unchanged benchmark/config/checkpoint with launcher stdout/stderr redirected to run-local logs.

- 2026-07-30 16:25:00: Recovery attempt 2 requested by user. Starting unchanged launcher through hidden WScript.Shell.Run to detach from Codex terminal lifecycle.

- 2026-07-30 16:27:00: Recovery launcher failed before model start because runner log redirection preceded creation of the model log directory. Fixed launch.ps1 to create the directory first; final bounded restart uses unchanged evaluation inputs.

- 2026-07-30 16:28:00: Recovery launcher treated a redirected Transformers FutureWarning as a terminating PowerShell error before inference. Scoped native-command stderr handling to Continue and retain exit-code based failure detection. Final bounded recovery restart keeps all evaluation inputs unchanged.

