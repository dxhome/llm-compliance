#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RUN_DIR/../.." && pwd)"

cd "$REPO_ROOT"

if command -v pwsh >/dev/null 2>&1; then
  POWERSHELL_BIN=pwsh
elif command -v powershell.exe >/dev/null 2>&1; then
  POWERSHELL_BIN=powershell.exe
else
  echo "[launch] PowerShell is required for the canonical run workflow." >&2
  exit 1
fi

"$POWERSHELL_BIN" -NoProfile -ExecutionPolicy Bypass \
  -File "$REPO_ROOT/scripts/run_phase2_workflow.ps1" \
  -RunName "phase2_2_full_500" \
  -RunDir "$RUN_DIR" \
  -Config "$RUN_DIR/configs/train.yaml" \
  -SmokeConfig "$RUN_DIR/configs/smoke.yaml"
