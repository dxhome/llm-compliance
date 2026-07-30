"""Run the Phase 2.3 checkpoint/resume smoke validation.

This runner intentionally performs a tiny two-leg training sequence:

1. train 10 logical steps and require checkpoint_step_10 + latest;
2. resume from the discovered latest checkpoint and train 5 more steps;
3. write a Chinese Markdown report and a JSON report under the run directory.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_and_log(args: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write("\n" + "=" * 80 + "\n")
        log.write("COMMAND: " + " ".join(args) + "\n")
        log.write("=" * 80 + "\n")
        proc = subprocess.run(
            args,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    return {
        "returncode": proc.returncode,
        "seconds": time.perf_counter() - start,
        "log": str(log_path),
    }


def _run_json(args: list[str], log_path: Path, json_path: Path) -> dict[str, Any]:
    run_result = _run_and_log(args + ["--json-out", str(json_path)], log_path)
    if run_result["returncode"] != 0 or not json_path.exists():
        return {
            "returncode": run_result["returncode"],
            "seconds": run_result["seconds"],
            "log": run_result["log"],
            "json": str(json_path),
            "avg_loss": None,
        }
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data.update({
        "returncode": run_result["returncode"],
        "seconds": run_result["seconds"],
        "log": run_result["log"],
        "json": str(json_path),
    })
    return data


def _latest_step_checkpoint(out_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in out_dir.glob("checkpoint_step_*.safetensors"):
        match = re.search(r"checkpoint_step_(\d+)\.safetensors$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def _discover_checkpoint(out_dir: Path) -> tuple[Path | None, str]:
    latest = out_dir / "latest.safetensors"
    if latest.exists():
        return latest, "latest.safetensors"
    step = _latest_step_checkpoint(out_dir)
    if step:
        return step, "checkpoint_step_*.safetensors"
    compat = out_dir / "lora_phase2_3_resume_smoke_latest.safetensors"
    if compat.exists():
        return compat, "compat_partial_name"
    return None, "missing"


def _extract_global_steps(log_path: Path) -> list[int]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return [int(m.group(1)) for m in re.finditer(r"\(global\s+(\d+)/", text)]


def _extract_avg_losses(log_path: Path) -> list[float]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return [float(m.group(1)) for m in re.finditer(r"\bloss=([0-9.]+)", text)]


def _clean_previous_smoke(run_dir: Path, out_dir: Path) -> None:
    for path in [
        run_dir / "logs" / "resume_smoke_first_leg.log",
        run_dir / "logs" / "resume_smoke_resume_leg.log",
        run_dir / "phase2_3_resume_smoke_report.md",
    ]:
        if path.exists():
            path.unlink()
    for pattern in [
        "*.safetensors",
        "*.state.pt",
        "train_summary.json",
        "phase2_3_resume_smoke_report.json",
    ]:
        for path in out_dir.glob(pattern):
            path.unlink()


def _render_md(report: dict[str, Any]) -> str:
    checks = report["checks"]
    lines = [
        "# Phase 2.3 第 4 步 Resume Smoke 报告",
        "",
        f"- 运行目录：`{report['run_dir']}`",
        f"- 总耗时：{report['total_seconds']:.1f} 秒",
        f"- 总结论：{report['verdict']}",
        "",
        "## 执行过程",
        "",
        f"- 第一段：跑 10 step，returncode={report['first_leg']['returncode']}，耗时 {report['first_leg']['seconds']:.1f} 秒。",
        f"- checkpoint 发现：{report['discovery_method']} -> `{report['discovered_checkpoint']}`",
        f"- 第二段：从 checkpoint 恢复后再跑 5 step，returncode={report['resume_leg']['returncode']}，耗时 {report['resume_leg']['seconds']:.1f} 秒。",
        "",
        "## 验收检查",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {name}：{'通过' if value else '未通过'}")
    lines.extend([
        "",
        "## 关键证据",
        "",
        f"- 第一段 global step：{report['first_leg_global_steps']}",
        f"- 第二段 global step：{report['resume_leg_global_steps']}",
        f"- 第一段 loss 日志：{report['first_leg_losses']}",
        f"- 第二段 loss 日志：{report['resume_leg_losses']}",
        f"- 断点 loss / 恢复 loss：{report['checkpoint_loss']} / {report['resume_loss']}",
        f"- 固定 probe loss：step10={report['probe_step10'].get('avg_loss')}，latest@step10={report['probe_latest_step10'].get('avg_loss')}，resume_final={report['probe_resume_final'].get('avg_loss')}",
        f"- 第一段日志：`{report['first_leg']['log']}`",
        f"- 第二段日志：`{report['resume_leg']['log']}`",
        "",
        "## 下一步",
        "",
        "- 如果本报告通过，可以进入正式训练前的训练期 compare / launcher 调度实现与最终 preflight。",
        "- 如果未通过，需要先修复失败项，不能启动正式长训练。",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "configs" / "resume_smoke.yaml"
    cfg = _load_yaml(config_path)
    config_dir = config_path.parent
    training = cfg.get("training", {}) or {}
    smoke = cfg.get("resume_smoke", {}) or {}
    out_dir = _resolve(cfg["io"]["out_dir"], config_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_previous_smoke(run_dir, out_dir)

    first_steps = int(smoke.get("first_leg_steps", 10))
    resume_steps = int(smoke.get("resume_leg_steps", 5))
    save_every = int(training.get("save_every", 10))
    partial_name = str(training.get("partial_name", "lora_phase2_3_resume_smoke_latest.safetensors"))
    checkpoint_name = str(training.get("checkpoint_name", "lora_phase2_3_resume_smoke.safetensors"))

    logs_dir = run_dir / "logs"
    first_log = logs_dir / "resume_smoke_first_leg.log"
    resume_log = logs_dir / "resume_smoke_resume_leg.log"

    python = str((REPO_ROOT / ".venv" / "Scripts" / "python.exe").resolve())
    if not Path(python).exists():
        python = sys.executable

    first_cmd = [
        python,
        "-u",
        "scripts/train.py",
        "--config",
        str(config_path),
        "--max-train-steps",
        str(first_steps),
        "--checkpoint-name",
        checkpoint_name,
        "--save-every",
        str(save_every),
        "--partial-name",
        partial_name,
    ]
    first = _run_and_log(first_cmd, first_log)
    discovered, method = _discover_checkpoint(out_dir)
    probe_json_dir = out_dir / "probe_loss"
    step10_ckpt = out_dir / f"checkpoint_step_{first_steps}.safetensors"
    smoke_jsonl = _resolve(cfg["io"]["train_jsonl"], config_dir)
    probe_step10 = {"avg_loss": None, "returncode": 99}
    probe_latest_step10 = {"avg_loss": None, "returncode": 99}
    if first["returncode"] == 0 and step10_ckpt.exists() and discovered is not None:
        base_probe_cmd = [
            python,
            "-u",
            "scripts/probe_checkpoint_loss.py",
            "--config",
            str(config_path),
            "--jsonl",
            str(smoke_jsonl),
            "--max-records",
            "18",
        ]
        probe_step10 = _run_json(
            base_probe_cmd + ["--checkpoint", str(step10_ckpt)],
            logs_dir / "resume_smoke_probe_step10.log",
            probe_json_dir / "probe_step10.json",
        )
        probe_latest_step10 = _run_json(
            base_probe_cmd + ["--checkpoint", str(discovered)],
            logs_dir / "resume_smoke_probe_latest_step10.log",
            probe_json_dir / "probe_latest_step10.json",
        )

    resume: dict[str, Any]
    if first["returncode"] == 0 and discovered is not None:
        resume_cmd = [
            python,
            "-u",
            "scripts/train.py",
            "--config",
            str(config_path),
            "--resume-from",
            str(discovered),
            "--resume-global-step",
            str(first_steps),
            "--skip-train-batches",
            str(first_steps),
            "--max-train-steps",
            str(resume_steps),
            "--checkpoint-name",
            checkpoint_name,
            "--save-every",
            str(save_every),
            "--partial-name",
            partial_name,
        ]
        resume = _run_and_log(resume_cmd, resume_log)
    else:
        resume = {
            "returncode": 99,
            "seconds": 0.0,
            "log": str(resume_log),
            "skipped_reason": "first leg failed or checkpoint discovery failed",
        }
    probe_resume_final = {"avg_loss": None, "returncode": 99}
    final_ckpt = out_dir / checkpoint_name
    if resume["returncode"] == 0 and final_ckpt.exists():
        probe_resume_final = _run_json(
            [
                python,
                "-u",
                "scripts/probe_checkpoint_loss.py",
                "--config",
                str(config_path),
                "--jsonl",
                str(smoke_jsonl),
                "--max-records",
                "18",
                "--checkpoint",
                str(final_ckpt),
            ],
            logs_dir / "resume_smoke_probe_final.log",
            probe_json_dir / "probe_resume_final.json",
        )

    first_steps_seen = _extract_global_steps(first_log)
    resume_steps_seen = _extract_global_steps(resume_log)
    first_losses = _extract_avg_losses(first_log)
    resume_losses = _extract_avg_losses(resume_log)
    checkpoint_loss = first_losses[-1] if first_losses else None
    resume_loss = resume_losses[-1] if resume_losses else None
    checks = {
        "first_leg_returncode_zero": first["returncode"] == 0,
        "checkpoint_step_10_exists": (out_dir / f"checkpoint_step_{first_steps}.safetensors").exists(),
        "latest_exists": (out_dir / "latest.safetensors").exists(),
        "checkpoint_discovered": discovered is not None,
        "resume_leg_returncode_zero": resume["returncode"] == 0,
        "resume_started_after_step_10": bool(resume_steps_seen) and min(resume_steps_seen) >= first_steps + 1,
        "resume_reached_step_15": bool(resume_steps_seen) and max(resume_steps_seen) >= first_steps + resume_steps,
        "final_checkpoint_exists": (out_dir / checkpoint_name).exists(),
        "optimizer_state_sidecar_exists": (out_dir / "latest.state.pt").exists(),
        "optimizer_state_loaded": "optimizer state resumed" in resume_log.read_text(encoding="utf-8", errors="replace") if resume_log.exists() else False,
        "probe_step10_returncode_zero": probe_step10.get("returncode") == 0,
        "probe_latest_step10_returncode_zero": probe_latest_step10.get("returncode") == 0,
        "probe_resume_final_returncode_zero": probe_resume_final.get("returncode") == 0,
        "probe_latest_matches_step10": (
            probe_step10.get("avg_loss") is not None
            and probe_latest_step10.get("avg_loss") is not None
            and abs(float(probe_step10["avg_loss"]) - float(probe_latest_step10["avg_loss"])) <= 1e-5
        ),
        "probe_final_loss_not_higher_than_step10": (
            probe_step10.get("avg_loss") is not None
            and probe_resume_final.get("avg_loss") is not None
            and float(probe_resume_final["avg_loss"]) <= float(probe_step10["avg_loss"])
        ),
        "logs_are_separate": first_log.exists() and resume_log.exists() and first_log != resume_log,
        "total_under_1h": first["seconds"] + resume["seconds"] <= 3600,
    }
    verdict = "通过" if all(checks.values()) else "未通过"
    report = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "out_dir": str(out_dir),
        "first_leg": first,
        "resume_leg": resume,
        "discovered_checkpoint": str(discovered) if discovered else None,
        "discovery_method": method,
        "first_leg_global_steps": first_steps_seen,
        "resume_leg_global_steps": resume_steps_seen,
        "first_leg_losses": first_losses,
        "resume_leg_losses": resume_losses,
        "checkpoint_loss": checkpoint_loss,
        "resume_loss": resume_loss,
        "training_window_loss_note": (
            "训练窗口 loss 来自不同 batch，仅作为观察值；严格恢复验收使用固定 probe loss。"
        ),
        "probe_step10": probe_step10,
        "probe_latest_step10": probe_latest_step10,
        "probe_resume_final": probe_resume_final,
        "checks": checks,
        "verdict": verdict,
        "total_seconds": first["seconds"] + resume["seconds"],
    }
    json_path = out_dir / "phase2_3_resume_smoke_report.json"
    md_path = run_dir / "phase2_3_resume_smoke_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps({
        "verdict": report["verdict"],
        "checks": report["checks"],
        "report_json": str(Path(report["out_dir"]) / "phase2_3_resume_smoke_report.json"),
        "report_md": str(Path(report["run_dir"]) / "phase2_3_resume_smoke_report.md"),
    }, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == "通过" else 1


if __name__ == "__main__":
    raise SystemExit(main())
