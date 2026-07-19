"""Render a compact plain-text Phase 2.2 progress check."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def _read_tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    content = [
        line for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if line.strip()
    ]
    return content[-lines:]


def _fmt_seconds(value: Any) -> str:
    try:
        seconds = int(round(float(value)))
    except Exception:
        return "\u4e0d\u53ef\u7528"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}\u5c0f\u65f6{m}\u5206{s}\u79d2"
    if m:
        return f"{m}\u5206{s}\u79d2"
    return f"{s}\u79d2"


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _phase_kind(step_id: str, step_name: str) -> str:
    token = f"{step_id} {step_name}".lower()
    if "smoke" in token:
        return "smoke"
    if "eval" in token:
        return "eval"
    if "compare" in token:
        return "compare"
    if "package" in token:
        return "package"
    if "offline" in token:
        return "offline_smoke"
    if "train" in token:
        return "train"
    return "other"


def _extract_train_metrics(lines: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "steps": [],
        "latest_step": None,
        "latest_total_steps": None,
        "latest_loss": None,
        "latest_eta_seconds": None,
    }
    pattern = re.compile(
        r"step\s+(?P<step>\d+)/(?P<total>\d+).*?loss=(?P<loss>[0-9.]+).*?ETA=(?P<eta>[0-9a-zA-Z.]+)"
    )
    for line in lines:
        match = pattern.search(line)
        if not match:
            continue
        step = int(match.group("step"))
        total = int(match.group("total"))
        loss = _safe_float(match.group("loss"))
        eta_token = match.group("eta")
        eta_seconds = None if eta_token.lower().startswith("nan") else _safe_float(eta_token)
        metrics["steps"].append(
            {
                "step": step,
                "total": total,
                "loss": loss,
                "eta_seconds": eta_seconds,
            }
        )
    if metrics["steps"]:
        latest = metrics["steps"][-1]
        metrics["latest_step"] = latest["step"]
        metrics["latest_total_steps"] = latest["total"]
        metrics["latest_loss"] = latest["loss"]
        metrics["latest_eta_seconds"] = latest["eta_seconds"]
    return metrics


def _loss_trend(metrics: dict[str, Any]) -> str:
    steps = metrics.get("steps", [])
    if len(steps) < 2:
        return "\u4fe1\u53f7\u4e0d\u8db3"
    recent = steps[-4:]
    losses = [item["loss"] for item in recent if item["loss"] is not None]
    if len(losses) < 2:
        return "\u4fe1\u53f7\u4e0d\u8db3"
    delta = losses[-1] - losses[0]
    if delta < -0.2:
        return "\u6574\u4f53\u5728\u6536\u655b"
    if delta > 0.2:
        return "\u8fd1\u671f\u6709\u6ce2\u52a8"
    return "\u57fa\u672c\u6301\u5e73\u6216\u7f13\u6162\u53d8\u597d"


def _training_judgement(metrics: dict[str, Any]) -> str:
    latest_loss = metrics.get("latest_loss")
    if latest_loss is None:
        return "\u5f53\u524d\u8fd8\u770b\u4e0d\u5230\u8db3\u591f\u7684\u8bad\u7ec3\u6307\u6807"
    trend = _loss_trend(metrics)
    if trend == "\u6574\u4f53\u5728\u6536\u655b":
        return "loss\u5728\u4e0b\u964d\uff0c\u76ee\u524d\u770b\u8d77\u6765\u5c5e\u4e8e\u6b63\u5e38\u6536\u655b"
    if trend == "\u8fd1\u671f\u6709\u6ce2\u52a8":
        return "loss\u8fd1\u671f\u6709\u4e00\u5b9a\u6ce2\u52a8\uff0c\u4f46\u8fd8\u9700\u7ee7\u7eed\u89c2\u5bdf"
    return "loss\u76ee\u524d\u6ca1\u770b\u5230\u660e\u663e\u5f02\u5e38"


def _contains_real_error(stderr_tail: list[str]) -> bool:
    blob = "\n".join(stderr_tail).lower()
    return any(token in blob for token in ("traceback", "error:", "exception"))


def _health_verdict(status: dict[str, Any], stderr_tail: list[str], stdout_tail: list[str]) -> tuple[str, str]:
    if status.get("status") == "failed" or status.get("step_status") == "failed":
        return "\u9700\u8981\u4ecb\u5165", "run\u6216\u5f53\u524d\u9636\u6bb5\u5df2\u5931\u8d25"
    if _contains_real_error(stderr_tail):
        return "\u9700\u8981\u4ecb\u5165", "stderr\u51fa\u73b0\u4e86\u660e\u786e\u62a5\u9519"
    if status.get("status") == "completed":
        return "\u6301\u7eed\u89c2\u5bdf", "run\u663e\u793a\u5df2\u5b8c\u6210\uff0c\u9700\u786e\u8ba4\u662f\u5426\u7b26\u5408\u9884\u671f"
    if not stdout_tail and not status.get("message"):
        return "\u6301\u7eed\u89c2\u5bdf", "\u5f53\u524d\u9636\u6bb5\u8fd8\u5728running\uff0c\u4f46\u6700\u65b0\u8f93\u51fa\u4fe1\u53f7\u504f\u5c11"
    return "\u5065\u5eb7", "\u5f53\u524d\u9636\u6bb5\u4ecd\u5728\u63a8\u8fdb\uff0c\u53ea\u770b\u5230\u4f4e\u98ce\u9669\u544a\u8b66"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a periodic Phase 2.2 progress check")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tail-lines", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = parse_args()
    run_dir = args.run_dir.resolve()
    status_path = run_dir / "status.json"
    if not status_path.exists():
        raise SystemExit(f"status.json not found: {status_path}")

    status = _read_json(status_path)
    stdout_path = Path(status.get("stdout", ""))
    stderr_path = Path(status.get("stderr", ""))
    stdout_tail = _read_tail(stdout_path, args.tail_lines)
    stderr_tail = _read_tail(stderr_path, args.tail_lines)
    full_stdout = _read_tail(stdout_path, 200)

    step_id = str(status.get("step_id", "unknown"))
    step_name = str(status.get("step_name", "unknown"))
    phase_kind = _phase_kind(step_id, step_name)
    verdict, verdict_reason = _health_verdict(status, stderr_tail, stdout_tail)

    started_at = None
    if status.get("log"):
        started_file = Path(status.get("log") + ".started")
        if started_file.exists():
            started_at = _parse_timestamp(started_file.read_text(encoding="utf-8-sig", errors="replace"))

    phase_elapsed_seconds = (datetime.now() - started_at).total_seconds() if started_at else _safe_float(status.get("elapsed_seconds"))
    phase_estimate_seconds = _safe_float(status.get("estimate_seconds"))
    phase_remaining_seconds = None
    if phase_estimate_seconds is not None and phase_elapsed_seconds is not None:
        phase_remaining_seconds = max(phase_estimate_seconds - phase_elapsed_seconds, 0.0)

    train_metrics = _extract_train_metrics(full_stdout)
    latest_step = train_metrics.get("latest_step")
    latest_total = train_metrics.get("latest_total_steps")
    latest_loss = train_metrics.get("latest_loss")
    latest_eta = train_metrics.get("latest_eta_seconds")

    print("\u5de1\u68c0\u65f6\u95f4\uff1a" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("\u603b\u4f53\u72b6\u6001\uff1a" + str(status.get("status", "unknown")))
    print("\u5f53\u524d\u9636\u6bb5\uff1a" + f"{step_id} | {step_name}")
    print("\u9636\u6bb5\u72b6\u6001\uff1a" + str(status.get("step_status", "unknown")))
    print("\u5065\u5eb7\u7ed3\u8bba\uff1a" + verdict)
    print("")

    print("\u5f53\u524d\u60c5\u51b5")
    print("- \u672c\u9636\u6bb5\u5df2\u8fd0\u884c\uff1a" + _fmt_seconds(phase_elapsed_seconds))
    print("- \u672c\u9636\u6bb5\u9884\u8ba1\u603b\u65f6\u957f\uff1a" + _fmt_seconds(phase_estimate_seconds))
    print("- \u672c\u9636\u6bb5\u9884\u8ba1\u5269\u4f59\uff1a" + _fmt_seconds(phase_remaining_seconds))
    print("- \u7b80\u8981\u5224\u65ad\uff1a" + verdict_reason)
    print("")

    print("\u9636\u6bb5\u5206\u6790")
    if phase_kind in {"train", "smoke"}:
        progress = f"{latest_step}/{latest_total}" if latest_step is not None and latest_total is not None else "\u6682\u4e0d\u53ef\u7528"
        loss_text = f"{latest_loss:.4f}" if latest_loss is not None else "\u6682\u4e0d\u53ef\u7528"
        eta_text = _fmt_seconds(latest_eta) if latest_eta is not None else "\u6682\u4e0d\u53ef\u7528"
        print("- \u8bad\u7ec3\u8fdb\u5ea6\uff1a" + progress)
        print("- \u6700\u65b0loss\uff1a" + loss_text)
        print("- loss\u8d8b\u52bf\uff1a" + _loss_trend(train_metrics))
        print("- \u8bad\u7ec3\u5224\u65ad\uff1a" + _training_judgement(train_metrics))
        print("- ETA\uff1a" + eta_text)
    elif phase_kind in {"eval", "compare"}:
        print("- \u9636\u6bb5\u7c7b\u578b\uff1aeval/compare")
        print("- \u5224\u65ad\uff1a\u76ee\u524d\u4ec5\u505a\u9ad8\u5c42\u68c0\u67e5\uff0c\u65e0\u660e\u663e\u5f02\u5e38")
    else:
        print("- \u9636\u6bb5\u7c7b\u578b\uff1a" + phase_kind)
        print("- \u5224\u65ad\uff1a\u76ee\u524d\u4ec5\u505a\u9ad8\u5c42\u68c0\u67e5\uff0c\u65e0\u660e\u663e\u5f02\u5e38")
    print("")

    print("\u5173\u952e\u4fe1\u606f")
    print("- \u62a5\u9519\u4fe1\u53f7\uff1a" + ("\u5b58\u5728\u660e\u786e\u62a5\u9519" if _contains_real_error(stderr_tail) else "\u672a\u89c1\u660e\u786e\u62a5\u9519"))
    print("")

    print("\u4e0b\u4e00\u6b65\u5efa\u8bae")
    if verdict == "\u9700\u8981\u4ecb\u5165":
        print("- \u7acb\u5373\u67e5\u5f53\u524d\u9636\u6bb5\u65e5\u5fd7\u548cstderr\uff0c\u5148\u5b9a\u4f4d\u62a5\u9519")
    elif verdict == "\u6301\u7eed\u89c2\u5bdf":
        print("- \u4e0b\u4e00\u6b21\u5de1\u68c0\u7ee7\u7eed\u786e\u8ba4\u662f\u5426\u4ecd\u5728\u63a8\u8fdb\uff0c\u6682\u4e0d\u8981\u91cd\u542f")
    else:
        print("- \u7ee7\u7eed\u8fd0\u884c\uff0c\u4e0b\u4e00\u6b21\u5de1\u68c0\u91cd\u70b9\u770b\u8fdb\u5ea6\u548closs\u6536\u655b\u60c5\u51b5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
