"""Local session state for the write → run → insight loop."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from autoresearch.loop.paths import (
    archive_failed_diff,
    revert_coder_files,
    snapshot_coder_files,
    workspace_path,
)

STATUSES = (
    "idle",
    "coder_failed",
    "lint_failed",
    "needs_screen",
    "needs_loso_subset",
    "needs_loso_full",
    "awaiting_new_code",
    "promoted",
    "target_reached",
)

STAGE_FOR_STATUS = {
    "needs_screen": "screen",
    "needs_loso_subset": "loso-subset",
    "needs_loso_full": "loso-full",
}

NEXT_STATUS_ON_PASS = {
    "screen": "needs_loso_subset",
    "loso-subset": "needs_loso_full",
}


def default_workspace() -> dict[str, Any]:
    return {
        "iteration": 0,
        "status": "idle",
        "hypothesis": "",
        "current_name": "",
        "files_changed": [],
        "last_results": {},
        "last_insight": None,
        "coder_ok": None,
        "coder_error": "",
        "lint_ok": None,
        "lint_report": None,
        "history": [],
        "last_win": None,
        "ruled_out": [],
    }


def _record_ruled_out(data: dict[str, Any], entry: dict[str, Any]) -> None:
    """Append or merge a rejected mechanism. Same trial name updates in place."""
    items = list(data.get("ruled_out") or [])
    name = entry.get("name")
    merged = {key: value for key, value in entry.items() if value not in (None, "")}
    if name:
        for index in range(len(items) - 1, -1, -1):
            if items[index].get("name") == name:
                items[index] = {**items[index], **merged}
                data["ruled_out"] = items
                return
    items.append(merged)
    data["ruled_out"] = items


def load_workspace() -> dict[str, Any]:
    path = workspace_path()
    if not path.exists():
        return default_workspace()
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = default_workspace()
    merged.update(data)
    return merged


def save_workspace(data: dict[str, Any]) -> None:
    path = workspace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def target_accuracy() -> float:
    raw = os.environ.get("AUTORESEARCH_LOOP_TARGET_ACC", "0.80")
    try:
        return float(raw)
    except ValueError:
        return 0.80


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:32] or "change"


def required_stage(workspace: dict[str, Any] | None = None) -> str | None:
    data = workspace if workspace is not None else load_workspace()
    return STAGE_FOR_STATUS.get(str(data.get("status") or "idle"))


def note_code_change(path: str, hypothesis: str = "") -> dict[str, Any]:
    """Record a coder edit. Starts a new iteration unless one is already open."""
    data = load_workspace()
    status = data.get("status") or "idle"
    opening = status in {
        "idle",
        "awaiting_new_code",
        "promoted",
        "lint_failed",
        "coder_failed",
    }
    if opening:
        data["iteration"] = int(data.get("iteration") or 0) + 1
        data["last_results"] = {}
        data["files_changed"] = []
        if hypothesis.strip():
            data["hypothesis"] = hypothesis.strip()
        elif not data.get("hypothesis"):
            data["hypothesis"] = "code change"
        data["current_name"] = f"iter{data['iteration']}_{slugify(data['hypothesis'])}"
    elif hypothesis.strip():
        data["hypothesis"] = hypothesis.strip()
    if path not in data["files_changed"]:
        data["files_changed"].append(path)
    data["status"] = "needs_screen"
    data["coder_ok"] = False
    data["lint_ok"] = False
    data["lint_report"] = None
    save_workspace(data)
    return data


def mark_coder_outcome(ok: bool, error: str = "") -> dict[str, Any]:
    """Record whether the coder instance finished. Failures must not be measured."""
    data = load_workspace()
    data["coder_ok"] = bool(ok)
    data["coder_error"] = error
    if not ok:
        data["status"] = "coder_failed"
    elif data.get("status") == "coder_failed" and data.get("files_changed"):
        data["status"] = "needs_screen"
    save_workspace(data)
    return data


def coder_finished_cleanly(workspace: dict[str, Any] | None = None) -> bool:
    data = workspace if workspace is not None else load_workspace()
    return data.get("coder_ok") is True


def lint_passed(workspace: dict[str, Any] | None = None) -> bool:
    data = workspace if workspace is not None else load_workspace()
    return data.get("lint_ok") is True


def record_lint(report: dict[str, Any]) -> dict[str, Any]:
    """Store the linter verdict. FAIL sends the loop back to the coder."""
    data = load_workspace()
    passed = bool(report.get("passed"))
    data["lint_ok"] = passed
    data["lint_report"] = report
    if passed:
        if data.get("files_changed") and data.get("coder_ok"):
            data["status"] = "needs_screen"
    else:
        data["status"] = "lint_failed"
        data["last_insight"] = {
            "source": "linter",
            "passed": False,
            "errors": report.get("errors") or [],
            "next_code_change": "fix the lint errors, then stop. Do not start a new mechanism.",
            "narrative": "Lint FAIL. Code is not qualified to train.\n"
            + "\n".join(str(item) for item in (report.get("errors") or [])),
        }
    save_workspace(data)
    return data


def apply_trial_outcome(
    stage: str,
    name: str,
    verdict: dict[str, Any],
    summary: dict[str, Any],
    protocol_failed: bool,
) -> dict[str, Any]:
    data = load_workspace()
    gate_passed = bool(verdict.get("passed")) and not protocol_failed
    auc = summary.get("auc_mean")
    acc = summary.get("accuracy_mean")
    last_win = data.get("last_win")
    improved = True
    if gate_passed and stage == "loso-full" and isinstance(last_win, dict):
        prev_auc = last_win.get("auc_mean")
        if prev_auc is not None and auc is not None and float(auc) <= float(prev_auc):
            improved = False
    won = gate_passed and improved
    data.setdefault("last_results", {})[stage] = {
        "name": name,
        "passed": won,
        "auc_mean": auc,
        "accuracy_mean": acc,
        "margin": verdict.get("margin"),
        "protocol_failed": protocol_failed,
        "did_not_beat_last_win": gate_passed and not improved,
    }
    if not won:
        files = list(data.get("files_changed") or [])
        archived = archive_failed_diff(name, files)
        _record_ruled_out(
            data,
            {
                "name": name,
                "stage": stage,
                "hypothesis": data.get("hypothesis") or "",
                "auc_mean": auc,
                "accuracy_mean": acc,
                "margin": verdict.get("margin"),
                "patch": archived,
            },
        )
        data["status"] = "awaiting_new_code"
        data["coder_ok"] = None
        data["lint_ok"] = None
        data["lint_report"] = None
        data["archived"] = archived
        data["reverted"] = revert_coder_files(files)
    elif stage == "loso-full":
        data["last_win"] = {
            "name": name,
            "auc_mean": auc,
            "accuracy_mean": acc,
        }
        data["snapshotted"] = snapshot_coder_files(list(data.get("files_changed") or []))
        data["reverted"] = []
        data["coder_ok"] = None
        data["lint_ok"] = None
        data["lint_report"] = None
        if acc is not None and float(acc) >= target_accuracy():
            data["status"] = "target_reached"
        else:
            data["status"] = "awaiting_new_code"
    else:
        data["status"] = NEXT_STATUS_ON_PASS[stage]
        data["reverted"] = []
    data["history"] = list(data.get("history") or []) + [
        {
            "name": name,
            "stage": stage,
            "passed": data["last_results"][stage]["passed"],
            "auc_mean": auc,
            "accuracy_mean": acc,
        }
    ]
    save_workspace(data)
    return data


def next_role(workspace: dict[str, Any] | None = None) -> str | None:
    """Who must run next. ``None`` means accuracy target reached."""
    data = workspace if workspace is not None else load_workspace()
    status = str(data.get("status") or "idle")
    if status in {"target_reached", "promoted"}:
        return None
    if status == "coder_failed" or data.get("coder_ok") is False:
        return "coder"
    if status == "lint_failed":
        return "coder"
    if data.get("coder_ok") is True and data.get("lint_ok") is not True:
        return "linter"
    if status in STAGE_FOR_STATUS and data.get("lint_ok") is True:
        return "experimenter"
    return "coder"


def save_insight(insight: dict[str, Any]) -> dict[str, Any]:
    data = load_workspace()
    data["last_insight"] = insight
    if insight.get("passed") is False and insight.get("source") != "linter":
        _record_ruled_out(
            data,
            {
                "name": insight.get("name") or data.get("current_name"),
                "stage": insight.get("stage"),
                "summary": str(insight.get("ruled_out") or "")[:400],
            },
        )
    save_workspace(data)
    return data
