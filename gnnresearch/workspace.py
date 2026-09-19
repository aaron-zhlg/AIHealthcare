"""Local session state for the write → run → insight loop."""

from __future__ import annotations

import json
import re
from typing import Any

from gnnresearch.paths import workspace_path

STATUSES = (
    "idle",
    "needs_screen",
    "needs_loso_subset",
    "needs_loso_full",
    "awaiting_new_code",
    "promoted",
)

STAGE_FOR_STATUS = {
    "needs_screen": "screen",
    "needs_loso_subset": "loso-subset",
    "needs_loso_full": "loso-full",
}

NEXT_STATUS_ON_PASS = {
    "screen": "needs_loso_subset",
    "loso-subset": "needs_loso_full",
    "loso-full": "promoted",
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
        "history": [],
    }


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
    opening = status in {"idle", "awaiting_new_code", "promoted"}
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
    data.setdefault("last_results", {})[stage] = {
        "name": name,
        "passed": bool(verdict.get("passed")) and not protocol_failed,
        "auc_mean": summary.get("auc_mean"),
        "margin": verdict.get("margin"),
        "protocol_failed": protocol_failed,
    }
    if protocol_failed or not verdict.get("passed"):
        data["status"] = "awaiting_new_code"
    else:
        data["status"] = NEXT_STATUS_ON_PASS[stage]
    data["history"] = list(data.get("history") or []) + [
        {
            "name": name,
            "stage": stage,
            "passed": data["last_results"][stage]["passed"],
            "auc_mean": summary.get("auc_mean"),
        }
    ]
    save_workspace(data)
    return data


def next_role(workspace: dict[str, Any] | None = None) -> str | None:
    """Who must run next. ``None`` means the loop is done (loso-full PASS)."""
    status = str(
        (workspace if workspace is not None else load_workspace()).get("status") or "idle"
    )
    if status in STAGE_FOR_STATUS:
        return "experimenter"
    if status == "promoted":
        return None
    return "coder"


def save_insight(insight: dict[str, Any]) -> dict[str, Any]:
    data = load_workspace()
    data["last_insight"] = insight
    save_workspace(data)
    return data
