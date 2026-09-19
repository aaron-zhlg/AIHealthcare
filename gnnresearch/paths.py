"""Repo-rooted path helpers for the GNN research agents."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NEUROASD_DIR = REPO_ROOT / "neuroasd"
TRIAL_PY = REPO_ROOT / "autoresearch" / "trial.py"
GATES_PATH = REPO_ROOT / "autoresearch" / "gates.json"
PROGRAM_PATH = REPO_ROOT / "autoresearch" / "program.md"
LEDGER_PATH = REPO_ROOT / "outputs" / "autoresearch" / "ledger.jsonl"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def workspace_path() -> Path:
    override = os.environ.get("GNNRESEARCH_WORKSPACE")
    if override:
        return Path(override)
    return REPO_ROOT / "outputs" / "gnnresearch" / "workspace.json"


def trials_dir() -> Path:
    override = os.environ.get("GNNRESEARCH_TRIALS_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / "outputs" / "autoresearch" / "trials"


def results_dir() -> Path:
    override = os.environ.get("GNNRESEARCH_RESULTS_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / "autoresearch" / "results"


# Back-compat names used by existing modules.
WORKSPACE_PATH = workspace_path()
TRIALS_DIR = trials_dir()
RESULTS_DIR = results_dir()


def _writable_extra() -> Path | None:
    raw = os.environ.get("GNNRESEARCH_WRITE_DIR")
    return Path(raw).resolve() if raw else None


def resolve_repo_path(user_path: str) -> Path:
    """Resolve a user-supplied path and reject escapes from the repo."""
    raw = Path(user_path)
    path = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    extra = _writable_extra()
    if extra and path.is_relative_to(extra):
        return path
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"path escapes the repository: {user_path}")
    if ".git" in path.parts:
        raise ValueError("refusing to touch .git")
    return path


def is_writable(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    resolved = path.resolve()
    extra = _writable_extra()
    if extra and resolved.is_relative_to(extra):
        return True
    if resolved == TRIAL_PY.resolve():
        return True
    return resolved.is_relative_to(NEUROASD_DIR.resolve())


def is_readable(path: Path) -> bool:
    if path.name in {".env", "credentials.json"}:
        return False
    extra = _writable_extra()
    if extra and path.resolve().is_relative_to(extra):
        return True
    return path.is_relative_to(REPO_ROOT) and ".git" not in path.parts


def rel(path: Path) -> str:
    extra = _writable_extra()
    resolved = path.resolve()
    if extra and resolved.is_relative_to(extra):
        return str(resolved)
    return str(resolved.relative_to(REPO_ROOT))
