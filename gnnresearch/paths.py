"""Repo-rooted path helpers for the GNN research agents."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NEUROASD_DIR = REPO_ROOT / "neuroasd"
TRIAL_PY = REPO_ROOT / "autoresearch" / "trial.py"
GATES_PATH = REPO_ROOT / "autoresearch" / "gates.json"
PROGRAM_PATH = REPO_ROOT / "autoresearch" / "program.md"
LEDGER_PATH = REPO_ROOT / "outputs" / "autoresearch" / "ledger.jsonl"
TRIALS_DIR = REPO_ROOT / "outputs" / "autoresearch" / "trials"
RESULTS_DIR = REPO_ROOT / "autoresearch" / "results"
WORKSPACE_PATH = REPO_ROOT / "outputs" / "gnnresearch" / "workspace.json"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Coder may edit the training / evaluation path. Nothing else.
_WRITABLE_FILES = frozenset({TRIAL_PY.resolve()})


def resolve_repo_path(user_path: str) -> Path:
    """Resolve a user-supplied path and reject escapes from the repo."""
    raw = Path(user_path)
    path = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"path escapes the repository: {user_path}")
    if ".git" in path.parts:
        raise ValueError("refusing to touch .git")
    return path


def is_writable(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    if path.resolve() in _WRITABLE_FILES:
        return True
    return path.resolve().is_relative_to(NEUROASD_DIR.resolve())


def is_readable(path: Path) -> bool:
    if path.name in {".env", "credentials.json"}:
        return False
    return path.is_relative_to(REPO_ROOT) and ".git" not in path.parts


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))
