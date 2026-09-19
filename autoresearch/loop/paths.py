"""Repo-rooted path helpers for the GNN research agents."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NEUROASD_DIR = REPO_ROOT / "neuroasd"
TRIAL_PY = REPO_ROOT / "autoresearch" / "trial.py"
GATES_PATH = REPO_ROOT / "autoresearch" / "gates.json"
PROGRAM_PATH = REPO_ROOT / "autoresearch" / "program.md"
LEDGER_PATH = REPO_ROOT / "outputs" / "autoresearch" / "ledger.jsonl"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _env_path(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def workspace_path() -> Path:
    override = _env_path("AUTORESEARCH_LOOP_WORKSPACE")
    if override:
        return Path(override)
    return REPO_ROOT / "outputs" / "autoresearch" / "loop" / "workspace.json"


def trials_dir() -> Path:
    override = _env_path("AUTORESEARCH_LOOP_TRIALS_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / "outputs" / "autoresearch" / "trials"


def results_dir() -> Path:
    override = _env_path("AUTORESEARCH_LOOP_RESULTS_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / "autoresearch" / "results"


# Back-compat names used by existing modules.
WORKSPACE_PATH = workspace_path()
TRIALS_DIR = trials_dir()
RESULTS_DIR = results_dir()


def _writable_extra() -> Path | None:
    raw = os.environ.get("AUTORESEARCH_LOOP_WRITE_DIR")
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


def baseline_dir() -> Path:
    return workspace_path().parent / "baseline"


def clear_baseline() -> None:
    import shutil

    path = baseline_dir()
    if path.exists():
        shutil.rmtree(path)


def _baseline_copy_path(path: Path) -> Path:
    extra = _writable_extra()
    if extra and path.is_relative_to(extra):
        return baseline_dir() / "_scratch" / path.relative_to(extra)
    return baseline_dir() / rel(path)


def snapshot_coder_files(rel_paths: list[str]) -> list[str]:
    """Keep a copy of the last loso-full winner. Later FAILs revert to this, not main."""
    saved: list[str] = []
    for user_path in rel_paths:
        try:
            path = resolve_repo_path(user_path)
        except ValueError:
            continue
        if not path.is_file():
            continue
        dest = _baseline_copy_path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        saved.append(rel(path))
    return saved


def revert_coder_files(rel_paths: list[str]) -> list[str]:
    """Restore coder-writable files after a FAIL.

    Prefer the last winning snapshot. If none, tracked files go back to HEAD and
    new files are deleted.
    """
    import subprocess

    reverted: list[str] = []
    extra = _writable_extra()
    for user_path in rel_paths:
        try:
            path = resolve_repo_path(user_path)
        except ValueError:
            continue
        snap = _baseline_copy_path(path)
        if snap.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(snap.read_bytes())
            reverted.append(rel(path))
            continue
        if extra and path.is_relative_to(extra):
            if path.is_file():
                path.unlink()
                reverted.append(rel(path))
            continue
        if not is_writable(path):
            continue
        rel_str = rel(path)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel_str],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if tracked.returncode == 0:
            subprocess.run(
                ["git", "checkout", "HEAD", "--", rel_str],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
            )
            reverted.append(rel_str)
        elif path.is_file():
            path.unlink()
            reverted.append(rel_str)
    return reverted


def rel(path: Path) -> str:
    extra = _writable_extra()
    resolved = path.resolve()
    if extra and resolved.is_relative_to(extra):
        return str(resolved)
    return str(resolved.relative_to(REPO_ROOT))
