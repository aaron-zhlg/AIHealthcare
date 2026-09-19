"""Promote a loso-full PASS: freeze results and open a review branch.

Commits only the training / evaluation diff plus the frozen experiment folder.
Never merges to main and never edits gates.json.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

from gnnresearch.paths import EXPERIMENTS_DIR, REPO_ROOT, RESULTS_DIR, rel
from gnnresearch.workspace import load_workspace


def _git(*args: str) -> str:
    ran = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if ran.returncode != 0:
        raise RuntimeError(ran.stderr.strip() or ran.stdout.strip() or "git failed")
    return ran.stdout.strip()


def _changed_promotable_files() -> list[Path]:
    raw = _git("status", "--porcelain")
    paths: list[Path] = []
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        rel_path = line[3:].strip()
        if " -> " in rel_path:
            rel_path = rel_path.split(" -> ", 1)[1]
        path = (REPO_ROOT / rel_path).resolve()
        if path.is_relative_to((REPO_ROOT / "neuroasd").resolve()) or path == (
            REPO_ROOT / "autoresearch" / "trial.py"
        ).resolve():
            paths.append(path)
    return paths


def write_experiment_freeze(result: dict[str, Any]) -> Path:
    """Write experiments/<name>_v1/ so the number can be reproduced later."""
    name = str(result["name"])
    folder = EXPERIMENTS_DIR / f"{name}_v1"
    folder.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    config = result["config"]
    (folder / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    run_config = {
        "name": f"{name}_v1",
        "validation": result["stage"],
        "model_selection": result.get(
            "model_selection", "final epoch (no selection on the evaluation set)"
        ),
        "data": {
            "dataset": "ABIDE Preprocessed rois_ho",
            "num_subjects": 884,
            "num_rois": 111,
            "fc_method": "Pearson correlation",
            "num_sites": 20,
        },
        "hyperparameters": config,
        "reproduce": (
            f"uv run python -m autoresearch.trial --name {name} "
            f"--stage {result['stage']}"
        ),
    }
    (folder / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    workspace = load_workspace()
    hypothesis = workspace.get("hypothesis") or result.get("note") or ""
    files = ", ".join(workspace.get("files_changed") or []) or "(see git diff)"
    readme = f"""# {name} v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** {date.today().isoformat()}
> **Hypothesis:** {hypothesis}
> **Files changed:** {files}

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f} |
| **AUC** | **{summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}** |
| F1 | {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f} |

Diagnostic best-epoch AUC (not reported): {summary.get('best_auc_mean', 'n/a')}.

## Reproduce

```bash
{run_config['reproduce']}
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
"""
    (folder / "README.md").write_text(readme, encoding="utf-8")
    return folder


def promote(result: dict[str, Any], *, push: bool) -> dict[str, Any]:
    """Copy the score, freeze experiments/, commit a narrow trial branch."""
    name = str(result["name"])
    stage = str(result["stage"])
    if stage != "loso-full":
        raise ValueError("promotion is only allowed after a loso-full PASS")

    staged_result = RESULTS_DIR / f"{stage}__{name}" / "result.json"
    staged_result.parent.mkdir(parents=True, exist_ok=True)
    staged_result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    experiment_dir = write_experiment_freeze(result)

    to_add = {rel(path) for path in _changed_promotable_files()}
    to_add.add(rel(staged_result))
    to_add.add(rel(experiment_dir / "results.json"))
    to_add.add(rel(experiment_dir / "run_config.json"))
    to_add.add(rel(experiment_dir / "README.md"))

    branch = f"experiment/trial-{name}"
    current = _git("rev-parse", "--abbrev-ref", "HEAD")
    if current != branch:
        existing = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=REPO_ROOT,
        )
        if existing.returncode == 0:
            raise RuntimeError(f"branch {branch} already exists; review it by hand")
        _git("checkout", "-b", branch)

    _git("add", "--", *sorted(to_add))
    note = result.get("note") or load_workspace().get("hypothesis") or "no note"
    _git("commit", "-m", f"Trial {name} ({stage}): {note}")

    pushed = False
    push_error = ""
    remotes = _git("remote")
    if push and "origin" in remotes.split():
        try:
            _git("push", "-u", "origin", branch)
            pushed = True
        except RuntimeError as exc:
            push_error = str(exc)

    return {
        "branch": branch,
        "files": sorted(to_add),
        "experiment_dir": rel(experiment_dir),
        "pushed": pushed,
        "push_error": push_error,
        "merged_to_main": False,
        "gates_updated": False,
        "note": "Human review required before merging to main or raising gates.",
    }
