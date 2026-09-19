"""Coder sub-agent: edits training / model code, one change at a time."""

from __future__ import annotations

from typing import Any

from orchestra import SubAgent

from autoresearch.loop.paths import NEUROASD_DIR, TRIAL_PY, is_writable, rel, resolve_repo_path
from autoresearch.loop.protocol import leak_reasons_in_source
from autoresearch.loop.workspace import load_workspace, note_code_change, required_stage

INSTRUCTIONS = """\
You write code that might improve the ABIDE GNN (ASD vs control). You do not \
run training and you do not judge AUC.

You are a FRESH instance each turn. You cannot see the experimenter conversation. \
If a previous trial exists, you MUST call read_last_insight (and read_workspace) \
before any edit, then implement ONE new change that follows that insight.

Rules:
- Interpret the insight: what was ruled out, what the next_code_change says, \
what not to repeat. Do not retry a ruled-out mechanism.
- Change ONE thing per turn (one mechanism). Keep the first edit tiny: one \
function or a few lines. Do not attempt DANN / multi-file rewrites in one turn.
- Failed edits revert to the last loso-full winner (or HEAD if none). \
Implement the new mechanism on that baseline; do not restack a ruled-out change. \
After a loso-full PASS, add the next mechanism on top of the winning code.
- The experimenter scores `autoresearch/trial.py` (`run_fold`), which imports \
`SimpleGCN` and `train_one_epoch`. If a training-step change is not visible \
there, the trial will not measure it — edit `trial.py` or `gcn.py` accordingly.
- Prefer small, testable edits: class weights, site harmonization, edge thresholding, \
Fisher z, attention pooling, a slightly wider/deeper GCN. 884 subjects will not \
support a new foundation model.
- You MAY edit files under neuroasd/ and autoresearch/trial.py. You may NOT edit \
gates.json, data/, experiments/, autoresearch/loop/, or medresearch/.
- NEVER select a model using the evaluation / held-out set. Do not gate, report, \
or save the "best epoch" on val/test/LOSO as the official score. That leak once \
reported LOSO AUC 0.707 instead of the honest final-epoch 0.623.
- After editing, state the single hypothesis you just implemented.

If the workspace status is needs_loso_subset or needs_loso_full, do not edit: \
that code is frozen until the current change is fully measured.
"""


class CodeTools:
    """File tools restricted to the GNN training path."""

    def __init__(self) -> None:
        self.touched: list[str] = []
        self._read_insight = False

    def as_tools(self) -> list:
        return [
            self.read_workspace,
            self.read_last_insight,
            self.list_trainable_files,
            self.read_file,
            self.replace_in_file,
            self.write_file,
            self.record_hypothesis,
        ]

    def sources(self) -> list[str]:
        return list(self.touched)

    def _track(self, path) -> str:
        name = rel(path)
        if name not in self.touched:
            self.touched.append(name)
        return name

    def read_workspace(self) -> dict[str, Any]:
        """Return session state: status, current hypothesis, files, last insight."""
        workspace = load_workspace()
        if workspace.get("last_insight"):
            self._read_insight = True
        return workspace

    def read_last_insight(self) -> dict[str, Any]:
        """Return the experimenter's latest insight so you can act on it.

        Call this before editing whenever a previous trial exists. This is the
        only memory of the measurement round; you do not inherit that conversation.
        """
        self._read_insight = True
        insight = load_workspace().get("last_insight")
        if not insight:
            return {"ok": True, "last_insight": None, "note": "no prior trial in this session"}
        return {"ok": True, "last_insight": insight}

    def list_trainable_files(self) -> list[str]:
        """List Python files the coder is allowed to edit.

        Returns:
            Paths relative to the repo root.
        """
        files = sorted(rel(path) for path in NEUROASD_DIR.glob("*.py"))
        files.append(rel(TRIAL_PY))
        return files

    def read_file(self, path: str) -> str:
        """Read a text file in this repository.

        Args:
            path: Repo-relative path, e.g. 'neuroasd/gcn.py'.
        """
        target = resolve_repo_path(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        text = target.read_text(encoding="utf-8")
        self._track(target)
        return text

    def record_hypothesis(self, hypothesis: str) -> dict[str, Any]:
        """Set the one-line hypothesis for the change you are about to make.

        Call this before the first edit of a new iteration.

        Args:
            hypothesis: One sentence, one mechanism, e.g. 'class-weighted loss'.
        """
        workspace = load_workspace()
        if required_stage(workspace) in {"loso-subset", "loso-full"}:
            return {
                "ok": False,
                "error": "code is frozen until the current change finishes LOSO",
                "workspace": workspace,
            }
        workspace["hypothesis"] = hypothesis.strip()
        from autoresearch.loop.workspace import save_workspace

        save_workspace(workspace)
        return {"ok": True, "hypothesis": hypothesis.strip()}

    def _guard_write(self, target, content: str) -> None:
        if not is_writable(target):
            raise PermissionError(
                f"not writable: {rel(target)}. Only neuroasd/*.py and "
                "autoresearch/trial.py may be edited."
            )
        reasons = leak_reasons_in_source(content)
        if reasons:
            raise ValueError(
                "refusing edit that reintroduces best-epoch evaluation leak: "
                + "; ".join(reasons)
            )
        workspace = load_workspace()
        if required_stage(workspace) in {"loso-subset", "loso-full"}:
            raise PermissionError(
                "code is frozen (status "
                f"{workspace.get('status')}); wait for the experimenter to finish"
            )
        if workspace.get("last_insight") and not self._read_insight:
            raise PermissionError(
                "call read_last_insight before editing so the next change "
                "follows the experimenter's write-up"
            )

    def replace_in_file(self, path: str, old: str, new: str) -> dict[str, Any]:
        """Replace one unique substring in a writable training file.

        Args:
            path: Repo-relative path to edit.
            old: Exact text to find (must occur once).
            new: Replacement text.
        """
        target = resolve_repo_path(path)
        current = target.read_text(encoding="utf-8")
        count = current.count(old)
        if count != 1:
            raise ValueError(f"old text occurs {count} times; need exactly one match")
        updated = current.replace(old, new, 1)
        self._guard_write(target, updated)
        target.write_text(updated, encoding="utf-8")
        workspace = note_code_change(rel(target))
        return {
            "ok": True,
            "path": self._track(target),
            "iteration": workspace["iteration"],
            "name": workspace["current_name"],
            "status": workspace["status"],
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Overwrite a writable training file with new contents.

        Prefer replace_in_file for small edits.

        Args:
            path: Repo-relative path to write.
            content: Full new file text.
        """
        target = resolve_repo_path(path)
        self._guard_write(target, content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        workspace = note_code_change(rel(target))
        return {
            "ok": True,
            "path": self._track(target),
            "iteration": workspace["iteration"],
            "name": workspace["current_name"],
            "status": workspace["status"],
        }


class CoderAgent(SubAgent):
    """Writes one GNN / training change per assignment."""

    name = "coder"
    description = (
        "Edits the GNN and its training code (neuroasd/ and autoresearch/trial.py). "
        "Use this to implement exactly one hypothesized improvement. Not for "
        "running trials or scoring AUC."
    )
    instructions = INSTRUCTIONS

    def __init__(self, *, promote: bool = True, push: bool = True, **kwargs: Any):
        kwargs.setdefault("max_tool_rounds", 32)
        super().__init__(**kwargs)

    def create_tools(self) -> CodeTools:
        return CodeTools()
