"""Lead agent for the GNN write → measure → insight → rewrite loop."""

from __future__ import annotations

import json
import sys
from typing import Any

from orchestra import Assignment, LLMError, Orchestrator, OrchestratorReport, SubAgentResult

from autoresearch.loop.coder import CoderAgent
from autoresearch.loop.experimenter import ExperimenterAgent
from autoresearch.loop.linter import LintAgent
from autoresearch.loop.paths import workspace_path
from autoresearch.loop.workspace import load_workspace, mark_coder_outcome, next_role, record_lint

DEFAULT_GOAL = (
    "Improve the GNN's cross-site ASD vs control AUC on ABIDE by editing training "
    "or model code, measuring each change with final-epoch gates, and iterating on "
    "the experimenter's insight. Change one mechanism at a time."
)

MISSION = """\
You coordinate a GNN research loop whose only job is a better, honest classifier. \
Sequence is write code → lint → run a trial → write an insight → write the next change. \
Never train until lint PASS. Never run coder and experimenter in the same round. \
Never treat a screen PASS as a better model. Never use best-epoch scores \
(that leak once inflated LOSO by ~0.07 AUC).
"""

PLANNER_INSTRUCTIONS = """\
You are the lead of a GNN experiment loop. You do not edit code or train yourself.

The loop is sequential. First assignment this round must be a SINGLE subagent:
- If the coder has not finished a clean edit, dispatch coder.
- If an edit is waiting to be linted, dispatch linter.
- If lint passed and a stage is waiting, dispatch experimenter.
Never dispatch more than one.

When briefing the coder, demand ONE tiny mechanism (a few lines). Do not ask \
for DANN, multi-file rewrites, or new training flags on the first turn. The \
scored path is autoresearch/trial.py + neuroasd/gcn.py.

Available subagent types:
{roster}

Respond with ONLY a JSON object (no prose, no code fence):
{{
  "complexity": "simple|moderate|complex",
  "reasoning": "one or two sentences",
  "assignments": [
    {{
      "subagent": "coder|linter|experimenter",
      "objective": "<self-contained task>",
      "output_format": "<what the worker should return>"
    }}
  ]
}}
"""

EVALUATOR_INSTRUCTIONS = """\
You inspect findings and decide the next SINGLE step of the loop.

- After coder FAIL: spawn coder again. Do not lint or train a partial edit.
- After coder PASS: spawn linter. Do not train yet.
- After lint FAIL: spawn coder with the lint errors.
- After lint PASS: spawn experimenter.
- After experimenter, if the same code still needs loso-subset or loso-full: \
spawn experimenter again.
- After experimenter FAIL (or a completed stage that needs a new idea): spawn \
coder. Put the insight into the coder objective: ruled out, next_code_change, \
final-epoch AUC vs gate. The coder is a fresh instance and cannot see this chat.
- After loso-full PASS: complete is true. Do not spawn more work.
- Never claim a best-epoch number as progress.

Available subagent types:
{roster}

Respond with ONLY a JSON object (no prose, no code fence):
{{
  "complete": true|false,
  "reasoning": "brief justification",
  "follow_up": [
    {{
      "subagent": "coder|linter|experimenter",
      "objective": "<specific next task, including insight when routing to coder>",
      "output_format": "<what the worker should return>"
    }}
  ]
}}
If complete is true, "follow_up" must be an empty list.
"""

SYNTHESIZER_INSTRUCTIONS = """\
Write the session report from the subagent findings only.

Open with whether a better GNN was found (loso-full PASS only). Then:
- Each change tried, final-epoch AUC, gate margin, PASS/FAIL.
- What was ruled out.
- If a review branch was opened, say so; a human still has to merge.
- Diagnostic best-epoch AUC is not a result.
Do not invent numbers. If only screen passed, say the search is unfinished.
"""

CITATION_INSTRUCTIONS = """\
Return the draft unchanged except for fixing file-path citations that the \
subagents actually produced. Do not add papers.
"""


def _insight_block() -> str:
    insight = load_workspace().get("last_insight") or {}
    if not insight:
        return "No prior insight in this session. Start with one small, testable change."
    return (
        "Latest experimenter insight (you must read_last_insight, then act on this):\n"
        + json.dumps(insight, indent=2)
    )


def _coder_assignment(goal: str, proposed: Assignment | None = None) -> Assignment:
    extra = proposed.objective if proposed and proposed.subagent == "coder" else goal
    return Assignment(
        "coder",
        (
            f"{extra}\n\n{_insight_block()}\n\n"
            "Implement exactly one new mechanism. Do not repeat a ruled-out idea."
        ),
        "Name the hypothesis, files edited, and a 3-line summary of the diff.",
    )


def _linter_assignment(proposed: Assignment | None = None) -> Assignment:
    objective = (
        proposed.objective
        if proposed and proposed.subagent == "linter"
        else "Lint the pending coder edit. Call lint_changed_files. The tool decides PASS/FAIL."
    )
    return Assignment(
        "linter",
        objective,
        "Return the lint report unchanged: passed, errors, files_checked.",
    )


def _experimenter_assignment(proposed: Assignment | None = None) -> Assignment:
    objective = (
        proposed.objective
        if proposed and proposed.subagent == "experimenter"
        else "Measure the pending code change at the workspace-required stage."
    )
    return Assignment(
        "experimenter",
        (
            f"{objective}\n\n"
            "Run the required stage, then store_insight so the next coder can "
            "read it. Official metric: final-epoch auc_mean."
        ),
        (
            "stage, auc_mean, gate margin, PASS/FAIL, what is ruled out, "
            "and next_code_change or next stage."
        ),
    )


def _forced_assignment(proposed: Assignment | None = None, goal: str = "") -> Assignment | None:
    role = next_role()
    if role is None:
        return None
    if role == "experimenter":
        return _experimenter_assignment(proposed)
    if role == "linter":
        return _linter_assignment(proposed)
    return _coder_assignment(goal, proposed)


class GNNLead(Orchestrator):
    """Orchestrator that keeps write → measure → insight → rewrite in order."""

    def __init__(
        self,
        subagents: Any = None,
        *,
        verbose: bool = True,
        push: bool = True,
        promote: bool = True,
        subagent_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if subagents is None:
            subagents = [CoderAgent, LintAgent, ExperimenterAgent]
        kwargs.setdefault("preamble", MISSION)
        kwargs.setdefault("planner_instructions", PLANNER_INSTRUCTIONS)
        kwargs.setdefault("evaluator_instructions", EVALUATOR_INSTRUCTIONS)
        kwargs.setdefault("synthesizer_instructions", SYNTHESIZER_INSTRUCTIONS)
        kwargs.setdefault("citation_instructions", CITATION_INSTRUCTIONS)
        kwargs.setdefault("max_rounds", 0)
        kwargs.setdefault("max_parallel", 1)
        kwargs.setdefault("add_citations", False)
        if subagent_kwargs is None:
            subagent_kwargs = {
                "verbose": verbose,
                "promote": promote,
                "push": push,
            }
        super().__init__(
            subagents,
            verbose=verbose,
            subagent_kwargs=subagent_kwargs,
            **kwargs,
        )

    def _plan(self, goal: str) -> tuple[str, list[Assignment]]:
        complexity, assignments = super()._plan(goal)
        forced = _forced_assignment(assignments[0] if assignments else None, goal)
        if forced is None:
            return complexity, []
        return complexity, [forced]

    def _evaluate(self, goal: str, results: list[SubAgentResult]) -> list[Assignment]:
        last = results[-1] if results else None
        if last is not None and last.subagent == "coder":
            mark_coder_outcome(last.ok, last.error or "")
        if last is not None and last.subagent == "linter":
            workspace = load_workspace()
            if not last.ok or workspace.get("lint_ok") is not True:
                record_lint(
                    {
                        "passed": False,
                        "errors": [
                            last.error
                            or "linter finished without a passing lint_changed_files result"
                        ],
                    }
                )
        if next_role() is None:
            return []
        follow_up = super()._evaluate(goal, results)
        proposed = follow_up[0] if follow_up else None
        forced = _forced_assignment(proposed, goal)
        return [forced] if forced is not None else []

    def run(self, goal: str) -> OrchestratorReport:
        """Loop write → lint → trial until loso-full PASS, max_rounds, or Ctrl-C.

        ``max_rounds <= 0`` means no cap. Workspace is kept across process
        restarts, so a killed run can be continued by starting again.
        """
        self._reset_logs()
        unlimited = self.max_rounds <= 0
        self._log(f"\n[lead] planning: {goal}")
        if unlimited:
            self._log("[lead] max_rounds=0; run until loso-full PASS or Ctrl-C")
        complexity, assignments = self._plan(goal)
        self._log(f"[lead] complexity={complexity}; {len(assignments)} initial task(s)")

        all_results: list[SubAgentResult] = []
        rounds = 0
        try:
            while assignments:
                rounds += 1
                self._log(
                    f"[lead] round {rounds}: dispatching {len(assignments)} "
                    f"subagent(s) ({assignments[0].subagent})"
                )
                all_results.extend(self._dispatch(assignments, rounds))
                follow_up = self._evaluate(goal, all_results)
                if not follow_up:
                    self._log("[lead] evaluation: stop (promoted or no next role)")
                    break
                if not unlimited and rounds >= self.max_rounds:
                    self._log(
                        f"[lead] reached max_rounds={self.max_rounds}; "
                        "synthesizing (re-run to continue from workspace)"
                    )
                    break
                spawned = ", ".join(a.subagent for a in follow_up)
                self._log(f"[lead] next: {spawned}")
                assignments = follow_up
        except KeyboardInterrupt:
            self._log("[lead] Ctrl-C; synthesizing what we have")

        if not all_results:
            return OrchestratorReport(
                goal=goal, answer="Nothing to do.", complexity=complexity, rounds=0
            )

        self._log("[lead] synthesizing final answer")
        draft = self._synthesize(goal, all_results)
        answer = self._cite(goal, draft, all_results) if self.add_citations else draft
        return OrchestratorReport(
            goal=goal,
            answer=answer,
            complexity=complexity,
            rounds=rounds,
            results=all_results,
        )

    def research(self, goal: str) -> OrchestratorReport:
        return self.run(goal)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="GNN write→run→insight loop (coder + experimenter)."
    )
    parser.add_argument("goal", nargs="*", help="Research goal; omit for the default.")
    parser.add_argument("--lead-model", default=None, help="Model for the lead.")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        metavar="N",
        help="Max write/lint/run rounds for a local smoke. Omit or 0 = until loso-full PASS or Ctrl-C.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the session workspace and start from idle.",
    )
    parser.add_argument("--no-push", action="store_true", help="Do not git push on loso-full PASS.")
    parser.add_argument("--no-promote", action="store_true", help="Do not open a review branch.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-dir", default="outputs/autoresearch/loop/logs")
    args = parser.parse_args()

    def build() -> GNNLead:
        return GNNLead(
            lead_model=args.lead_model,
            max_rounds=args.max_rounds,
            push=not args.no_push,
            promote=not args.no_promote,
            verbose=not args.quiet,
            log_dir=args.log_dir,
        )

    if args.fresh:
        path = workspace_path()
        if path.exists():
            path.unlink()
            print(f"cleared {path}", file=sys.stderr)

    goal = " ".join(args.goal).strip() or DEFAULT_GOAL
    try:
        report = build().research(goal)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("\n" + "=" * 80)
    print(report.answer)
    print("=" * 80)
    print(f"[rounds={report.rounds}, workers={len(report.results)}]")


if __name__ == "__main__":
    main()
