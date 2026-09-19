"""A multi-agent medical research system (orchestrator-worker pattern).

This is the *medical* instantiation of the domain-agnostic :mod:`orchestra`
framework. All the orchestration machinery — the LLM client, the tool-calling
loop, the plan → dispatch → evaluate → synthesize control flow, parallel
subagents, dynamic spawning, trajectory logging — lives in :mod:`orchestra`. This
module only supplies the medical pieces:

* the two domain subagents (:class:`~aihealthcare.agents.literature.MedicalLiteratureAgent`
  for PubMed and :class:`~aihealthcare.agents.trials.ClinicalTrialsAgent` for
  ClinicalTrials.gov), and
* medical-tuned phase prompts (planner / evaluator / synthesizer / citation) plus
  a project ``MISSION`` preamble injected into every phase of the lead agent.

The lead agent is intelligence-driven: it decides the decomposition, routing, how
many of each subagent to spin up, and when to stop — there is no hardcoded graph.

    export DEEPSEEK_API_KEY=sk-...
    uv run python -m aihealthcare.agents "Do GLP-1 receptor agonists reduce major adverse cardiovascular events in type 2 diabetes, and what trials support this?"

Programmatic use (backwards-compatible with the pre-orchestra API)::

    from aihealthcare.agents import LeadResearcher

    report = LeadResearcher().research("...")
    print(report.report)      # the final answer text
    print(report.sources)     # every PMID / NCT id cited
"""

from __future__ import annotations

import sys
from typing import Any

from orchestra import (
    LLMError,
    Orchestrator,
    OrchestratorReport,
    SubAgentSpec,
    resolve_endpoint,
)

from aihealthcare.agents.literature import MedicalLiteratureAgent
from aihealthcare.agents.trials import ClinicalTrialsAgent

# Best-effort default model (for run metadata/labels). Resolved from the same
# provider env the framework uses; falls back to a sensible alias when no key is
# configured at import time.
try:
    _, _, DEFAULT_MODEL = resolve_endpoint()
except Exception:
    DEFAULT_MODEL = "deepseek-flash"


# --------------------------------------------------------------------------- #
# Medical phase prompts (injected into the domain-agnostic Orchestrator)
# --------------------------------------------------------------------------- #

#: Project-level system prompt, prepended to every lead-agent phase.
MISSION = """\
You coordinate a medical research assistant. Be rigorous and conservative: weigh \
the level of evidence, prefer high-quality sources, and never overstate findings.
"""

PLANNER_INSTRUCTIONS = """\
You are the lead researcher of a medical multi-agent system. You do not search \
yourself; instead you plan and delegate to specialized subagents that run in \
parallel, each with its own context window and tools.

Your job now: turn the user's research goal into a concrete delegation plan.

Guidelines (mirror how an expert human researcher works):
- Judge complexity and SCALE EFFORT accordingly. Do not over-invest in simple \
questions:
    * simple  (one fact / one angle): 1 subagent task.
    * moderate (comparison / a couple of angles): 2-3 subagent tasks.
    * complex (broad, multi-part): 3-5 subagent tasks, clearly divided.
- Give each subagent a DISTINCT slice of the problem so they do not duplicate \
work. Each task must include a precise objective and, implicitly, its output \
expectation.
- Route each task to the most appropriate subagent TYPE by its described strengths.
- You control instantiation. You MAY assign the SAME subagent type to several \
tasks — each runs as a SEPARATE instance with its own context window — when the \
goal has multiple independent sub-questions of that kind (e.g. two distinct \
literature sub-questions -> two pubmed_literature tasks). You MAY also OMIT any \
subagent type the goal does not need (e.g. no clinical_trials task if trials are \
irrelevant). Only spin up what the goal actually requires.
- Start wide, then narrow: prefer tasks that first map the landscape.

Available subagent types:
{roster}

For EACH task give: the subagent type, a detailed self-contained objective, and \
an explicit output_format telling the worker exactly what to return (fields, \
structure, and that every claim must carry its source id). Clear task boundaries \
prevent duplicated work and gaps.

Respond with ONLY a JSON object of this shape (no prose, no code fence):
{{
  "complexity": "simple|moderate|complex",
  "reasoning": "one or two sentences on your decomposition",
  "assignments": [
    {{
      "subagent": "<one of the subagent names above>",
      "objective": "<detailed, self-contained task with clear boundaries>",
      "output_format": "<what the worker should return, e.g. 'a structured bullet list of findings, each with (PMID: ...); note evidence strength'>"
    }}
  ]
}}
"""

EVALUATOR_INSTRUCTIONS = """\
You are the lead researcher coordinating a medical multi-agent system. You have \
received findings from the subagents you dispatched. Inspect the collected data \
and decide, autonomously, whether it is sufficient to write a complete, \
well-supported answer — or whether the data has revealed a NEW sub-goal or gap \
worth spawning more subagents for.

You may dynamically spin up additional subagents now, based on what the data \
showed. This includes: a subagent TYPE you have not used yet (e.g. the findings \
mention a pivotal trial, so spawn a clinical_trials task), or additional \
instances of a type you already used, each with a sharp, gap-closing objective.

Be judicious: only spawn more work to close a REAL, important gap (a missing \
mechanism, a contradiction to resolve, a newly surfaced entity/trial, or an \
un-searched angle the goal clearly requires). Do NOT spawn more just to be \
thorough, and keep the number of follow-ups small.

Available subagent types:
{roster}

Respond with ONLY a JSON object (no prose, no code fence):
{{
  "complete": true|false,
  "reasoning": "brief justification",
  "follow_up": [
    {{
      "subagent": "<name>",
      "objective": "<specific gap-closing task with clear boundaries>",
      "output_format": "<what the worker should return>"
    }}
  ]
}}
If complete is true, "follow_up" must be an empty list.
"""

SYNTHESIZER_INSTRUCTIONS = """\
You are the lead researcher of a medical multi-agent system. Using ONLY the \
findings gathered by your subagents (provided below), write the final answer to \
the user's goal.

Requirements:
- Open with a direct, decision-useful answer (2-5 sentences).
- Follow with well-organized sections / bullets covering the key evidence.
- Preserve every citation token from the findings verbatim: PubMed as \
(PMID: 12345678) and trials as (NCT01234567). Attribute each claim to its source.
- Weigh the strength/level of evidence; note disagreements, caveats, and gaps.
- Be precise and neutral. Do not invent citations or facts beyond the findings. \
If evidence is thin, say so.
Do not add a references list; a later step handles that.
"""

CITATION_INSTRUCTIONS = """\
You are the CitationAgent for a medical research system. You are given a draft \
report and the list of sources the subagents actually retrieved.

Your job:
- Verify that factual claims carry an inline citation token that exists in the \
allowed source list. Remove or flag any citation token not in the list.
- Do NOT change the substance of the report; only fix/normalize citations and \
append a final "References" section listing every source that is actually cited \
in the report, one per line, using the identifier and its URL:
    * PMID:NNNN  -> https://pubmed.ncbi.nlm.nih.gov/NNNN/
    * NCTNNNN    -> https://clinicaltrials.gov/study/NCTNNNN
Return the full, final report text.
"""


# --------------------------------------------------------------------------- #
# Subagent registry (the medical subagent *types* shipped with this system)
# --------------------------------------------------------------------------- #


def default_subagent_specs(**kwargs: Any) -> list[SubAgentSpec]:
    """The medical subagent *types* shipped with the system, as orchestra specs.

    Each spec's factory builds a brand-new agent instance per task, so the lead
    can instantiate any number of them (or none). ``kwargs`` (e.g. ``verbose``,
    ``model``) are forwarded to every agent constructor.
    """
    return [
        MedicalLiteratureAgent.spec(**kwargs),
        ClinicalTrialsAgent.spec(**kwargs),
    ]


# --------------------------------------------------------------------------- #
# Report (backwards-compatible view over orchestra's OrchestratorReport)
# --------------------------------------------------------------------------- #


class ResearchReport(OrchestratorReport):
    """An :class:`orchestra.OrchestratorReport` that also exposes ``.report``.

    ``.report`` is an alias for ``.answer`` kept so existing callers/harnesses
    (e.g. ``tests/run_goal.py``) continue to work unchanged.
    """

    @property
    def report(self) -> str:
        return self.answer


# --------------------------------------------------------------------------- #
# The orchestrator (medical specialization of orchestra.Orchestrator)
# --------------------------------------------------------------------------- #


class LeadResearcher(Orchestrator):
    """Medical lead researcher: an :class:`orchestra.Orchestrator` pre-wired with
    the PubMed + ClinicalTrials subagents and medical-tuned phase prompts.

    Everything the lead does (planning, parallel dispatch, dynamic spawning,
    synthesis, citation) is inherited from :class:`orchestra.Orchestrator`; this
    subclass only injects the medical defaults and keeps the ``.research()`` /
    ``ResearchReport.report`` API the rest of the project expects.
    """

    def __init__(
        self,
        subagents: Any = None,
        *,
        verbose: bool = True,
        subagent_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if subagents is None:
            subagents = [MedicalLiteratureAgent, ClinicalTrialsAgent]
        # Inject the medical phase prompts + mission preamble (overridable).
        kwargs.setdefault("preamble", MISSION)
        kwargs.setdefault("planner_instructions", PLANNER_INSTRUCTIONS)
        kwargs.setdefault("evaluator_instructions", EVALUATOR_INSTRUCTIONS)
        kwargs.setdefault("synthesizer_instructions", SYNTHESIZER_INSTRUCTIONS)
        kwargs.setdefault("citation_instructions", CITATION_INSTRUCTIONS)
        # By default subagents inherit the lead's verbosity (and its HTTP client,
        # which orchestra wires in automatically).
        if subagent_kwargs is None:
            subagent_kwargs = {"verbose": verbose}
        super().__init__(subagents, verbose=verbose, subagent_kwargs=subagent_kwargs, **kwargs)

    def research(self, goal: str) -> ResearchReport:
        """Run the full multi-agent loop for ``goal`` and return a report.

        Thin wrapper over :meth:`orchestra.Orchestrator.run` that returns a
        :class:`ResearchReport` (so ``.report`` is available).
        """
        r = self.run(goal)
        return ResearchReport(
            goal=r.goal,
            answer=r.answer,
            complexity=r.complexity,
            rounds=r.rounds,
            results=r.results,
        )


def deep_research(goal: str, **kwargs: Any) -> ResearchReport:
    """One-shot convenience wrapper around :class:`LeadResearcher`."""
    return LeadResearcher(**kwargs).research(goal)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-agent medical research system (orchestrator + subagents)."
    )
    parser.add_argument("goal", nargs="*", help="Research goal; omit for an interactive session.")
    parser.add_argument("--lead-model", default=None, help="Model for the lead/orchestrator.")
    parser.add_argument("--max-rounds", type=int, default=2, help="Max research rounds.")
    parser.add_argument("--effort", default=None, help="Lead thinking effort: none/low/medium/high/max.")
    parser.add_argument("--no-citations", action="store_true", help="Skip the CitationAgent pass.")
    parser.add_argument("--quiet", action="store_true", help="Suppress orchestration logs.")
    args = parser.parse_args()

    def build() -> LeadResearcher:
        return LeadResearcher(
            lead_model=args.lead_model,
            max_rounds=args.max_rounds,
            add_citations=not args.no_citations,
            verbose=not args.quiet,
            reasoning_effort=args.effort,
        )

    def answer(goal: str) -> None:
        try:
            report = build().research(goal)
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return
        print("\n" + "=" * 80)
        print(report.report)
        print("=" * 80)
        print(f"[complexity={report.complexity}, rounds={report.rounds}, sources={len(report.sources)}]")

    if args.goal:
        answer(" ".join(args.goal))
        return

    print("Multi-agent medical research. Ctrl-C or an empty line to quit.")
    while True:
        try:
            goal = input("\ngoal > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not goal:
            return
        answer(goal)


if __name__ == "__main__":
    main()
