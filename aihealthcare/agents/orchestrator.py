"""A multi-agent medical research system (orchestrator-worker pattern).

This implements the architecture described in Anthropic's "How we built our
multi-agent research system" (https://www.anthropic.com/engineering/multi-agent-research-system),
adapted to the medical domain and built on the DeepSeek agent loop in
:mod:`aihealthcare.agents.deepseek`.

Roles
-----
* **LeadResearcher (orchestrator)** — analyzes the goal, *plans* by decomposing
  it into subtasks, *delegates* those subtasks to specialized subagents that run
  **in parallel** (each with its own context window and tools), *evaluates* the
  returned findings and decides whether another round of research is needed,
  then *synthesizes* the final report. Pure reasoning; holds no search tools.
* **Subagents (workers)** — LLMs autonomously using tools in a loop. Each is a
  narrow specialist that acts as an intelligent filter/compressor, returning a
  condensed, cited summary rather than raw data. Shipped subagents:
    - :class:`aihealthcare.agents.literature.MedicalLiteratureAgent` (PubMed)
    - :class:`aihealthcare.agents.trials.ClinicalTrialsAgent` (ClinicalTrials.gov)
  More can be registered via the :class:`SubAgent` interface.
* **CitationAgent** — a final pass that attributes every claim in the report to
  a retrieved source and compiles the reference list.

Key principles borrowed from the article: scale effort to query complexity,
give each subagent a clear objective/output-format/boundaries, start wide then
narrow, run subagents in parallel, and degrade gracefully when a tool fails.

    export DEEPSEEK_API_KEY=sk-...
    uv run python -m aihealthcare.agents "Do GLP-1 receptor agonists reduce major adverse cardiovascular events in type 2 diabetes, and what trials support this?"
"""

from __future__ import annotations

import json
import re
import sys
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from aihealthcare.agents.deepseek import DeepSeekError, ResponsesClient, output_text
from aihealthcare.agents.literature import MedicalLiteratureAgent
from aihealthcare.agents.trials import ClinicalTrialsAgent


# --------------------------------------------------------------------------- #
# Subagent interface + adapters
# --------------------------------------------------------------------------- #


@dataclass
class SubAgentResult:
    """The condensed output a worker returns to the lead agent."""

    subagent: str
    objective: str
    findings: str
    sources: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class SubAgent(typing.Protocol):
    """A worker the lead can delegate to.

    Concrete subagents expose ``name`` and ``description`` (used by the lead to
    route work) and a ``run_subtask`` method that executes one objective inside
    its own tool-calling loop and returns a :class:`SubAgentResult`.
    """

    name: str
    description: str

    def run_subtask(self, objective: str) -> SubAgentResult: ...


class LiteratureSubAgent:
    """Adapter exposing :class:`MedicalLiteratureAgent` as a :class:`SubAgent`."""

    def __init__(self, agent: MedicalLiteratureAgent | None = None, **kwargs: Any):
        self.agent = agent or MedicalLiteratureAgent(**kwargs)
        self.name = self.agent.name
        self.description = self.agent.description

    def run_subtask(self, objective: str) -> SubAgentResult:
        report = self.agent.run(objective)
        sources = [f"PMID:{p}" for p in report.pmids]
        return SubAgentResult(self.name, objective, report.summary, sources)


class TrialsSubAgent:
    """Adapter exposing :class:`ClinicalTrialsAgent` as a :class:`SubAgent`."""

    def __init__(self, agent: ClinicalTrialsAgent | None = None, **kwargs: Any):
        self.agent = agent or ClinicalTrialsAgent(**kwargs)
        self.name = self.agent.name
        self.description = self.agent.description

    def run_subtask(self, objective: str) -> SubAgentResult:
        summary, ncts = self.agent.run(objective)
        return SubAgentResult(self.name, objective, summary, list(ncts))


def default_subagents(**kwargs: Any) -> list[SubAgent]:
    """The medical subagent roster shipped with the system."""
    return [LiteratureSubAgent(**kwargs), TrialsSubAgent(**kwargs)]


# --------------------------------------------------------------------------- #
# Plan / report data
# --------------------------------------------------------------------------- #


@dataclass
class Assignment:
    """One subtask the lead hands to a specific subagent."""

    subagent: str
    objective: str


@dataclass
class ResearchReport:
    """The end product of a :class:`LeadResearcher` run."""

    goal: str
    report: str
    complexity: str = "unknown"
    rounds: int = 0
    results: list[SubAgentResult] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for r in self.results:
            for s in r.sources:
                if s not in seen:
                    seen.append(s)
        return seen

    def __str__(self) -> str:
        return self.report


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _extract_json(text: str) -> Any:
    """Best-effort parse of a JSON object/array embedded in model output."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from model output: {text[:200]!r}")


# --------------------------------------------------------------------------- #
# The orchestrator
# --------------------------------------------------------------------------- #

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
- Route each task to the most appropriate subagent by its described strengths.
- Start wide, then narrow: prefer tasks that first map the landscape.

Available subagents:
{roster}

Respond with ONLY a JSON object of this shape (no prose, no code fence):
{{
  "complexity": "simple|moderate|complex",
  "reasoning": "one or two sentences on your decomposition",
  "assignments": [
    {{"subagent": "<one of the subagent names above>", "objective": "<detailed, self-contained task>"}}
  ]
}}
"""

EVALUATOR_INSTRUCTIONS = """\
You are the lead researcher coordinating a medical multi-agent system. You have \
received findings from your subagents for the user's goal. Decide whether the \
evidence is sufficient to write a complete, well-supported answer, or whether ONE \
more round of targeted research is warranted.

Be judicious: only request more research to close a real, important gap (e.g. a \
missing mechanism, a contradicting result to resolve, or an un-searched angle the \
goal clearly requires). Do NOT request more research just to be thorough. Never \
exceed a small number of follow-ups.

Available subagents:
{roster}

Respond with ONLY a JSON object (no prose, no code fence):
{{
  "complete": true|false,
  "reasoning": "brief justification",
  "follow_up": [
    {{"subagent": "<name>", "objective": "<specific gap-closing task>"}}
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


class LeadResearcher:
    """Orchestrator implementing the plan → dispatch → evaluate → synthesize loop."""

    def __init__(
        self,
        subagents: list[SubAgent] | None = None,
        *,
        client: ResponsesClient | None = None,
        lead_model: str | None = None,
        max_rounds: int = 2,
        max_parallel: int = 5,
        add_citations: bool = True,
        verbose: bool = True,
        reasoning_effort: str | None = None,
    ):
        self.subagents: dict[str, SubAgent] = {}
        for sa in subagents if subagents is not None else default_subagents(verbose=verbose):
            self.subagents[sa.name] = sa
        if not self.subagents:
            raise ValueError("at least one subagent is required")
        self.client = client or ResponsesClient()
        self.lead_model = lead_model
        self.max_rounds = max_rounds
        self.max_parallel = max_parallel
        self.add_citations = add_citations
        self.verbose = verbose
        self.reasoning_effort = reasoning_effort

    # -- observability ----------------------------------------------------- #

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr, flush=True)

    # -- lead-agent LLM calls ---------------------------------------------- #

    @property
    def _roster(self) -> str:
        return "\n".join(f"- {sa.name}: {sa.description}" for sa in self.subagents.values())

    def _think(self, instructions: str, prompt: str) -> str:
        params: dict[str, Any] = {}
        if self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.create(
            input=prompt, instructions=instructions, model=self.lead_model, **params
        )
        return output_text(response)

    def _plan(self, goal: str) -> tuple[str, list[Assignment]]:
        instructions = PLANNER_INSTRUCTIONS.format(roster=self._roster)
        raw = self._think(instructions, f"Research goal:\n{goal}")
        data = _extract_json(raw)
        complexity = str(data.get("complexity", "unknown"))
        assignments = self._coerce_assignments(data.get("assignments", []))
        if not assignments:  # never leave the lead with nothing to do
            assignments = [Assignment(next(iter(self.subagents)), goal)]
        return complexity, assignments

    def _evaluate(self, goal: str, results: list[SubAgentResult]) -> list[Assignment]:
        instructions = EVALUATOR_INSTRUCTIONS.format(roster=self._roster)
        prompt = f"Research goal:\n{goal}\n\nFindings so far:\n{_render_findings(results)}"
        try:
            data = _extract_json(self._think(instructions, prompt))
        except ValueError:
            return []  # unparseable => treat as complete
        if data.get("complete", True):
            return []
        return self._coerce_assignments(data.get("follow_up", []))

    def _coerce_assignments(self, items: Any) -> list[Assignment]:
        out: list[Assignment] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("subagent", "")).strip()
            objective = str(item.get("objective", "")).strip()
            if not objective:
                continue
            if name not in self.subagents:  # route unknown names to the first agent
                name = next(iter(self.subagents))
            out.append(Assignment(name, objective))
        return out[: self.max_parallel]

    def _synthesize(self, goal: str, results: list[SubAgentResult]) -> str:
        prompt = f"User goal:\n{goal}\n\nSubagent findings:\n{_render_findings(results)}"
        return self._think(SYNTHESIZER_INSTRUCTIONS, prompt)

    def _cite(self, goal: str, draft: str, results: list[SubAgentResult]) -> str:
        allowed = sorted({s for r in results for s in r.sources})
        prompt = (
            f"User goal:\n{goal}\n\nAllowed sources (only these may be cited):\n"
            + ("\n".join(allowed) if allowed else "(none)")
            + f"\n\nDraft report:\n{draft}"
        )
        return self._think(CITATION_INSTRUCTIONS, prompt)

    # -- dispatch (parallel workers) --------------------------------------- #

    def _dispatch(self, assignments: list[Assignment]) -> list[SubAgentResult]:
        results: list[SubAgentResult] = [None] * len(assignments)  # type: ignore[list-item]

        def work(index: int, a: Assignment) -> tuple[int, SubAgentResult]:
            agent = self.subagents[a.subagent]
            self._log(f"   → [{a.subagent}] {a.objective}")
            try:
                return index, agent.run_subtask(a.objective)
            except Exception as exc:  # a failing worker must not sink the run
                return index, SubAgentResult(a.subagent, a.objective, "", error=str(exc))

        workers = min(self.max_parallel, len(assignments)) or 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i, a) for i, a in enumerate(assignments)]
            for fut in as_completed(futures):
                index, result = fut.result()
                results[index] = result
                status = "ok" if result.ok else f"ERROR: {result.error}"
                self._log(f"   ← [{result.subagent}] done ({len(result.sources)} sources, {status})")
        return results

    # -- public API -------------------------------------------------------- #

    def research(self, goal: str) -> ResearchReport:
        """Run the full multi-agent loop for ``goal`` and return the final report."""
        self._log(f"\n[lead] planning: {goal}")
        complexity, assignments = self._plan(goal)
        self._log(f"[lead] complexity={complexity}; {len(assignments)} initial task(s)")

        all_results: list[SubAgentResult] = []
        rounds = 0
        for rnd in range(1, self.max_rounds + 1):
            rounds = rnd
            self._log(f"[lead] round {rnd}: dispatching {len(assignments)} subagent(s) in parallel")
            all_results.extend(self._dispatch(assignments))

            if rnd >= self.max_rounds:
                break
            follow_up = self._evaluate(goal, all_results)
            if not follow_up:
                self._log("[lead] evaluation: sufficient, proceeding to synthesis")
                break
            self._log(f"[lead] evaluation: {len(follow_up)} follow-up task(s)")
            assignments = follow_up

        self._log("[lead] synthesizing final report")
        draft = self._synthesize(goal, all_results)
        report = self._cite(goal, draft, all_results) if self.add_citations else draft

        return ResearchReport(
            goal=goal, report=report, complexity=complexity, rounds=rounds, results=all_results
        )


def _render_findings(results: list[SubAgentResult]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        header = f"### Finding {i} — subagent: {r.subagent}\nObjective: {r.objective}"
        if r.error:
            blocks.append(f"{header}\n[FAILED: {r.error}]")
        else:
            src = ", ".join(r.sources) if r.sources else "(none)"
            blocks.append(f"{header}\nSources: {src}\n{r.findings}")
    return "\n\n".join(blocks) if blocks else "(no findings)"


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
        except DeepSeekError as exc:
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
