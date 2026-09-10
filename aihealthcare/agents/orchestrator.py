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

import datetime as _dt
import json
import re
import sys
import threading
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aihealthcare.agents.deepseek import DeepSeekError, ResponsesClient, output_text
from aihealthcare.agents.literature import MedicalLiteratureAgent
from aihealthcare.agents.trials import ClinicalTrialsAgent


# --------------------------------------------------------------------------- #
# Subagent interface: specs (types the lead can instantiate) + workers
# --------------------------------------------------------------------------- #

#: Signature of a per-run trajectory sink: ``(tool_name, arguments_json, result)``.
ToolCallSink = typing.Callable[[str, str, str], None]


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


class SubAgentWorker(typing.Protocol):
    """One live worker instance: runs a single objective in its own tool loop."""

    def run_subtask(
        self, objective: str, on_tool_call: ToolCallSink | None = None
    ) -> SubAgentResult: ...


@dataclass
class SubAgentSpec:
    """A *type* of subagent the lead can instantiate on demand.

    The lead sees ``name`` + ``description`` to decide routing, then calls
    :meth:`create` **once per task**. This is what lets the lead autonomously:
    spin up the same type multiple times (one independent instance, with its own
    context window, per sub-question), or skip a type entirely when the goal does
    not need it. A spec is a lightweight factory; nothing is instantiated until
    the lead actually assigns work to it.
    """

    name: str
    description: str
    factory: typing.Callable[[], SubAgentWorker]

    def create(self) -> SubAgentWorker:
        return self.factory()


class LiteratureWorker:
    """A single PubMed literature worker instance (fresh context per task)."""

    def __init__(self, agent: MedicalLiteratureAgent):
        self.agent = agent

    def run_subtask(self, objective: str, on_tool_call: ToolCallSink | None = None) -> SubAgentResult:
        report = self.agent.run(objective, on_tool_call=on_tool_call)
        sources = [f"PMID:{p}" for p in report.pmids]
        return SubAgentResult(MedicalLiteratureAgent.name, objective, report.summary, sources)


class TrialsWorker:
    """A single ClinicalTrials.gov worker instance (fresh context per task)."""

    def __init__(self, agent: ClinicalTrialsAgent):
        self.agent = agent

    def run_subtask(self, objective: str, on_tool_call: ToolCallSink | None = None) -> SubAgentResult:
        summary, ncts = self.agent.run(objective, on_tool_call=on_tool_call)
        return SubAgentResult(ClinicalTrialsAgent.name, objective, summary, list(ncts))


def default_subagent_specs(**kwargs: Any) -> list[SubAgentSpec]:
    """The medical subagent *types* shipped with the system.

    Each spec's factory builds a brand-new agent instance per task, so the lead
    can instantiate any number of them (or none). ``kwargs`` (e.g. ``verbose``,
    ``model``) are forwarded to every agent constructor.
    """
    return [
        SubAgentSpec(
            MedicalLiteratureAgent.name,
            MedicalLiteratureAgent.description,
            lambda: LiteratureWorker(MedicalLiteratureAgent(**kwargs)),
        ),
        SubAgentSpec(
            ClinicalTrialsAgent.name,
            ClinicalTrialsAgent.description,
            lambda: TrialsWorker(ClinicalTrialsAgent(**kwargs)),
        ),
    ]


# --------------------------------------------------------------------------- #
# Plan / report data
# --------------------------------------------------------------------------- #


@dataclass
class Assignment:
    """One subtask the lead hands to a specific subagent.

    Following the article's delegation advice, an assignment carries not just an
    objective but also an explicit ``output_format`` describing what the worker
    should return, which reduces misinterpretation and duplicated work.
    """

    subagent: str
    objective: str
    output_format: str = ""

    def task_prompt(self) -> str:
        """The full instruction handed to the worker (objective + output format)."""
        if self.output_format:
            return f"{self.objective}\n\nRequired output format:\n{self.output_format}"
        return self.objective


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


def _ts() -> str:
    """A compact local timestamp prefix for log lines."""
    return _dt.datetime.now().strftime("%H:%M:%S")


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


class LeadResearcher:
    """Orchestrator implementing the plan → dispatch → evaluate → synthesize loop.

    The lead autonomously decides which subagent *types* to instantiate, how many
    of each, and when to stop. After each wave of parallel subagents it inspects
    the collected findings and may **dynamically spawn** more subagents to pursue
    a newly discovered sub-goal (a new type, or more instances) — repeating until
    it judges the evidence sufficient or the ``max_rounds`` safety cap is hit.
    """

    def __init__(
        self,
        subagents: list[SubAgentSpec] | None = None,
        *,
        client: ResponsesClient | None = None,
        lead_model: str | None = None,
        max_rounds: int = 3,
        max_parallel: int = 5,
        add_citations: bool = True,
        verbose: bool = True,
        reasoning_effort: str | None = None,
        log_dir: str | Path | None = None,
    ):
        # Registry of subagent *types* (specs). The lead instantiates them on
        # demand, so nothing here is a live agent until work is assigned.
        self.specs: dict[str, SubAgentSpec] = {}
        for spec in subagents if subagents is not None else default_subagent_specs(verbose=verbose):
            self.specs[spec.name] = spec
        if not self.specs:
            raise ValueError("at least one subagent spec is required")
        self.client = client or ResponsesClient()
        self.lead_model = lead_model
        self.max_rounds = max_rounds
        self.max_parallel = max_parallel
        self.add_citations = add_citations
        self.verbose = verbose
        self.reasoning_effort = reasoning_effort
        # When set, the lead writes logs/lead.log and each subagent instance
        # streams its full trajectory to logs/<type>_<NNN>.log, where NNN is a
        # per-type instance counter (the lead may init several of one type).
        self.log_dir = Path(log_dir) if log_dir else None
        self._lead_log_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._instance_counts: dict[str, int] = {}
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # -- observability ----------------------------------------------------- #

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr, flush=True)
        if self.log_dir:
            with self._lead_log_lock:
                with (self.log_dir / "lead.log").open("a", encoding="utf-8") as fh:
                    fh.write(f"{_ts()} {message}\n")

    # -- lead-agent LLM calls ---------------------------------------------- #

    @property
    def _roster(self) -> str:
        return "\n".join(f"- {spec.name}: {spec.description}" for spec in self.specs.values())

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
            assignments = [Assignment(next(iter(self.specs)), goal)]
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
            output_format = str(item.get("output_format", "")).strip()
            if not objective:
                continue
            if name not in self.specs:  # route unknown names to the first spec
                name = next(iter(self.specs))
            out.append(Assignment(name, objective, output_format))
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

    def _trajectory_sink(
        self, round_idx: int, index: int, a: Assignment
    ) -> tuple[ToolCallSink | None, typing.Callable[[SubAgentResult], None]]:
        """Return (per-tool-call sink, finalizer) that stream one subagent
        instance's full trajectory to ``logs/<type>_<NNN>.log``, where NNN counts
        instances of that type across the whole run. No-ops without ``log_dir``."""
        if not self.log_dir:
            return None, lambda result: None

        with self._counter_lock:
            self._instance_counts[a.subagent] = self._instance_counts.get(a.subagent, 0) + 1
            instance_no = self._instance_counts[a.subagent]

        path = self.log_dir / f"{a.subagent}_{instance_no:03d}.log"
        fh = path.open("w", encoding="utf-8")
        fh.write(f"{_ts()} === subagent: {a.subagent} #{instance_no:03d} (round {round_idx}) ===\n")
        fh.write(f"{_ts()} OBJECTIVE:\n{a.objective}\n\n")
        fh.flush()
        lock = threading.Lock()
        step = {"n": 0}

        def sink(name: str, arguments: str, result: str) -> None:
            with lock:
                step["n"] += 1
                fh.write(f"{_ts()} [tool #{step['n']}] {name}({arguments})\n")
                fh.write(f"{_ts()}   -> {result}\n\n")
                fh.flush()

        def closer(result: SubAgentResult) -> None:
            with lock:
                if result.error:
                    fh.write(f"{_ts()} [FAILED] {result.error}\n")
                else:
                    fh.write(f"{_ts()} [DONE] {step['n']} tool call(s), {len(result.sources)} source(s)\n")
                    fh.write(f"{_ts()} SOURCES: {', '.join(result.sources) or '(none)'}\n\n")
                    fh.write(f"{_ts()} FINDINGS:\n{result.findings}\n")
                fh.close()

        return sink, closer

    def _dispatch(self, assignments: list[Assignment], round_idx: int = 1) -> list[SubAgentResult]:
        results: list[SubAgentResult] = [None] * len(assignments)  # type: ignore[list-item]

        def work(index: int, a: Assignment) -> tuple[int, SubAgentResult]:
            # Instantiate a fresh worker for this task (this is the lead "init"-ing
            # a subagent on demand — same type can be spun up many times).
            worker = self.specs[a.subagent].create()
            self._log(f"   → init [{a.subagent}] {a.objective}")
            sink, closer = self._trajectory_sink(round_idx, index, a)
            try:
                result = worker.run_subtask(a.task_prompt(), on_tool_call=sink)
            except Exception as exc:  # a failing worker must not sink the run
                result = SubAgentResult(a.subagent, a.objective, "", error=str(exc))
            closer(result)
            return index, result

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

    def _reset_logs(self) -> None:
        """Clear historical logs so each run starts a clean trajectory."""
        with self._counter_lock:
            self._instance_counts.clear()
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            for old in self.log_dir.glob("*.log"):
                try:
                    old.unlink()
                except OSError:
                    pass

    def research(self, goal: str) -> ResearchReport:
        """Run the full multi-agent loop for ``goal`` and return the final report."""
        self._reset_logs()
        self._log(f"\n[lead] planning: {goal}")
        complexity, assignments = self._plan(goal)
        self._log(f"[lead] complexity={complexity}; {len(assignments)} initial task(s)")

        all_results: list[SubAgentResult] = []
        rounds = 0
        for rnd in range(1, self.max_rounds + 1):
            rounds = rnd
            self._log(f"[lead] round {rnd}: dispatching {len(assignments)} subagent(s) in parallel")
            all_results.extend(self._dispatch(assignments, rnd))

            if rnd >= self.max_rounds:
                self._log(f"[lead] reached max_rounds={self.max_rounds}; proceeding to synthesis")
                break
            follow_up = self._evaluate(goal, all_results)
            if not follow_up:
                self._log("[lead] evaluation: evidence sufficient, proceeding to synthesis")
                break
            spawned = ", ".join(a.subagent for a in follow_up)
            self._log(f"[lead] evaluation: dynamically spawning {len(follow_up)} subagent(s): {spawned}")
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
