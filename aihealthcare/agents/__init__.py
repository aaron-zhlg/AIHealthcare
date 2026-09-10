"""Multi-agent medical research system.

An orchestrator-worker architecture (see Anthropic's "How we built our
multi-agent research system") built on the DeepSeek agent loop:

* :mod:`aihealthcare.agents.deepseek`      — LLM backend (client + tool-calling loop).
* :mod:`aihealthcare.agents.literature`    — PubMed literature sub-agent.
* :mod:`aihealthcare.agents.trials`        — ClinicalTrials.gov sub-agent.
* :mod:`aihealthcare.agents.orchestrator`  — LeadResearcher + CitationAgent.

Quick start::

    from aihealthcare.agents import LeadResearcher

    report = LeadResearcher().research("Do GLP-1 agonists reduce MACE in T2D?")
    print(report.report)

Or run one sub-agent on its own::

    from aihealthcare.agents import MedicalLiteratureAgent
    print(MedicalLiteratureAgent().run("Recent RCTs on semaglutide for weight loss"))
"""

from aihealthcare.agents.deepseek import Conversation, DeepSeekError, ResponsesClient
from aihealthcare.agents.literature import MedicalLiteratureAgent, SearchReport
from aihealthcare.agents.orchestrator import (
    Assignment,
    LeadResearcher,
    LiteratureWorker,
    ResearchReport,
    SubAgentResult,
    SubAgentSpec,
    SubAgentWorker,
    TrialsWorker,
    deep_research,
    default_subagent_specs,
)
from aihealthcare.agents.trials import ClinicalTrialsAgent

__all__ = [
    "Conversation",
    "ResponsesClient",
    "DeepSeekError",
    "MedicalLiteratureAgent",
    "SearchReport",
    "ClinicalTrialsAgent",
    "LeadResearcher",
    "ResearchReport",
    "SubAgentSpec",
    "SubAgentWorker",
    "SubAgentResult",
    "Assignment",
    "LiteratureWorker",
    "TrialsWorker",
    "default_subagent_specs",
    "deep_research",
]
