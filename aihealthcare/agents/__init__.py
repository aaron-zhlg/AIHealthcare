"""Multi-agent medical research system.

An orchestrator-worker architecture built on the reusable :mod:`orchestra`
framework (LLM client + tool-calling loop + orchestrator). This package supplies
only the medical specialization:

* :mod:`aihealthcare.agents.literature`    — PubMed literature sub-agent.
* :mod:`aihealthcare.agents.trials`        — ClinicalTrials.gov sub-agent.
* :mod:`aihealthcare.agents.orchestrator`  — LeadResearcher + medical prompts.

Quick start::

    from aihealthcare.agents import LeadResearcher

    report = LeadResearcher().research("Do GLP-1 agonists reduce MACE in T2D?")
    print(report.report)

Or run one sub-agent on its own::

    from aihealthcare.agents import MedicalLiteratureAgent
    print(MedicalLiteratureAgent().run("Recent RCTs on semaglutide for weight loss").findings)
"""

# Framework types (re-exported for convenience) live in orchestra.
from orchestra import (
    Assignment,
    ChatClient,
    Conversation,
    LLMError,
    SubAgent,
    SubAgentResult,
    SubAgentSpec,
)

from aihealthcare.agents.literature import MedicalLiteratureAgent, PubMedTools, search_literature
from aihealthcare.agents.orchestrator import (
    DEFAULT_MODEL,
    LeadResearcher,
    ResearchReport,
    deep_research,
    default_subagent_specs,
)
from aihealthcare.agents.trials import ClinicalTrialsAgent, ClinicalTrialsTools

__all__ = [
    # framework (from orchestra)
    "Conversation",
    "ChatClient",
    "LLMError",
    "SubAgent",
    "SubAgentResult",
    "SubAgentSpec",
    "Assignment",
    # medical subagents
    "MedicalLiteratureAgent",
    "PubMedTools",
    "search_literature",
    "ClinicalTrialsAgent",
    "ClinicalTrialsTools",
    # orchestrator
    "LeadResearcher",
    "ResearchReport",
    "default_subagent_specs",
    "deep_research",
    "DEFAULT_MODEL",
]
