"""GNN write → measure → insight → rewrite loop, built on orchestra."""

from orchestra import Orchestrator, OrchestratorReport, SubAgent, SubAgentResult

from autoresearch.loop.coder import CoderAgent
from autoresearch.loop.experimenter import ExperimenterAgent
from autoresearch.loop.linter import LintAgent
from autoresearch.loop.orchestrator import DEFAULT_GOAL, GNNLead

__all__ = [
    "CoderAgent",
    "DEFAULT_GOAL",
    "ExperimenterAgent",
    "GNNLead",
    "LintAgent",
    "Orchestrator",
    "OrchestratorReport",
    "SubAgent",
    "SubAgentResult",
]
