"""GNN write → measure → insight → rewrite loop, built on orchestra."""

from orchestra import Orchestrator, OrchestratorReport, SubAgent, SubAgentResult

from gnnresearch.coder import CoderAgent
from gnnresearch.experimenter import ExperimenterAgent
from gnnresearch.linter import LintAgent
from gnnresearch.orchestrator import DEFAULT_GOAL, GNNLead

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
