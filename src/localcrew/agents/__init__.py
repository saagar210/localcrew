"""Agent definitions for LocalCrew."""

# Task Decomposition Crew agents
from localcrew.agents.analyzer import create_analyzer_agent
from localcrew.agents.planner import create_planner_agent
from localcrew.agents.validator import create_validator_agent

# Research Crew agents
from localcrew.agents.query_decomposer import create_query_decomposer_agent
from localcrew.agents.gatherer import create_gatherer_agent
from localcrew.agents.synthesizer import create_synthesizer_agent
from localcrew.agents.reporter import create_reporter_agent

__all__ = [
    # Task Decomposition
    "create_analyzer_agent",
    "create_planner_agent",
    "create_validator_agent",
    # Research
    "create_query_decomposer_agent",
    "create_gatherer_agent",
    "create_synthesizer_agent",
    "create_reporter_agent",
]
