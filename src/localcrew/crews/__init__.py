"""CrewAI crew definitions for LocalCrew."""

from localcrew.crews.decomposition import (
    DecompositionState,
    TaskDecompositionFlow,
    run_decomposition,
)
from localcrew.crews.research import (
    ResearchState,
    ResearchFlow,
    run_research,
)

__all__ = [
    # Task Decomposition
    "DecompositionState",
    "TaskDecompositionFlow",
    "run_decomposition",
    # Research
    "ResearchState",
    "ResearchFlow",
    "run_research",
]
