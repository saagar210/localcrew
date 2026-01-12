"""External integrations for LocalCrew."""

from localcrew.integrations.crewai_llm import MLXLLM
from localcrew.integrations.mlx import MLXInference
from localcrew.integrations.taskmaster import TaskMasterIntegration, get_taskmaster

__all__ = [
    "MLXLLM",
    "MLXInference",
    "TaskMasterIntegration",
    "get_taskmaster",
]
