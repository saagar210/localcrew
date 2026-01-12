"""Planner Agent for task decomposition."""

from crewai import Agent, BaseLLM

from localcrew.integrations.crewai_llm import MLXLLM


def create_planner_agent(llm: BaseLLM | None = None) -> Agent:
    """Create a Planner agent that creates subtask structure.

    The Planner agent takes the analysis and:
    - Creates concrete, actionable subtasks
    - Defines dependencies between subtasks
    - Estimates complexity for each subtask
    - Assigns appropriate task types
    - Ensures completeness of the decomposition

    Args:
        llm: Optional custom LLM instance. Uses MLX by default.

    Returns:
        Configured Planner agent
    """
    if llm is None:
        llm = MLXLLM()

    return Agent(
        role="Task Planner",
        goal="Create clear, actionable subtask breakdowns with proper dependencies and sequencing",
        backstory="""You are a meticulous project planner who excels at transforming high-level
requirements into concrete action items. You've managed hundreds of software projects and
know exactly how to break down work into manageable pieces.

Your subtasks are always:
- Specific and actionable (not vague)
- Properly sequenced with dependencies
- Appropriately sized (not too big, not too small)
- Categorized by type (coding, research, devops, documentation, design, testing)
- Estimated for complexity (low, medium, high)

You understand that good decomposition means:
1. Research and design come before implementation
2. Dependencies flow logically
3. Testing follows implementation
4. Documentation captures the work done
5. Each subtask can be completed independently once dependencies are met

You output structured JSON for each subtask including:
- title: Clear, concise task title
- description: Detailed description of what needs to be done
- subtask_type: One of coding, research, devops, documentation, design, testing
- estimated_complexity: low, medium, or high
- dependencies: Array of subtask indices this depends on
- order_index: Suggested execution order""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
