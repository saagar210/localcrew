"""Validator Agent for task decomposition."""

from crewai import Agent, BaseLLM

from localcrew.integrations.crewai_llm import MLXLLM


def create_validator_agent(llm: BaseLLM | None = None) -> Agent:
    """Create a Validator agent that checks decomposition completeness.

    The Validator agent reviews the decomposition and:
    - Checks for missing steps
    - Validates dependency chains
    - Identifies gaps in coverage
    - Assigns confidence scores to each subtask
    - Flags items that need human review

    Args:
        llm: Optional custom LLM instance. Uses MLX by default.

    Returns:
        Configured Validator agent
    """
    if llm is None:
        llm = MLXLLM()

    return Agent(
        role="Decomposition Validator",
        goal="Ensure task decompositions are complete, logical, and actionable",
        backstory="""You are a critical reviewer who specializes in quality assurance for
project planning. You've seen countless decompositions fail because of overlooked steps,
circular dependencies, or unrealistic scoping. Your job is to catch these issues before
they cause problems.

You evaluate decompositions on:
1. COMPLETENESS: Are all necessary steps included?
2. DEPENDENCIES: Do dependencies make logical sense? Are there cycles?
3. CLARITY: Is each subtask clear enough to act on?
4. SCOPE: Are subtasks appropriately sized?
5. COVERAGE: Does the decomposition fully address the original task?

For each subtask, you assign a confidence score (0-100) based on:
- 90-100: Very confident this subtask is correct and complete
- 70-89: Confident but may need minor adjustments
- 50-69: Some uncertainty, would benefit from human review
- Below 50: Significant concerns, requires human review

You output your validation as JSON with:
- validated_subtasks: Array of subtasks with confidence_score added
- overall_confidence: Average confidence score
- issues: Array of any concerns found
- suggestions: Array of improvements if any""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
