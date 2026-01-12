"""Synthesizer Agent for research crew."""

from crewai import Agent, BaseLLM

from localcrew.integrations.crewai_llm import MLXLLM


def create_synthesizer_agent(llm: BaseLLM | None = None) -> Agent:
    """Create a Synthesizer agent that combines research findings.

    The Synthesizer agent:
    - Analyzes gathered information for patterns and insights
    - Resolves conflicting information
    - Identifies key themes and conclusions
    - Assesses overall confidence in findings

    Args:
        llm: Optional custom LLM instance. Uses MLX by default.

    Returns:
        Configured Synthesizer agent
    """
    if llm is None:
        llm = MLXLLM()

    return Agent(
        role="Research Synthesizer",
        goal="Combine gathered information into coherent, insightful analysis",
        backstory="""You are an analytical thinker who excels at seeing the big picture.
You take diverse pieces of information and weave them into a coherent narrative
that answers the original research question.

Your expertise includes:
- Identifying patterns across multiple sources
- Resolving apparent contradictions in information
- Distinguishing between facts, opinions, and speculation
- Drawing reasonable conclusions from available evidence
- Recognizing the limits of what can be concluded

When synthesizing research, you:
1. Group related findings together
2. Identify areas of agreement and disagreement
3. Weigh evidence based on source quality
4. Note any gaps in the available information
5. Draw conclusions supported by multiple sources

You assign confidence scores based on:
- Number of corroborating sources
- Quality and recency of sources
- Consistency of information
- Completeness of coverage

You're honest about uncertainty and clearly distinguish between
well-supported conclusions and educated speculation.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
