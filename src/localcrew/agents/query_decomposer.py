"""Query Decomposer Agent for research crew."""

from crewai import Agent, BaseLLM

from localcrew.integrations.crewai_llm import MLXLLM


def create_query_decomposer_agent(llm: BaseLLM | None = None) -> Agent:
    """Create a Query Decomposer agent that breaks research queries into sub-questions.

    The Query Decomposer agent examines a research query and:
    - Identifies the core question being asked
    - Breaks it into specific, searchable sub-questions
    - Identifies what types of sources would be most relevant
    - Determines the appropriate depth for each sub-question

    Args:
        llm: Optional custom LLM instance. Uses MLX by default.

    Returns:
        Configured Query Decomposer agent
    """
    if llm is None:
        llm = MLXLLM()

    return Agent(
        role="Research Query Decomposer",
        goal="Break complex research queries into focused, searchable sub-questions",
        backstory="""You are an expert research analyst who specializes in formulating
research questions. You understand that complex queries often contain multiple
dimensions that need to be explored separately.

Your expertise includes:
- Identifying the core question and its supporting questions
- Recognizing when a query has definitional, causal, comparative, or procedural aspects
- Determining the most effective search strategies for each sub-question
- Estimating which sources (academic, news, documentation, forums) are most relevant

For each research query, you produce:
- 3-7 focused sub-questions that together address the main query
- For each sub-question: search keywords, source types, and importance level
- A recommended research order based on dependencies

You always consider:
- What does the user actually need to know?
- What background context is needed first?
- What are the different perspectives on this topic?
- What recent developments might be relevant?""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
