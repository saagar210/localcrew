"""Analyzer Agent for task decomposition."""

from crewai import Agent, BaseLLM

from localcrew.integrations.crewai_llm import MLXLLM


def create_analyzer_agent(llm: BaseLLM | None = None) -> Agent:
    """Create an Analyzer agent that understands task scope.

    The Analyzer agent examines the input task and:
    - Identifies the domain (coding, research, devops, documentation, design, testing)
    - Determines task complexity
    - Extracts key requirements and constraints
    - Identifies dependencies on external systems or knowledge

    Args:
        llm: Optional custom LLM instance. Uses MLX by default.

    Returns:
        Configured Analyzer agent
    """
    if llm is None:
        llm = MLXLLM()

    return Agent(
        role="Task Analyzer",
        goal="Thoroughly understand the scope, requirements, and complexity of incoming tasks",
        backstory="""You are an expert task analyst with years of experience breaking down
complex projects. You have a keen eye for identifying hidden requirements, potential
blockers, and the true scope of work. You're particularly skilled at recognizing when
a task seems simple but has underlying complexity, and when a complex-sounding task
is actually straightforward.

Your analysis forms the foundation for all subsequent planning, so you take care to
be thorough but concise. You categorize tasks into domains (coding, research, devops,
documentation, design, testing) and assess complexity levels (low, medium, high).

You always consider:
- What domain does this task belong to?
- What are the explicit requirements?
- What are the implicit requirements?
- What external dependencies exist?
- What could go wrong?
- How complex is this really?""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
