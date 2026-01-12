"""Task Decomposition Crew using CrewAI Flows."""

import json
import logging
import re
from typing import Any

from crewai import Crew, Process, Task
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field

from localcrew.agents import (
    create_analyzer_agent,
    create_planner_agent,
    create_validator_agent,
)
from localcrew.integrations.crewai_llm import MLXLLM
from localcrew.models.subtask import SubtaskType

logger = logging.getLogger(__name__)


class DecompositionState(BaseModel):
    """State for the decomposition flow."""

    task_text: str = ""
    project_context: str | None = None
    taskmaster_context: dict[str, Any] | None = None

    # Analysis output
    analysis: dict[str, Any] = Field(default_factory=dict)

    # Planning output
    planned_subtasks: list[dict[str, Any]] = Field(default_factory=list)

    # Validation output
    validated_subtasks: list[dict[str, Any]] = Field(default_factory=list)
    overall_confidence: int = 0
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class TaskDecompositionFlow(Flow[DecompositionState]):
    """Flow for decomposing tasks into subtasks using multiple agents."""

    def __init__(self, llm: MLXLLM | None = None) -> None:
        """Initialize the decomposition flow.

        Args:
            llm: Optional custom LLM. Uses MLX by default.
        """
        super().__init__()
        self.llm = llm or MLXLLM()

    @start()
    def analyze_task(self) -> str:
        """Analyze the task to understand scope and requirements."""
        logger.info(f"Analyzing task: {self.state.task_text[:100]}...")

        analyzer = create_analyzer_agent(self.llm)

        # Build context string
        context_parts = [f"Task to analyze: {self.state.task_text}"]
        if self.state.project_context:
            context_parts.append(f"Project context: {self.state.project_context}")
        if self.state.taskmaster_context:
            context_parts.append(
                f"Existing tasks context: {json.dumps(self.state.taskmaster_context)}"
            )

        analysis_task = Task(
            description=f"""Analyze the following task and provide a structured analysis:

{chr(10).join(context_parts)}

Provide your analysis as JSON with:
- domain: The primary domain (coding, research, devops, documentation, design, testing)
- complexity: Overall complexity (low, medium, high)
- explicit_requirements: List of clearly stated requirements
- implicit_requirements: List of inferred requirements
- dependencies: External dependencies or prerequisites
- risks: Potential issues or blockers
- estimated_subtask_count: How many subtasks this should decompose into (3-10)""",
            expected_output="JSON analysis of the task",
            agent=analyzer,
        )

        analysis_crew = Crew(
            agents=[analyzer],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=True,
        )

        result = analysis_crew.kickoff()

        # Parse the analysis from the result
        try:
            analysis = self._extract_json(result.raw)
            self.state.analysis = analysis
            logger.info(f"Analysis complete: {analysis.get('domain', 'unknown')} domain")
        except Exception as e:
            logger.warning(f"Could not parse analysis JSON: {e}")
            self.state.analysis = {
                "domain": "coding",
                "complexity": "medium",
                "explicit_requirements": [self.state.task_text],
                "implicit_requirements": [],
                "dependencies": [],
                "risks": [],
                "estimated_subtask_count": 5,
            }

        return "analysis_complete"

    @listen(analyze_task)
    def plan_subtasks(self, _: str) -> str:
        """Create the subtask breakdown based on analysis."""
        logger.info("Planning subtasks...")

        planner = create_planner_agent(self.llm)

        planning_task = Task(
            description=f"""Based on the following task and analysis, create a detailed subtask breakdown.

Original Task: {self.state.task_text}

Analysis:
{json.dumps(self.state.analysis, indent=2)}

Create {self.state.analysis.get('estimated_subtask_count', 5)} subtasks.

For each subtask, provide JSON with:
- title: Clear, concise title (imperative form, e.g., "Implement user authentication")
- description: Detailed description of what needs to be done
- subtask_type: One of: {', '.join([t.value for t in SubtaskType])}
- estimated_complexity: low, medium, or high
- dependencies: Array of indices of subtasks this depends on (0-indexed)
- order_index: Execution order (0, 1, 2, ...)

Output as a JSON array of subtask objects.""",
            expected_output="JSON array of subtasks",
            agent=planner,
        )

        planning_crew = Crew(
            agents=[planner],
            tasks=[planning_task],
            process=Process.sequential,
            verbose=True,
        )

        result = planning_crew.kickoff()

        # Parse the subtasks
        try:
            subtasks = self._extract_json(result.raw)
            if isinstance(subtasks, dict) and "subtasks" in subtasks:
                subtasks = subtasks["subtasks"]
            if not isinstance(subtasks, list):
                subtasks = [subtasks]
            self.state.planned_subtasks = subtasks
            logger.info(f"Planned {len(subtasks)} subtasks")
        except Exception as e:
            logger.warning(f"Could not parse subtasks JSON: {e}")
            self.state.planned_subtasks = self._create_fallback_subtasks()

        return "planning_complete"

    @listen(plan_subtasks)
    def validate_subtasks(self, _: str) -> str:
        """Validate the subtask breakdown and assign confidence scores."""
        logger.info("Validating subtasks...")

        validator = create_validator_agent(self.llm)

        validation_task = Task(
            description=f"""Validate the following task decomposition and assign confidence scores.

Original Task: {self.state.task_text}

Planned Subtasks:
{json.dumps(self.state.planned_subtasks, indent=2)}

For each subtask:
1. Verify it's clear and actionable
2. Check dependencies are logical
3. Assign a confidence score (0-100):
   - 90-100: Very confident, no changes needed
   - 70-89: Confident, minor adjustments possible
   - 50-69: Uncertain, would benefit from human review
   - Below 50: Significant concerns, requires review

Output JSON with:
- validated_subtasks: Array of subtasks with confidence_score added
- overall_confidence: Average confidence (0-100)
- issues: Array of any concerns found
- suggestions: Array of improvements if any""",
            expected_output="JSON validation results with confidence scores",
            agent=validator,
        )

        validation_crew = Crew(
            agents=[validator],
            tasks=[validation_task],
            process=Process.sequential,
            verbose=True,
        )

        result = validation_crew.kickoff()

        # Parse validation results
        try:
            validation = self._extract_json(result.raw)
            self.state.validated_subtasks = validation.get(
                "validated_subtasks", self.state.planned_subtasks
            )
            self.state.overall_confidence = validation.get("overall_confidence", 75)
            self.state.issues = validation.get("issues", [])
            self.state.suggestions = validation.get("suggestions", [])
            logger.info(f"Validation complete. Confidence: {self.state.overall_confidence}%")
        except Exception as e:
            logger.warning(f"Could not parse validation JSON: {e}")
            # Assign default confidence scores
            self.state.validated_subtasks = [
                {**s, "confidence_score": 75} for s in self.state.planned_subtasks
            ]
            self.state.overall_confidence = 75

        return "validation_complete"

    def _extract_json(self, text: str) -> dict | list:
        """Extract JSON from LLM response text.

        Handles cases where JSON is wrapped in markdown code blocks.
        """
        # Try to find JSON in code blocks first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON array or object
        text = text.strip()

        # Find first [ or {
        start_bracket = -1
        start_brace = -1

        for i, c in enumerate(text):
            if c == "[" and start_bracket == -1:
                start_bracket = i
            elif c == "{" and start_brace == -1:
                start_brace = i
            if start_bracket != -1 and start_brace != -1:
                break

        # Use whichever comes first
        if start_bracket == -1 and start_brace == -1:
            raise ValueError("No JSON found in text")

        if start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
            # Array
            end = text.rfind("]")
            if end != -1:
                text = text[start_bracket : end + 1]
        else:
            # Object
            end = text.rfind("}")
            if end != -1:
                text = text[start_brace : end + 1]

        return json.loads(text)

    def _create_fallback_subtasks(self) -> list[dict[str, Any]]:
        """Create fallback subtasks if planning fails."""
        return [
            {
                "title": f"Research: {self.state.task_text[:50]}",
                "description": "Research requirements and best practices",
                "subtask_type": SubtaskType.RESEARCH.value,
                "estimated_complexity": "low",
                "dependencies": [],
                "order_index": 0,
            },
            {
                "title": "Design solution approach",
                "description": "Create technical design based on research",
                "subtask_type": SubtaskType.DESIGN.value,
                "estimated_complexity": "medium",
                "dependencies": [0],
                "order_index": 1,
            },
            {
                "title": "Implement core functionality",
                "description": "Build the main implementation",
                "subtask_type": SubtaskType.CODING.value,
                "estimated_complexity": "high",
                "dependencies": [1],
                "order_index": 2,
            },
            {
                "title": "Write tests",
                "description": "Create unit and integration tests",
                "subtask_type": SubtaskType.TESTING.value,
                "estimated_complexity": "medium",
                "dependencies": [2],
                "order_index": 3,
            },
            {
                "title": "Document implementation",
                "description": "Write documentation and update README",
                "subtask_type": SubtaskType.DOCUMENTATION.value,
                "estimated_complexity": "low",
                "dependencies": [2],
                "order_index": 4,
            },
        ]


async def run_decomposition(
    task_text: str,
    project_context: str | None = None,
    taskmaster_context: dict[str, Any] | None = None,
) -> DecompositionState:
    """Run the task decomposition flow.

    Args:
        task_text: The task to decompose
        project_context: Optional project context
        taskmaster_context: Optional Task Master context

    Returns:
        DecompositionState with validated subtasks and confidence scores
    """
    flow = TaskDecompositionFlow()
    flow.state.task_text = task_text
    flow.state.project_context = project_context
    flow.state.taskmaster_context = taskmaster_context

    await flow.kickoff_async()

    return flow.state
