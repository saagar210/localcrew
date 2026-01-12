"""Crew execution endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from localcrew.core.database import get_session
from localcrew.models.execution import ExecutionCreate, ExecutionRead, ExecutionStatus

router = APIRouter()


class DecomposeRequest(BaseModel):
    """Request schema for task decomposition."""

    task: str = Field(..., min_length=10, max_length=5000, description="Task to decompose")
    project: str | None = Field(default=None, description="Project context from Task Master")
    include_taskmaster_context: bool = Field(default=True, description="Include recent Task Master history")
    auto_sync: bool = Field(default=True, description="Auto-sync subtasks to Task Master")


class DecomposeResponse(BaseModel):
    """Response schema for task decomposition."""

    execution_id: UUID
    status: ExecutionStatus
    message: str


class ResearchRequest(BaseModel):
    """Request schema for research."""

    query: str = Field(..., min_length=5, max_length=2000, description="Research query")
    depth: str = Field(default="medium", pattern="^(shallow|medium|deep)$")
    output_format: str = Field(default="markdown", pattern="^(markdown|json|plain)$")
    store_to_kas: bool = Field(default=False, description="Store findings to KAS")


class ResearchResponse(BaseModel):
    """Response schema for research."""

    execution_id: UUID
    status: ExecutionStatus
    message: str


@router.post("/decompose", response_model=DecomposeResponse)
async def decompose_task(
    request: DecomposeRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> DecomposeResponse:
    """
    Decompose a complex task into actionable subtasks.

    The task decomposition crew will:
    1. Analyze the task scope and requirements
    2. Break it into structured subtasks with types and dependencies
    3. Assign confidence scores to each subtask
    4. Auto-sync to Task Master (if enabled)
    5. Flag low-confidence items for human review
    """
    from localcrew.services.decomposition import DecompositionService

    service = DecompositionService(session)
    execution = await service.create_execution(
        crew_type="task_decomposition",
        input_text=request.task,
        input_config={
            "project": request.project,
            "include_taskmaster_context": request.include_taskmaster_context,
            "auto_sync": request.auto_sync,
        },
    )

    # Run decomposition in background
    background_tasks.add_task(service.run_decomposition, execution.id)

    return DecomposeResponse(
        execution_id=execution.id,
        status=ExecutionStatus.PENDING,
        message="Task decomposition started. Check /executions/{id} for results.",
    )


@router.post("/research", response_model=ResearchResponse)
async def research_topic(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> ResearchResponse:
    """
    Research a topic using the research crew.

    The research crew will:
    1. Break the query into sub-questions
    2. Gather information from multiple sources
    3. Synthesize findings
    4. Generate a structured report with citations
    """
    from localcrew.services.research import ResearchService

    service = ResearchService(session)
    execution = await service.create_execution(
        crew_type="research",
        input_text=request.query,
        input_config={
            "depth": request.depth,
            "output_format": request.output_format,
            "store_to_kas": request.store_to_kas,
        },
    )

    # Run research in background
    background_tasks.add_task(service.run_research, execution.id)

    return ResearchResponse(
        execution_id=execution.id,
        status=ExecutionStatus.PENDING,
        message="Research started. Check /executions/{id} for results.",
    )


@router.get("/types")
async def list_crew_types() -> dict[str, Any]:
    """List available crew types and their capabilities."""
    return {
        "crews": [
            {
                "type": "task_decomposition",
                "name": "Task Decomposition Crew",
                "description": "Breaks complex tasks into actionable subtasks",
                "agents": ["Analyzer", "Planner", "Validator"],
                "task_types": ["coding", "research", "devops", "documentation"],
            },
            {
                "type": "research",
                "name": "Research Crew",
                "description": "Deep research with synthesis and reporting",
                "agents": ["QueryDecomposer", "Gatherer", "Synthesizer", "Reporter"],
                "depths": ["shallow", "medium", "deep"],
            },
        ]
    }
