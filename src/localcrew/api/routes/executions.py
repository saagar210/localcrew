"""Execution history and status endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from localcrew.core.database import get_session
from localcrew.models.execution import Execution, ExecutionRead, ExecutionStatus
from localcrew.models.subtask import Subtask, SubtaskRead

router = APIRouter()


@router.get("", response_model=list[ExecutionRead])
async def list_executions(
    skip: int = 0,
    limit: int = 50,
    status: ExecutionStatus | None = None,
    crew_type: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[ExecutionRead]:
    """List all executions with optional filters."""
    query = select(Execution)

    if status:
        query = query.where(Execution.status == status)
    if crew_type:
        query = query.where(Execution.crew_type == crew_type)

    query = query.offset(skip).limit(limit).order_by(Execution.created_at.desc())

    result = await session.execute(query)
    executions = result.scalars().all()
    return [ExecutionRead.model_validate(e) for e in executions]


@router.get("/{execution_id}", response_model=ExecutionRead)
async def get_execution(
    execution_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ExecutionRead:
    """Get execution details by ID."""
    result = await session.execute(select(Execution).where(Execution.id == execution_id))
    execution = result.scalar_one_or_none()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    return ExecutionRead.model_validate(execution)


@router.get("/{execution_id}/subtasks", response_model=list[SubtaskRead])
async def get_execution_subtasks(
    execution_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> list[SubtaskRead]:
    """Get all subtasks for an execution."""
    # Verify execution exists
    result = await session.execute(select(Execution).where(Execution.id == execution_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Execution not found")

    # Get subtasks
    result = await session.execute(
        select(Subtask)
        .where(Subtask.execution_id == execution_id)
        .order_by(Subtask.order_index)
    )
    subtasks = result.scalars().all()
    return [SubtaskRead.model_validate(s) for s in subtasks]


@router.post("/{execution_id}/retry")
async def retry_execution(
    execution_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Retry a failed execution."""
    result = await session.execute(select(Execution).where(Execution.id == execution_id))
    execution = result.scalar_one_or_none()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    if execution.status not in [ExecutionStatus.FAILED, ExecutionStatus.REVIEW_REQUIRED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retry execution with status: {execution.status}"
        )

    # TODO: Implement retry logic
    return {"message": "Retry not yet implemented", "execution_id": str(execution_id)}
