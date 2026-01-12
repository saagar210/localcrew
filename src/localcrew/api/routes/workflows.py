"""Workflow management endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from localcrew.core.database import get_session
from localcrew.models.workflow import Workflow, WorkflowCreate, WorkflowRead

router = APIRouter()


@router.get("", response_model=list[WorkflowRead])
async def list_workflows(
    skip: int = 0,
    limit: int = 50,
    active_only: bool = True,
    session: AsyncSession = Depends(get_session),
) -> list[WorkflowRead]:
    """List all workflows."""
    query = select(Workflow)
    if active_only:
        query = query.where(Workflow.is_active.is_(True))
    query = query.offset(skip).limit(limit).order_by(Workflow.created_at.desc())

    result = await session.execute(query)
    workflows = result.scalars().all()
    return [WorkflowRead.model_validate(w) for w in workflows]


@router.post("", response_model=WorkflowRead)
async def create_workflow(
    workflow: WorkflowCreate,
    session: AsyncSession = Depends(get_session),
) -> WorkflowRead:
    """Create a new workflow."""
    db_workflow = Workflow.model_validate(workflow)
    session.add(db_workflow)
    await session.commit()
    await session.refresh(db_workflow)
    return WorkflowRead.model_validate(db_workflow)


@router.get("/{workflow_id}", response_model=WorkflowRead)
async def get_workflow(
    workflow_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> WorkflowRead:
    """Get a workflow by ID."""
    result = await session.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return WorkflowRead.model_validate(workflow)


@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Soft delete a workflow (set inactive)."""
    result = await session.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow.is_active = False
    await session.commit()
    return {"message": "Workflow deactivated", "id": str(workflow_id)}
