"""Human review endpoints."""

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from localcrew.core.database import get_session
from localcrew.core.types import utcnow
from localcrew.models.review import Review, ReviewCreate, ReviewRead, ReviewDecision
from localcrew.models.feedback import Feedback, FeedbackType

logger = logging.getLogger(__name__)
router = APIRouter()


async def _store_feedback(session: AsyncSession, review: Review) -> None:
    """Store feedback from a review for prompt improvement analysis."""
    # Map review decision to feedback type
    feedback_type_map = {
        ReviewDecision.APPROVED: FeedbackType.APPROVAL,
        ReviewDecision.MODIFIED: FeedbackType.MODIFICATION,
        ReviewDecision.REJECTED: FeedbackType.REJECTION,
        ReviewDecision.RERUN: FeedbackType.RERUN,
    }

    feedback_type = feedback_type_map.get(review.decision)
    if not feedback_type:
        return  # Don't store feedback for pending reviews

    feedback = Feedback(
        review_id=review.id,
        execution_id=review.execution_id,
        feedback_type=feedback_type,
        feedback_text=review.feedback,
        confidence_score=review.confidence_score,
        original_content=review.original_content,
        modified_content=review.modified_content,
    )

    session.add(feedback)
    await session.commit()
    logger.info(f"Stored feedback for review {review.id}: {feedback_type.value}")


class ReviewSubmission(BaseModel):
    """Schema for submitting a review decision."""

    decision: ReviewDecision
    modified_content: dict | None = Field(default=None)
    feedback: str | None = Field(default=None, max_length=2000)


@router.get("/pending", response_model=list[ReviewRead])
async def list_pending_reviews(
    skip: int = 0,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
) -> list[ReviewRead]:
    """List all pending reviews."""
    query = (
        select(Review)
        .where(Review.decision == ReviewDecision.PENDING)
        .offset(skip)
        .limit(limit)
        .order_by(Review.created_at.asc())
    )

    result = await session.execute(query)
    reviews = result.scalars().all()
    return [ReviewRead.model_validate(r) for r in reviews]


@router.get("/stats")
async def review_stats(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get review statistics."""
    from sqlalchemy import func

    # Count by decision
    result = await session.execute(
        select(Review.decision, func.count(Review.id))
        .group_by(Review.decision)
    )
    counts = {str(row[0].value): row[1] for row in result.all()}

    return {
        "total": sum(counts.values()),
        "by_decision": counts,
        "pending": counts.get("pending", 0),
    }


@router.get("/{review_id}", response_model=ReviewRead)
async def get_review(
    review_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ReviewRead:
    """Get a review by ID."""
    result = await session.execute(select(Review).where(Review.id == review_id))
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return ReviewRead.model_validate(review)


@router.post("/{review_id}/submit", response_model=ReviewRead)
async def submit_review(
    review_id: UUID,
    submission: ReviewSubmission,
    session: AsyncSession = Depends(get_session),
) -> ReviewRead:
    """Submit a review decision."""
    result = await session.execute(select(Review).where(Review.id == review_id))
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    if review.decision != ReviewDecision.PENDING:
        raise HTTPException(status_code=400, detail="Review already submitted")

    review.decision = submission.decision
    review.modified_content = submission.modified_content
    review.feedback = submission.feedback
    review.reviewed_at = utcnow()

    await session.commit()
    await session.refresh(review)

    # Store feedback for prompt improvement
    if submission.feedback:
        await _store_feedback(session, review)

    return ReviewRead.model_validate(review)


@router.post("/{review_id}/sync")
async def sync_review_to_taskmaster(
    review_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Sync an approved/modified review to Task Master."""
    result = await session.execute(select(Review).where(Review.id == review_id))
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    if review.decision not in (ReviewDecision.APPROVED, ReviewDecision.MODIFIED):
        raise HTTPException(
            status_code=400,
            detail="Only approved or modified reviews can be synced"
        )

    # Get content to sync
    content = review.modified_content or review.original_content
    if not content:
        raise HTTPException(status_code=400, detail="No content to sync")

    # Sync to Task Master
    from localcrew.integrations.taskmaster import get_taskmaster
    taskmaster = get_taskmaster()

    try:
        # If content is a subtask, create it in Task Master
        if isinstance(content, dict) and "title" in content:
            task_id = await taskmaster._create_task(
                title=content.get("title", "Untitled"),
                description=content.get("description"),
            )
            return {"synced": True, "task_id": task_id}

        # If content is a list of subtasks
        if isinstance(content, list):
            synced_ids = await taskmaster.sync_subtasks(
                execution_id=review.execution_id,
                subtasks=content,
            )
            return {"synced": True, "task_ids": synced_ids}

        return {"synced": False, "reason": "Unknown content format"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")


class RerunRequest(BaseModel):
    """Schema for rerun request."""
    guidance: str = Field(..., min_length=1, max_length=2000)


@router.post("/{review_id}/rerun")
async def rerun_review(
    review_id: UUID,
    request: RerunRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Rerun the decomposition with additional guidance."""
    from localcrew.models.execution import Execution

    result = await session.execute(select(Review).where(Review.id == review_id))
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    # Get the original execution
    exec_result = await session.execute(
        select(Execution).where(Execution.id == review.execution_id)
    )
    original_execution = exec_result.scalar_one_or_none()
    if not original_execution:
        raise HTTPException(status_code=404, detail="Original execution not found")

    # Create a new execution with the guidance
    from uuid import uuid4
    from localcrew.models.execution import ExecutionStatus

    new_execution = Execution(
        id=uuid4(),
        crew_type=original_execution.crew_type,
        input_text=original_execution.input_text,
        input_config={
            **(original_execution.input_config or {}),
            "rerun_guidance": request.guidance,
            "original_execution_id": str(original_execution.id),
        },
        status=ExecutionStatus.PENDING,
    )
    session.add(new_execution)
    await session.commit()

    # Trigger the decomposition in background using FastAPI's BackgroundTasks
    from localcrew.services.decomposition import DecompositionService

    service = DecompositionService(session)
    background_tasks.add_task(service.run_decomposition, new_execution.id)

    return {
        "execution_id": str(new_execution.id),
        "status": "pending",
        "guidance": request.guidance,
    }
