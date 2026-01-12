"""Human review endpoints."""

import json
import logging
import re
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from localcrew.core.database import get_session
from localcrew.core.types import utcnow
from localcrew.integrations.kas import get_kas
from localcrew.models.review import Review, ReviewCreate, ReviewRead, ReviewDecision
from localcrew.models.feedback import Feedback, FeedbackType

logger = logging.getLogger(__name__)
router = APIRouter()


class SyncableTaskContent(BaseModel):
    """Validated content for Task Master sync.

    Sanitizes input to prevent JSON injection and ensures safe values.
    """

    title: str = Field(..., min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=5000)

    @field_validator("title", "description", mode="before")
    @classmethod
    def sanitize_text(cls, v: Any) -> str | None:
        """Remove control characters and sanitize text."""
        if v is None:
            return None
        if not isinstance(v, str):
            v = str(v)
        # Remove null bytes and control characters (except newline, tab)
        sanitized = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", v)
        return sanitized.strip()


def _validate_sync_content(content: dict | list) -> list[SyncableTaskContent]:
    """Validate and sanitize content before syncing to Task Master.

    Args:
        content: Raw content from review (dict or list of dicts)

    Returns:
        List of validated SyncableTaskContent objects

    Raises:
        ValueError: If content is invalid or contains malicious data
    """
    if isinstance(content, dict):
        content = [content]

    if not isinstance(content, list):
        raise ValueError("Content must be a dict or list of dicts")

    validated = []
    for item in content:
        if not isinstance(item, dict):
            raise ValueError("Each item must be a dict")
        if "title" not in item:
            raise ValueError("Each item must have a 'title' field")
        validated.append(SyncableTaskContent(
            title=item.get("title"),
            description=item.get("description"),
        ))

    return validated


async def _store_pattern_to_kas(review: Review) -> str | None:
    """Store an approved/modified pattern to KAS for future reference.

    When a human approves or modifies a decomposition, the validated
    pattern is valuable knowledge that can inform future decompositions.

    Args:
        review: The approved/modified review

    Returns:
        KAS content_id if stored, None otherwise
    """
    kas = get_kas()
    if kas is None:
        return None

    # Get the validated content (modified or original)
    content = review.modified_content or review.original_content
    if not content:
        return None

    try:
        # Build pattern title based on content
        if isinstance(content, dict):
            title = f"Pattern: {content.get('title', 'Task Decomposition')}"
        elif isinstance(content, list) and content:
            first_item = content[0]
            title = f"Pattern: {first_item.get('title', 'Task Decomposition')}"
        else:
            title = "Pattern: Task Decomposition"

        # Build markdown content for the pattern
        pattern_content = f"""# Validated Task Pattern

## Context
- **Review ID**: {review.id}
- **Execution ID**: {review.execution_id}
- **Decision**: {review.decision.value}
- **Confidence**: {review.confidence_score}%

## Pattern Details
```json
{json.dumps(content, indent=2)}
```

## Human Feedback
{review.feedback or "No additional feedback provided."}
"""

        # Determine domain from content if available
        domain = "general"
        if isinstance(content, dict):
            domain = content.get("domain", content.get("analysis", {}).get("domain", "general"))

        # Store to KAS
        content_id = await kas.ingest_research(
            title=title,
            content=pattern_content,
            tags=["pattern", "decomposition", f"domain:{domain}", "validated"],
            metadata={
                "review_id": str(review.id),
                "execution_id": str(review.execution_id),
                "confidence_score": review.confidence_score,
                "decision": review.decision.value,
            },
        )

        if content_id:
            logger.info(f"Stored validated pattern to KAS: {content_id}")
        return content_id

    except Exception as e:
        logger.warning(f"Failed to store pattern to KAS: {e}")
        return None


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

    # Store validated patterns to KAS for future decompositions
    if submission.decision in (ReviewDecision.APPROVED, ReviewDecision.MODIFIED):
        await _store_pattern_to_kas(review)

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

    # Validate and sanitize content before syncing
    try:
        validated_tasks = _validate_sync_content(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid content: {e}")

    # Sync to Task Master
    from localcrew.integrations.taskmaster import get_taskmaster
    taskmaster = get_taskmaster()

    try:
        synced_ids = []
        for task in validated_tasks:
            task_id = await taskmaster._create_task(
                title=task.title,
                description=task.description,
            )
            synced_ids.append(task_id)

        if len(synced_ids) == 1:
            return {"synced": True, "task_id": synced_ids[0]}
        return {"synced": True, "task_ids": synced_ids}

    except Exception as e:
        logger.error(f"Task Master sync failed: {e}")
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
