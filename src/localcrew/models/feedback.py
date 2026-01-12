"""Feedback model for storing human review feedback for prompt improvement."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime
from sqlmodel import Column, Field, SQLModel

from localcrew.core.types import JSONType, utcnow


class FeedbackType(str, Enum):
    """Type of feedback."""

    APPROVAL = "approval"  # Approved without changes
    MODIFICATION = "modification"  # Approved with changes
    REJECTION = "rejection"  # Rejected
    RERUN = "rerun"  # Requested rerun with guidance


class FeedbackBase(SQLModel):
    """Base feedback fields (no JSON columns)."""

    review_id: UUID = Field(foreign_key="reviews.id", index=True)
    execution_id: UUID = Field(foreign_key="executions.id", index=True)
    feedback_type: FeedbackType
    feedback_text: str | None = Field(default=None, max_length=2000)
    confidence_score: int = Field(ge=0, le=100)


class Feedback(FeedbackBase, table=True):
    """Feedback database model for storing human review feedback.

    This data is used to analyze patterns in human corrections
    and improve prompts over time.
    """

    __tablename__ = "feedback"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    original_content: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONType))
    modified_content: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONType))
    created_at: datetime = Field(default_factory=utcnow, sa_column=Column(DateTime(timezone=True)))


class FeedbackCreate(SQLModel):
    """Schema for creating feedback."""

    review_id: UUID
    execution_id: UUID
    feedback_type: FeedbackType
    feedback_text: str | None = None
    confidence_score: int
    original_content: dict[str, Any] | None = None
    modified_content: dict[str, Any] | None = None


class FeedbackRead(FeedbackBase):
    """Schema for reading feedback."""

    id: UUID
    original_content: dict[str, Any] | None = None
    modified_content: dict[str, Any] | None = None
    created_at: datetime


class FeedbackSummary(SQLModel):
    """Summary of feedback for analysis."""

    total_count: int
    approval_rate: float
    modification_rate: float
    rejection_rate: float
    avg_confidence_on_approval: float
    avg_confidence_on_rejection: float
    common_feedback_themes: list[str]
