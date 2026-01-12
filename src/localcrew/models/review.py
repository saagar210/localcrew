"""Review model for human-in-the-loop feedback."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime
from sqlmodel import Column, Field, SQLModel

from localcrew.core.types import JSONType, utcnow


class ReviewDecision(str, Enum):
    """Human review decision."""

    PENDING = "pending"
    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"
    RERUN = "rerun"


class ReviewBase(SQLModel):
    """Base review fields (no JSON columns)."""

    execution_id: UUID = Field(foreign_key="executions.id", index=True)
    subtask_id: UUID | None = Field(default=None, foreign_key="subtasks.id", index=True)
    decision: ReviewDecision = Field(default=ReviewDecision.PENDING)
    feedback: str | None = Field(default=None, max_length=2000)
    confidence_score: int = Field(ge=0, le=100)


class Review(ReviewBase, table=True):
    """Review database model."""

    __tablename__ = "reviews"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    original_content: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONType))
    modified_content: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONType))
    created_at: datetime = Field(default_factory=utcnow, sa_column=Column(DateTime(timezone=True)))
    reviewed_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))


class ReviewCreate(SQLModel):
    """Schema for creating a review."""

    execution_id: UUID
    subtask_id: UUID | None = None
    original_content: dict[str, Any]
    confidence_score: int


class ReviewRead(ReviewBase):
    """Schema for reading a review."""

    id: UUID
    original_content: dict[str, Any] | None = None
    modified_content: dict[str, Any] | None = None
    created_at: datetime
    reviewed_at: datetime | None
