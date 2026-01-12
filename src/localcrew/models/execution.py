"""Execution model for tracking workflow runs."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime
from sqlmodel import Column, Field, SQLModel

from localcrew.core.types import JSONType, utcnow


class ExecutionStatus(str, Enum):
    """Status of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class ExecutionBase(SQLModel):
    """Base execution fields (no JSON columns)."""

    workflow_id: UUID | None = Field(default=None, foreign_key="workflows.id", index=True)
    crew_type: str = Field(max_length=50)
    input_text: str
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    error_message: str | None = Field(default=None, max_length=2000)
    confidence_score: int | None = Field(default=None, ge=0, le=100)
    duration_ms: int | None = Field(default=None)
    model_used: str | None = Field(default=None, max_length=100)
    tokens_used: int | None = Field(default=None)
    kas_content_id: str | None = Field(default=None, max_length=100, description="KAS content ID if stored")


class Execution(ExecutionBase, table=True):
    """Execution database model."""

    __tablename__ = "executions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    input_config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONType))
    output: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONType))
    created_at: datetime = Field(default_factory=utcnow, sa_column=Column(DateTime(timezone=True)))
    completed_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))


class ExecutionCreate(SQLModel):
    """Schema for creating an execution."""

    workflow_id: UUID | None = None
    crew_type: str
    input_text: str
    input_config: dict[str, Any] | None = None


class ExecutionRead(ExecutionBase):
    """Schema for reading an execution."""

    id: UUID
    input_config: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    created_at: datetime
    completed_at: datetime | None
