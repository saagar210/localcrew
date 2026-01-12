"""Subtask model for decomposed task outputs."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime
from sqlmodel import Column, Field, SQLModel

from localcrew.core.types import JSONType, utcnow


class SubtaskType(str, Enum):
    """Type of subtask."""

    CODING = "coding"
    RESEARCH = "research"
    DEVOPS = "devops"
    DOCUMENTATION = "documentation"
    DESIGN = "design"
    TESTING = "testing"


class SubtaskBase(SQLModel):
    """Base subtask fields (no JSON columns)."""

    execution_id: UUID = Field(foreign_key="executions.id", index=True)
    title: str = Field(max_length=500)
    description: str | None = Field(default=None, max_length=2000)
    subtask_type: SubtaskType
    estimated_complexity: str = Field(default="medium", max_length=20)  # low, medium, high
    confidence_score: int = Field(ge=0, le=100)
    order_index: int = Field(default=0)

    # Task Master sync
    taskmaster_id: str | None = Field(default=None, max_length=50)
    synced_to_taskmaster: bool = Field(default=False)


class Subtask(SubtaskBase, table=True):
    """Subtask database model."""

    __tablename__ = "subtasks"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    dependencies: list[int] | None = Field(default=None, sa_column=Column(JSONType))
    extra_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONType))
    created_at: datetime = Field(default_factory=utcnow, sa_column=Column(DateTime(timezone=True)))


class SubtaskCreate(SQLModel):
    """Schema for creating a subtask."""

    execution_id: UUID
    title: str
    description: str | None = None
    subtask_type: SubtaskType
    estimated_complexity: str = "medium"
    dependencies: list[int] | None = None
    confidence_score: int
    order_index: int = 0
    extra_data: dict[str, Any] | None = None


class SubtaskRead(SubtaskBase):
    """Schema for reading a subtask."""

    id: UUID
    dependencies: list[int] | None = None
    extra_data: dict[str, Any] | None = None
    created_at: datetime
