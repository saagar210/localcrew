"""Workflow model for storing workflow definitions."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime
from sqlmodel import Column, Field, SQLModel

from localcrew.core.types import JSONType, utcnow


class WorkflowBase(SQLModel):
    """Base workflow fields (no JSON columns)."""

    name: str = Field(index=True, max_length=255)
    description: str | None = Field(default=None, max_length=1000)
    crew_type: str = Field(max_length=50)  # task_decomposition, research, etc.
    is_active: bool = Field(default=True)


class Workflow(WorkflowBase, table=True):
    """Workflow database model."""

    __tablename__ = "workflows"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONType))
    created_at: datetime = Field(default_factory=utcnow, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime = Field(default_factory=utcnow, sa_column=Column(DateTime(timezone=True)))


class WorkflowCreate(WorkflowBase):
    """Schema for creating a workflow."""

    config: dict[str, Any] | None = None


class WorkflowRead(WorkflowBase):
    """Schema for reading a workflow."""

    id: UUID
    config: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
