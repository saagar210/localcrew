"""Base service class with common functionality."""

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from localcrew.core.types import utcnow
from localcrew.models.execution import Execution, ExecutionStatus


class BaseCrewService:
    """Base class for crew services."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_execution(
        self,
        crew_type: str,
        input_text: str,
        input_config: dict[str, Any] | None = None,
    ) -> Execution:
        """Create a new execution record."""
        execution = Execution(
            crew_type=crew_type,
            input_text=input_text,
            input_config=input_config or {},
            status=ExecutionStatus.PENDING,
        )
        self.session.add(execution)
        await self.session.commit()
        await self.session.refresh(execution)
        return execution

    async def update_execution_status(
        self,
        execution_id: UUID,
        status: ExecutionStatus,
        output: dict[str, Any] | None = None,
        error_message: str | None = None,
        confidence_score: int | None = None,
        duration_ms: int | None = None,
        model_used: str | None = None,
        tokens_used: int | None = None,
    ) -> None:
        """Update execution status and metadata."""
        from sqlalchemy import select

        result = await self.session.execute(
            select(Execution).where(Execution.id == execution_id)
        )
        execution = result.scalar_one()

        execution.status = status
        if output:
            execution.output = output
        if error_message:
            execution.error_message = error_message
        if confidence_score is not None:
            execution.confidence_score = confidence_score
        if duration_ms is not None:
            execution.duration_ms = duration_ms
        if model_used:
            execution.model_used = model_used
        if tokens_used is not None:
            execution.tokens_used = tokens_used

        if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
            execution.completed_at = utcnow()

        await self.session.commit()
