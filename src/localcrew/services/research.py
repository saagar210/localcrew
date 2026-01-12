"""Research service using CrewAI."""

import time
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from localcrew.core.config import settings
from localcrew.models.execution import ExecutionStatus
from localcrew.services.base import BaseCrewService


class ResearchService(BaseCrewService):
    """Service for research using CrewAI."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def run_research(self, execution_id: UUID) -> None:
        """
        Run the research crew.

        This is called as a background task.
        """
        start_time = time.time()

        try:
            # Update status to running
            await self.update_execution_status(execution_id, ExecutionStatus.RUNNING)

            # Get execution details
            from sqlalchemy import select
            from localcrew.models.execution import Execution

            result = await self.session.execute(
                select(Execution).where(Execution.id == execution_id)
            )
            execution = result.scalar_one()

            # Run the research crew
            research_result = await self._execute_crew(
                query=execution.input_text,
                config=execution.input_config,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            await self.update_execution_status(
                execution_id=execution_id,
                status=ExecutionStatus.COMPLETED,
                output=research_result,
                confidence_score=research_result.get("confidence_score", 80),
                duration_ms=duration_ms,
                model_used=settings.mlx_model_id,
            )

            # Store to KAS if enabled
            if execution.input_config.get("store_to_kas", False):
                await self._store_to_kas(research_result)

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            await self.update_execution_status(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms,
            )
            raise

    async def _execute_crew(
        self,
        query: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute the CrewAI research crew.

        Uses the ResearchFlow with Query Decomposer, Gatherer, Synthesizer,
        and Reporter agents.
        """
        from localcrew.crews.research import run_research

        depth = config.get("depth", "medium")
        output_format = config.get("output_format", "markdown")

        # Run the research flow
        result = await run_research(
            query=query,
            depth=depth,
            output_format=output_format,
        )

        # Format the result
        return {
            "query": query,
            "depth": depth,
            "format": output_format,
            "confidence_score": result.confidence_score,
            "sub_questions": [sq.question for sq in result.sub_questions],
            "findings": [
                {
                    "source": f.source_url,
                    "title": f.source_title,
                    "retrieved_at": f.retrieved_at,
                    "summary": f.content[:200] + "..." if len(f.content) > 200 else f.content,
                }
                for f in result.findings
            ],
            "synthesis": result.report,
            "sources": result.sources,
        }

    async def _store_to_kas(self, research_result: dict[str, Any]) -> None:
        """
        Store research findings to KAS.

        TODO: Implement KAS integration.
        """
        # TODO: Integrate with KAS API
        pass
