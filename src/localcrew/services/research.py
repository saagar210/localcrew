"""Research service using CrewAI."""

import logging
import time
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from localcrew.core.config import settings
from localcrew.integrations.kas import get_kas
from localcrew.models.execution import ExecutionStatus
from localcrew.services.base import BaseCrewService

logger = logging.getLogger(__name__)


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

            # Auto-ingest to KAS if enabled (always when KAS is available)
            if settings.kas_enabled:
                await self._store_to_kas(execution, research_result)

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

    async def _store_to_kas(self, execution: Any, research_result: dict[str, Any]) -> str | None:
        """
        Store research findings to KAS.

        Auto-ingests the research report into the personal knowledge base
        for future reference and searchability.

        Args:
            execution: The Execution model instance
            research_result: The formatted research result dict

        Returns:
            KAS content_id if stored, None otherwise
        """
        kas = get_kas()
        if kas is None:
            return None

        try:
            # Extract topic tags from sub-questions
            topic_tags = self._extract_topic_tags(research_result)

            content_id = await kas.ingest_research(
                title=f"Research: {research_result['query'][:100]}",
                content=research_result.get("synthesis", ""),
                tags=["research", "localcrew"] + topic_tags,
                metadata={
                    "crew_type": "research",
                    "execution_id": str(execution.id),
                    "confidence": research_result.get("confidence_score", 0),
                    "query": research_result.get("query", ""),
                    "depth": research_result.get("depth", "medium"),
                },
            )

            if content_id:
                logger.info(f"Research ingested to KAS: {content_id}")
                # Update execution with KAS content ID
                execution.kas_content_id = content_id
                await self.session.commit()
                return content_id
            return None
        except Exception as e:
            # Don't fail execution - KAS is optional
            logger.warning(f"Failed to ingest to KAS: {e}")
            return None

    def _extract_topic_tags(self, research_result: dict[str, Any]) -> list[str]:
        """
        Extract topic tags from research result.

        Derives tags from sub-questions and synthesis themes.

        Args:
            research_result: The formatted research result dict

        Returns:
            List of topic tags (deduplicated)
        """
        tags = set()

        # Extract keywords from sub-questions
        for question in research_result.get("sub_questions", []):
            # Simple keyword extraction: split and filter short/common words
            words = question.lower().split()
            for word in words:
                # Keep words that are likely meaningful (3+ chars, alphanumeric)
                if len(word) >= 4 and word.isalnum():
                    tags.add(word)

        # Limit to top 10 tags
        return list(tags)[:10]
