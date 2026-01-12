"""Task decomposition service using CrewAI."""

import time
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from localcrew.core.config import settings
from localcrew.models.execution import ExecutionStatus
from localcrew.models.subtask import Subtask, SubtaskType
from localcrew.models.review import Review
from localcrew.services.base import BaseCrewService


class DecompositionService(BaseCrewService):
    """Service for task decomposition using CrewAI."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def run_decomposition(self, execution_id: UUID) -> None:
        """
        Run the task decomposition crew.

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

            # Run the decomposition crew
            subtasks = await self._execute_crew(
                task_text=execution.input_text,
                config=execution.input_config,
            )

            # Calculate overall confidence
            overall_confidence = self._calculate_confidence(subtasks)

            # Save subtasks
            await self._save_subtasks(execution_id, subtasks)

            # Check if review is needed
            needs_review = overall_confidence < settings.confidence_threshold
            low_confidence_subtasks = [
                s for s in subtasks if s["confidence_score"] < settings.confidence_threshold
            ]

            if needs_review:
                # Create review records for low-confidence subtasks
                await self._create_reviews(execution_id, subtasks, low_confidence_subtasks)
                status = ExecutionStatus.REVIEW_REQUIRED
            else:
                status = ExecutionStatus.COMPLETED
                # Auto-sync to Task Master if enabled
                if execution.input_config.get("auto_sync", True):
                    await self._sync_to_taskmaster(execution_id, subtasks)

            duration_ms = int((time.time() - start_time) * 1000)

            await self.update_execution_status(
                execution_id=execution_id,
                status=status,
                output={"subtasks": subtasks, "needs_review": needs_review},
                confidence_score=overall_confidence,
                duration_ms=duration_ms,
                model_used=settings.mlx_model_id,
            )

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
        task_text: str,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Execute the CrewAI decomposition crew.

        Uses the TaskDecompositionFlow with Analyzer, Planner, and Validator agents.
        """
        from localcrew.crews.decomposition import run_decomposition

        # Get Task Master context if requested
        taskmaster_context = None
        if config.get("include_taskmaster_context"):
            taskmaster_context = await self._get_taskmaster_context(config.get("project"))

        # Run the decomposition flow
        result = await run_decomposition(
            task_text=task_text,
            project_context=config.get("project"),
            taskmaster_context=taskmaster_context,
        )

        # Return validated subtasks with confidence scores
        return result.validated_subtasks

    async def _get_taskmaster_context(self, project: str | None) -> dict[str, Any] | None:
        """Fetch context from Task Master AI.

        Gets recent and active tasks to provide context for decomposition.
        """
        from localcrew.integrations.taskmaster import get_taskmaster

        try:
            taskmaster = get_taskmaster()
            return await taskmaster.get_project_context(project)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to get Task Master context: {e}")
            return None

    def _calculate_confidence(self, subtasks: list[dict[str, Any]]) -> int:
        """Calculate overall confidence score from subtasks."""
        if not subtasks:
            return 0
        scores = [s.get("confidence_score", 0) for s in subtasks]
        return int(sum(scores) / len(scores))

    async def _save_subtasks(
        self,
        execution_id: UUID,
        subtasks: list[dict[str, Any]],
    ) -> None:
        """Save subtasks to database."""
        for subtask_data in subtasks:
            subtask = Subtask(
                execution_id=execution_id,
                title=subtask_data["title"],
                description=subtask_data.get("description"),
                subtask_type=SubtaskType(subtask_data["subtask_type"]),
                estimated_complexity=subtask_data.get("estimated_complexity", "medium"),
                dependencies=subtask_data.get("dependencies"),
                confidence_score=subtask_data["confidence_score"],
                order_index=subtask_data.get("order_index", 0),
            )
            self.session.add(subtask)

        await self.session.commit()

    async def _create_reviews(
        self,
        execution_id: UUID,
        all_subtasks: list[dict[str, Any]],
        low_confidence_subtasks: list[dict[str, Any]],
    ) -> None:
        """Create review records for low-confidence subtasks."""
        for subtask in low_confidence_subtasks:
            review = Review(
                execution_id=execution_id,
                original_content=subtask,
                confidence_score=subtask["confidence_score"],
            )
            self.session.add(review)

        await self.session.commit()

    async def _sync_to_taskmaster(
        self,
        execution_id: UUID,
        subtasks: list[dict[str, Any]],
    ) -> None:
        """
        Sync subtasks to Task Master AI.

        Creates subtasks in Task Master's tasks.json file.
        """
        from localcrew.integrations.taskmaster import get_taskmaster

        try:
            taskmaster = get_taskmaster()
            synced_ids = await taskmaster.sync_subtasks(
                execution_id=execution_id,
                subtasks=subtasks,
            )

            # Log the sync result
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Synced {len(synced_ids)} subtasks to Task Master for execution {execution_id}"
            )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to sync to Task Master: {e}")
