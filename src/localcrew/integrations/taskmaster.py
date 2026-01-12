"""Task Master AI integration for task synchronization."""

import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from localcrew.models.subtask import SubtaskType

logger = logging.getLogger(__name__)


class TaskMasterIntegration:
    """Integration with Task Master AI via MCP tools.

    Task Master uses a file-based task storage in .taskmaster/tasks/tasks.json.
    We integrate via the task-master CLI and direct file access.
    """

    def __init__(self, project_root: str | None = None) -> None:
        """Initialize Task Master integration.

        Args:
            project_root: Root directory of the project with .taskmaster folder.
                         If not provided, uses current working directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self._tasks_file = self.project_root / ".taskmaster" / "tasks" / "tasks.json"

    def _load_tasks(self) -> dict[str, Any]:
        """Load tasks from Task Master tasks.json file."""
        if not self._tasks_file.exists():
            logger.warning(f"Task Master tasks file not found: {self._tasks_file}")
            return {"tasks": []}

        try:
            with open(self._tasks_file) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tasks.json: {e}")
            return {"tasks": []}

    def _save_tasks(self, data: dict[str, Any]) -> bool:
        """Save tasks to Task Master tasks.json file."""
        try:
            self._tasks_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._tasks_file, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save tasks.json: {e}")
            return False

    async def get_project_context(self, project_name: str | None = None) -> dict[str, Any]:
        """
        Get recent task history from Task Master for context.

        This provides context for better task decomposition.
        """
        logger.info(f"Getting Task Master context for project: {project_name}")

        data = self._load_tasks()
        tasks = data.get("tasks", [])

        # Get active and recent tasks
        active_tasks = [t for t in tasks if t.get("status") in ("pending", "in-progress")]
        completed_tasks = [t for t in tasks if t.get("status") == "done"][-10:]  # Last 10

        return {
            "recent_tasks": [
                {"id": t.get("id"), "title": t.get("title"), "status": t.get("status")}
                for t in completed_tasks
            ],
            "active_tasks": [
                {
                    "id": t.get("id"),
                    "title": t.get("title"),
                    "status": t.get("status"),
                    "description": t.get("description", "")[:200],
                }
                for t in active_tasks[:10]
            ],
            "total_tasks": len(tasks),
        }

    async def create_subtask(
        self,
        parent_task_id: str,
        title: str,
        description: str | None = None,
        subtask_type: SubtaskType | None = None,
        dependencies: list[str] | None = None,
    ) -> str | None:
        """
        Create a subtask in Task Master.

        Returns the created subtask ID or None if failed.
        """
        logger.info(f"Creating subtask in Task Master: {title}")

        data = self._load_tasks()
        tasks = data.get("tasks", [])

        # Find the parent task
        parent_task = None
        for task in tasks:
            if str(task.get("id")) == str(parent_task_id):
                parent_task = task
                break

        if not parent_task:
            logger.warning(f"Parent task {parent_task_id} not found")
            # Create as a new top-level task instead
            return await self._create_task(title, description, subtask_type, dependencies)

        # Add subtask to parent
        if "subtasks" not in parent_task:
            parent_task["subtasks"] = []

        subtask_id = len(parent_task["subtasks"]) + 1
        subtask = {
            "id": subtask_id,
            "title": title,
            "status": "pending",
        }
        if description:
            subtask["description"] = description
        if dependencies:
            subtask["dependencies"] = dependencies

        parent_task["subtasks"].append(subtask)

        if self._save_tasks(data):
            return f"{parent_task_id}.{subtask_id}"
        return None

    async def _create_task(
        self,
        title: str,
        description: str | None = None,
        subtask_type: SubtaskType | None = None,
        dependencies: list[str] | None = None,
    ) -> str | None:
        """Create a new top-level task."""
        data = self._load_tasks()
        tasks = data.get("tasks", [])

        # Find next task ID
        max_id = max([t.get("id", 0) for t in tasks], default=0)
        new_id = max_id + 1

        task = {
            "id": new_id,
            "title": title,
            "status": "pending",
        }
        if description:
            task["description"] = description
        if subtask_type:
            task["type"] = subtask_type.value
        if dependencies:
            task["dependencies"] = dependencies

        tasks.append(task)
        data["tasks"] = tasks

        if self._save_tasks(data):
            return str(new_id)
        return None

    async def sync_subtasks(
        self,
        execution_id: UUID,
        subtasks: list[dict[str, Any]],
        parent_task_id: str | None = None,
    ) -> dict[str, str]:
        """
        Sync multiple subtasks to Task Master.

        If parent_task_id is provided, creates subtasks under that task.
        Otherwise, creates them as new top-level tasks.

        Returns mapping of local subtask index to Task Master task ID.
        """
        logger.info(f"Syncing {len(subtasks)} subtasks to Task Master")
        synced_ids: dict[str, str] = {}

        if parent_task_id:
            # Create as subtasks of parent
            for i, subtask in enumerate(subtasks):
                task_id = await self.create_subtask(
                    parent_task_id=parent_task_id,
                    title=subtask["title"],
                    description=subtask.get("description"),
                    subtask_type=SubtaskType(subtask["subtask_type"]) if subtask.get("subtask_type") else None,
                    dependencies=[str(d) for d in subtask.get("dependencies", [])],
                )
                if task_id:
                    synced_ids[str(i)] = task_id
        else:
            # Create as top-level tasks
            for i, subtask in enumerate(subtasks):
                task_id = await self._create_task(
                    title=subtask["title"],
                    description=subtask.get("description"),
                    subtask_type=SubtaskType(subtask["subtask_type"]) if subtask.get("subtask_type") else None,
                    dependencies=[str(d) for d in subtask.get("dependencies", [])],
                )
                if task_id:
                    synced_ids[str(i)] = task_id

        logger.info(f"Synced {len(synced_ids)}/{len(subtasks)} subtasks to Task Master")
        return synced_ids

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        notes: str | None = None,
    ) -> bool:
        """
        Update a task's status in Task Master.

        Returns True if successful.
        """
        logger.info(f"Updating Task Master task {task_id} status to: {status}")

        data = self._load_tasks()
        tasks = data.get("tasks", [])

        # Handle subtask IDs (e.g., "5.2")
        if "." in task_id:
            parent_id, subtask_id = task_id.split(".", 1)
            for task in tasks:
                if str(task.get("id")) == parent_id:
                    subtasks = task.get("subtasks", [])
                    for subtask in subtasks:
                        if str(subtask.get("id")) == subtask_id:
                            subtask["status"] = status
                            if notes:
                                subtask["notes"] = notes
                            return self._save_tasks(data)
            return False

        # Handle regular task IDs
        for task in tasks:
            if str(task.get("id")) == task_id:
                task["status"] = status
                if notes:
                    task["notes"] = notes
                return self._save_tasks(data)

        logger.warning(f"Task {task_id} not found")
        return False

    async def get_next_task(self) -> dict[str, Any] | None:
        """
        Get the next recommended task from Task Master.

        Finds the first pending task with all dependencies completed.
        """
        logger.info("Getting next task from Task Master")

        data = self._load_tasks()
        tasks = data.get("tasks", [])

        # Get completed task IDs
        completed_ids = {str(t.get("id")) for t in tasks if t.get("status") == "done"}

        # Find first pending task with dependencies met
        for task in tasks:
            if task.get("status") != "pending":
                continue

            deps = task.get("dependencies", [])
            if all(str(d) in completed_ids for d in deps):
                return {
                    "id": task.get("id"),
                    "title": task.get("title"),
                    "description": task.get("description", ""),
                    "status": task.get("status"),
                    "dependencies": deps,
                }

        return None

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get a specific task by ID."""
        data = self._load_tasks()
        tasks = data.get("tasks", [])

        for task in tasks:
            if str(task.get("id")) == task_id:
                return task

        return None


# Singleton instance
_taskmaster_instance: TaskMasterIntegration | None = None


def get_taskmaster(project_root: str | None = None) -> TaskMasterIntegration:
    """Get or create Task Master integration instance."""
    global _taskmaster_instance
    if _taskmaster_instance is None or project_root:
        _taskmaster_instance = TaskMasterIntegration(project_root)
    return _taskmaster_instance
