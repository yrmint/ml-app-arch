import logging
from fastapi import APIRouter, HTTPException
from uuid import UUID

from backend.app.models.task_model import TaskStatusResponse
from backend.app.services.task_service import task_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/tasks",
    tags=["Tasks"],
)


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: UUID):
    """
    Returns current status of task with task_id.
    """
    task = await task_service.get_task(task_id)

    if not task:
        logger.warning(
            "Task not found | task_id=%s",
            task_id
        )
        raise HTTPException(
            status_code=404,
            detail=f"Task with id {task_id} not found"
        )

    return task
