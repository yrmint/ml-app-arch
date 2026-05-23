from uuid import UUID
import datetime
import json
import redis.asyncio as redis

from backend.app.models.task_model import TaskStatusResponse
from backend.app.core.config import settings


class TaskService:
    def __init__(self):
        self.redis = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            encoding="utf-8"
        )

    async def create_task(self, task_id: UUID, filename: str):
        """Creates new task with pending status"""
        task_data = {
            "task_id": str(task_id),
            "status": "pending",
            "filename": filename,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        await self.redis.hmset(f"task:{task_id}", task_data)
        await self.redis.expire(f"task:{task_id}", settings.REDIS_TASK_TTL)

    async def update_status(
            self,
            task_id: UUID,
            status: str,
            result: dict = None,
            error: str = None
    ):
        """Updates task status"""
        data = {
            "status": status,
            "processed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        if result:
            data["result"] = json.dumps(result)
        if error:
            data["error_message"] = error

        await self.redis.hmset(f"task:{task_id}", data)

    async def get_task(self, task_id: UUID) -> TaskStatusResponse | None:
        """Gets task status"""
        data = await self.redis.hgetall(f"task:{task_id}")
        if not data:
            return None

        if data.get("result"):
            data["result"] = json.loads(data["result"])

        return TaskStatusResponse(**data)


task_service = TaskService()
