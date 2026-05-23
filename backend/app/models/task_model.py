from uuid import UUID
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class TaskAcceptedResponse(BaseModel):
    task_id: UUID
    status: str = "accepted"
    message: str = "Task accepted"


class TaskStatusResponse(BaseModel):
    task_id: UUID
    status: str                    # pending, processing, completed, failed
    filename: str
    created_at: datetime
    processed_at: Optional[datetime] = None
    result: Optional[dict] = None  # will be filled when completed
    error_message: Optional[str] = None
