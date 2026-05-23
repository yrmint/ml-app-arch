from pydantic import BaseModel, Field
from uuid import UUID
import datetime


class AudioTask(BaseModel):
    task_id: UUID
    filename: str
    file_path: str
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
    status: str = "pending"  # pending, processing, completed, failed

    class Config:
        arbitrary_types_allowed = True
