from pydantic import BaseModel
from uuid import UUID
import datetime


class AudioTask(BaseModel):
    task_id: UUID
    filename: str
    audio_bytes: bytes
    created_at: datetime = datetime.datetime.now(datetime.UTC)
    status: str = "pending"  # pending, processing, completed, failed

    class Config:
        arbitrary_types_allowed = True
