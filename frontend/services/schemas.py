from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID


class GenrePrediction(BaseModel):
    """Схема одного предсказания."""
    genre: str
    confidence: float


class PredictResponse(BaseModel):
    """Схема всего ответа от бэкенда."""
    prediction: str = Field(..., alias="predicted_genre")
    confidence: float = Field(..., alias="confidence")
    top_3: List[GenrePrediction] = Field(..., alias="top_3")


class TaskAcceptedResponse(BaseModel):
    """Ответ бэкенда при успешной постановке задачи в RabbitMQ."""
    task_id: UUID
    status: str = "accepted"
    message: str = "Task accepted"


class TaskStatusResponse(BaseModel):
    """Ответ бэкенда при опросе статуса из Redis."""
    task_id: UUID
    status: str
    filename: str
    created_at: datetime
    processed_at: Optional[datetime] = None
    result: Optional[PredictResponse] = None
    error_message: Optional[str] = None
