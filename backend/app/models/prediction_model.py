from pydantic import BaseModel
from typing import List
from uuid import UUID


class TaskAcceptedResponse(BaseModel):
    task_id: UUID
    status: str = "accepted"
    message: str = "Task accepted"


class Top3Item(BaseModel):
    genre: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_genre: str
    confidence: float
    top_3: List[Top3Item]


class ErrorResponse(BaseModel):
    details: str
