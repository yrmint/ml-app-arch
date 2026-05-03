from pydantic import BaseModel, Field
from typing import List


class GenrePrediction(BaseModel):
    """Схема одного предсказания."""
    genre: str
    confidence: float


class PredictResponse(BaseModel):
    """Схема всего ответа от бэкенда."""
    predictions: List[GenrePrediction] = Field(..., alias="top_3")
