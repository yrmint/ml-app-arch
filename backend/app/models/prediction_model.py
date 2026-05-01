from pydantic import BaseModel
from typing import List


class Top3Item(BaseModel):
    genre: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_genre: str
    confidence: float
    top_3: List[Top3Item]


class ErrorResponse(BaseModel):
    details: str
