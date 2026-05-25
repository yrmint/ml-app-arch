from pydantic import BaseModel
from datetime import datetime

class FinetuneAcceptedResponse(BaseModel):
    message: str
    dataset_path: str
    timestamp: datetime
    