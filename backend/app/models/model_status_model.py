from pydantic import BaseModel


class ModelStatusResponse(BaseModel):
    model_loaded: bool
    device: str
    supported_formats: list[str]
    max_upload_size_mb: int
