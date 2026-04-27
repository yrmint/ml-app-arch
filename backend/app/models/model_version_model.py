from pydantic import BaseModel


class ModelVersionItem(BaseModel):
    version: str
    is_active: bool
    description: str


class ModelVersionsResponse(BaseModel):
    current_version: str
    versions: list[ModelVersionItem]