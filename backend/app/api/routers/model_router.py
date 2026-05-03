from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.models.model_status_model import ModelStatusResponse
from backend.app.models.model_version_model import (
    ModelVersionItem,
    ModelVersionsResponse,
)
from backend.app.services.genre_classifier import GenreClassifier


router = APIRouter(
    prefix="/model",
    tags=["Model"],
)

classifier = GenreClassifier()


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Returns current model and inference configuration status.
    """
    return ModelStatusResponse(
        model_loaded=classifier.is_model_loaded(),
        device=str(classifier.device),
        supported_formats=list(settings.SUPPORTED_AUDIO_EXTENSIONS),
        max_upload_size_mb=settings.MAX_UPLOAD_SIZE_MB,
    )


@router.get("/versions", response_model=ModelVersionsResponse)
async def get_model_versions():
    """
    Returns available model versions for future update and rollback support.
    """
    return ModelVersionsResponse(
        current_version=classifier.get_current_model_version(),
        versions=[
            ModelVersionItem(**version)
            for version in classifier.get_available_model_versions()
        ],
    )
