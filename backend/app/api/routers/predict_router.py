import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.app.core.config import settings
from backend.app.models.prediction_model import PredictionResponse, Top3Item
from backend.app.services.genre_classifier_facade import GenreClassifierFacade
from backend.app.services.genre_service import get_genre_classifier


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


def _get_supported_formats_text() -> str:
    return ", ".join(
        extension.lstrip(".")
        for extension in settings.SUPPORTED_AUDIO_EXTENSIONS
    )


@router.post("/", response_model=PredictionResponse)
async def predict_genre(
    audio_file: UploadFile = File(...),
    classifier: GenreClassifierFacade = Depends(get_genre_classifier),
):
    """
    Receives an audio file and returns predicted genre + top-3.

    :param audio_file: audio file to be classified
    """
    filename = audio_file.filename or ""
    file_extension = Path(filename).suffix.lower()

    logger.info(
        "Prediction request received | filename=%s | extension=%s",
        filename,
        file_extension,
    )

    if file_extension not in settings.SUPPORTED_AUDIO_EXTENSIONS:
        logger.warning(
            "Unsupported audio format rejected | filename=%s | extension=%s",
            filename,
            file_extension,
        )
        raise HTTPException(
            status_code=415,
            detail=(
                "Only supported formats: "
                f"{_get_supported_formats_text()}"
            ),
        )

    audio_bytes = await audio_file.read()

    if len(audio_bytes) == 0:
        logger.warning("Empty audio upload rejected | filename=%s", filename)
        raise HTTPException(
            status_code=422,
            detail="Uploaded audio file is empty",
        )

    max_upload_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if len(audio_bytes) > max_upload_size_bytes:
        logger.warning(
            "Large audio upload rejected | filename=%s | size_bytes=%s | "
            "max_size_bytes=%s",
            filename,
            len(audio_bytes),
            max_upload_size_bytes,
        )
        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded audio file is too large. "
                f"Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB"
            ),
        )

    try:
        predicted_genre, confidence, top_3 = classifier.predict(
            audio_bytes=audio_bytes,
            filename=filename,
        )

        logger.info(
            "Prediction completed | filename=%s | predicted_genre=%s | "
            "confidence=%.4f",
            filename,
            predicted_genre,
            confidence,
        )

        return PredictionResponse(
            predicted_genre=predicted_genre,
            confidence=confidence,
            top_3=[Top3Item(**item) for item in top_3],
        )

    except Exception as error:
        logger.exception(
            "Prediction failed | filename=%s | error=%s",
            filename,
            error,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing error: {error}",
        ) from error
