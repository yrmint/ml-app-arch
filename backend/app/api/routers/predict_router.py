from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.models.prediction_model import PredictionResponse, Top3Item
from app.services.genre_classifier import GenreClassifier


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)

classifier = GenreClassifier()


def _get_supported_formats_text() -> str:
    return ", ".join(
        extension.lstrip(".")
        for extension in settings.SUPPORTED_AUDIO_EXTENSIONS
    )


@router.post("/", response_model=PredictionResponse)
async def predict_genre(audio_file: UploadFile = File(...)):
    """
    Receives an audio file and returns predicted genre + top-3.

    :param audio_file: audio file to be classified
    """
    filename = audio_file.filename or ""
    file_extension = Path(filename).suffix.lower()

    if file_extension not in settings.SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                "Only supported formats: "
                f"{_get_supported_formats_text()}"
            ),
        )

    audio_bytes = await audio_file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=422,
            detail="Uploaded audio file is empty",
        )

    max_upload_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if len(audio_bytes) > max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded audio file is too large. "
                f"Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB"
            ),
        )

    try:
        # Temporary mock call.
        # Later this service will receive audio_bytes and filename.
        predicted_genre, confidence, top_3 = classifier.predict(
              audio_bytes=audio_bytes,
              filename=filename,
        )
        return PredictionResponse(
            predicted_genre=predicted_genre,
            confidence=confidence,
            top_3=[Top3Item(**item) for item in top_3],
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing error: {error}",
        ) from error
