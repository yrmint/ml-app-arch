import logging
from fastapi import HTTPException

from backend.app.services.genre_service import get_genre_classifier
from backend.app.models.prediction_model import PredictionResponse, Top3Item

logger = logging.getLogger(__name__)


async def process_audio_file(
        audio_bytes: bytes,
        filename: str) -> PredictionResponse:
    """
    Main logic of audio file processing.
    """
    logger.info("Starting audio processing | filename=%s", filename)

    try:
        classifier = get_genre_classifier()
        predicted_genre, confidence, top_3 = classifier.predict(
            audio_bytes=audio_bytes,
            filename=filename,
        )

        logger.info(
            "Prediction successful | filename=%s | genre=%s | confidence=%.4f",
            filename, predicted_genre, confidence
        )

        return PredictionResponse(
            predicted_genre=predicted_genre,
            confidence=confidence,
            top_3=[Top3Item(**item) for item in top_3],
        )

    except Exception as e:
        logger.exception("Prediction failed | filename=%s", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing error: {e}",
        ) from e
