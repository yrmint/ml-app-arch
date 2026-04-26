from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.prediction_model import PredictionResponse, Top3Item
from app.services.genre_classifier import GenreClassifier

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)
classifier = GenreClassifier()


@router.post("/", response_model=PredictionResponse)
async def predict_genre(audio_file: UploadFile = File(...)):
    """
    Receives an audio file and returns predicted genre + top-3.
    :param audio_file: audio file to be classified
    """
    if not (audio_file.filename.lower().
            endswith(('.mp3', '.wav', '.flac', '.ogg'))):
        raise HTTPException(
            status_code=415,
            detail="Only supported formats: mp3, wav, flac, ogg")

    audio_bytes = await audio_file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=422,
            detail="Uploaded audio file is empty")

    try:
        # call audio preprocessor later
        # placeholder, pass tensor later
        predicted_genre, confidence, top_3 = classifier.predict(None)

        return PredictionResponse(
            predicted_genre=predicted_genre,
            confidence=confidence,
            top_3=[Top3Item(**item) for item in top_3],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing error: {e}")
