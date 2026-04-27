from io import BytesIO
from pathlib import Path
from typing import Any

import librosa
from transformers import pipeline

from ml.core.config import settings


class GenreClassifier:
    def __init__(self, classifier: Any | None = None) -> None:
        self.model_name = settings.MODEL_NAME
        self.classifier = classifier

    def load_model(self) -> Any:
        if self.classifier is None:
            self.classifier = pipeline(
                task="audio-classification",
                model=self.model_name,
                top_k=settings.TOP_K,
            )

        return self.classifier

    def validate_audio(self, audio_bytes: bytes, filename: str) -> None:
        if not audio_bytes:
            raise ValueError("Audio bytes are empty.")

        suffix = Path(filename).suffix.lower()

        if suffix not in settings.SUPPORTED_EXTENSIONS:
            supported = ", ".join(settings.SUPPORTED_EXTENSIONS)
            raise ValueError(
                f"Unsupported audio format: {suffix}. "
                f"Supported formats: {supported}"
            )

    def load_audio(self, audio_bytes: bytes) -> tuple[Any, int]:
        audio_file = BytesIO(audio_bytes)

        audio_array, sampling_rate = librosa.load(
            audio_file,
            sr=settings.SAMPLE_RATE,
            mono=True,
        )

        return audio_array, sampling_rate

    def format_predictions(self, predictions: list[dict[str, Any]]) -> dict:
        if not predictions:
            raise ValueError("Model returned no predictions.")

        formatted_predictions = [
            {
                "genre": prediction["label"],
                "confidence": float(prediction["score"]),
            }
            for prediction in predictions
        ]

        best_prediction = formatted_predictions[0]
        alternative_predictions = formatted_predictions[1:4]

        return {
            "genre": best_prediction["genre"],
            "confidence": best_prediction["confidence"],
            "top_predictions": alternative_predictions,
            "model": self.model_name,
        }

    def predict(self, audio_bytes: bytes, filename: str) -> dict:
        self.validate_audio(audio_bytes, filename)

        audio_array, sampling_rate = self.load_audio(audio_bytes)
        classifier = self.load_model()

        predictions = classifier(
            {
                "array": audio_array,
                "sampling_rate": sampling_rate,
            }
        )

        return self.format_predictions(predictions)
