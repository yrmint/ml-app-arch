from typing import Any, Dict, List, Tuple

from ml.core.config import settings as ml_settings
from ml.inference.genre_classifier import GenreClassifier as MLGenreClassifier


class GenreClassifier:
    def __init__(self, ml_classifier: MLGenreClassifier | None = None) -> None:
        self.ml_classifier = ml_classifier or MLGenreClassifier()
        self.device = ml_settings.DEVICE

    def predict(
        self,
        audio_bytes: bytes,
        filename: str,
    ) -> Tuple[str, float, List[Dict[str, float]]]:
        """
        Predicts music genre using the shared ML package classifier.

        Backend response format is kept unchanged for API compatibility.
        """
        result = self.ml_classifier.predict(
            audio_bytes=audio_bytes,
            filename=filename,
        )

        top_3 = [
            {
                "genre": result["genre"],
                "confidence": result["confidence"],
            },
            *result["top_predictions"][:2],
        ]

        return result["genre"], result["confidence"], top_3

    def is_model_loaded(self) -> bool:
        """
        Returns whether the wrapped ML classifier is initialized.
        """
        return self.ml_classifier is not None

    def get_current_model_version(self) -> str:
        """
        Returns the currently active model identifier.
        """
        return ml_settings.MODEL_NAME

    def get_available_model_versions(self) -> list[dict[str, Any]]:
        """
        Returns available model versions.

        The current backend exposes the active ML model as the only
        available version until real model registry support is implemented.
        """
        return [
            {
                "version": ml_settings.MODEL_NAME,
                "is_active": True,
                "description": "Active ML package audio classification model",
            },
        ]
