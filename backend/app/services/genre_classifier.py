import time
from typing import Dict, List, Tuple

import torch

from backend.app.core.config import settings


class GenreClassifier:
    def __init__(self):
        self.model = None
        self.device = torch.device(settings.DEVICE)
        self.genres = [
            "Blues",
            "Classical",
            "Country",
            "Disco",
            "Hip-Hop",
            "Jazz",
            "Metal",
            "Pop",
            "Reggae",
            "Rock",
        ]
        self.load_model()

    def load_model(self):
        print(f"[{settings.APP_NAME}] Model loaded to device: {self.device}")
        self.model = "mock_model"  # using mocks for now

    def predict(
        self,
        audio_bytes: bytes,
        filename: str,
    ) -> Tuple[str, float, List[Dict[str, float]]]:
        """
        Placeholder prediction method.

        The method already receives raw audio bytes and filename,
        so the backend is ready for future integration with a real ML model.
        """
        if not audio_bytes:
            raise ValueError("Audio bytes are empty")

        if not filename:
            raise ValueError("Audio filename is missing")

        time.sleep(1)  # imitate model inference

        top_3 = [
            {"genre": "Rock", "confidence": 0.87},
            {"genre": "Metal", "confidence": 0.09},
            {"genre": "Pop", "confidence": 0.03},
        ]

        predicted_genre = top_3[0]["genre"]
        confidence = top_3[0]["confidence"]

        return predicted_genre, confidence, top_3

    def is_model_loaded(self) -> bool:
        """
        Returns whether the classifier currently has a loaded model object.
        """
        return self.model is not None

    def get_current_model_version(self) -> str:
        """
        Returns the currently active model version.
        """
        return "mock-v1"

    def get_available_model_versions(self) -> list[dict[str, str | bool]]:
        """
        Returns available model versions.

        This is a temporary versioning skeleton for future model update
        and rollback support.
        """
        return [
            {
                "version": "mock-v1",
                "is_active": True,
                "description": "Initial mock backend inference model",
            },
            {
                "version": "future-ml-v1",
                "is_active": False,
                "description": (
                    "Reserved for future trained ML model "
                    "integration"
                ),
            },
        ]
