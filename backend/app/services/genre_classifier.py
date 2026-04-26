from app.core.config import settings
from typing import Tuple, List, Dict
import torch
import time


class GenreClassifier:
    def __init__(self):
        self.model = None
        self.device = torch.device(settings.DEVICE)
        self.genres = ["Blues", "Classical", "Country", "Disco", "Hip-Hop",
                       "Jazz", "Metal", "Pop", "Reggae", "Rock"]
        self.load_model()

    def load_model(self):
        print(f"[{settings.APP_NAME}] Model loaded to device: {self.device}")
        self.model = "mock_model"   # using mocks for now

    def predict(self, audio_tensor) -> Tuple[str, float, List[Dict[str, float]]]:
        """
        Placeholder prediction.
        In the future: audio preprocessing + model inference
        """
        time.sleep(1)  # imitate model inference

        # mock result
        top_3 = [
            {"genre": "Rock", "confidence": 0.87},
            {"genre": "Metal", "confidence": 0.09},
            {"genre": "Pop", "confidence": 0.03}
        ]

        prediction_genre = top_3[0]["genre"]
        confidence = top_3[0]["confidence"]

        return prediction_genre, confidence, top_3
