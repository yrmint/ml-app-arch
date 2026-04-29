from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import subprocess

import librosa
import numpy as np
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

    def load_audio_with_ffmpeg(
        self,
        audio_bytes: bytes,
        filename: str,
    ) -> tuple[np.ndarray, int]:
        suffix = Path(filename).suffix.lower()
        target_sample_rate = settings.SAMPLE_RATE

        temp_input_path: str | None = None

        try:
            with NamedTemporaryFile(suffix=suffix, delete=False) as temp_input:
                temp_input.write(audio_bytes)
                temp_input_path = temp_input.name

            command = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                temp_input_path,
                "-t",
                str(settings.MAX_AUDIO_LENGTH_SEC),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                str(target_sample_rate),
                "-",
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            audio_array = np.frombuffer(result.stdout, dtype=np.float32)

            if audio_array.size == 0:
                raise ValueError("Decoded audio is empty.")

            return audio_array, target_sample_rate

        except FileNotFoundError as exc:
            raise ValueError(
                "FFmpeg is required to decode this audio format, "
                "but it was not found on the system PATH."
            ) from exc

        except subprocess.CalledProcessError as exc:
            error_message = exc.stderr.decode(errors="ignore")
            raise ValueError(f"FFmpeg failed to decode audio: {error_message}") from exc

        finally:
            if temp_input_path is not None:
                Path(temp_input_path).unlink(missing_ok=True)

    def load_audio(self, audio_bytes: bytes, filename: str) -> tuple[Any, int]:
        suffix = Path(filename).suffix.lower()

        if suffix == ".m4a":
            return self.load_audio_with_ffmpeg(audio_bytes, filename)

        audio_file = BytesIO(audio_bytes)

        audio_array, sampling_rate = librosa.load(
            audio_file,
            sr=settings.SAMPLE_RATE,
            mono=True,
            duration=settings.MAX_AUDIO_LENGTH_SEC,
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

        audio_array, sampling_rate = self.load_audio(audio_bytes, filename)
        classifier = self.load_model()

        predictions = classifier(
            {
                "array": audio_array,
                "sampling_rate": sampling_rate,
            }
        )

        return self.format_predictions(predictions)
    