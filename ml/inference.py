import argparse
import json
from pathlib import Path
from functools import lru_cache

import librosa
from transformers import pipeline


MODEL_NAME = "dima806/music_genres_classification"
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def validate_audio_path(audio_path: str) -> Path:
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {audio_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: {path.suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return path


@lru_cache(maxsize=1)
def get_classifier():
    return pipeline(
        task="audio-classification",
        model=MODEL_NAME,
        top_k=5,
    )


def predict_genre(audio_path: str) -> dict:
    path = validate_audio_path(audio_path)

    classifier = get_classifier()

    audio_array, sampling_rate = librosa.load(path, sr=16000, mono=True)

    predictions = classifier(
        {
            "array": audio_array,
            "sampling_rate": sampling_rate,
        }
    )

    top_predictions = [
        {
            "label": prediction["label"],
            "score": float(prediction["score"]),
        }
        for prediction in predictions
    ]

    best_prediction = top_predictions[0]

    return {
        "genre": best_prediction["label"],
        "confidence": best_prediction["score"],
        "top_predictions": top_predictions,
        "model": MODEL_NAME,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict music genre from an audio file."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file.",
    )

    args = parser.parse_args()

    try:
        result = predict_genre(args.audio_path)
        print(json.dumps(result, indent=2))
    except Exception as error:
        print(json.dumps({"error": str(error)}, indent=2))


if __name__ == "__main__":
    main()
