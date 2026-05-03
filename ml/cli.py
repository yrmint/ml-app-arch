import argparse
import json
from pathlib import Path

from ml.inference.genre_classifier import GenreClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict music genre from an audio file."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file.",
    )

    args = parser.parse_args()
    audio_path = Path(args.audio_path)

    try:
        audio_bytes = audio_path.read_bytes()
        classifier = GenreClassifier()
        result = classifier.predict(audio_bytes, filename=audio_path.name)
        print(json.dumps(result, indent=2))
    except Exception as error:
        print(json.dumps({"error": str(error)}, indent=2))


if __name__ == "__main__":
    main()
