from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from ml.inference.genre_classifier import GenreClassifier


def create_test_audio_bytes() -> bytes:
    sample_rate = 16000
    duration_seconds = 1
    time_axis = np.linspace(
        0,
        duration_seconds,
        sample_rate * duration_seconds,
        endpoint=False,
    )
    audio = 0.2 * np.sin(2 * np.pi * 440 * time_axis)

    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def test_empty_audio_bytes_raise_error():
    classifier = GenreClassifier(classifier=lambda _: [])

    with pytest.raises(ValueError):
        classifier.predict(b"", filename="test.wav")


def test_unsupported_file_format_raises_error():
    classifier = GenreClassifier(classifier=lambda _: [])

    with pytest.raises(ValueError):
        classifier.predict(b"fake audio", filename="test.txt")


def test_prediction_output_format():
    mock_predictions = [
        {"label": "Rock", "score": 0.87},
        {"label": "Metal", "score": 0.08},
        {"label": "Pop", "score": 0.03},
        {"label": "Blues", "score": 0.02},
    ]

    classifier = GenreClassifier(classifier=lambda _: mock_predictions)
    audio_bytes = create_test_audio_bytes()

    result = classifier.predict(audio_bytes, filename="test.wav")

    assert result["genre"] == "Rock"
    assert result["confidence"] == 0.87
    assert len(result["top_predictions"]) == 3
    assert result["top_predictions"][0]["genre"] == "Metal"
    assert result["top_predictions"][1]["genre"] == "Pop"
    assert result["top_predictions"][2]["genre"] == "Blues"
    assert result["model"] == "dima806/music_genres_classification"
