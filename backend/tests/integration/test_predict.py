import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app

client = TestClient(app)


def test_predict_endpoint_success():
    """Test successful prediction with valid audio file"""
    test_audio_content = b"fake audio data for testing" * 1000  # ~10KB

    response = client.post(
        "/predict",
        files={"audio_file": (
            "test_audio.wav",
            test_audio_content,
            "audio/wav"
        )}
    )

    assert response.status_code == 200
    data = response.json()

    assert "predicted_genre" in data
    assert "confidence" in data
    assert "top_3" in data

    assert isinstance(data["predicted_genre"], str)
    assert isinstance(data["confidence"], (int, float))
    assert 0 <= data["confidence"] <= 1
    assert isinstance(data["top_3"], list)
    assert len(data["top_3"]) == 3

    for item in data["top_3"]:
        assert "genre" in item
        assert "confidence" in item
        assert isinstance(item["genre"], str)
        assert isinstance(item["confidence"], (int, float))


def test_predict_endpoint_empty_file():
    """Test passing empty file"""
    response = client.post(
        "/predict",
        files={"audio_file": ("empty.wav", b"", "audio/wav")}
    )

    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_predict_endpoint_unsupported_format():
    """Test passing file with unsupported file format"""
    response = client.post(
        "/predict",
        files={"audio_file": ("document.txt", b"not an audio", "text/plain")}
    )

    assert response.status_code == 415
    assert "supported formats" in response.json()["detail"].lower()


def test_predict_endpoint_no_file():
    """Test passing no file"""
    response = client.post("/predict")
    assert response.status_code == 422


@pytest.mark.parametrize("filename, content_type", [
    ("song.mp3", "audio/mpeg"),
    ("track.flac", "audio/flac"),
    ("music.ogg", "audio/ogg"),
])
def test_predict_supported_formats(filename: str, content_type: str):
    """Test different supported formats"""
    test_content = b"test audio content" * 500

    response = client.post(
        "/predict",
        files={"audio_file": (filename, test_content, content_type)}
    )

    assert response.status_code == 200
    assert "predicted_genre" in response.json()


def test_predict_endpoint_large_file_rejected(monkeypatch):
    """Test passing an audio file that exceeds upload size limit"""
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)

    large_audio_content = b"x" * (1 * 1024 * 1024 + 1)

    response = client.post(
        "/predict",
        files={
            "audio_file": (
                "large_audio.wav",
                large_audio_content,
                "audio/wav",
            )
        },
    )

    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()
