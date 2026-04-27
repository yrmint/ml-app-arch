from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_model_status_endpoint_success():
    """Test model status endpoint returns current inference configuration."""
    response = client.get("/model/status")

    assert response.status_code == 200

    data = response.json()

    assert data["model_loaded"] is True
    assert data["device"] == "cpu"
    assert data["supported_formats"] == [".mp3", ".wav", ".flac", ".ogg"]
    assert data["max_upload_size_mb"] == 25