from fastapi.testclient import TestClient

from backend.app.main import app


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


def test_model_versions_endpoint_success():
    """Test model versions endpoint returns available model versions."""
    response = client.get("/model/versions")

    assert response.status_code == 200

    data = response.json()

    assert data["current_version"] == "mock-v1"
    assert "versions" in data
    assert isinstance(data["versions"], list)
    assert len(data["versions"]) == 2

    active_versions = [
        version for version in data["versions"]
        if version["is_active"] is True
    ]

    assert len(active_versions) == 1
    assert active_versions[0]["version"] == "mock-v1"

    for version in data["versions"]:
        assert "version" in version
        assert "is_active" in version
        assert "description" in version
