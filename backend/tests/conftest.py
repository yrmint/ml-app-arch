import pytest

from backend.app.main import app
from backend.app.services.genre_service import get_genre_classifier


class FakeGenreClassifier:
    device = "cpu"

    def predict(self, audio_bytes: bytes, filename: str):
        return (
            "Rock",
            0.87,
            [
                {"genre": "Rock", "confidence": 0.87},
                {"genre": "Metal", "confidence": 0.09},
                {"genre": "Pop", "confidence": 0.03},
            ],
        )

    def is_model_loaded(self) -> bool:
        return True

    def get_current_model_version(self) -> str:
        return "test-model-v1"

    def get_available_model_versions(self) -> list[dict[str, str | bool]]:
        return [
            {
                "version": "test-model-v1",
                "is_active": True,
                "description": "Fake classifier used in backend tests",
            },
        ]


@pytest.fixture(autouse=True)
def override_genre_classifier_dependency():
    def get_fake_genre_classifier():
        return FakeGenreClassifier()

    app.dependency_overrides[get_genre_classifier] = get_fake_genre_classifier
    yield
    app.dependency_overrides.clear()
