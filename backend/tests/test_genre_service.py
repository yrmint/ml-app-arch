from backend.app.services.genre_classifier_facade import GenreClassifierFacade
from backend.app.services.genre_service import get_genre_classifier


class FakeMLClassifier:
    def predict(self, audio_bytes: bytes, filename: str) -> dict:
        return {
            "genre": "rock",
            "confidence": 0.45,
            "top_predictions": [
                {"genre": "blues", "confidence": 0.41},
                {"genre": "disco", "confidence": 0.05},
                {"genre": "reggae", "confidence": 0.01},
            ],
            "model": "test-model",
        }


def test_get_genre_classifier_returns_shared_instance():
    """Test genre classifier dependency returns the cached instance."""
    first_classifier = get_genre_classifier()
    second_classifier = get_genre_classifier()

    assert isinstance(first_classifier, GenreClassifierFacade)
    assert first_classifier is second_classifier


def test_genre_classifier_facade_uses_ml_top_predictions_unchanged():
    """Test facade does not duplicate predicted genre in top predictions."""
    facade = GenreClassifierFacade(ml_classifier=FakeMLClassifier())

    predicted_genre, confidence, top_3 = facade.predict(
        audio_bytes=b"fake audio",
        filename="test.wav",
    )

    assert predicted_genre == "rock"
    assert confidence == 0.45
    assert top_3 == [
        {"genre": "blues", "confidence": 0.41},
        {"genre": "disco", "confidence": 0.05},
        {"genre": "reggae", "confidence": 0.01},
    ]
