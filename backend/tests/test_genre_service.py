from backend.app.services.genre_classifier import GenreClassifier
from backend.app.services.genre_service import get_genre_classifier


def test_get_genre_classifier_returns_shared_instance():
    """Test genre classifier dependency returns the cached instance."""
    first_classifier = get_genre_classifier()
    second_classifier = get_genre_classifier()

    assert isinstance(first_classifier, GenreClassifier)
    assert first_classifier is second_classifier
