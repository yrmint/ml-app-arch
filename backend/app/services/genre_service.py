from functools import lru_cache

from backend.app.services.genre_classifier import GenreClassifier


@lru_cache(maxsize=1)
def get_genre_classifier() -> GenreClassifier:
    """
    Returns a shared GenreClassifier instance for FastAPI dependencies.

    The instance is cached to avoid creating GenreClassifier multiple times
    in different routers.
    """
    return GenreClassifier()
