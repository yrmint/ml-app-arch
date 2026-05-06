from functools import lru_cache

from backend.app.services.genre_classifier_facade import GenreClassifierFacade


@lru_cache(maxsize=1)
def get_genre_classifier() -> GenreClassifierFacade:
    """
    Returns a shared GenreClassifierFacade instance for FastAPI dependencies.

    The instance is cached to avoid creating the ML classifier multiple times
    in different routers.
    """
    return GenreClassifierFacade()
