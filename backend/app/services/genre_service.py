import logging

from backend.app.services.genre_classifier_facade import GenreClassifierFacade


logger = logging.getLogger(__name__)
_classifier: GenreClassifierFacade | None = None


def get_genre_classifier() -> GenreClassifierFacade:
    global _classifier

    if _classifier is None:
        logger.info("Initializing genre classifier...")

        _classifier = GenreClassifierFacade()

        logger.info("Loading ML model into memory...")
        _classifier.ml_classifier.load_model()

        logger.info("ML model loaded successfully")

    return _classifier


def preload_model() -> None:
    """
    Forces model loading during worker startup.
    """
    get_genre_classifier()
