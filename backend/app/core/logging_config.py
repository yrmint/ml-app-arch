import logging
import sys


LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(message)s"
)


def configure_logging() -> None:
    """
    Configures backend application logging.

    Logging is intentionally limited to backend-level events.
    Model-level logging should stay inside the ML package.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
