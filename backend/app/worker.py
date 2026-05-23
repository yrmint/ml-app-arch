import asyncio
import logging

from backend.app.core.logging_config import configure_logging
from backend.app.services.genre_service import preload_model
from backend.app.services.rabbitmq_consumer import consume_tasks

configure_logging()
logger = logging.getLogger(__name__)


async def main():
    logger.info("Preloading ML model...")

    preload_model()

    logger.info("ML model ready")

    await consume_tasks()


if __name__ == "__main__":
    asyncio.run(main())
