import logging
from aio_pika import connect_robust, RobustConnection

from backend.app.core.config import rabbitmq_settings


logger = logging.getLogger(__name__)
_rabbitmq_connection: RobustConnection | None = None


async def get_rabbitmq_connection() -> RobustConnection:
    """Returns active RabbitMQ connection (singleton)."""
    global _rabbitmq_connection

    if _rabbitmq_connection is None or _rabbitmq_connection.is_closed:
        _rabbitmq_connection = await connect_robust(
            host=rabbitmq_settings.HOST,
            port=rabbitmq_settings.PORT,
            login=rabbitmq_settings.USER,
            password=rabbitmq_settings.PASSWORD,
            virtualhost=rabbitmq_settings.VHOST,
        )
        logger.info("RabbitMQ connection established")

    return _rabbitmq_connection
