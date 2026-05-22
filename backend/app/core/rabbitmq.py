from aio_pika import connect_robust, RobustConnection
from functools import lru_cache

from backend.app.core.config import rabbitmq_settings


@lru_cache(maxsize=1)
async def get_rabbitmq_connection() -> RobustConnection:
    """Creates and caches RabbitMQ connection"""
    connection = await connect_robust(
        host=rabbitmq_settings.HOST,
        port=rabbitmq_settings.PORT,
        login=rabbitmq_settings.USER,
        password=rabbitmq_settings.PASSWORD,
        virtualhost=rabbitmq_settings.VHOST,
    )
    return connection
