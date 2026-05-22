import asyncio
import json
import logging
from aio_pika import IncomingMessage

from backend.app.core.rabbitmq import get_rabbitmq_connection
from backend.app.core.config import rabbitmq_settings
from backend.app.services.task_service import task_service
from backend.app.services.audio_processor import process_audio_file
from backend.app.tasks.audio_tasks import AudioTask

logger = logging.getLogger(__name__)


async def consume_tasks():
    """Consumer that processes tasks from queue"""
    connection = await get_rabbitmq_connection()

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        queue = await channel.declare_queue(
            rabbitmq_settings.QUEUE_NAME,
            durable=True
        )

        logger.info("Consumer started, waiting for tasks...")

        async def on_message(message: IncomingMessage):
            async with message.process():
                try:
                    body = json.loads(message.body.decode())
                    task = AudioTask(**body)

                    with open(task.file_path, "rb") as file:
                        audio_bytes = file.read()

                    logger.info(
                        f"Processing task {task.task_id} "
                        f"| file: {task.filename}"
                    )

                    await task_service.update_status(
                        task.task_id,
                        "processing")

                    result = await process_audio_file(
                        audio_bytes,
                        task.filename)

                    # save result
                    await task_service.update_status(
                        task_id=task.task_id,
                        status="completed",
                        result=result.model_dump()
                    )

                    logger.info(f"Task {task.task_id} completed successfully")

                except Exception as e:
                    logger.error(
                        f"Task failed | task_id={task.task_id} | error={e}"
                    )
                    await task_service.update_status(
                        task_id=task.task_id,
                        status="failed",
                        error=str(e)
                    )

        await queue.consume(on_message)
        await asyncio.Future()  # keep consumer run
