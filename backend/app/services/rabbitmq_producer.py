import json
import logging

from pathlib import Path
from uuid import uuid4

from aio_pika import Message, DeliveryMode

from backend.app.core.rabbitmq import get_rabbitmq_connection
from backend.app.core.config import rabbitmq_settings
from backend.app.tasks.audio_tasks import AudioTask
from backend.app.services.task_service import task_service


logger = logging.getLogger(__name__)

SHARED_AUDIO_DIR = Path("/shared_audio")


class RabbitMQProducer:
    """Service for sending tasks to RabbitMQ"""

    async def send_audio_task(
            self,
            filename: str,
            audio_bytes: bytes
    ) -> AudioTask:
        """
        Saves audio file and sends task to queue.
        """

        task_id = uuid4()
        file_extension = Path(filename).suffix
        stored_filename = f"{task_id}{file_extension}"
        file_path = SHARED_AUDIO_DIR / stored_filename

        # create directory if not exists
        SHARED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        # save audio to shared volume
        with open(file_path, "wb") as file:
            file.write(audio_bytes)

        logger.info("File saved to shared volume | file_path=%s",
                    file_path)

        task = AudioTask(
            task_id=task_id,
            filename=filename,
            file_path=str(file_path),
        )

        connection = await get_rabbitmq_connection()

        async with connection.channel() as channel:
            await channel.declare_queue(
                rabbitmq_settings.QUEUE_NAME,
                durable=True
            )

            # serialize metadata only
            message_body = json.dumps(
                task.model_dump(mode="json")
            ).encode()

            # create message
            message = Message(
                body=message_body,
                delivery_mode=DeliveryMode.PERSISTENT,  # saved to disk
                headers={"task_id": str(task.task_id)}
            )

            # send to queue
            await channel.default_exchange.publish(
                message,
                routing_key=rabbitmq_settings.QUEUE_NAME
            )

            await task_service.create_task(
                task_id=task.task_id,
                filename=filename
            )

            logger.info(
                "Task sent to queue | task_id=%s | file=%s | path=%s",
                task.task_id,
                filename,
                file_path
            )
            return task


# singleton
rabbitmq_producer = RabbitMQProducer()
