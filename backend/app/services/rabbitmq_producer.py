from uuid import uuid4
from aio_pika import Message, DeliveryMode

from backend.app.core.rabbitmq import get_rabbitmq_connection
from backend.app.core.config import rabbitmq_settings
from backend.app.tasks.audio_tasks import AudioTask


class RabbitMQProducer:
    """Service for sending tasks to RabbitMQ"""

    async def send_audio_task(
            self,
            filename: str,
            audio_bytes: bytes
    ) -> AudioTask:
        """
        Sends a task to queue.
        """
        task = AudioTask(
            task_id=uuid4(),
            filename=filename,
            audio_bytes=audio_bytes
        )

        connection = await get_rabbitmq_connection()

        async with connection:
            channel = await connection.channel()

            # declare queue (durable = persists after reload)
            await channel.declare_queue(
                rabbitmq_settings.QUEUE_NAME,
                durable=True
            )

            # convert task to JSON
            message_body = task.model_dump_json(
                exclude={"audio_bytes"}
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

            print(f"Task sent to queue: {task.task_id} | File: {filename}")
            return task


# singleton
rabbitmq_producer = RabbitMQProducer()
