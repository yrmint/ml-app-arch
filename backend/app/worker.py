import asyncio
from backend.app.services.rabbitmq_consumer import consume_tasks

if __name__ == "__main__":
    asyncio.run(consume_tasks())
