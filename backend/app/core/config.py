from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Music Genre Classifier"

    MODEL_PATH: str = "models/current_model.pth"
    DEVICE: str = "cpu"

    SAMPLE_RATE: int = 16000
    MAX_AUDIO_LENGTH_SEC: float = 30.0

    SUPPORTED_AUDIO_EXTENSIONS: tuple[str, ...] = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    )
    MAX_UPLOAD_SIZE_MB: int = 25

    # Redis settings
    REDIS_URL: str = "redis://redis:6379/0"
    REDIS_TASK_TTL: int = 3600  # 1 hour

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class RabbitMQSettings(BaseSettings):
    HOST: str = "localhost"
    PORT: int = 5672
    USER: str = "guest"
    PASSWORD: str = "guest"
    VHOST: str = "/"
    QUEUE_NAME: str = "audio_tasks"
    EXCHANGE_NAME: str = "audio_exchange"
    ROUTING_KEY: str = "audio.process"

    model_config = SettingsConfigDict(
        env_prefix="RABBITMQ_",
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
rabbitmq_settings = RabbitMQSettings()
