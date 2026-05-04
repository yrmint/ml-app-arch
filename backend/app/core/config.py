from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Music Genre Classifier"

    MODEL_PATH: str = "models/current_model.pth"
    DEVICE: str = "cpu"

    SAMPLE_RATE: int = 22050
    MAX_AUDIO_LENGTH_SEC: float = 30.0

    SUPPORTED_AUDIO_EXTENSIONS: tuple[str, ...] = (
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
    )
    MAX_UPLOAD_SIZE_MB: int = 25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
