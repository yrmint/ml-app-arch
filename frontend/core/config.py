from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    """
    Настройки фронтенд-приложения.
    """

    APP_TITLE: str = "Music Genre Classifier"
    BACKEND_URL: str = "http://localhost:8000"
    MAX_UPLOAD_SIZE_MB: int = 25
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    )
    MAX_AUDIO_LENGTH_SEC: float = 30.0
    TIMEOUT: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = FrontendSettings()
