from pydantic_settings import BaseSettings, SettingsConfigDict


class MLSettings(BaseSettings):
    MODEL_NAME: str = "dima806/music_genres_classification"
    DEVICE: str = "cpu"
    SAMPLE_RATE: int = 16000
    TOP_K: int = 4
    MAX_AUDIO_LENGTH_SEC: float = 30.0
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    )
    APP_NAME: str = "Music Genre Classifier ML"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = MLSettings()
