from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.cuda import is_available


class MLSettings(BaseSettings):
    # MODEL_NAME: str = "dima806/music_genres_classification"
    # Can we do changing model easier somehow?
    MODEL_NAME: str = "CNN"
    DEVICE: str = "cuda" if is_available() else "cpu"
    SAMPLE_RATE: int = 16000
    TOP_K: int = 4
    MAX_AUDIO_LENGTH_SEC: float = 300.0
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    )
    APP_NAME: str = "Music Genre Classifier ML"
    GENRE_LABELS: tuple[str, ...] = [
        "blues",
        "classical",
        "country",
        "disco",
        "hip-hop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = MLSettings()
