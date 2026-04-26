from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models/current_model.pth"
    DEVICE: str = "cpu"
    SAMPLE_RATE: int = 22050
    MAX_AUDIO_LENGTH_SEC: float = 30.0
    APP_NAME: str = "Music Genre Classifier"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()