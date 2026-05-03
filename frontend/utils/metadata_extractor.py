import io
from PIL import Image, UnidentifiedImageError
from mutagen import MutagenError
from mutagen.mp3 import MP3
from mutagen.id3 import APIC
from mutagen.flac import FLAC
from frontend.core.config import settings


def get_album_art(file_bytes: bytes, file_name: str) -> Image.Image:
    """
    Извлекает обложку из аудиофайла.
    Возвращает базовую обложку в случае неудачи.
    """
    try:
        audio_file = io.BytesIO(file_bytes)
        if file_name.lower().endswith('.mp3'):
            audio = MP3(audio_file)
            if audio.tags:
                for tag in audio.tags.values():
                    if isinstance(tag, APIC):
                        return Image.open(io.BytesIO(tag.data))
        elif file_name.lower().endswith('.flac'):
            audio = FLAC(audio_file)
            if audio.pictures:
                return Image.open(io.BytesIO(audio.pictures[0].data))
    except (MutagenError, UnidentifiedImageError, ValueError, KeyError):
        pass

    return Image.open(settings.DEFAULT_COVER_PATH)
