import zipfile
from pathlib import Path
from fastapi import HTTPException

from backend.app.core.config import settings


def validate_finetune_archive(archive_path: Path) -> None:
    if archive_path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(archive_path)
    else:
        raise HTTPException(
            status_code=415,
            detail="Only supported format is .zip"
        )

    with archive:
        all_files = archive.namelist()

        if not all_files:
            raise HTTPException(status_code=400, detail="Archive empty")

        directories = set()
        has_audio = False
        audio_extensions = {ext.lower()
                            for ext in settings.SUPPORTED_AUDIO_EXTENSIONS}

        for name in all_files:
            if not name or name.endswith('/'):
                continue

            path = Path(name)

            if path.suffix.lower() in audio_extensions:
                has_audio = True

            for parent in path.parents:
                if parent.name:
                    directories.add(parent.name.lower())

        if not has_audio:
            raise HTTPException(
                status_code=400,
                detail="No audio files found in archive"
            )

        genre_folders = directories.intersection(settings.SUPPORTED_GENRES)

        if not genre_folders:
            raise HTTPException(
                status_code=400,
                detail="No folder with supported genre found in archive. "
                       f"Supported genres: "
                       f"{', '.join(sorted(settings.SUPPORTED_GENRES))}"
            )
