import zipfile
import logging
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from backend.app.models.finetune_model import FinetuneAcceptedResponse
from backend.app.core.config import settings
from backend.app.services.archive_validator import validate_finetune_archive


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/finetune",
    tags=["Finetune"],
)


@router.post(
    "/",
    response_model=FinetuneAcceptedResponse,
    status_code=status.HTTP_200_OK
)
async def finetune_model(
        archive_file: UploadFile = File(...)
):
    """
    Accepts an archive with audios for model finetuning.
    """
    file_ext = Path(archive_file.filename).suffix.lower()
    if file_ext not in settings.SUPPORTED_ARCHIVES:
        raise HTTPException(
            status_code=415,
            detail="Only .zip archives supported."
        )

    if archive_file.size > settings.MAX_ARCHIVE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too heavy (max {settings.MAX_ARCHIVE_SIZE_MB} Mb)"
        )

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = Path("data/finetune") / f"finetune-{timestamp}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory() as temp_dir:
            temp_archive_path = Path(temp_dir) / archive_file.filename

            content = await archive_file.read()

            with open(temp_archive_path, "wb") as f:
                f.write(content)

            validate_finetune_archive(temp_archive_path)

            with zipfile.ZipFile(temp_archive_path) as z:
                logger.info("Extracting files | dataset_dir=%s",
                            dataset_dir)
                z.extractall(dataset_dir)

        return FinetuneAcceptedResponse(
            message="Dataset loaded successfully",
            dataset_path=str(dataset_dir),
            timestamp=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing archive: {str(e)}"
        )
