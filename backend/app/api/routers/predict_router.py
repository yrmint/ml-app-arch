import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.core.config import settings
from backend.app.models.prediction_model import TaskAcceptedResponse
from backend.app.services.rabbitmq_producer import rabbitmq_producer
from backend.app.tasks.audio_tasks import AudioTask

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


def _get_supported_formats_text() -> str:
    return ", ".join(
        extension.lstrip(".")
        for extension in settings.SUPPORTED_AUDIO_EXTENSIONS
    )


@router.post(
    "/",
    response_model=TaskAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def predict_genre(
    audio_file: UploadFile = File(...),
):
    """
    Receives an audio file and returns predicted genre + top-3.

    :param audio_file: audio file to be classified
    """
    filename = audio_file.filename or ""
    file_extension = Path(filename).suffix.lower()

    logger.info(
        "Prediction request received | filename=%s | extension=%s",
        filename,
        file_extension,
    )

    if file_extension not in settings.SUPPORTED_AUDIO_EXTENSIONS:
        logger.warning(
            "Unsupported audio format rejected | filename=%s | extension=%s",
            filename,
            file_extension,
        )
        raise HTTPException(
            status_code=415,
            detail=(
                "Only supported formats: "
                f"{_get_supported_formats_text()}"
            ),
        )

    audio_bytes = await audio_file.read()

    if len(audio_bytes) == 0:
        logger.warning("Empty audio upload rejected | filename=%s", filename)
        raise HTTPException(
            status_code=422,
            detail="Uploaded audio file is empty",
        )

    max_upload_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if len(audio_bytes) > max_upload_size_bytes:
        logger.warning(
            "Large audio upload rejected | filename=%s | size_bytes=%s | "
            "max_size_bytes=%s",
            filename,
            len(audio_bytes),
            max_upload_size_bytes,
        )
        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded audio file is too large. "
                f"Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB"
            ),
        )

    try:
        task: AudioTask = await rabbitmq_producer.send_audio_task(
            filename,
            audio_bytes
        )

        return TaskAcceptedResponse(
            task_id=task.task_id,
            message="Task accepted"
        )

    except Exception as error:
        logger.exception(
            "Failed to queue task | filename=%s | error=%s",
            filename,
            error,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Task processing error: {error}",
        ) from error
