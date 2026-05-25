import requests
from pydantic import ValidationError
from frontend.core.config import settings
from frontend.services.schemas import TaskAcceptedResponse, TaskStatusResponse

TASK_STATUS_MESSAGES = {
    "pending": "⏳ Композиция ожидает своей очереди на обработку...",
    "processing": "⚡ Нейросеть распознает музыкальные особенности...",
    "failed": "❌ Произошел сбой при обработке аудиофайла.",
}

NETWORK_ERRORS = {
    "timeout": "⚠️ Время ожидания ответа от сервера истекло.",
    "connection":
        "⚠️ Проблемы с соединением. Не удалось связаться с сервером.",
    "unexpected":
        "⚠️ Произошла непредвиденная ошибка при обращении к серверу.",
}

API_ERRORS = {
    "validation_fail": "⚠️ Не удалось обработать ответ сервера.",
    "send_fail": "⚠️ Не удалось отправить файл на сервер.",
    "status_validation_fail": "⚠️ Не удалось обработать статус задачи.",
    "status_update_fail": "⚠️ Не удалось обновить статус анализа трека.",
}

FINETUNE_MESSAGES = {
    "success": "✅ Архив успешно загружен и принят в работу.",
    "auth_error": "❌ Ошибка авторизации (неверный логин или пароль).",
    "server_error": "❌ Ошибка сервера: ",
    "connection_error": "⚠️ Не удалось подключиться к серверу: "
}


def send_audio_to_queue(
        file_bytes: bytes,
        file_name: str,
        file_type: str
) -> tuple[TaskAcceptedResponse | None, str | None]:
    """Отправляет аудиофайл на бэкенд для постановки в очередь RabbitMQ."""
    url = f"{settings.BACKEND_URL}/predict/"
    files = {"audio_file": (file_name, file_bytes, file_type)}
    timeout = getattr(settings, "TIMEOUT", 60)

    try:
        response = requests.post(url, files=files, timeout=timeout)
        if response.status_code in (200, 202):
            try:
                data = TaskAcceptedResponse.model_validate(response.json())
                return data, None
            except ValidationError:
                return None, API_ERRORS["validation_fail"]

        return None, API_ERRORS["send_fail"]

    except requests.exceptions.Timeout:
        return None, NETWORK_ERRORS["timeout"]
    except requests.exceptions.ConnectionError:
        return None, NETWORK_ERRORS["connection"]
    except requests.exceptions.RequestException:
        return None, NETWORK_ERRORS["unexpected"]


def get_task_status(task_id: str) -> tuple[TaskStatusResponse | None,
                                           str | None]:
    """Запрашивает текущий статус выполнения задачи из Redis."""
    url = f"{settings.BACKEND_URL}/tasks/{task_id}"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            try:
                data = TaskStatusResponse.model_validate(response.json())
                return data, None
            except ValidationError:
                return None, API_ERRORS["status_validation_fail"]

        return None, API_ERRORS["status_update_fail"]

    except requests.exceptions.Timeout:
        return None, NETWORK_ERRORS["timeout"]
    except requests.exceptions.ConnectionError:
        return None, NETWORK_ERRORS["connection"]
    except requests.exceptions.RequestException:
        return None, NETWORK_ERRORS["unexpected"]


def upload_finetune_archive(
        file_bytes: bytes,
        file_name: str,
        username: str,
        password: str
) -> tuple[bool, str]:
    """Отправляет архив на эндпоинт /finetune с Basic Auth."""
    url = f"{settings.BACKEND_URL}/finetune"
    files = {"archive_file": (file_name, file_bytes, "application/octet-stream")}

    try:
        response = requests.post(
            url,
            files=files,
            auth=(username, password),
            timeout=300
        )
        if response.status_code == 200:
            return True, FINETUNE_MESSAGES["success"]
        elif response.status_code == 401:
            return False, FINETUNE_MESSAGES["auth_error"]
        else:
            return False, f"{FINETUNE_MESSAGES['server_error']}{response.text}"
    except Exception as e:
        return False, f"{FINETUNE_MESSAGES['connection_error']}{str(e)}"
