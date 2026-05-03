import requests
from pydantic import ValidationError
from frontend.core.config import settings
from frontend.services.schemas import PredictResponse

ERROR_MESSAGES = {
    413: "Файл слишком велик. Пожалуйста, загрузите файл меньшего размера.",
    415: "Данный формат аудио не поддерживается.",
    429: "Слишком много запросов. Пожалуйста, подождите некоторое время.",
    500: ("Произошла ошибка при анализе композиции. "
          "Мы уже занимаемся её устранением."),
    503: "Система временно перегружена. Пожалуйста, повторите попытку позже."
}


def predict_genre(
        file_bytes: bytes,
        file_name: str,
        file_type: str
) -> tuple[PredictResponse | None, str | None]:
    try:
        files = {"audio_file": (file_name, file_bytes, file_type)}
        timeout = getattr(settings, "TIMEOUT", 60)
        response = requests.post(
            f"{settings.BACKEND_URL}/predict",
            files=files,
            timeout=timeout
        )

        if response.status_code == 200:
            try:
                data = PredictResponse.model_validate(response.json())
                return data, None
            except ValidationError as e:
                return None, f"Ошибка валидации данных от сервера: {e}"

        try:
            is_json = (
                    response.headers.get('Content-Type') == 'application/json'
            )
            res_data = response.json() if is_json else {}
            error_detail = res_data.get("detail")
        except (requests.exceptions.JSONDecodeError, ValueError):
            error_detail = None

        msg = (
            error_detail if error_detail
            else ERROR_MESSAGES.get(response.status_code, "Сервис недоступен.")
        )

        return None, f"Ошибка {response.status_code}: {msg}"

    except requests.exceptions.Timeout:
        return None, "Сервис отвечает слишком долго. Попробуйте еще раз."
    except requests.exceptions.ConnectionError:
        return None, ("Не удалось связаться с сервером. "
                      "Проверьте интернет-соединение.")
    except requests.exceptions.RequestException as e:
        return None, f"Непредвиденная сетевая ошибка: {e}"
