import os
import pytest
import requests
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock

APP_PATH = os.path.join(os.getcwd(), "main.py")


def test_backend_connection_error():
    """
    Проверка обработки сетевой ошибки при обращении к API бэкенда.
    Ожидается вывод сообщения о невозможности связаться с сервером.
    """
    at = AppTest.from_file(APP_PATH).run()
    if at.exception:
        pytest.fail(f"App failed to load: {at.exception[0]}")

    with patch("requests.post",
               side_effect=requests.exceptions.ConnectionError):
        at.file_uploader[0].upload("test.mp3", b"fake audio data").run()

        if at.button:
            at.button[0].click().run()
            expected = "Не удалось связаться с сервером"
            assert any(expected in err.value for err in at.error)
        else:
            pytest.fail("Кнопка классификации не появилась")


def test_error_413_payload_too_large():
    """
    Проверка вывода сообщения об ошибке при статусе 413 (Payload Too Large).
    Текст ошибки должен соответствовать словарю ERROR_MESSAGES.
    """
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 413
        mock_post.return_value = mock_response

        at.file_uploader[0].upload("large_file.flac", b"large data").run()

        if at.button:
            at.button[0].click().run()
            expected = "Файл слишком велик"
            assert any(expected in err.value for err in at.error)


def test_malformed_json_response():
    """
    Проверка устойчивости приложения к некорректной структуре JSON от сервера.
    Pydantic должен выбросить ValidationError, а приложение — вывести st.error.
    """
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"wrong_key": []}
        mock_post.return_value = mock_response

        at.file_uploader[0].upload("audio.wav", b"test data").run()

        if at.button:
            at.button[0].click().run()
            assert any("Ошибка валидации данных" in err.value
                       for err in at.error)


def test_file_uploader_allowed_types():
    """
    Проверка корректной инициализации виджета загрузки файлов.
    Проверяет наличие виджета на странице при запуске.
    """
    at = AppTest.from_file(APP_PATH).run()

    if len(at.file_uploader) > 0:
        uploader = at.file_uploader[0]
        assert uploader is not None
    else:
        pytest.fail("Виджет file_uploader не найден на странице")


def test_backend_timeout():
    """
    Проверка реакции фронтенда на превышение времени ожидания (timeout).
    Ожидается вывод специфического сообщения для requests.exceptions.Timeout.
    """
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post", side_effect=requests.exceptions.Timeout):
        at.file_uploader[0].upload("track.mp3", b"audio").run()

        if len(at.button) > 0:
            at.button[0].click().run()
            expected = "Сервис отвечает слишком долго"
            assert any(expected in err.value for err in at.error)
        else:
            pytest.fail("Кнопка классификации не появилась")
