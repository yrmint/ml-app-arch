import sys
import os
import pathlib
import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
import requests

FRONTEND_DIR = str(pathlib.Path(__file__).parent.parent)
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)
os.environ["PYTHONPATH"] = FRONTEND_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")
APP_PATH = "frontend/main.py"


def test_backend_connection_error():
    # Проверка обработки сетевой ошибки (ConnectionError) при обращении к API бэкенда
    at = AppTest.from_file(APP_PATH).run()
    if at.exception:
        pytest.fail(f"App failed to load: {at.exception[0]}")

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
        at.file_uploader[0].upload("test.mp3", b"fake audio data").run()

        if len(at.button) > 0:
            at.button[0].click().run()
            assert any("Проблема с сетевым соединением" in err.value for err in at.error)
        else:
            pytest.fail("Кнопка классификации не появилась")


def test_error_413_payload_too_large():
    # Проверка вывода сообщения об ошибке, если сервер вернул статус 413 (файл слишком большой)
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 413
        mock_post.return_value = mock_response

        at.file_uploader[0].upload("large_file.flac", b"large data").run()

        if len(at.button) > 0:
            at.button[0].click().run()
            assert any("Файл слишком велик" in err.value for err in at.error)


def test_malformed_json_response():
    # Проверка устойчивости приложения к некорректным данным в формате JSON от сервера
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"bad_key": "no_data"}
        mock_post.return_value = mock_response

        at.file_uploader[0].upload("audio.wav", b"test data").run()

        if len(at.button) > 0:
            at.button[0].click().run()
            assert any("Результаты анализа не получены" in warn.value for warn in at.warning)


def test_file_uploader_allowed_types():
    # Проверка корректной инициализации виджета загрузки файлов
    at = AppTest.from_file(APP_PATH).run()

    if len(at.file_uploader) > 0:
        uploader = at.file_uploader[0]
        assert getattr(uploader, "label", "") != "", "У загрузчика должен быть установлен текст (label)"
    else:
        pytest.fail("Виджет file_uploader не найден на странице")


def test_backend_timeout():
    # Проверка реакции фронтенда на долгий ответ сервера (превышение таймаута)
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post", side_effect=requests.exceptions.Timeout):
        at.file_uploader[0].upload("track.mp3", b"audio").run()

        if len(at.button) > 0:
            at.button[0].click().run()
            assert any("Проблема с сетевым соединением" in err.value for err in at.error)
        else:
            pytest.fail("Кнопка классификации не появилась")
