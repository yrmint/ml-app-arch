import sys
import os
import pathlib
import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock

FRONTEND_DIR = str(pathlib.Path(__file__).parent.parent)
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)
os.environ["PYTHONPATH"] = FRONTEND_DIR + os.pathsep + os.environ.get(
    "PYTHONPATH", "")
APP_PATH = "frontend/main.py"


def test_initial_ui_state():
    # Проверка, что при запуске все базовые элементы интерфейса на месте
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception, "Приложение упало при запуске"
    assert len(at.title) > 0, "Заголовок страницы не найден"
    assert len(at.file_uploader) > 0, "Виджет загрузки файлов не отображается"
    # Кнопка не должна появиться, пока не загружен файл
    assert len(at.button) == 0


def test_successful_classification_flow():
    # Проверка основного сценария
    # Загрузка файла -> нажатие кнопки -> отображеие результата
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post") as mock_post:
        # Имитируем идеальный ответ от бэкенда
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predicted_genre": "classical",
            "confidence": 0.95,
            "top_3": [
                {"genre": "classical", "confidence": 0.95},
                {"genre": "jazz", "confidence": 0.03},
                {"genre": "pop", "confidence": 0.02}
            ]
        }
        mock_post.return_value = mock_response

        at.file_uploader[0].upload("track.mp3", b"fake_audio_bytes").run()
        if len(at.button) > 0:
            at.button[0].click().run()
        else:
            pytest.fail(
                "Кнопка классификации не появилась после загрузки файла")
        results_found = any("Classical" in s.value for s in at.success) or any(
            "Classical" in w.value for w in at.markdown)

        assert results_found, (
            "Результат классификации не отобразился на странице"
        )
        assert not at.exception
