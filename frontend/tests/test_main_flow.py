import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
from frontend.core.config import settings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_PATH = str(BASE_DIR / "main.py")


def test_initial_ui_state():
    """
    Проверка базового состояния интерфейса: наличие заголовка и загрузчика,
    отсутствие кнопки классификации до загрузки файла.
    """
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert at.title[0].value == f"🎵 {settings.APP_TITLE}"
    assert len(at.get("file_uploader")) > 0
    assert len(at.get("button")) == 0


def test_successful_classification_flow():
    """
    Проверка полного цикла: загрузка файла, успешный ответ бэкенда по схеме
    PredictResponse и корректное отображение результата классификации.
    """
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "top_3": [
                {"genre": "classical", "confidence": 0.95},
                {"genre": "jazz", "confidence": 0.03},
                {"genre": "pop", "confidence": 0.02}
            ]
        }
        mock_post.return_value = mock_response

        at.get("file_uploader")[0].upload("track.mp3", b"fake_bytes").run()

        buttons = at.get("button")
        if buttons:
            buttons[0].click().run()
        else:
            pytest.fail(
                "Кнопка классификации не появилась после загрузки файла")

        success_messages = [s.value for s in at.get("success")]
        results_found = any(
            "CLASSICAL" in msg.upper() for msg in success_messages)

        msg = "Результат классификации не найден в блоке успеха"
        assert results_found, msg
        assert not at.exception
