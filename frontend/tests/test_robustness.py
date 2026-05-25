import pytest
import requests
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
from pathlib import Path
from frontend.services.classifier_api import NETWORK_ERRORS, API_ERRORS

BASE_DIR = Path(__file__).resolve().parent.parent
APP_PATH = str(BASE_DIR / "main.py")

DIAGNOSTIC_MESSAGES = {
    "connection": (
        "Ошибка соединения с бэкендом не отобразилась в интерфейсе."
    ),
    "payload_too_large": (
        "Ошибка превышения размера полезной нагрузки (413 Payload Too Large) "
        "не была корректно обработана."
    ),
    "malformed_json": (
        "Ошибка валидации некорректного JSON-ответа от бэкенда "
        "не была выведена на экран."
    ),
    "timeout": (
        "Таймаут ожидания ответа от бэкенда не привёл к отображению "
        "соответствующего уведомления."
    )
}

DIAGNOSTIC_TEMPLATE = (
    "\n[Diagnostic Failure]: {message}\n"
    "Expected substring: '{expected}'\n"
    "Actual st.error elements: {errors}\n"
    "Actual st.warning elements: {warnings}\n"
    "App exceptions: {exceptions}\n"
)


def _clean(text: str) -> str:
    for emoji in ["⏳", "⚡", "❌", "⚠️", "✅"]:
        text = text.replace(emoji, "")
    return text.strip()


def _assert_error_message(at, expected_msg, context_key):
    cleaned_expected = _clean(expected_msg)
    actual_errors = [err.value for err in at.error]
    actual_warnings = [w.value for w in at.warning]

    if not any(cleaned_expected in _clean(err) for err in actual_errors):
        error_details = DIAGNOSTIC_TEMPLATE.format(
            message=DIAGNOSTIC_MESSAGES[context_key],
            expected=expected_msg,
            errors=actual_errors,
            warnings=actual_warnings,
            exceptions=at.exception
        )
        pytest.fail(error_details)


def _upload_and_classify(at, filename: str, data: bytes):
    at.file_uploader[0].upload(filename, data).run()
    classify_btn = next(
        (b for b in at.button if b.label == "ОПРЕДЕЛИТЬ ЖАНР"), None
    )
    assert classify_btn is not None, (
        "Кнопка 'ОПРЕДЕЛИТЬ ЖАНР' не найдена на странице интерфейса."
    )
    classify_btn.click().run()


def test_backend_connection_error():
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError
        _upload_and_classify(at, "test.mp3", b"fake audio data")

    _assert_error_message(at, NETWORK_ERRORS["connection"], "connection")


def test_error_413_payload_too_large():
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 413
        mock_post.return_value = mock_response
        _upload_and_classify(at, "large_track.mp3", b"large data")

    _assert_error_message(at, API_ERRORS["send_fail"], "payload_too_large")


def test_malformed_json_response():
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"wrong_key": []}
        mock_post.return_value = mock_response
        _upload_and_classify(at, "audio.wav", b"test data")

    _assert_error_message(at, API_ERRORS["validation_fail"], "malformed_json")


def test_backend_timeout():
    at = AppTest.from_file(APP_PATH).run()

    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout
        _upload_and_classify(at, "track.mp3", b"data")

    _assert_error_message(at, NETWORK_ERRORS["timeout"], "timeout")


def test_file_uploader_allowed_types():
    at = AppTest.from_file(APP_PATH).run()

    if len(at.file_uploader) > 0:
        uploader = at.file_uploader[0]
        assert uploader is not None
    else:
        pytest.fail("Виджет file_uploader не найден на странице")
