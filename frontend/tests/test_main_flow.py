from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
from frontend.core.config import settings
from pathlib import Path
import uuid
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
APP_PATH = str(BASE_DIR / "main.py")


def test_initial_ui_state():
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    assert at.title[0].value == f"🎵 {settings.APP_TITLE}"
    assert len(at.file_uploader) > 0
    assert len(at.button) == 2


@patch("frontend.services.classifier_api.requests.get")
@patch("frontend.services.classifier_api.requests.post")
def test_successful_classification_flow(mock_post, mock_get):
    at = AppTest.from_file(APP_PATH).run()
    task_uuid = str(uuid.uuid4())

    mock_post_response = MagicMock()
    mock_post_response.status_code = 202
    mock_post_response.json.return_value = {
        "task_id": task_uuid,
        "status": "accepted",
        "message": "Task accepted"
    }
    mock_post.return_value = mock_post_response

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = {
        "task_id": task_uuid,
        "status": "completed",
        "filename": "track.mp3",
        "created_at": datetime.now().isoformat(),
        "result": {
            "predicted_genre": "classical",
            "confidence": 0.95,
            "top_3": [
                {"genre": "reggae", "confidence": 0.02},
                {"genre": "jazz", "confidence": 0.02},
                {"genre": "pop", "confidence": 0.01}
            ]
        }
    }
    mock_get.return_value = mock_get_response

    at.file_uploader[0].upload("track.mp3", b"fake_bytes").run()

    classify_btn = next((b for b in at.button if b.label == "ОПРЕДЕЛИТЬ ЖАНР"),
                        None)
    assert classify_btn is not None, "Кнопка классификации не появилась"

    classify_btn.click().run()

    success_messages = [s.value for s in at.success]
    assert any("CLASSICAL" in msg for msg in
               success_messages), "Результат классификации не найден"
    assert not at.exception
