import pytest

from inference import validate_audio_path


def test_missing_audio_file_raises_error():
    with pytest.raises(FileNotFoundError):
        validate_audio_path("missing_file.wav")


def test_unsupported_file_format_raises_error(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("not an audio file")

    with pytest.raises(ValueError):
        validate_audio_path(str(test_file))
