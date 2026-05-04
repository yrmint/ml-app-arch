import sys
import subprocess
from pathlib import Path

try:
    from frontend.core.config import settings
except ImportError:
    print("Ошибка: Не удалось найти frontend.core.config.")
    sys.exit(1)


def run():
    """Запуск интерфейса Streamlit с параметрами из конфига."""
    current_dir = Path(__file__).parent
    app_path = current_dir / "main.py"

    if not app_path.exists():
        print(f"Ошибка: Файл интерфейса не найден по пути {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.maxUploadSize",
        str(settings.MAX_UPLOAD_SIZE_MB),
        "--server.port",
        str(settings.FRONTEND_PORT),
    ]

    print(f"Запуск {settings.APP_TITLE}...")
    print(f"Локальный порт: {settings.FRONTEND_PORT}")
    print(f"Лимит загрузки: {settings.MAX_UPLOAD_SIZE_MB} MB")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nПриложение остановлено пользователем.")
    except Exception as e:
        print(f"Ошибка при запуске Streamlit: {e}")


if __name__ == "__main__":
    run()
