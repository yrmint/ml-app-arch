import streamlit as st
import requests
import io
from PIL import Image, UnidentifiedImageError
from mutagen import MutagenError
from mutagen.mp3 import MP3
from mutagen.id3 import APIC
from mutagen.flac import FLAC
from core.config import settings

st.set_page_config(
    page_title=f"{settings.APP_TITLE}",
    page_icon="🎵",
    layout="centered"
)

ERROR_MESSAGES = {
    413: "Файл слишком велик. Пожалуйста, загрузите файл меньшего размера.",
    415: "Данный формат аудио не поддерживается.",
    429: "Слишком много запросов. Пожалуйста, подождите некоторое время.",
    500: ("Произошла ошибка при анализе композиции. "
          "Мы уже занимаемся её устранением."
          ),
    503: "Система временно перегружена. Пожалуйста, повторите попытку позже."
}


def get_album_art(file_bytes: bytes, file_name: str):
    """
    Извлекает обложку из аудиофайла.
    Возвращает путь к заглушке в случае неудачи.
    """
    try:
        audio_file = io.BytesIO(file_bytes)
        if file_name.lower().endswith('.mp3'):
            audio = MP3(audio_file)
            if audio.tags:
                for tag in audio.tags.values():
                    if isinstance(tag, APIC):
                        return Image.open(io.BytesIO(tag.data))
        elif file_name.lower().endswith('.flac'):
            audio = FLAC(audio_file)
            if audio.pictures:
                return Image.open(io.BytesIO(audio.pictures[0].data))
    except (MutagenError, UnidentifiedImageError, ValueError, KeyError):
        pass

    return settings.DEFAULT_COVER_PATH


def main():
    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
        }

        div.stAudio { margin-bottom: 2rem; }

        [data-testid="stHorizontalBlock"] {
            display: flex;
            align-items: stretch;
            padding-top: 0.5rem;
        }

        [data-testid="column"] {
            display: flex;
        }

        .feature-card {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
            border: 1px solid #e0e0e0;
            width: 100%;
            font-size: 0.9rem;
            word-wrap: break-word;
        }

        .spacer {
            margin-top: 1.5rem;
        }

        .file-info {
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }

        .cta-wrapper {
            padding-top: 2rem;
            padding-bottom: 0.5rem;
        }

        .cta-text {
            font-weight: 600;
            color: #31333F;
            margin: 0;
        }

        .description-block h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1.75rem;
        }

        .description-block p {
            margin-bottom: 0;
        }

        hr {
            margin-top: 0.5rem !important;
            margin-bottom: 1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title(f"🎵 {settings.APP_TITLE}")

    if "history" not in st.session_state:
        st.session_state.history = []

    allowed_types = [ext.replace(".", "") for ext in
                     settings.SUPPORTED_EXTENSIONS]

    if st.session_state.get("uploader") is None:
        st.markdown("""
        <div class="description-block">
            <h3>Автоматическое определение музыкальных жанров</h3>
            <p>Профессиональный сервис для мгновенной классификации 
            аудиокомпозиций с использованием алгоритмов глубокого обучения.</p>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            st.markdown(
                '<div class="feature-card"><b>🚀 Оперативность</b>'
                '<br>Получение результата анализа в течение нескольких секунд'
                '</div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                '<div class="feature-card"><b>📊 Наглядность</b>'
                '<br>Детальный расчет вероятностей по списку направлений'
                '</div>',
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                '<div class="feature-card">'
                '<b>🔒 Безопасность</b><br>Конфиденциальная обработка данных '
                'без сохранения файлов</div>',
                unsafe_allow_html=True)

        st.markdown(
            '<div class="cta-wrapper"><p class="cta-text">Выберите аудиофайл '
            'для определения жанра:</p></div>',
            unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Загрузите аудиофайл",
        type=allowed_types,
        label_visibility="collapsed",
        key="uploader"
    )

    if uploaded_file is None:
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        st.info(
            f"Допустимые форматы: {', '.join(allowed_types).upper()} "
            f"(ограничение размера до {settings.MAX_UPLOAD_SIZE_MB} МБ)")

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        col1, col2 = st.columns([1, 2.5], gap="medium")

        with col1:
            art = get_album_art(file_bytes, uploaded_file.name)
            st.image(art, width='stretch')

        with col2:
            st.markdown(
                f"<div class='file-info'><b>Файл:</b> "
                f"<code>{uploaded_file.name}</code><br>"
                f"<b>Размер:</b> "
                f"<code>{uploaded_file.size / (1024 * 1024):.2f} "
                f"МБ</code></div>",
                unsafe_allow_html=True)
            st.audio(uploaded_file)

        if st.button("ОПРЕДЕЛИТЬ ЖАНР", type="primary",
                     use_container_width=True):
            with st.spinner("Выполняется анализ жанра композиции..."):
                try:
                    files = {"audio_file": (
                        uploaded_file.name, file_bytes, uploaded_file.type)}
                    timeout = getattr(settings, "TIMEOUT", 60)
                    response = requests.post(f"{settings.BACKEND_URL}/predict",
                                             files=files, timeout=timeout)

                    if response.status_code == 200:
                        res = response.json()
                        predictions_data = res.get("top_3") or res.get(
                            "top_k") or res.get("predictions")

                        if predictions_data:
                            items = list(
                                predictions_data.values()) if isinstance(
                                predictions_data,
                                dict) else predictions_data
                            items = sorted(items,
                                           key=lambda x: x.get('confidence',
                                                               0),
                                           reverse=True)
                            winner = items[0]

                            st.success(
                                f"Наиболее вероятный жанр: "
                                f"**{winner['genre'].upper()}** "
                                f"({winner['confidence']:.1%})")
                            st.session_state.history.insert(0, {
                                "file": uploaded_file.name,
                                "genre": winner['genre']})

                            with st.expander(
                                    f"Детальное распределение "
                                    f"(Топ-{len(items)})"):
                                for item in items:
                                    c1_d, c2_d = st.columns([1, 4])
                                    c1_d.write(
                                        f"**{item['genre'].capitalize()}**")
                                    c2_d.caption(f"{item['confidence']:.1%}")
                                    c2_d.progress(float(item['confidence']))
                                    st.markdown(" ")
                        else:
                            st.warning("Результаты анализа не получены в "
                                       "ожидаемом формате.")
                    else:
                        try:
                            res_data = response.json() if response.headers.get(
                                'Content-Type') == 'application/json' else {}
                            error_detail = res_data.get("detail")
                        except (
                                requests.exceptions.JSONDecodeError,
                                ValueError):
                            error_detail = None

                        msg = error_detail if error_detail else \
                            ERROR_MESSAGES.get(response.status_code,
                                               "Сервис временно недоступен.")
                        st.error(f"Ошибка {response.status_code}: {msg}")

                except requests.exceptions.RequestException:
                    st.error(
                        "Проблема с сетевым соединением. "
                        "Убедитесь, что сервер запущен.")

    if st.session_state.history:
        st.markdown("---")
        st.subheader("История запросов")
        for h in st.session_state.history[:5]:
            st.write(f"🕒 `{h['genre'].upper()}` — {h['file']}")


if __name__ == "__main__":
    main()
