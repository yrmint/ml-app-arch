import os
from typing import Union, Optional, Any
import streamlit as st
from frontend.core.config import settings


def load_css():
    """Загрузка пользовательских стилей."""
    if os.path.exists(settings.CSS_PATH):
        with open(settings.CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header():
    """Отрисовка заголовка и приветственного блока."""
    st.title(f"🎵 {settings.APP_TITLE}")

    if st.session_state.get("uploader") is None:
        st.markdown(
            '<div class="description-block">'
            '<h3>Автоматическое определение музыкальных жанров</h3>'
            '<p>Профессиональный сервис для мгновенной классификации '
            'аудиокомпозиций с использованием алгоритмов '
            'глубокого обучения.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            with st.container(border=True):
                st.markdown(
                    '<div class="card-header">🚀 Оперативность</div>'
                    '<div class="card-content">Получение результата '
                    'анализа в течение нескольких секунд</div>',
                    unsafe_allow_html=True)
        with c2:
            with st.container(border=True):
                st.markdown(
                    '<div class="card-header">📊 Наглядность</div>'
                    '<div class="card-content">Детальный расчет '
                    'вероятностей по каждому жанру</div>',
                    unsafe_allow_html=True
                )
        with c3:
            with st.container(border=True):
                st.markdown(
                    '<div class="card-header">🔒 Безопасность</div>'
                    '<div class="card-content">Конфиденциальная обработка '
                    'данных без сохранения файлов</div>',
                    unsafe_allow_html=True
                )

        st.markdown(
            '<div class="cta-wrapper"><p class="cta-text">Выберите аудиофайл '
            'для определения жанра:</p></div>',
            unsafe_allow_html=True)


def render_file_details(uploaded_file):
    """Отображение информации о загруженном файле."""
    st.markdown(
        f"<div class='file-info'><b>Файл:</b> "
        f"<code>{uploaded_file.name}</code><br>"
        f"<b>Размер:</b> <code>"
        f"{uploaded_file.size / (1024 * 1024):.2f} МБ</code></div>",
        unsafe_allow_html=True
    )
    st.audio(uploaded_file)


def render_predictions(predictions_data: Union[list, Any]) -> Optional[str]:
    """Обработка и визуализация результатов классификации."""
    items = predictions_data
    items = sorted(items, key=lambda x: x.confidence, reverse=True)
    if not items:
        st.warning("Результаты анализа не получены в ожидаемом формате.")
        return None

    winner = items[0]
    st.success(
        f"Наиболее вероятный жанр: "
        f"**{winner.genre.upper()}** ({winner.confidence:.1%})"
    )

    with st.expander(f"Детальное распределение (Топ-{len(items)})"):
        for item in items:
            c1, c2 = st.columns([1, 4])
            c1.write(f"**{item.genre.capitalize()}**")
            c2.caption(f"{item.confidence:.1%}")
            conf = min(1.0, max(0.0, float(item.confidence)))
            c2.progress(conf)

    return winner.genre


def render_history():
    """Отрисовка истории последних запросов."""
    if st.session_state.get("history"):
        st.markdown("---")
        st.subheader("История запросов")
        for h in st.session_state.history[:5]:
            st.write(f"🕒 `{h['genre'].upper()}` — {h['file']}")
