import os
import streamlit as st
from frontend.core.config import settings
from frontend.services.schemas import PredictResponse


def load_css():
    """Загрузка пользовательских стилей."""
    if os.path.exists(settings.CSS_PATH):
        with open(settings.CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header():
    """Отрисовка заголовка и приветственного блока."""
    st.title(f"{settings.APP_TITLE}")

    if st.session_state.get("uploader") is None and not st.session_state.get(
            "is_admin"):
        st.markdown(
            '<div class="description-block">'
            '<h3>Автоматическое определение музыкальных жанров</h3>'
            '<p>Cервис для классификации '
            'аудиокомпозиций с использованием алгоритмов '
            'глубокого обучения.</p>'
            '</div>',
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


def render_predictions(predictions_data: PredictResponse) -> None:
    """Обработка и визуализация результатов классификации."""
    items = predictions_data.top_3
    items = sorted(items, key=lambda x: x.confidence, reverse=True)
    if not items:
        st.warning("Результаты анализа не получены в ожидаемом формате.")
        return None

    prediction = predictions_data.prediction
    confidence = predictions_data.confidence
    st.success(
        f"Наиболее вероятный жанр: "
        f"**{prediction.upper()}** ({confidence:.1%})"
    )

    with st.expander(f"Дополнительные варианты (Топ-{len(items)})"):
        for item in items:
            c1, c2 = st.columns([1, 4])
            c1.write(f"**{item.genre.capitalize()}**")
            c2.caption(f"{item.confidence:.1%}")
            conf = min(1.0, max(0.0, float(item.confidence)))
            c2.progress(conf)


def render_history():
    """Отрисовка истории последних запросов."""
    if st.session_state.get("history"):
        st.markdown("---")
        st.subheader("История запросов")
        for h in st.session_state.history[:5]:
            st.write(f"🕒 `{h['genre'].upper()}` — {h['file']}")


def render_admin_login():
    """Отрисовка формы входа администратора."""
    st.subheader("🔑 Панель администратора")
    with st.form("admin_login_form"):
        username = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")
        submit = st.form_submit_button("Войти")
        if submit:
            if username == "admin" and password == "admin":
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("Неверные учетные данные")


def render_admin_upload():
    """Отрисовка панели загрузки архива с использованием формы."""
    st.subheader("⚙️ Управление Fine-tuning")
    st.info("Загрузите архив (.zip) с размеченными данными для дообучения.")

    with st.form("finetune_form"):
        uploaded_file = st.file_uploader("Архив с данными", type=["zip"])
        submitted = st.form_submit_button("Запустить обучение")

    if submitted:
        if uploaded_file is not None:
            from frontend.services.classifier_api import \
                upload_finetune_archive

            with st.spinner("Загрузка архива на сервер..."):
                success, msg = upload_finetune_archive(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    "admin",
                    "admin"
                )

            if success:
                st.success(msg)
            else:
                st.error(msg)
        else:
            st.warning("Пожалуйста, сначала выберите файл.")
