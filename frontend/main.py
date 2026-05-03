import streamlit as st
from frontend.core.config import settings
from frontend.utils.metadata_extractor import get_album_art
from frontend.services.classifier_api import predict_genre
from frontend.components.layout import (
    load_css,
    render_header,
    render_history,
    render_file_details,
    render_predictions
)

st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🎵",
    layout="centered"
)


def main():
    """
    Основная точка входа в приложение: инициализация интерфейса,
    обработка загрузки файла и координация вызова API классификатора.
    """
    load_css()
    render_header()

    if "history" not in st.session_state:
        st.session_state.history = []

    allowed = [ext.replace(".", "") for ext in settings.SUPPORTED_EXTENSIONS]
    uploaded_file = st.file_uploader(
        "Загрузите аудиофайл",
        type=allowed,
        label_visibility="collapsed",
        key="uploader"
    )

    if uploaded_file is None:
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        st.info(f"Допустимые форматы: {', '.join(allowed).upper()} "
                f"(до {settings.MAX_UPLOAD_SIZE_MB} МБ)")
        render_history()
        return

    file_bytes = uploaded_file.getvalue()
    col1, col2 = st.columns([1, 2.5], gap="medium")
    with col1:
        art = get_album_art(file_bytes, uploaded_file.name)
        st.image(art, width='stretch')
    with col2:
        render_file_details(uploaded_file)

    if st.button("ОПРЕДЕЛИТЬ ЖАНР", type="primary", width='stretch'):
        with st.spinner("Выполняется анализ жанра композиции..."):
            res, error = predict_genre(
                file_bytes,
                uploaded_file.name,
                uploaded_file.type
            )
            if error:
                st.error(error)
            elif res:
                winner_genre = render_predictions(res.predictions)
                if winner_genre:
                    st.session_state.history.insert(0, {
                        "file": uploaded_file.name,
                        "genre": winner_genre
                    })

    render_history()


if __name__ == "__main__":
    main()
