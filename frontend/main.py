import time
import streamlit as st
from frontend.core.config import settings
from frontend.utils.metadata_extractor import get_album_art
from frontend.services.classifier_api import (
    send_audio_to_queue,
    get_task_status,
    TASK_STATUS_MESSAGES,
    NETWORK_ERRORS,
    API_ERRORS
)
from frontend.components.layout import (
    load_css,
    render_header,
    render_history,
    render_file_details,
    render_predictions,
    render_admin_login,
    render_admin_upload
)

st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🎵",
    layout="centered"
)


def main():
    load_css()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    with st.sidebar:
        st.title("Меню")
        if st.button("🏠 Главная"):
            st.session_state.is_admin = False
            st.rerun()
        if st.button("🔒 Панель управления"):
            st.session_state.is_admin = True
            st.rerun()

    if st.session_state.is_admin:
        if st.session_state.get("admin_logged_in"):
            render_admin_upload()
            if st.button("Выйти"):
                st.session_state.is_admin = False
                st.rerun()
        else:
            render_admin_login()
        return

    render_header()
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
        st.image(art, width="stretch")
    with col2:
        render_file_details(uploaded_file)

    if st.button("ОПРЕДЕЛИТЬ ЖАНР", type="primary", use_container_width=True):
        with st.spinner("Пожалуйста, подождите. Файл отправляется..."):
            task_res, error = send_audio_to_queue(
                file_bytes,
                uploaded_file.name,
                uploaded_file.type
            )

        if error:
            st.error(error)
        elif task_res:
            task_id = str(task_res.task_id)
            status_container = st.empty()
            max_retries = 120
            retries = 0

            while retries < max_retries:
                status_res, status_err = get_task_status(task_id)
                if status_err:
                    status_container.error(status_err)
                    break

                current_status = status_res.status
                if current_status == "completed":
                    status_container.empty()
                    if status_res.result:
                        render_predictions(status_res.result)
                        st.session_state.history.insert(0, {
                            "file": uploaded_file.name,
                            "genre": status_res.result.prediction
                        })
                    else:
                        status_container.error(API_ERRORS["missing_result"])
                    break
                elif current_status in TASK_STATUS_MESSAGES:
                    if current_status == "failed":
                        status_container.error(
                            TASK_STATUS_MESSAGES[current_status])
                        break
                    status_container.warning(
                        TASK_STATUS_MESSAGES[current_status])

                time.sleep(1.0)
                retries += 1
            if retries >= max_retries:
                status_container.error(NETWORK_ERRORS["timeout"])

    render_history()


if __name__ == "__main__":
    main()
