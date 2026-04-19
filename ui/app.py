"""RAG Assistant — Streamlit main entry point."""

import streamlit as st

from ui_style import FAVICON_PATH, inject_global_styles, LOGO_PATH

st.set_page_config(
    page_title="RAG Assistant",
    page_icon=FAVICON_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("ui_theme", "light")
inject_global_styles()

# st.logo() expects a path, URL, PIL image, or numpy array — raw bytes are not reliably supported.
if LOGO_PATH:
    st.logo(LOGO_PATH, size="large")

from i18n import init_lang, t

init_lang()

with st.sidebar:
    st.caption(t("theme_label"))
    _choice = st.radio(
        "theme_mode",
        options=["light", "dark"],
        format_func=lambda x: t("theme_light") if x == "light" else t("theme_dark"),
        index=0 if st.session_state.ui_theme == "light" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    if _choice != st.session_state.ui_theme:
        st.session_state.ui_theme = _choice
        st.rerun()

pg = st.navigation([
    st.Page("pages/chat.py", title=t("page_chat"), icon=":material/chat:", default=True),
    st.Page("pages/corpus.py", title=t("page_corpus"), icon=":material/folder_open:"),
    st.Page("pages/images.py", title=t("page_images"), icon=":material/image:"),
    st.Page("pages/benchmark.py", title=t("page_benchmark"), icon=":material/analytics:"),
    st.Page("pages/benchmark_review.py", title=t("page_review"), icon=":material/fact_check:"),
    st.Page("pages/onboarding.py", title=t("page_onboarding"), icon=":material/domain:"),
    st.Page("pages/settings.py", title=t("page_settings"), icon=":material/settings:"),
])

pg.run()
