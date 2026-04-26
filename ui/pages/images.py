"""WP16 §16.1 / WP14 §14.2 — Image gallery: search docs, load images only for selection."""

import os
from io import BytesIO
from urllib.parse import quote

import httpx
import streamlit as st
from i18n import t
from progress_helpers import image_fetch_progress, run_with_progress
from ui_style import banner, page_heading, section_header

API_URL = os.getenv("API_URL", "http://localhost:8000")
_MAX_IMAGES = 40


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_documents(api_base: str) -> dict:
    r = httpx.get(f"{api_base.rstrip('/')}/library/documents", timeout=30)
    r.raise_for_status()
    return r.json()


def _doc_matches(doc: dict, q: str) -> bool:
    if not q.strip():
        return True
    qn = q.strip().lower()
    parts = [
        str(doc.get("doc_id", "") or ""),
        str(doc.get("source_file", "") or ""),
        str(doc.get("doc_type", "") or ""),
        str(doc.get("module_id", "") or ""),
    ]
    return any(qn in p.lower() for p in parts)


page_heading(t("images_title"), "image")
st.caption(t("page_desc_images"))


def _images_body():
    try:
        doc_payload = run_with_progress(t("page_loading"), lambda: _fetch_documents(API_URL))
    except Exception as e:
        banner(f"{t('images_load_error')}: {e}", variant="error", icon_name="error")
        st.stop()

    all_docs = [d for d in doc_payload.get("documents", []) if d.get("media_files")]
    if not all_docs:
        banner(t("images_no_media"), variant="warn", icon_name="warning")
        st.stop()

    search = st.text_input(
        t("images_search_label"),
        value="",
        placeholder=t("images_search_placeholder"),
        key="images_search_query",
    )

    filtered = [d for d in all_docs if _doc_matches(d, search)]
    if not filtered:
        banner(t("images_no_match"), variant="info", icon_name="info")
        st.stop()

    options = [d["doc_id"] for d in filtered]

    def _format_doc(did: str) -> str:
        d = next((x for x in filtered if x["doc_id"] == did), None)
        if not d:
            return did
        src = d.get("source_file") or ""
        n = len(d.get("media_files") or [])
        return f"{did} — {src} ({t('images_n_images').format(n=n)})"

    selected = st.selectbox(
        t("images_select_doc"),
        options=options,
        format_func=_format_doc,
        index=0,
        key="images_selected_doc",
    )

    doc = next((x for x in filtered if x["doc_id"] == selected), None)
    if not doc:
        st.stop()

    media = doc.get("media_files") or []
    st.divider()
    section_header(doc.get("source_file") or doc["doc_id"], "image")
    st.caption(t("images_n_images").format(n=len(media)))

    slice_m = media[:_MAX_IMAGES]
    n_img = len(slice_m)
    bar = st.empty()
    for i, name in enumerate(slice_m):
        image_fetch_progress(bar, t("images_loading"), i, n_img)
        doc_seg = quote(doc["doc_id"], safe="")
        name_seg = quote(name, safe="")
        url = f"{API_URL.rstrip('/')}/corpus/{doc_seg}/media/{name_seg}"
        st.markdown(f"**{name}**")
        try:
            ir = httpx.get(url, timeout=60)
            if ir.status_code == 200 and ir.content:
                st.image(BytesIO(ir.content))
            else:
                st.caption(f"{t('images_http_error')}: {ir.status_code} — `{url}`")
        except Exception as ex:
            st.caption(f"{t('images_fetch_failed')}: {ex}")
    bar.empty()

    if len(media) > _MAX_IMAGES:
        st.caption(t("images_truncated"))


_images_body()
