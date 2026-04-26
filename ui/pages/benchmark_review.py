"""WP14 §14.2 — Human review of benchmark pairs (accept / reject / optional edit)."""

import os

import httpx
import streamlit as st
from i18n import t
from progress_helpers import run_with_progress
from ui_style import banner, page_heading, section_header

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _post_review(pair_index: int, action: str, edited: str | None, notes: str | None) -> None:
    body = {
        "pair_index": pair_index,
        "action": action,
        "edited_answer": edited,
        "notes": notes,
    }

    def _send():
        r = httpx.post(f"{API_URL}/benchmark/review", json=body, timeout=30)
        r.raise_for_status()
        return True

    try:
        run_with_progress(t("page_loading"), _send)
        banner(t("review_stored"), variant="ok", icon_name="save")
    except Exception as e:
        banner(f"{t('error')}: {e}", variant="error", icon_name="error")


page_heading(t("review_title"), "fact_check")
st.caption(t("page_desc_benchmark_review"))


def _review_body():
    col_a, col_b = st.columns(2)
    with col_a:
        offset = st.number_input(t("review_offset"), min_value=0, value=0, step=1)
    with col_b:
        limit = st.number_input(t("review_limit"), min_value=1, max_value=50, value=5, step=1)

    section_header(t("review_title"), "fact_check")
    if st.button(t("review_load"), icon=":material/folder_open:"):
        off = int(offset)
        lim = int(limit)

        def _load_pairs():
            r = httpx.get(
                f"{API_URL}/benchmark/pairs",
                params={"offset": off, "limit": lim},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()

        try:
            st.session_state["review_batch"] = run_with_progress(t("page_loading"), _load_pairs)
        except Exception as e:
            banner(f"{t('error')}: {e}", variant="error", icon_name="error")

    batch = st.session_state.get("review_batch")
    if batch:
        st.caption(t("review_total").format(n=batch.get("total", 0)))
        for p in batch.get("pairs", []):
            idx = p.get("_line_index", 0)
            with st.expander(f"#{idx} — {p.get('question', '')[:80]}…"):
                st.markdown(f"**{t('review_question')}** {p.get('question', '')}")
                st.markdown(f"**{t('review_answer')}** {p.get('answer', '')}")
                prev = p.get("chunk_text_preview") or ""
                if prev:
                    st.text_area(t("review_chunk_preview"), prev, height=160, disabled=True, key=f"ch_{idx}")
                hn1 = p.get("hard_neg_a")
                hn2 = p.get("hard_neg_b")
                if hn1 or hn2:
                    st.markdown(f"**{t('review_hard_neg')}**")
                    if hn1:
                        st.caption(hn1[:600])
                    if hn2:
                        st.caption(hn2[:600])
                edit = st.text_input(t("review_edited_answer"), key=f"ed_{idx}", value="")
                notes = st.text_input(t("review_notes"), key=f"nt_{idx}", value="")
                c1, c2 = st.columns(2)
                if c1.button(t("review_accept"), key=f"a_{idx}", icon=":material/check:"):
                    _post_review(idx, "accept", edit or None, notes or None)
                if c2.button(t("review_reject"), key=f"r_{idx}", icon=":material/close:"):
                    _post_review(idx, "reject", edit or None, notes or None)


_review_body()
