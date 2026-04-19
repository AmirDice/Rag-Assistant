"""Chat interface with citations, feedback, and optional LLM generation (WP16 §16.2-16.3)."""

import os
import re
import streamlit as st
import httpx
from i18n import t, t_list
from progress_helpers import run_with_progress
from ui_style import page_heading, render_citation_row

API_URL = os.getenv("API_URL", "http://localhost:8000")


_PICTURE_MARKER_RE = re.compile(
    r"\*?\*?\s*==>\s*picture\s*\[\d+\s*x\s*\d+\]\s*intentionally omitted\s*<==\s*\*?\*?",
    re.IGNORECASE,
)
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\(/corpus/[^\)]+\)\n?")


def _clean_chunk_text(text: str) -> str:
    """Strip pymupdf4llm markers and raw image markdown from chunk text."""
    text = _PICTURE_MARKER_RE.sub("", text)
    text = _IMAGE_MD_RE.sub("", text)
    return text.strip()


def render_answer_chunk(chunk: dict, idx: int):
    with st.container():
        text = _clean_chunk_text(chunk["text"])
        st.markdown(text)

        render_citation_row(chunk, t("image_badge"))
        st.markdown(
            f'<div class="vk-relevance-note">{t("relevance")}: {chunk.get("score", 0):.3f}</div>',
            unsafe_allow_html=True,
        )


def render_response(resp: dict):
    """Render a full response — synthesized answer + source chunks."""
    if resp.get("cache_hit"):
        st.info(t("cache_hit"), icon=":material/bolt:")

    if resp.get("synthesized_answer"):
        st.markdown(resp["synthesized_answer"])
        st.divider()
        with st.expander(f"{t('source_chunks')} ({len(resp.get('answer_chunks', []))})"):
            for i, chunk in enumerate(resp.get("answer_chunks", [])):
                render_answer_chunk(chunk, i)
                if i < len(resp["answer_chunks"]) - 1:
                    st.divider()
    else:
        for i, chunk in enumerate(resp.get("answer_chunks", [])):
            render_answer_chunk(chunk, i)
            if i < len(resp["answer_chunks"]) - 1:
                st.divider()

    if resp.get("related_docs"):
        with st.expander(t("see_also")):
            for rd in resp["related_docs"]:
                st.markdown(f"- **{rd['doc']}** — {rd['relevance']}")

    render_feedback(resp.get("query_id", ""))


def render_feedback(query_id: str):
    feedback_key = f"feedback_{query_id}"

    if feedback_key in st.session_state and st.session_state[feedback_key] == "sent":
        st.success(t("feedback_thanks"), icon=":material/check_circle:")
        return

    cols = st.columns([1, 1, 8])
    with cols[0]:
        if st.button(t("btn_useful"), key=f"ok_{query_id}", icon=":material/thumb_up:"):
            _send_feedback(query_id, "ok")
            st.session_state[feedback_key] = "sent"
            st.rerun()
    with cols[1]:
        if st.button(t("btn_not_useful"), key=f"nok_{query_id}", icon=":material/thumb_down:"):
            st.session_state[f"show_feedback_form_{query_id}"] = True

    if st.session_state.get(f"show_feedback_form_{query_id}"):
        reason = st.selectbox(
            t("feedback_reason"),
            t_list("feedback_reasons"),
            key=f"reason_{query_id}",
        )
        correction = st.text_area(
            t("feedback_correction"),
            key=f"correction_{query_id}",
        )
        if st.button(t("feedback_send"), key=f"send_{query_id}"):
            _send_feedback(query_id, "not_ok", reason, correction or None)
            st.session_state[feedback_key] = "sent"
            st.session_state[f"show_feedback_form_{query_id}"] = False
            st.rerun()


def _send_feedback(query_id: str, rating: str, reason: str = None, correction: str = None):
    try:
        payload = {
            "query_id": query_id,
            "rating": rating,
            "tenant_id": st.session_state.get("tenant_id", "demo"),
        }
        if reason:
            payload["reason"] = reason
        if correction:
            payload["correction"] = correction
        httpx.post(f"{API_URL}/feedback", json=payload, timeout=5)
    except Exception:
        pass


# ── Page layout ──────────────────────────────────────────────

page_heading(t("chat_title"), "chat")
st.caption(t("page_desc_chat"))

with st.sidebar:
    st.session_state.setdefault("tenant_id", "demo")
    st.session_state["tenant_id"] = st.text_input(
        t("tenant_label"), value=st.session_state["tenant_id"]
    )
    top_k = st.slider(t("results_label"), min_value=1, max_value=10, value=5)

    generate_mode = st.toggle(
        t("mode_generate"),
        value=st.session_state.get("generate_mode", False),
        key="_generate_toggle",
    )
    st.session_state["generate_mode"] = generate_mode


def _chat_body():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(
            msg["role"],
            avatar=":material/person:" if msg["role"] == "user" else ":material/psychology:",
        ):
            if msg["role"] == "assistant" and "response" in msg:
                render_response(msg["response"])
            else:
                st.markdown(msg["content"])

    if question := st.chat_input(t("input_placeholder")):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(question)

        with st.chat_message("assistant", avatar=":material/psychology:"):
            label = t("generating") if generate_mode else t("searching")

            def _query():
                raw = httpx.post(
                    f"{API_URL}/query",
                    json={
                        "question": question,
                        "tenant_id": st.session_state.get("tenant_id", "demo"),
                        "top_k": top_k,
                        "generate": generate_mode,
                    },
                    timeout=60,
                )
                raw.raise_for_status()
                return raw.json()

            try:
                resp = run_with_progress(label, _query)
                if not resp.get("answer_chunks"):
                    st.warning(t("no_results"))
                else:
                    render_response(resp)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "...",
                    "response": resp,
                })
            except httpx.ConnectError:
                st.error(t("api_error"))
            except httpx.HTTPStatusError as e:
                st.error(f"{t('error')}: API returned {e.response.status_code}")
            except Exception as e:
                st.error(f"{t('error')}: {e}")


_chat_body()

