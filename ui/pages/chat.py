"""Chat interface with citations, feedback, and optional LLM generation (WP16 §16.2-16.3)."""

import html as html_module
import os
import re

import httpx
import streamlit as st
from i18n import t, t_list
from progress_helpers import run_with_progress
from ui_style import (
    clean_source_display_name,
    inject_global_styles,
    page_heading,
    render_citation_row,
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Must match ``api.core.generation_catalog.GENERATION_MODEL_IDS`` (UI cannot import ``api`` in Docker).
_VALID_UI_GENERATION_MODELS = frozenset({
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gpt-4",
    "gpt-4o-mini",
})


def _ensure_ui_prefs_loaded() -> None:
    """Load persisted reranker / generation_model from API once per session."""
    if st.session_state.get("_ui_prefs_loaded"):
        return
    try:
        r = httpx.get(f"{API_URL}/config/ui-preferences", timeout=5)
        r.raise_for_status()
        d = r.json()
        st.session_state["ui_reranker"] = d.get("reranker") or None
        gm = d.get("generation_model")
        if isinstance(gm, str):
            gm = gm.strip() or None
        if gm not in _VALID_UI_GENERATION_MODELS:
            gm = None
        st.session_state["ui_generation_model"] = gm
    except Exception:
        st.session_state.setdefault("ui_reranker", None)
        st.session_state.setdefault("ui_generation_model", None)
    st.session_state["_ui_prefs_loaded"] = True


_PICTURE_MARKER_RE = re.compile(
    r"\*?\*?\s*==>\s*picture\s*\[\d+\s*x\s*\d+\]\s*intentionally omitted\s*<==\s*\*?\*?",
    re.IGNORECASE,
)
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\(/corpus/[^\)]+\)\n?")
_MD_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_MD_HEAD_RE = re.compile(r"^#+\s*", re.MULTILINE)


def _clean_chunk_text(text: str) -> str:
    """Strip pymupdf4llm markers and raw image markdown from chunk text."""
    text = _PICTURE_MARKER_RE.sub("", text)
    text = _IMAGE_MD_RE.sub("", text)
    return text.strip()


def _snippet_plain(text: str, max_len: int = 380) -> str:
    """One-line style snippet for source cards (no markdown noise)."""
    t = _clean_chunk_text(text)
    t = _MD_HEAD_RE.sub("", t)
    t = _MD_BOLD_RE.sub(r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return (t[:max_len] + "…") if len(t) > max_len else t


def _relevance_pct(score: float, top_score: float) -> tuple[int, str, str]:
    """Percentage vs top chunk + CSS classes for bar and label (match reference tiers)."""
    sc = float(score)
    ref = float(top_score) if top_score and top_score > 0 else 0.0
    if ref > 0:
        pct = int(max(1, min(100, round(100 * sc / ref))))
    elif sc <= 1.0:
        pct = int(max(1, min(100, round(sc * 100))))
    else:
        pct = int(max(1, min(100, round(sc))))
    if pct >= 82:
        return pct, "", "vk-pct-good"
    if pct >= 65:
        return pct, "vk-bar-warn", "vk-pct-warn"
    return pct, "vk-bar-low", "vk-pct-low"


def _widget_key_safe(query_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", query_id)[:48] or "q"


def _relevance_badge_html(pct: int) -> str:
    """Rounded badge: Relevance:N with green / yellow / red tier."""
    inject_global_styles()
    if pct >= 82:
        cls = "vk-rel-badge vk-rel-badge--good"
    elif pct >= 65:
        cls = "vk-rel-badge vk-rel-badge--ok"
    else:
        cls = "vk-rel-badge vk-rel-badge--bad"
    lab = html_module.escape(t("relevance"))
    return f'<div class="{cls}">{lab}:{pct}</div>'


def _render_secondary_source_card(
    ch: dict,
    top_score: float,
    query_id: str,
    idx: int,
) -> None:
    """Compact card + popover with full chunk text."""
    score = float(ch.get("score") or 0.0)
    pct, _, _ = _relevance_pct(score, top_score)
    doc_raw = str(ch.get("source_doc") or "—")
    doc_display = clean_source_display_name(doc_raw).replace("`", "'")
    page = ch.get("source_page")
    key_base = _widget_key_safe(query_id)
    with st.container(border=True):
        inject_global_styles()
        safe_doc = html_module.escape(doc_display)
        st.markdown(
            f'<div class="vk-source-doc-line">'
            f'<span class="material-symbols-outlined vk-source-doc-line__ic">description</span>'
            f'<span class="vk-source-doc-title">{safe_doc}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if page is not None:
            st.caption(f"p.{page}")
        st.markdown(_relevance_badge_html(pct), unsafe_allow_html=True)
        st.caption(_snippet_plain(ch.get("text") or ""))
        with st.popover(
            t("show_more"),
            icon=":material/unfold_more:",
            width="stretch",
            key=f"src_{key_base}_{idx}",
        ):
            st.markdown(_clean_chunk_text(ch.get("text") or ""))
            render_citation_row(ch, t("image_badge"))
            pct_in, _, _ = _relevance_pct(score, top_score)
            st.markdown(_relevance_badge_html(pct_in), unsafe_allow_html=True)


def _render_secondary_sources_row(rest: list[dict], top_score: float, query_id: str) -> None:
    """Up to 6 cards per horizontal row; each opens a popover for the full passage."""
    st.markdown(
        f'<div class="vk-sources-heading">{html_module.escape(t("sources_panel_title"))}</div>',
        unsafe_allow_html=True,
    )
    max_cols = 6
    n = len(rest)
    for row_start in range(0, n, max_cols):
        row = rest[row_start : row_start + max_cols]
        cols = st.columns(len(row), gap="small")
        for i, ch in enumerate(row):
            with cols[i]:
                _render_secondary_source_card(
                    ch, top_score, query_id, row_start + i
                )


def render_answer_chunk(chunk: dict, idx: int):
    with st.container():
        text = _clean_chunk_text(chunk["text"])
        st.markdown(text)

        render_citation_row(chunk, t("image_badge"))
        sc = float(chunk.get("score") or 0.0)
        pct_top, _, _ = _relevance_pct(sc, sc)
        st.markdown(_relevance_badge_html(pct_top), unsafe_allow_html=True)


def render_response(resp: dict):
    """Top chunk (and optional synthesis) first; other chunks as source cards stacked below."""
    if resp.get("cache_hit"):
        st.info(t("cache_hit"), icon=":material/bolt:")

    chunks = resp.get("answer_chunks") or []

    if not chunks:
        if resp.get("synthesized_answer"):
            st.markdown(resp["synthesized_answer"])
        if resp.get("related_docs"):
            with st.expander(t("see_also")):
                for rd in resp["related_docs"]:
                    st.markdown(f"- **{rd['doc']}** — {rd['relevance']}")
        render_feedback(resp.get("query_id", ""))
        return

    top = chunks[0]
    rest = chunks[1:]

    def _main_answer_block():
        if resp.get("synthesized_answer"):
            st.markdown(resp["synthesized_answer"])
            st.divider()
        render_answer_chunk(top, 0)

    _main_answer_block()
    if rest:
        _render_secondary_sources_row(
            rest,
            float(top.get("score") or 0.0),
            str(resp.get("query_id") or "unknown"),
        )

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


def _chat_body():
    """Scrollable history + footer toggle; ``st.chat_input`` is last so it stays page-bottom."""
    _ensure_ui_prefs_loaded()
    st.session_state.setdefault("chat_generate_toggle", True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    _CHAT_SCROLL_PX = 560
    pending = st.session_state.pop("_pending_question", None)
    # Frozen at send time (same run as ``st.toggle``, before ``st.chat_input``) — avoids reading
    # generate intent on the *pending* rerun before the toggle widget executes.
    pending_generate = st.session_state.pop("_pending_generate", None)

    with st.container(height=_CHAT_SCROLL_PX):
        for msg in st.session_state.messages:
            with st.chat_message(
                msg["role"],
                avatar=":material/person:" if msg["role"] == "user" else ":material/psychology:",
            ):
                if msg["role"] == "assistant" and "response" in msg:
                    render_response(msg["response"])
                else:
                    st.markdown(msg["content"])

        if pending is not None:
            gen = (
                bool(pending_generate)
                if pending_generate is not None
                else False
            )
            label = t("generating") if gen else t("searching")

            with st.chat_message("assistant", avatar=":material/psychology:"):

                def _query():
                    body: dict = {
                        "question": pending,
                        "tenant_id": st.session_state.get("tenant_id", "demo"),
                        "top_k": 5,
                        "generate": gen,
                    }
                    rr = st.session_state.get("ui_reranker")
                    if rr:
                        body["reranker"] = rr
                    gm = st.session_state.get("ui_generation_model")
                    if gm:
                        body["generation_model"] = gm
                    raw = httpx.post(
                        f"{API_URL}/query",
                        json=body,
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

    _tog_sp, _tog_col = st.columns([5, 1])
    with _tog_col:
        st.toggle(
            t("mode_generate_short"),
            key="chat_generate_toggle",
        )

    if prompt := st.chat_input(t("input_placeholder")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state["_pending_question"] = prompt
        st.session_state["_pending_generate"] = bool(
            st.session_state.get("chat_generate_toggle", True)
        )
        st.rerun()


_chat_body()
