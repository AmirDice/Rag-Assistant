"""Chat interface with citations, feedback, and optional LLM generation (WP16 §16.2-16.3)."""

import base64
import html as html_module
import io
import os
import re

import httpx
import streamlit as st
from i18n import t, t_list
from progress_helpers import run_with_progress
from ui_style import (
    banner,
    clean_source_display_name,
    inject_global_styles,
    page_heading,
    render_citation_row,
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Must match ``api.core.generation_catalog`` available_models ids (UI cannot import ``api`` in Docker).
_VALID_UI_GENERATION_MODELS = frozenset({
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gpt-4",
    "gpt-4o-mini",
    "llama3.1-local",
})

# Local labels for the answer-meta surfaces (kept here so we don't touch i18n.py).
_SURFACE_LABELS = {
    "es": {
        "grounded_ok": "Verificado: respaldado por las fuentes",
        "grounded_low": "Atención: respaldo débil en las fuentes",
        "coverage": "cobertura",
        "est_cost": "Coste estimado",
        "query_understanding": "Procesado de la consulta",
        "matched_terms": "Términos del glosario detectados",
        "retrieval_query": "Consulta usada en la búsqueda",
        "clarification": "Necesito una aclaración",
        "new_chat": "Nuevo chat",
        "your_chats": "Conversación",
        "chat_default_title": "Nuevo chat",
        "delete_chat": "Eliminar",
        "empty_hint": "Escribe tu pregunta abajo para empezar.",
    },
    "en": {
        "grounded_ok": "Verified: supported by sources",
        "grounded_low": "Heads up: weak support in sources",
        "coverage": "coverage",
        "est_cost": "Estimated cost",
        "query_understanding": "Query processing",
        "matched_terms": "Glossary terms detected",
        "retrieval_query": "Query used for search",
        "clarification": "I need a clarification",
        "new_chat": "New chat",
        "your_chats": "Conversation",
        "chat_default_title": "New chat",
        "delete_chat": "Delete",
        "empty_hint": "Type your question below to get started.",
    },
}


def _lbl(key: str) -> str:
    lang = st.session_state.get("ui_lang", "en")
    return _SURFACE_LABELS.get(lang, _SURFACE_LABELS["en"]).get(key, key)


def _render_answer_meta(resp: dict) -> None:
    """Subtle grounding line + a small query-processing note. (Cost lives on the
    Usage dashboard, not inline in chat.)"""
    grounded = resp.get("grounded")
    if grounded is not None:
        ratio = resp.get("grounding_ratio")
        ratio_txt = ""
        if ratio is not None:
            try:
                ratio_txt = f" · {int(round(float(ratio) * 100))}% {_lbl('coverage')}"
            except (TypeError, ValueError):
                ratio_txt = ""
        if grounded:
            st.caption(f":green[●] {_lbl('grounded_ok')}{ratio_txt}")
        else:
            st.caption(f":orange[●] {_lbl('grounded_low')}{ratio_txt}")

    matched = resp.get("matched_terms") or []
    orig = (resp.get("original_query") or "").strip()
    rq = (resp.get("retrieval_query") or "").strip()
    if matched or (rq and rq.lower() != orig.lower()):
        with st.expander(_lbl("query_understanding")):
            if matched:
                st.markdown(f"**{_lbl('matched_terms')}:** " + ", ".join(matched))
            if rq and rq.lower() != orig.lower():
                st.markdown(f"**{_lbl('retrieval_query')}:** {rq}")


def _render_sources_expander(chunks: list[dict], query_id: str) -> None:
    """All retrieved chunks in ONE tidy expander — replaces the bordered-card grid."""
    if not chunks:
        return
    top_score = float(chunks[0].get("score") or 0.0)
    key_base = _widget_key_safe(query_id)
    with st.expander(f":material/menu_book: {t('sources_panel_title')} ({len(chunks)})"):
        for i, ch in enumerate(chunks):
            doc = clean_source_display_name(str(ch.get("source_doc") or "—")).replace("`", "'")
            page = ch.get("source_page")
            pct, _, _ = _relevance_pct(float(ch.get("score") or 0.0), top_score)
            loc = f" · p.{page}" if page is not None else ""
            st.markdown(f"**{i + 1}. {doc}**{loc}  ·  {t('relevance')} {pct}%")
            st.caption(_snippet_plain(ch.get("text") or "", max_len=240))
            with st.popover(t("show_more"), icon=":material/unfold_more:", key=f"src_{key_base}_{i}"):
                st.markdown(_clean_chunk_text(ch.get("text") or ""))
                render_citation_row(ch, t("image_badge"))
            if i < len(chunks) - 1:
                st.divider()


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
_IMAGE_PATH_RE = re.compile(r"!\[[^\]]*\]\((/corpus/[^\)]+)\)")
_HTML_CORPUS_IMG_RE = re.compile(
    r'(?:src|href)\s*=\s*"(/corpus/[^"]+)"',
    re.IGNORECASE,
)
_MD_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_MD_HEAD_RE = re.compile(r"^#+\s*", re.MULTILINE)

# Word / Office → markdown noise (Gemini echoes these instead of real images).
_OFFICE_DIM_DQ_RE = re.compile(
    r'\{\s*width\s*=\s*"[^"]+"\s+height\s*=\s*"[^"]+"\s*\}',
    re.IGNORECASE,
)
_OFFICE_DIM_SQ_RE = re.compile(
    r"\{\s*width\s*=\s*'[^']+'\s+height\s*=\s*'[^']+'\s*\}",
    re.IGNORECASE,
)
# Stray single-attribute braces sometimes left behind.
_OFFICE_DIM_ORPHAN_RE = re.compile(
    r'\{\s*(?:width|height)\s*=\s*"[^"]+"\s*\}',
    re.IGNORECASE,
)
_GEMINI_IMG_DISCLAIMER_ES_RE = re.compile(
    r"El\s+contenido\s+generado\s+por\s+IA\s+puede\s+ser\s+incorrecto\.?",
    re.IGNORECASE,
)
_GEMINI_IMG_DISCLAIMER_EN_RE = re.compile(
    r"The\s+AI[- ]?generated\s+content\s+may\s+be\s+incorrect\.?",
    re.IGNORECASE,
)

# Gemini echoes Word/vision “UI strings” inside paragraphs — remove inline (not only full lines).
# Longest / most specific patterns first.
_INLINE_UI_JUNK_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\s*Graphical user interface, Text, Application, Email\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*Graphical user interface, Text, Application\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*Graphical user interface, Text, Email\s*",
        re.IGNORECASE,
    ),
    re.compile(r"\s*Graphical user interface, Text\s*", re.IGNORECASE),
    re.compile(r"\s*Graphical user interface\s*", re.IGNORECASE),
    re.compile(
        r"\s*Interfaz de usuario gráfica, Texto, Aplicación, Correo electrónico\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*Interfaz de usuario gráfica, Texto, Aplicación\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*Interfaz de usuario gráfica, Texto, Correo electrónico\s*",
        re.IGNORECASE,
    ),
    re.compile(r"\s*Interfaz de usuario gráfica, Texto\s*", re.IGNORECASE),
    re.compile(r"\s*Interfaz de usuario gráfica\s*", re.IGNORECASE),
    # Match even when Word/Gemini glues the next word (e.g. “…backgroundalong with”).
    re.compile(r"A set of black letters on a white background", re.IGNORECASE),
    re.compile(r"Un conjunto de letras negras en un fondo blanco", re.IGNORECASE),
    # OCR table title stuck after “website”
    re.compile(r"\s+Board\s+\.", re.IGNORECASE),
    re.compile(
        r"\s*Image containing object, first aid kit, board",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*Imagen que contiene objeto, botiquín de primeros auxilios[^.]*\.\s*",
        re.IGNORECASE,
    ),
)


def _strip_inline_ui_junk(text: str) -> str:
    """Remove embedded UI / vision-caption fragments (EN/ES) left after Office cleanup."""
    t = text
    for _ in range(8):
        prev = t
        for rx in _INLINE_UI_JUNK_RES:
            t = rx.sub(" ", t)
        if t == prev:
            break
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return t


_JUNK_LINE_PREFIXES = (
    "interfaz de usuario gráfica",
    "tabla el contenido",
    "imagen que contiene",
    "un conjunto de letras negras",
    "dibujo de",
    "diagrama de",
    "graphical user interface",
    "a set of black letters",
    "image containing",
)


def _is_junk_ui_line(st: str) -> bool:
    """Drop Gemini / Word UI-description lines that are not real prose."""
    if not st:
        return True
    low = st.lower()
    if "el contenido generado por ia puede ser incorrecto" in low and len(st) < 900:
        return True
    if "the ai-generated content may be incorrect" in low and len(st) < 900:
        return True
    if len(st) < 500:
        for p in _JUNK_LINE_PREFIXES:
            if low.startswith(p):
                return True
    return False


def _filter_junk_lines(text: str) -> str:
    lines: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if _is_junk_ui_line(s):
            continue
        lines.append(ln.rstrip())
    return "\n".join(lines).strip()


def _clean_chunk_text(text: str) -> str:
    """Strip pymupdf markers, corpus image markdown, Office dimension junk, and UI noise."""
    text = _PICTURE_MARKER_RE.sub("", text)
    text = _IMAGE_MD_RE.sub("", text)
    text = _OFFICE_DIM_DQ_RE.sub("", text)
    text = _OFFICE_DIM_SQ_RE.sub("", text)
    text = _OFFICE_DIM_ORPHAN_RE.sub("", text)
    text = _GEMINI_IMG_DISCLAIMER_ES_RE.sub("", text)
    text = _GEMINI_IMG_DISCLAIMER_EN_RE.sub("", text)
    text = _strip_inline_ui_junk(text)
    text = _filter_junk_lines(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _snippet_plain(text: str, max_len: int = 380) -> str:
    """Plain snippet for source cards (no markdown noise)."""
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


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _collect_image_paths(synth_raw: str | None, chunks: list[dict]) -> list[str]:
    """Paths like ``/corpus/{doc_id}/media/{file}`` from synthesis + chunk markdown."""
    parts: list[str] = []
    for blob in [synth_raw or ""] + [str(c.get("text") or "") for c in chunks]:
        parts.extend(_IMAGE_PATH_RE.findall(blob))
        parts.extend(_HTML_CORPUS_IMG_RE.findall(blob))
    return _dedupe_paths(parts)


def _abs_corpus_url(path: str) -> str:
    p = (path or "").strip()
    if p.startswith("http://") or p.startswith("https://"):
        return p
    if p.startswith("/corpus/"):
        return f"{API_URL.rstrip('/')}{p}"
    if p.startswith("corpus/"):
        return f"{API_URL.rstrip('/')}/{p}"
    return f"{API_URL.rstrip('/')}/{p.lstrip('/')}"


def _mime_from_bytes(b: bytes) -> str:
    if len(b) >= 3 and b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(b) >= 6 and b[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _fetch_corpus_image_bytes(abs_url: str, timeout: float = 45.0) -> bytes | None:
    """Load image via API (avoids browser CORS / wrong host for ``/corpus/...``)."""
    try:
        r = httpx.get(abs_url, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def _render_chat_image_gallery(paths: list[str], widget_key: str) -> None:
    """Small thumbs in the answer; full size only inside the popover."""
    if not paths:
        return
    inject_global_styles()
    abs_urls = [_abs_corpus_url(p) for p in paths]
    n = len(abs_urls)
    first = _fetch_corpus_image_bytes(abs_urls[0]) if abs_urls else None

    st.markdown(
        f'<div class="vk-sources-heading">{html_module.escape(t("chat_images_heading"))}</div>',
        unsafe_allow_html=True,
    )

    if n == 1:
        if not first:
            st.caption(t("chat_image_load_failed"))
            return
        c_thumb, c_pop = st.columns([6, 1], gap="small")
        with c_thumb:
            st.image(io.BytesIO(first), width=128)
        with c_pop:
            with st.popover(
                t("chat_image_view_full"),
                icon=":material/open_in_full:",
                width=920,
                key=f"gal_pop_{widget_key}_one",
            ):
                st.image(io.BytesIO(first), use_container_width=True)
        return

    c_prev, c_pop = st.columns([6, 1], gap="small")
    with c_prev:
        if first:
            mime = _mime_from_bytes(first)
            b64 = base64.standard_b64encode(first).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"
            badge = html_module.escape(f"+{n - 1}")
            st.markdown(
                f'<div class="vk-chat-img-preview-wrap vk-chat-img-preview--many">'
                f'<img src="{data_url}" alt="" />'
                f'<span class="vk-chat-img-badge">{badge}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(t("chat_image_load_failed"))

    pop_label = t("chat_images_view_all").format(n=n)
    with c_pop:
        with st.popover(
            pop_label,
            icon=":material/photo_library:",
            width=920,
            key=f"gal_pop_{widget_key}",
        ):
            for i, u in enumerate(abs_urls):
                data = _fetch_corpus_image_bytes(u)
                if data:
                    st.image(
                        io.BytesIO(data),
                        caption=f"{i + 1}/{n}",
                        use_container_width=True,
                    )
                else:
                    st.caption(f"{t('chat_image_load_failed')} — `{u}`")


def _relevance_badge_html(pct: int, *, compact: bool = False) -> str:
    """Rounded badge: Relevance:N with green / yellow / red tier."""
    inject_global_styles()
    if pct >= 82:
        cls = "vk-rel-badge vk-rel-badge--good"
    elif pct >= 65:
        cls = "vk-rel-badge vk-rel-badge--ok"
    else:
        cls = "vk-rel-badge vk-rel-badge--bad"
    if compact:
        cls += " vk-rel-badge--compact"
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
            f'<div class="vk-source-doc-line vk-source-doc-line--compact">'
            f'<span class="material-symbols-outlined vk-source-doc-line__ic">description</span>'
            f'<span class="vk-source-doc-title">{safe_doc}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if page is not None:
            st.caption(f"p.{page}")
        st.markdown(_relevance_badge_html(pct, compact=True), unsafe_allow_html=True)
        snip = html_module.escape(_snippet_plain(ch.get("text") or "", max_len=280))
        st.markdown(f'<p class="vk-snippet-compact">{snip}</p>', unsafe_allow_html=True)
        with st.popover(
            t("show_more"),
            icon=":material/unfold_more:",
            width="stretch",
            key=f"src_{key_base}_{idx}",
        ):
            st.markdown(_clean_chunk_text(ch.get("text") or ""))
            render_citation_row(ch, t("image_badge"))
            pct_in, _, _ = _relevance_pct(score, top_score)
            st.markdown(_relevance_badge_html(pct_in, compact=True), unsafe_allow_html=True)


def _render_secondary_sources_row(rest: list[dict], top_score: float, query_id: str) -> None:
    """Up to 6 source cards per row; each column uses full row share (readable width)."""
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


def render_response(resp: dict, *, gallery_key_suffix: str = "") -> None:
    """Clean layout: answer → grounding line → images → one Sources expander."""
    if resp.get("cache_hit"):
        st.caption(f":violet[:material/bolt:] {t('cache_hit')}")

    # Intent planner asked for clarification instead of answering.
    if resp.get("message_type") == "clarification_question" or resp.get("needs_clarification"):
        question = _clean_chunk_text(resp.get("synthesized_answer") or "")
        st.info(f"**{_lbl('clarification')}**\n\n{question}", icon=":material/help:")
        return

    chunks = resp.get("answer_chunks") or []
    synth_raw = resp.get("synthesized_answer") or ""
    image_paths = _collect_image_paths(synth_raw, chunks)
    qkey = _widget_key_safe(str(resp.get("query_id") or "unknown") + str(gallery_key_suffix))

    # Primary answer: synthesized text if present, else the top chunk.
    answer_text = _clean_chunk_text(synth_raw) if synth_raw.strip() else (
        _clean_chunk_text(chunks[0]["text"]) if chunks else ""
    )
    if answer_text.strip():
        st.markdown(answer_text)
    elif not chunks:
        st.caption(t("no_results"))

    _render_answer_meta(resp)
    _render_chat_image_gallery(image_paths, qkey)
    _render_sources_expander(chunks, str(resp.get("query_id") or "unknown"))

    if resp.get("related_docs"):
        with st.expander(t("see_also")):
            for rd in resp["related_docs"]:
                st.markdown(f"- **{rd['doc']}** — {rd['relevance']}")

    if chunks or synth_raw.strip():
        render_feedback(resp.get("query_id", ""))


def render_feedback(query_id: str):
    feedback_key = f"feedback_{query_id}"

    if feedback_key in st.session_state and st.session_state[feedback_key] == "sent":
        st.success(t("feedback_thanks"), icon=":material/check_circle:")
        return

    cols = st.columns([1, 1, 8])
    with cols[0]:
        if st.button("", key=f"ok_{query_id}", icon=":material/thumb_up:", help=t("btn_useful")):
            st.session_state[f"show_feedback_form_{query_id}"] = True
            st.session_state[f"feedback_rating_{query_id}"] = "ok"
    with cols[1]:
        if st.button("", key=f"nok_{query_id}", icon=":material/thumb_down:", help=t("btn_not_useful")):
            st.session_state[f"show_feedback_form_{query_id}"] = True
            st.session_state[f"feedback_rating_{query_id}"] = "not_ok"

    if st.session_state.get(f"show_feedback_form_{query_id}"):
        star = st.select_slider(
            t("feedback_stars"),
            options=[0, 1, 2, 3, 4, 5],
            value=5,
            format_func=lambda x: "★" * int(x) + ("☆" * (5 - int(x))),
            key=f"stars_{query_id}",
        )
        reason = None
        rating = st.session_state.get(f"feedback_rating_{query_id}", "not_ok")
        if rating == "not_ok":
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
            _send_feedback(query_id, rating, int(star), reason, correction or None)
            st.session_state[feedback_key] = "sent"
            st.session_state[f"show_feedback_form_{query_id}"] = False
            st.rerun()


def _send_feedback(
    query_id: str,
    rating: str,
    stars: int | None = None,
    reason: str = None,
    correction: str = None,
):
    try:
        payload = {
            "query_id": query_id,
            "rating": rating,
            "tenant_id": st.session_state.get("tenant_id", "demo"),
        }
        if stars is not None:
            payload["stars"] = int(stars)
        if reason:
            payload["reason"] = reason
        if correction:
            payload["correction"] = correction
        httpx.post(f"{API_URL}/feedback", json=payload, timeout=5)
    except Exception:
        pass


# ── Multi-chat session state ─────────────────────────────────

def _new_chat() -> str:
    """Create a fresh conversation and make it active. Returns its id."""
    n = st.session_state.get("_chat_counter", 0)
    st.session_state["_chat_counter"] = n + 1
    cid = f"chat-{n}"
    st.session_state.conversations[cid] = {"title": _lbl("chat_default_title"), "messages": []}
    st.session_state.active_chat = cid
    return cid


def _init_chats() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
        st.session_state.active_chat = None
    if not st.session_state.conversations or st.session_state.active_chat not in st.session_state.conversations:
        _new_chat()


def _active_conv() -> dict:
    _init_chats()
    return st.session_state.conversations[st.session_state.active_chat]


def _chat_switcher() -> None:
    """Conversation controls at the top of the Chat page (dropdown + new + delete).

    Kept out of the sidebar so it doesn't compete with the app's page navigation.
    """
    _init_chats()
    conv_ids = list(st.session_state.conversations.keys())
    titles = {
        cid: (st.session_state.conversations[cid]["title"] or _lbl("chat_default_title"))
        for cid in conv_ids
    }
    c_sel, c_new, c_del = st.columns([6, 2, 2], vertical_alignment="bottom")
    with c_sel:
        active = st.session_state.active_chat
        chosen = st.selectbox(
            _lbl("your_chats"),
            conv_ids,
            index=conv_ids.index(active) if active in conv_ids else 0,
            format_func=lambda c: titles.get(c, c),
            label_visibility="collapsed",
        )
        if chosen != active:
            st.session_state.active_chat = chosen
            st.rerun()
    with c_new:
        if st.button(_lbl("new_chat"), icon=":material/add:", use_container_width=True, type="primary"):
            _new_chat()
            st.rerun()
    with c_del:
        if st.button(
            _lbl("delete_chat"),
            icon=":material/delete:",
            use_container_width=True,
            disabled=len(conv_ids) <= 1,
        ):
            st.session_state.conversations.pop(st.session_state.active_chat, None)
            st.session_state.active_chat = next(iter(st.session_state.conversations))
            st.rerun()


def _prior_turns(messages: list[dict], limit: int = 3) -> list[dict]:
    """Completed user→assistant pairs (for multi-turn context / clarification)."""
    turns: list[dict] = []
    pending_q: str | None = None
    for m in messages:
        if m["role"] == "user":
            pending_q = m.get("content") or ""
        elif m["role"] == "assistant" and pending_q is not None:
            resp = m.get("response") or {}
            ans = (resp.get("synthesized_answer") or "").strip()
            if not ans:
                chs = resp.get("answer_chunks") or []
                ans = (chs[0].get("text") or "")[:500] if chs else ""
            turns.append({"question": pending_q, "answer": ans[:1000]})
            pending_q = None
    return turns[-limit:]


# ── Page layout ──────────────────────────────────────────────

page_heading(t("chat_title"), "chat")
st.caption(t("page_desc_chat"))

# Conversation controls live here (top of the page), not in the sidebar.
_chat_switcher()

with st.sidebar:
    st.session_state.setdefault("tenant_id", "demo")
    st.session_state["tenant_id"] = st.text_input(
        t("tenant_label"), value=st.session_state["tenant_id"]
    )


def _chat_body():
    """Scrollable history + footer toggle; ``st.chat_input`` is last so it stays page-bottom."""
    _ensure_ui_prefs_loaded()
    st.session_state.setdefault("chat_generate_toggle", True)

    conv = _active_conv()
    messages = conv["messages"]

    pending = st.session_state.pop("_pending_question", None)
    # Frozen at send time (same run as ``st.toggle``, before ``st.chat_input``) — avoids reading
    # generate intent on the *pending* rerun before the toggle widget executes.
    pending_generate = st.session_state.pop("_pending_generate", None)

    # Adaptive height: small on a fresh chat, taller once there's history.
    has_history = bool(messages) or pending is not None
    chat_box_height = 540 if has_history else 150

    with st.container(height=chat_box_height):
        if not has_history:
            st.caption(_lbl("empty_hint"))
        for mi, msg in enumerate(messages):
            with st.chat_message(
                msg["role"],
                avatar=":material/person:" if msg["role"] == "user" else ":material/psychology:",
            ):
                if msg["role"] == "assistant" and "response" in msg:
                    render_response(
                        msg["response"],
                        gallery_key_suffix=f"_h{mi}",
                    )
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
                        # Prior turns (excluding the just-sent message) give the
                        # backend multi-turn / clarification context.
                        "prior_turns": _prior_turns(messages[:-1]),
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
                    _gal_suf = f"_p{len(messages)}"
                    if not resp.get("answer_chunks"):
                        banner(t("no_results"), variant="warn", icon_name="search_off")
                    else:
                        if gen and not (resp.get("synthesized_answer") or "").strip():
                            banner(t("generation_fallback_warning"), variant="warn", icon_name="warning")
                        render_response(resp, gallery_key_suffix=_gal_suf)
                    messages.append({
                        "role": "assistant",
                        "content": "...",
                        "response": resp,
                    })
                except httpx.ConnectError:
                    banner(t("api_error"), variant="error", icon_name="error")
                except httpx.HTTPStatusError as e:
                    banner(f"{t('error')}: API returned {e.response.status_code}", variant="error", icon_name="error")
                except Exception as e:
                    banner(f"{t('error')}: {e}", variant="error", icon_name="error")

    _tog_sp, _tog_col = st.columns([5, 1])
    with _tog_col:
        st.toggle(
            t("mode_generate_short"),
            key="chat_generate_toggle",
        )

    if prompt := st.chat_input(t("input_placeholder")):
        messages.append({"role": "user", "content": prompt})
        # Name the conversation after its first question.
        if conv["title"] == _lbl("chat_default_title") or not conv["title"]:
            conv["title"] = (prompt[:38] + "…") if len(prompt) > 38 else prompt
        st.session_state["_pending_question"] = prompt
        st.session_state["_pending_generate"] = bool(
            st.session_state.get("chat_generate_toggle", True)
        )
        st.rerun()


_chat_body()
