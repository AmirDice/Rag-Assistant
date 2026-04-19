"""Global UI: Inter, Material Symbols, light/dark theme, minimal layout."""

from __future__ import annotations

import html as html_module
import re
from pathlib import Path

import streamlit as st


def clean_source_display_name(raw: str) -> str:
    """Strip upload/hash prefixes; keep the human-readable file name only."""
    if not raw or not str(raw).strip():
        return "—"
    base = Path(str(raw).strip()).name
    # Ingest upload pattern: 12-char hex + underscore + original filename
    m = re.match(r"^([a-f0-9]{12})_(.+)$", base, re.I)
    if m:
        base = m.group(2)
    # Numeric stem noise often left before the real title (e.g. 12__1_File.docx)
    base = re.sub(r"^\d+__\d+_+", "", base)
    base = re.sub(r"^\d+__\d+__+", "", base)
    return base.strip() or Path(str(raw).strip()).name

_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parent
FAVICON_PATH = str(_ROOT / "assets" / "favicon.svg")
# Optional wordmark image — do not fall back to legacy logo.png (avoids showing an old asset).
_LOGO_CANDIDATES = (
    _REPO_ROOT / "wordmark.png",
    _ROOT / "assets" / "wordmark.png",
)
LOGO_PATH: str | None = None
for _p in _LOGO_CANDIDATES:
    if _p.is_file():
        LOGO_PATH = str(_p.resolve())
        break


def render_brand_logo(*, width: int = 200) -> None:
    """Sidebar or header wordmark image."""
    if LOGO_PATH:
        st.image(LOGO_PATH, width=width)

_FONTS_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet"/>
"""


def _root_vars(theme: str) -> str:
    if theme == "dark":
        return """
          :root {
            --vk-app-bg: #0f172a;
            --vk-main-bg: #1e293b;
            --vk-text: #f1f5f9;
            --vk-text-muted: #94a3b8;
            --vk-border: #334155;
            --vk-sidebar-bg: #1e293b;
            --vk-input-bg: #334155;
            --vk-code-bg: #0f172a;
            --vk-accent: #38bdf8;
            --vk-ok: #2dd4bf;
            --vk-bad: #f87171;
            --vk-warn: #fbbf24;
            --vk-doc-title: #f8fafc;
          }
        """
    return """
      :root {
        --vk-app-bg: #f8fafc;
        --vk-main-bg: #ffffff;
        --vk-text: #0f172a;
        --vk-doc-title: #0a0a0a;
        --vk-text-muted: #64748b;
        --vk-border: #e2e8f0;
        --vk-sidebar-bg: #f8fafc;
        --vk-input-bg: #ffffff;
        --vk-code-bg: #f1f5f9;
        --vk-accent: #334155;
        --vk-ok: #0d9488;
        --vk-bad: #b91c1c;
        --vk-warn: #ca8a04;
      }
    """


def _layout_css() -> str:
    return """
          html, body, [class*="stApp"] {
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
          }
          .stApp, [data-testid="stAppViewContainer"] {
            background: var(--vk-app-bg) !important;
            color: var(--vk-text) !important;
          }
          [data-testid="stHeader"] {
            background: transparent !important;
            border-bottom: 1px solid var(--vk-border) !important;
          }
          section.main [data-testid="stVerticalBlock"] {
            color: var(--vk-text);
          }
          h1, h2, h3 {
            letter-spacing: -0.02em;
            font-weight: 600 !important;
            color: var(--vk-text) !important;
          }
          .stMarkdown, .stMarkdown p, label, [data-testid="stWidgetLabel"] p {
            color: var(--vk-text) !important;
          }
          [data-testid="stCaption"] {
            color: var(--vk-text-muted) !important;
          }
          [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: var(--vk-text) !important;
          }
          [data-testid="stSidebar"] {
            background: var(--vk-sidebar-bg) !important;
            border-right: 1px solid var(--vk-border) !important;
          }
          [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
            color: var(--vk-text) !important;
          }
          [data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
          [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
            background-color: var(--vk-input-bg) !important;
            color: var(--vk-text) !important;
            border-color: var(--vk-border) !important;
          }
          [data-baseweb="select"] > div {
            background-color: var(--vk-input-bg) !important;
            color: var(--vk-text) !important;
          }
          .stCodeBlock, pre code, [data-testid="stCodeBlock"] {
            background-color: var(--vk-code-bg) !important;
            color: var(--vk-text) !important;
          }
          [data-testid="stExpander"] {
            background-color: var(--vk-main-bg) !important;
            border: 1px solid var(--vk-border) !important;
            border-radius: 8px !important;
          }
          [data-testid="stDataFrame"] {
            border: 1px solid var(--vk-border) !important;
            border-radius: 8px !important;
          }
          .stButton > button[kind="primary"] {
            background-color: var(--vk-accent) !important;
            color: #ffffff !important;
            border: none !important;
          }
          .material-symbols-outlined {
            font-family: "Material Symbols Outlined" !important;
            font-weight: normal !important;
            font-style: normal !important;
            font-size: 1.5rem !important;
            line-height: 1 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            display: inline-block !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24;
          }
          .vk-page-head {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 0 0 0.25rem 0;
            padding: 0;
          }
          .vk-page-head__icon {
            font-size: 1.85rem !important;
            color: var(--vk-text-muted) !important;
            font-weight: 300 !important;
          }
          .vk-page-head__title {
            font-size: 1.6rem;
            font-weight: 600;
            margin: 0;
            padding: 0;
            line-height: 1.2;
            color: var(--vk-text);
            border: none;
          }
          .vk-citation {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.35rem;
            font-size: 0.88rem;
            color: var(--vk-text-muted);
            margin-top: 0.35rem;
          }
          .vk-citation .material-symbols-outlined {
            font-size: 1.05rem !important;
            color: var(--vk-text-muted) !important;
          }
          .vk-relevance-note {
            font-size: 0.85rem;
            color: var(--vk-text-muted);
          }
          .vk-sources-heading {
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            color: var(--vk-text);
            margin: 0 0 0.35rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid var(--vk-border);
          }
          .vk-source-doc-line {
            display: flex;
            align-items: flex-start;
            gap: 0.45rem;
            margin: 0 0 0.15rem 0;
          }
          .vk-source-doc-line__ic {
            font-size: 1.28rem !important;
            color: var(--vk-text-muted) !important;
            flex-shrink: 0;
            margin-top: 0.12rem;
          }
          .vk-source-doc-title {
            font-size: 1.08rem;
            font-weight: 700;
            color: var(--vk-doc-title);
            line-height: 1.35;
            word-break: break-word;
          }
          /* Chat: secondary “Sources consulted” cards — slightly smaller than main chunk */
          .vk-source-doc-line--compact {
            gap: 0.35rem;
            margin: 0 0 0.1rem 0;
          }
          .vk-source-doc-line--compact .vk-source-doc-line__ic {
            font-size: 1.05rem !important;
            margin-top: 0.08rem;
          }
          .vk-source-doc-line--compact .vk-source-doc-title {
            font-size: 0.92rem;
            font-weight: 600;
            line-height: 1.3;
            word-break: break-word;
            overflow-wrap: anywhere;
            hyphens: auto;
          }
          .vk-snippet-compact {
            font-size: 0.8rem;
            line-height: 1.42;
            color: var(--vk-text-muted);
            margin: 0.12rem 0 0.25rem 0;
            word-break: break-word;
            overflow-wrap: anywhere;
          }
          .vk-rel-badge--compact {
            padding: 0.28rem 0.52rem;
            font-size: 0.65rem;
            margin: 0.1rem 0 0.25rem 0;
            border-radius: 9px;
          }
          .vk-citation-docname {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--vk-doc-title);
          }
          .vk-rel-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            padding: 0.38rem 0.65rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            line-height: 1.15;
            margin: 0.2rem 0 0.45rem 0;
            width: fit-content;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.12);
          }
          .vk-rel-badge--good {
            background: linear-gradient(145deg, #0d9488, #2dd4bf);
            color: #ecfdf5;
          }
          .vk-rel-badge--ok {
            background: linear-gradient(145deg, #ca8a04, #facc15);
            color: #1c1917;
          }
          .vk-rel-badge--bad {
            background: linear-gradient(145deg, #b91c1c, #f87171);
            color: #fff7f7;
          }
          .vk-status-ok { color: var(--vk-ok); margin: 0.35rem 0; font-size: 0.95rem; }
          .vk-status-bad { color: var(--vk-bad); margin: 0.35rem 0; font-size: 0.95rem; }
          .vk-key-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.4rem 0;
            font-size: 0.95rem;
            color: var(--vk-text);
          }
          .vk-key-row .material-symbols-outlined {
            font-size: 1.2rem !important;
          }
          .vk-key-ok { color: var(--vk-ok); }
          .vk-key-miss { color: var(--vk-warn); }
          div[data-testid="stSpinner"] > div {
            border-color: var(--vk-accent) transparent transparent transparent !important;
          }
          /* st.chat_input: do not keep a fixed bar at the viewport bottom — scroll with content */
          div[data-testid="stBottom"],
          section[data-testid="stBottom"] {
            position: static !important;
            left: auto !important;
            right: auto !important;
            top: auto !important;
            bottom: auto !important;
            width: 100% !important;
            max-width: none !important;
            transform: none !important;
            z-index: auto !important;
            background: transparent !important;
            backdrop-filter: none !important;
            box-shadow: none !important;
            border-top: 1px solid var(--vk-border);
            padding: 0.5rem 0 0.25rem 0;
            margin-top: 0.5rem;
          }
          div[data-testid="stBottom"] > div,
          section[data-testid="stBottom"] > div {
            position: static !important;
          }
          div[data-testid="stChatFloatingInputContainer"] {
            position: static !important;
            bottom: auto !important;
            left: auto !important;
            right: auto !important;
            width: 100% !important;
            max-width: none !important;
            z-index: auto !important;
            background: transparent !important;
            box-shadow: none !important;
          }
          section[data-testid="stMain"] > div {
            padding-bottom: 0.5rem;
          }
          footer { visibility: hidden; height: 0; }
    """


def _chat_images_css() -> str:
    """Chat gallery: small in-result thumb; popover shows large crisp images."""
    return """
          .vk-chat-img-preview-wrap {
            position: relative;
            width: 128px;
            height: 96px;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--vk-border);
            background: var(--vk-code-bg);
          }
          .vk-chat-img-preview-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            filter: none;
            image-rendering: auto;
          }
          .vk-chat-img-preview-wrap.vk-chat-img-preview--many img {
            opacity: 0.88;
          }
          .vk-chat-img-preview-wrap .vk-chat-img-badge {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(15, 23, 42, 0.42);
            color: #f8fafc;
            font-weight: 700;
            font-size: 1.35rem;
            letter-spacing: 0.02em;
            pointer-events: none;
          }
          div[data-testid="stPopoverBody"]:has([data-testid="stImage"]) [data-testid="stImage"] img,
          div[data-testid="stPopoverBody"]:has([data-testid="stImage"]) [data-testid="stImage"] picture img {
            max-width: min(920px, 96vw) !important;
            max-height: min(78vh, 900px) !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
            filter: none !important;
            image-rendering: auto !important;
          }
    """


def _logo_css() -> str:
    """st.logo: preserve PNG transparency; 2× height vs Streamlit size=\"large\" (~32px → 64px)."""
    return """
          [data-testid="stSidebarHeader"],
          [data-testid="stSidebarHeader"] a,
          [data-testid="collapsedControl"],
          [data-testid="collapsedControl"] a,
          [data-testid="stLogo"],
          [data-testid="stLogo"] a {
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
          }
          [data-testid="stSidebarHeader"] img,
          [data-testid="collapsedControl"] img,
          [data-testid="stLogo"] img {
            height: 64px !important;
            max-height: 64px !important;
            width: auto !important;
            max-width: min(100%, 420px) !important;
            object-fit: contain !important;
            background: transparent !important;
            background-color: transparent !important;
          }
    """


def inject_global_styles() -> None:
    theme = st.session_state.get("ui_theme", "light")
    if theme not in ("light", "dark"):
        theme = "light"
    if not st.session_state.get("_vk_fonts_done"):
        st.markdown(_FONTS_HTML, unsafe_allow_html=True)
        st.session_state["_vk_fonts_done"] = True
    st.markdown(
        f"<style>{_root_vars(theme)}{_layout_css()}{_logo_css()}{_chat_images_css()}</style>",
        unsafe_allow_html=True,
    )


def page_heading(title: str, icon_name: str) -> None:
    inject_global_styles()
    safe_title = html_module.escape(title)
    safe_icon = html_module.escape(icon_name)
    st.markdown(
        f'<div class="vk-page-head">'
        f'<span class="material-symbols-outlined vk-page-head__icon">{safe_icon}</span>'
        f'<h1 class="vk-page-head__title">{safe_title}</h1>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_citation_row(chunk: dict, image_badge_label: str) -> None:
    inject_global_styles()
    parts: list[str] = []
    if chunk.get("source_doc"):
        doc = html_module.escape(clean_source_display_name(str(chunk["source_doc"])))
        parts.append(
            f'<span class="material-symbols-outlined">description</span> '
            f'<strong class="vk-citation-docname">{doc}</strong>'
        )
    if chunk.get("source_page"):
        parts.append(f"p. {html_module.escape(str(chunk['source_page']))}")
    if chunk.get("source_section"):
        parts.append(f"§ {html_module.escape(str(chunk['source_section']))}")
    line = " · ".join(parts)
    if chunk.get("has_image_caption"):
        badge = html_module.escape(image_badge_label)
        line += (
            f' <span class="material-symbols-outlined" style="font-size:1rem!important;vertical-align:middle;">'
            f'photo_camera</span> <em>{badge}</em>'
        )
    if not line.strip():
        return
    st.markdown(f'<div class="vk-citation">{line}</div>', unsafe_allow_html=True)


def render_health_row(service: str, status: str) -> None:
    inject_global_styles()
    ok = str(status).lower() == "ok"
    cls = "vk-status-ok" if ok else "vk-status-bad"
    sym = "check_circle" if ok else "cancel"
    s = html_module.escape(str(service))
    v = html_module.escape(str(status))
    st.markdown(
        f'<p class="{cls}">'
        f'<span class="material-symbols-outlined" style="font-size:1.1rem!important;vertical-align:text-bottom;">{sym}</span> '
        f"<strong>{s}</strong>: {v}</p>",
        unsafe_allow_html=True,
    )


def render_api_key_row(name: str, configured: bool, label_ok: str, label_miss: str) -> None:
    inject_global_styles()
    sym = "check_circle" if configured else "warning"
    cls = "vk-key-ok" if configured else "vk-key-miss"
    nm = html_module.escape(name)
    lb = html_module.escape(label_ok if configured else label_miss)
    st.markdown(
        f'<div class="vk-key-row {cls}">'
        f'<span class="material-symbols-outlined">{sym}</span>'
        f"<span><strong>{nm}</strong> — {lb}</span></div>",
        unsafe_allow_html=True,
    )
