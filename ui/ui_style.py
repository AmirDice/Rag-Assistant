"""Global UI: Inter, Material Symbols, light/dark theme, minimal layout."""

from __future__ import annotations

import html as html_module
from pathlib import Path

import streamlit as st

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
          }
        """
    return """
      :root {
        --vk-app-bg: #f8fafc;
        --vk-main-bg: #ffffff;
        --vk-text: #0f172a;
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
          footer { visibility: hidden; height: 0; }
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
        f"<style>{_root_vars(theme)}{_layout_css()}{_logo_css()}</style>",
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
        doc = html_module.escape(str(chunk["source_doc"]))
        parts.append(
            f'<span class="material-symbols-outlined">description</span> <strong>{doc}</strong>'
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
