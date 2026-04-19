"""Settings page — tenant config, model selection, cache (WP16 §16.1)."""

import os

import httpx
import streamlit as st
from i18n import t
from progress_helpers import run_with_progress
from ui_style import page_heading, render_api_key_row

API_URL = os.getenv("API_URL", "http://localhost:8000")

page_heading(t("settings_title"), "settings")
st.caption(t("page_desc_settings"))


def _settings_body():
    st.subheader(t("active_config"))

    try:

        def _health():
            return httpx.get(f"{API_URL}/health", timeout=5).json()

        health = run_with_progress(t("page_loading"), _health)
        st.json(health)
    except Exception:
        st.info(t("api_unavailable"))

    st.divider()
    st.subheader(t("tenant_config"))

    st.markdown(t("tenant_help"))
    st.caption(t("product_config_hint"))

    st.code("""
tenants:
  acme_corp:
    name: "Acme Corp"
    erp_version: 5.0
    contracted_modules:
      - "billing"
      - "inventory"
    has_robot_integration: false
    preferred_lang: "es"
""", language="yaml")

    st.divider()
    st.subheader(t("models_title"))

    st.markdown("""
| Component | Cloud | Local |
|-----------|-------|-------|
| **Embedding** | Voyage-4-large | Qwen3-Embedding-8B |
| **Reranker** | Cohere rerank-v4.0-pro | BGE-reranker-v2-m3 |
| **Vision** | Gemini 2.5 Flash | — |
| **Generation** | Gemini 2.5 Flash | — |
""")

    st.info(t("models_help"))

    st.divider()
    st.subheader(t("api_keys_title"))

    keys = {
        "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY", ""),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY", ""),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    }

    for name, value in keys.items():
        render_api_key_row(name, bool(value), t("key_set"), t("key_missing"))


_settings_body()
