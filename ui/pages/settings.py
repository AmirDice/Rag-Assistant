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
| **Generation** | Gemini 2.5 Flash / Lite, GPT-4, GPT-4o mini | — |
""")

    st.info(t("models_help"))

    st.subheader(t("prefs_chat_models_title"))
    st.caption(t("prefs_chat_models_help"))

    def _load_model_prefs():
        opt = httpx.get(f"{API_URL}/config/model-options", timeout=8)
        opt.raise_for_status()
        pref = httpx.get(f"{API_URL}/config/ui-preferences", timeout=8)
        pref.raise_for_status()
        return opt.json(), pref.json()

    try:
        model_opts, prefs = run_with_progress(t("page_loading"), _load_model_prefs)
    except Exception:
        st.warning(t("prefs_load_error"))
        model_opts, prefs = None, None

    if model_opts is not None and prefs is not None:
        rer_ids = [o["id"] for o in model_opts.get("reranker_options", [])]
        gen_opts = model_opts.get("generation_options", [])
        gen_ids = [o["id"] for o in gen_opts]
        gen_labels = {o["id"]: o.get("label", o["id"]) for o in gen_opts}
        yaml_rer = str(model_opts.get("reranker_yaml_active", ""))
        yaml_gen = str(model_opts.get("generation_yaml_default", ""))

        def _fmt_rer(x: str) -> str:
            if not x:
                return t("prefs_use_yaml_default") + (f" ({yaml_rer})" if yaml_rer else "")
            return x

        def _fmt_gen(x: str) -> str:
            if not x:
                return t("prefs_use_yaml_default") + (f" ({yaml_gen})" if yaml_gen else "")
            return gen_labels.get(x, x)

        cur_rer = (prefs.get("reranker") or "").strip().lower()
        cur_gen = (prefs.get("generation_model") or "").strip()

        rer_choices: list[str] = [""] + [r for r in rer_ids if r]
        if cur_rer and cur_rer not in rer_choices:
            rer_choices.append(cur_rer)
        idx_rer = rer_choices.index(cur_rer) if cur_rer in rer_choices else 0
        sel_rer = st.selectbox(
            t("prefs_reranker"),
            options=rer_choices,
            index=idx_rer,
            format_func=_fmt_rer,
            key="settings_pref_reranker",
        )

        gen_choices: list[str] = [""] + [g for g in gen_ids if g]
        if cur_gen and cur_gen not in gen_choices:
            gen_choices.append(cur_gen)
        idx_gen = gen_choices.index(cur_gen) if cur_gen in gen_choices else 0
        sel_gen = st.selectbox(
            t("prefs_generation_model"),
            options=gen_choices,
            index=idx_gen,
            format_func=_fmt_gen,
            key="settings_pref_generation",
        )

        if st.button(t("prefs_save"), type="primary", key="settings_pref_save"):
            put_body = {
                "reranker": sel_rer or None,
                "generation_model": sel_gen or None,
            }
            try:
                r = httpx.put(
                    f"{API_URL}/config/ui-preferences",
                    json=put_body,
                    timeout=8,
                )
                r.raise_for_status()
                st.session_state["ui_reranker"] = put_body["reranker"]
                st.session_state["ui_generation_model"] = put_body["generation_model"]
                st.session_state["_ui_prefs_loaded"] = True
                st.success(t("prefs_saved"))
            except Exception as e:
                st.error(f"{t('error')}: {e}")

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
