"""Settings page — tenant config, model selection, cache (WP16 §16.1)."""

import os
from pathlib import Path

import httpx
import streamlit as st
from i18n import t
from progress_helpers import run_with_progress
from ui_style import banner, page_heading, section_header, status_cards

API_URL = os.getenv("API_URL", "http://localhost:8000")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _REPO_ROOT / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return out
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        k = key.strip()
        if not k:
            continue
        v = value.strip()
        if " #" in v:
            v = v.split(" #", 1)[0].strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out[k] = v
    return out


def _admin_headers(env_file: dict[str, str]) -> dict[str, str]:
    token = (os.getenv("ADMIN_TOKEN", "") or env_file.get("ADMIN_TOKEN", "")).strip()
    return {"X-Admin-Token": token} if token else {}

page_heading(t("settings_title"), "settings")
st.caption(t("page_desc_settings"))


def _settings_body():
    env_file = _read_env_file(_DOTENV_PATH)
    admin_headers = _admin_headers(env_file)

    section_header(t("active_config"), "monitoring")

    try:

        def _health():
            return httpx.get(f"{API_URL}/health", timeout=5).json()

        health = run_with_progress(t("page_loading"), _health)
        status_cards(
            [
                {
                    "label": "API",
                    "state": "on" if str(health.get("api", "")).lower() == "ok" else "off",
                    "state_text": t("status_ok") if str(health.get("api", "")).lower() == "ok" else t("status_error"),
                },
                {
                    "label": "Qdrant",
                    "state": "on" if str(health.get("qdrant", "")).lower() == "ok" else "off",
                    "state_text": t("status_ok") if str(health.get("qdrant", "")).lower() == "ok" else t("status_error"),
                },
                {
                    "label": "Redis",
                    "state": "on" if str(health.get("redis", "")).lower() == "ok" else "off",
                    "state_text": t("status_ok") if str(health.get("redis", "")).lower() == "ok" else t("status_error"),
                },
            ]
        )
    except Exception:
        banner(t("api_unavailable"), variant="warn", icon_name="warning")

    st.divider()
    section_header(t("tenant_config"), "domain")

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
    section_header(t("models_title"), "model_training")

    st.markdown("""
| Component | Cloud | Local |
|-----------|-------|-------|
| **Embedding** | Voyage-4-large | Qwen3-Embedding-8B |
| **Reranker** | Cohere rerank-v4.0-pro | BGE-reranker-v2-m3 |
| **Vision** | Gemini 2.5 Flash | — |
| **Generation** | Gemini 2.5 Flash / Lite, GPT-4, GPT-4o mini | — |
""")

    banner(t("models_help"), variant="info", icon_name="info")

    section_header(t("prefs_chat_models_title"), "tune")
    st.caption(t("prefs_chat_models_help"))

    def _load_model_prefs():
        opt = httpx.get(f"{API_URL}/config/model-options", timeout=8, headers=admin_headers)
        opt.raise_for_status()
        pref = httpx.get(f"{API_URL}/config/ui-preferences", timeout=8, headers=admin_headers)
        pref.raise_for_status()
        return opt.json(), pref.json()

    try:
        model_opts, prefs = run_with_progress(t("page_loading"), _load_model_prefs)
    except Exception:
        banner(t("prefs_load_error"), variant="warn", icon_name="warning")
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
                    headers=admin_headers,
                    timeout=8,
                )
                r.raise_for_status()
                st.session_state["ui_reranker"] = put_body["reranker"]
                st.session_state["ui_generation_model"] = put_body["generation_model"]
                st.session_state["_ui_prefs_loaded"] = True
                banner(t("prefs_saved"), variant="ok", icon_name="check_circle")
            except Exception as e:
                banner(f"{t('error')}: {e}", variant="error", icon_name="error")

    st.divider()
    section_header(t("api_keys_title"), "vpn_key")

    keys = {
        "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY", "") or env_file.get("VOYAGE_API_KEY", ""),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY", "") or env_file.get("COHERE_API_KEY", ""),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", "") or env_file.get("GOOGLE_API_KEY", ""),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "") or env_file.get("OPENAI_API_KEY", ""),
    }
    status_cards(
        [
            {
                "label": name,
                "state": "on" if bool(value) else "off",
                "state_text": t("key_set") if bool(value) else t("key_missing"),
            }
            for name, value in keys.items()
        ]
    )

_settings_body()
