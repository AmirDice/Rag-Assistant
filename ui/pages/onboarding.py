"""Operator onboarding — tenant overlay via API (data/tenant_onboarding.json)."""

import os

import httpx
import streamlit as st
from i18n import t
from progress_helpers import run_with_progress
from ui_style import banner, page_heading, section_header, status_cards

API_URL = os.getenv("API_URL", "http://localhost:8000")

page_heading(t("onboarding_title"), "domain")
st.caption(t("page_desc_onboarding"))


def _onboarding_body():
    tenant_id = st.text_input(t("tenant_label"), value="demo")
    admin_token = st.text_input(t("onboarding_admin_token"), type="password", help=t("onboarding_admin_help"))
    prof = st.session_state.get("tenant_profile")

    status_cards(
        [
            {
                "label": t("onboarding_admin_token"),
                "state": "on" if bool(admin_token.strip()) else "off",
                "state_text": t("status_on") if bool(admin_token.strip()) else t("status_off"),
            },
            {
                "label": t("tenant_label"),
                "state": "on" if bool(tenant_id.strip()) else "off",
                "state_text": t("status_on") if bool(tenant_id.strip()) else t("status_off"),
                "hint": tenant_id.strip() or "—",
            },
            {
                "label": t("onboarding_merged"),
                "state": "on" if bool(prof) else "neutral",
                "state_text": t("status_on") if bool(prof) else t("status_idle"),
            },
        ]
    )

    if st.button(t("onboarding_load"), icon=":material/refresh:"):
        tid = tenant_id

        def _load():
            r = httpx.get(f"{API_URL}/tenant/{tid}/profile", timeout=30)
            r.raise_for_status()
            return r.json()

        try:
            st.session_state["tenant_profile"] = run_with_progress(t("page_loading"), _load)
        except Exception as e:
            banner(f"{t('error')}: {e}", variant="error", icon_name="error")

    if prof:
        section_header(t("onboarding_merged"), "account_tree")
        st.json(prof.get("merged", {}))

    st.divider()
    section_header(t("onboarding_update"), "edit")

    with st.form("onboarding_form"):
        erp_s = st.text_input("erp_version (optional)", placeholder="5.0")
        mods = st.text_input(t("onboarding_modules"), placeholder="rowa, dispensador")
        robot = st.checkbox("has_robot_integration", value=False)
        pref = st.selectbox("preferred_lang", ["es", "ca"], index=0)
        bench = st.selectbox("benchmark_lang", ["es", "ca"], index=0)
        submitted = st.form_submit_button(t("onboarding_save"))

    if submitted and admin_token:
        modules = [m.strip() for m in mods.split(",") if m.strip()] if mods else None
        payload = {}
        if erp_s.strip():
            try:
                payload["erp_version"] = float(erp_s.replace(",", "."))
            except ValueError:
                banner(t("onboarding_erp_invalid"), variant="error", icon_name="error")
                st.stop()
        if modules is not None:
            payload["contracted_modules"] = modules
        payload["has_robot_integration"] = robot
        payload["preferred_lang"] = pref
        payload["benchmark_lang"] = bench
        tid = tenant_id
        tok = admin_token

        def _save():
            r = httpx.put(
                f"{API_URL}/tenant/{tid}/onboarding",
                json=payload,
                headers={"X-Admin-Token": tok},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()

        try:
            result = run_with_progress(t("page_loading"), _save)
            banner(t("onboarding_saved"), variant="ok", icon_name="check_circle")
            st.session_state["tenant_profile"] = result
            st.json(result)
        except Exception as e:
            banner(f"{t('error')}: {e}", variant="error", icon_name="error")
    elif submitted and not admin_token:
        banner(t("onboarding_need_token"), variant="warn", icon_name="warning")


_onboarding_body()
