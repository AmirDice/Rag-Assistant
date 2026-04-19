"""Operator onboarding — tenant overlay via API (data/tenant_onboarding.json)."""

import os

import httpx
import streamlit as st
from i18n import t
from progress_helpers import run_with_progress
from ui_style import page_heading

API_URL = os.getenv("API_URL", "http://localhost:8000")

page_heading(t("onboarding_title"), "domain")
st.caption(t("page_desc_onboarding"))


def _onboarding_body():
    tenant_id = st.text_input(t("tenant_label"), value="demo")
    admin_token = st.text_input(t("onboarding_admin_token"), type="password", help=t("onboarding_admin_help"))

    if st.button(t("onboarding_load"), icon=":material/refresh:"):
        tid = tenant_id

        def _load():
            r = httpx.get(f"{API_URL}/tenant/{tid}/profile", timeout=30)
            r.raise_for_status()
            return r.json()

        try:
            st.session_state["tenant_profile"] = run_with_progress(t("page_loading"), _load)
        except Exception as e:
            st.error(f"{t('error')}: {e}")

    prof = st.session_state.get("tenant_profile")
    if prof:
        st.subheader(t("onboarding_merged"))
        st.json(prof.get("merged", {}))

    st.divider()
    st.subheader(t("onboarding_update"))

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
                st.error(t("onboarding_erp_invalid"))
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
            st.success(t("onboarding_saved"))
            st.session_state["tenant_profile"] = result
            st.json(result)
        except Exception as e:
            st.error(f"{t('error')}: {e}")
    elif submitted and not admin_token:
        st.warning(t("onboarding_need_token"))


_onboarding_body()
