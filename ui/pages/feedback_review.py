"""Admin review queue for chat feedback items."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import streamlit as st

from i18n import t
from progress_helpers import run_with_progress
from ui_style import banner, page_heading, section_header

API_URL = os.getenv("API_URL", "http://localhost:8000")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _REPO_ROOT / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return out
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _admin_headers() -> dict[str, str]:
    env = _read_env_file(_DOTENV_PATH)
    token = (os.getenv("ADMIN_TOKEN", "") or env.get("ADMIN_TOKEN", "")).strip()
    return {"X-Admin-Token": token} if token else {}


def _load_feedback(offset: int, limit: int, rating: str, status: str) -> dict:
    params: dict[str, str | int] = {"offset": offset, "limit": limit}
    if rating != "all":
        params["rating"] = rating
    if status != "all":
        params["status"] = status
    r = httpx.get(
        f"{API_URL}/feedback/list",
        params=params,
        headers=_admin_headers(),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _submit_review(feedback_id: str, action: str, notes: str | None) -> None:
    body = {"feedback_id": feedback_id, "action": action, "notes": notes or None}
    r = httpx.post(
        f"{API_URL}/feedback/review",
        json=body,
        headers=_admin_headers(),
        timeout=30,
    )
    r.raise_for_status()


page_heading(t("feedback_review_title"), "rate_review")
st.caption(t("page_desc_feedback_review"))

col_a, col_b, col_c = st.columns(3)
with col_a:
    offset = int(st.number_input(t("feedback_review_offset"), min_value=0, value=0, step=1))
with col_b:
    limit = int(st.number_input(t("feedback_review_limit"), min_value=1, max_value=100, value=10, step=1))
with col_c:
    rating = st.selectbox(
        t("feedback_review_rating"),
        options=["all", "ok", "not_ok"],
        index=0,
    )

status_filter = st.selectbox(
    t("feedback_review_status"),
    options=["all", "open", "acknowledged", "resolved", "dismissed"],
    index=0,
)

section_header(t("feedback_review_queue"), "inbox")
if st.button(t("feedback_review_load"), icon=":material/refresh:"):
    try:
        st.session_state["feedback_review_batch"] = run_with_progress(
            t("page_loading"),
            lambda: _load_feedback(offset, limit, rating, status_filter),
        )
    except Exception as e:
        banner(f"{t('error')}: {e}", variant="error", icon_name="error")

batch = st.session_state.get("feedback_review_batch")
if batch:
    st.caption(t("feedback_review_total").format(n=batch.get("total", 0)))
    items = batch.get("items", [])
    if not items:
        banner(t("feedback_review_empty"), variant="info", icon_name="info")
    for it in items:
        fid = it.get("feedback_id", "")
        title = f"{it.get('timestamp', '')[:19]} — {it.get('rating', '')} — {it.get('review_status', 'open')}"
        with st.expander(title):
            st.markdown(f"**query_id**: `{it.get('query_id', '')}`")
            st.markdown(f"**tenant_id**: `{it.get('tenant_id', '')}`")
            st.markdown(f"**stars**: `{it.get('stars', '—')}`")
            if it.get("reason"):
                st.markdown(f"**{t('feedback_reason')}**: {it.get('reason')}")
            if it.get("correction"):
                st.text_area(t("feedback_correction"), value=str(it.get("correction")), height=120, disabled=True, key=f"corr_{fid}")
            notes = st.text_input(t("review_notes"), value=str(it.get("review_notes") or ""), key=f"notes_{fid}")
            c1, c2, c3, c4 = st.columns(4)
            if c1.button(t("feedback_review_ack"), key=f"ack_{fid}"):
                try:
                    _submit_review(fid, "acknowledged", notes)
                    banner(t("review_stored"), variant="ok", icon_name="check_circle")
                except Exception as e:
                    banner(f"{t('error')}: {e}", variant="error", icon_name="error")
            if c2.button(t("feedback_review_resolve"), key=f"res_{fid}"):
                try:
                    _submit_review(fid, "resolved", notes)
                    banner(t("review_stored"), variant="ok", icon_name="check_circle")
                except Exception as e:
                    banner(f"{t('error')}: {e}", variant="error", icon_name="error")
            if c3.button(t("feedback_review_reopen"), key=f"reopen_{fid}"):
                try:
                    _submit_review(fid, "open", notes)
                    banner(t("review_stored"), variant="ok", icon_name="check_circle")
                except Exception as e:
                    banner(f"{t('error')}: {e}", variant="error", icon_name="error")
            if c4.button(t("feedback_review_dismiss"), key=f"dismiss_{fid}"):
                try:
                    _submit_review(fid, "dismissed", notes)
                    banner(t("review_stored"), variant="ok", icon_name="check_circle")
                except Exception as e:
                    banner(f"{t('error')}: {e}", variant="error", icon_name="error")
