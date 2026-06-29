"""Audio calls page — upload, track jobs, browse analyzed calls."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import streamlit as st

from i18n import t
from progress_helpers import run_with_progress
from ui_style import banner, page_heading, section_header

API_URL = os.getenv("API_URL", "http://localhost:8000")
_DEMO_LAST_INDEXED = "—"


def _demo_call_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": "demo-001",
            "call_id": "CALL-001",
            "farmacia": "Northwind Traders",
            "resolved": True,
            "indexed_at": "2026-04-26T22:38:14",
            "source": "acme_support_sso_setup.mp3",
        },
        {
            "id": "demo-002",
            "call_id": "CALL-002",
            "farmacia": "Globex Inc.",
            "resolved": False,
            "indexed_at": "2026-04-26T22:31:02",
            "source": "acme_support_upload_fail.mp3",
        },
        {
            "id": "demo-003",
            "call_id": "CALL-003",
            "farmacia": "Initech",
            "resolved": True,
            "indexed_at": "2026-04-26T22:27:58",
            "source": "acme_support_billing_plan.mp3",
        },
    ]


def _demo_call_detail(selected_id: str) -> dict[str, Any]:
    return {
        "id": selected_id,
        "problema_corto": "Single sign-on fails for all members after enabling SAML.",
        "resumen": "Customer enabled SAML SSO but members could not sign in; the email attribute was unmapped.",
        "resolucion": "Mapped the email and name attributes in Admin Settings > SSO and set a default role; sign-in worked.",
        "audio_available": False,
        "transcript": [
            {"start": 0.0, "end": 7.2, "speaker": "AGENT", "text": "Hi, how can we help today?"},
            {"start": 7.3, "end": 19.8, "speaker": "CALLER", "text": "I enabled single sign-on and now my team can't log in."},
            {"start": 20.0, "end": 33.1, "speaker": "AGENT", "text": "Let's check the SSO attribute mapping in Admin Settings."},
        ],
        "rag_qa": [
            {
                "question": "What was the main symptom?",
                "answer": "Members were redirected back with an error and could not sign in via SSO.",
                "category": "authentication",
                "confidence": 0.91,
            },
            {
                "question": "What action resolved the problem?",
                "answer": "Mapping the email/name attributes and setting a default role in Admin Settings > SSO.",
                "category": "resolution",
                "confidence": 0.88,
            },
        ],
    }

page_heading(t("calls_title"), "call")
st.caption(t("page_desc_calls"))

st.session_state.setdefault("calls_agent_id", "call_audio")


def _fetch_calls_bundle(agent_id: str, query: str, farmacia: str, tag: str, resolved: str) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": 200, "offset": 0}
    if agent_id.strip():
        params["agent_id"] = agent_id.strip()
    if query.strip():
        params["search"] = query.strip()
    if farmacia.strip():
        params["farmacia"] = farmacia.strip()
    if tag.strip():
        params["tag"] = tag.strip()
    if resolved == "Resolved":
        params["resolved"] = "true"
    elif resolved == "Unresolved":
        params["resolved"] = "false"

    stats = httpx.get(f"{API_URL}/calls/stats", params={"agent_id": agent_id.strip() or None}, timeout=10).json()
    filters = httpx.get(f"{API_URL}/calls/filters", params={"agent_id": agent_id.strip() or None}, timeout=10).json()
    calls = httpx.get(f"{API_URL}/calls", params=params, timeout=20).json()
    return {"stats": stats, "filters": filters, "calls": calls}


def _poll_job(job_id: str, agent_id: str) -> dict[str, Any]:
    while True:
        resp = httpx.get(
            f"{API_URL}/calls/jobs/{job_id}",
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        status = str(payload.get("status") or "")
        if status in {"completed", "failed"}:
            return payload
        time.sleep(2)


def _upload_audio(uploaded, agent_id: str) -> dict[str, Any]:
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
    data = {"agent_id": agent_id}
    r = httpx.post(f"{API_URL}/calls/upload", files=files, data=data, timeout=180)
    r.raise_for_status()
    return r.json()


def _load_call_detail(call_id: str) -> dict[str, Any]:
    r = httpx.get(f"{API_URL}/calls/{call_id}", timeout=20)
    r.raise_for_status()
    return r.json()


def _load_call_audio(call_id: str) -> bytes | None:
    try:
        r = httpx.get(f"{API_URL}/calls/{call_id}/audio", timeout=20)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def _delete_call(call_id: str) -> dict[str, Any]:
    r = httpx.delete(f"{API_URL}/calls/{call_id}", timeout=20)
    r.raise_for_status()
    return r.json()


with st.sidebar:
    st.session_state["calls_agent_id"] = st.text_input(
        t("calls_agent_id"),
        value=st.session_state.get("calls_agent_id", "call_audio"),
    )

agent_id = st.session_state.get("calls_agent_id", "call_audio")

section_header(t("calls_upload"), "upload_file")
up_cols = st.columns([3, 1])
with up_cols[0]:
    up = st.file_uploader(
        t("calls_upload_file"),
        type=["mp3", "wav", "m4a", "ogg", "flac", "webm", "mp4"],
        accept_multiple_files=False,
        key="calls_audio_upload",
    )
with up_cols[1]:
    if st.button(t("calls_upload_btn"), type="primary", disabled=up is None):
        try:
            started = run_with_progress(t("calls_upload_progress"), lambda: _upload_audio(up, agent_id))
            job = run_with_progress(t("calls_job_progress"), lambda: _poll_job(started["job_id"], agent_id))
            if job.get("status") == "completed":
                banner(t("calls_job_done").format(n=job.get("calls_created", 0)), variant="ok", icon_name="check_circle")
            else:
                banner(t("calls_job_failed").format(err=job.get("error", "unknown error")), variant="error", icon_name="error")
            st.rerun()
        except Exception as e:
            banner(f"{t('error')}: {e}", variant="error", icon_name="error")

st.divider()
section_header(t("calls_library"), "library_books")

f1, f2, f3 = st.columns(3)
with f1:
    search_q = st.text_input(t("calls_search"), value="")
with f2:
    farmacia_q = st.text_input(t("calls_farmacia"), value="")
with f3:
    resolved_q = st.selectbox(t("calls_resolved"), options=["All", "Resolved", "Unresolved"], index=0)

try:
    bundle = run_with_progress(
        t("page_loading"),
        lambda: _fetch_calls_bundle(agent_id, search_q, farmacia_q, "", resolved_q),
    )
except Exception as e:
    banner(f"{t('error')}: {e}", variant="error", icon_name="error")
    st.stop()

stats = bundle.get("stats") or {}
calls_payload = bundle.get("calls") or {}
calls = calls_payload.get("calls") or []

s1, s2, s3 = st.columns(3)
s1.metric(t("calls_metric_total"), int(stats.get("total", 0) or 0))
s2.metric(t("calls_metric_resolved"), int(stats.get("resolved", 0) or 0))
s3.metric(
    t("calls_metric_last"),
    ((stats.get("last_indexed_at") or "")[:19] or _DEMO_LAST_INDEXED),
)

if not calls:
    calls = _demo_call_rows()

rows = [
    {
        "id": c.get("id"),
        "call_id": c.get("call_id"),
        "customer": c.get("farmacia"),
        "resolved": bool(c.get("resolucion_exitosa")),
        "indexed_at": c.get("indexed_at"),
        "source": c.get("source_file"),
    }
    for c in calls
]
st.dataframe(rows, use_container_width=True, hide_index=True)

selected_id = st.selectbox(
    t("calls_pick"),
    options=[r["id"] for r in rows],
    format_func=lambda x: next((f"{r['call_id']} - {r['customer'] or 'N/A'}" for r in rows if r["id"] == x), x),
)

is_demo = selected_id.startswith("demo-")
detail = _demo_call_detail(selected_id) if is_demo else run_with_progress(
    t("page_loading"),
    lambda: _load_call_detail(selected_id),
)

section_header(t("calls_detail"), "description")
st.markdown(f"**{t('calls_problem')}**: {detail.get('problema_corto') or '—'}")
st.markdown(f"**{t('calls_summary')}**: {detail.get('resumen') or '—'}")
st.markdown(f"**{t('calls_resolution')}**: {detail.get('resolucion') or '—'}")

audio = _load_call_audio(selected_id) if detail.get("audio_available") else None
if audio:
    st.audio(audio)

with st.expander(t("calls_transcript"), expanded=False):
    for ln in detail.get("transcript", []):
        st.markdown(
            f"- [{float(ln.get('start', 0)):.1f}s - {float(ln.get('end', 0)):.1f}s] "
            f"**{ln.get('speaker', '?')}**: {ln.get('text', '')}"
        )

with st.expander(t("calls_rag_pairs"), expanded=False):
    for i, pair in enumerate(detail.get("rag_qa", []), 1):
        st.markdown(f"**Q{i}**: {pair.get('question', '')}")
        st.markdown(f"**A{i}**: {pair.get('answer', '')}")
        st.caption(f"{pair.get('category', '')} · confidence={pair.get('confidence', 0)}")
        st.divider()

if st.button(t("calls_delete_btn"), type="secondary", disabled=is_demo):
    try:
        run_with_progress(t("calls_delete_progress"), lambda: _delete_call(selected_id))
        banner(t("calls_deleted"), variant="ok", icon_name="delete")
        st.rerun()
    except Exception as e:
        banner(f"{t('error')}: {e}", variant="error", icon_name="error")
