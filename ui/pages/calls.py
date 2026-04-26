"""Audio calls page — upload, track jobs, browse analyzed calls."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import streamlit as st

from i18n import t
from progress_helpers import run_with_progress
from ui_style import page_heading

API_URL = os.getenv("API_URL", "http://localhost:8000")

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

st.subheader(t("calls_upload"))
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
                st.success(t("calls_job_done").format(n=job.get("calls_created", 0)))
            else:
                st.error(t("calls_job_failed").format(err=job.get("error", "unknown error")))
            st.rerun()
        except Exception as e:
            st.error(f"{t('error')}: {e}")

st.divider()
st.subheader(t("calls_library"))

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
    st.error(f"{t('error')}: {e}")
    st.stop()

stats = bundle.get("stats") or {}
calls_payload = bundle.get("calls") or {}
calls = calls_payload.get("calls") or []

s1, s2, s3 = st.columns(3)
s1.metric(t("calls_metric_total"), int(stats.get("total", 0) or 0))
s2.metric(t("calls_metric_resolved"), int(stats.get("resolved", 0) or 0))
s3.metric(t("calls_metric_last"), (stats.get("last_indexed_at") or "")[:19] or t("never"))

if not calls:
    st.info(t("calls_empty"))
    st.stop()

rows = [
    {
        "id": c.get("id"),
        "call_id": c.get("call_id"),
        "farmacia": c.get("farmacia"),
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
    format_func=lambda x: next((f"{r['call_id']} - {r['farmacia'] or 'N/A'}" for r in rows if r["id"] == x), x),
)

detail = run_with_progress(t("page_loading"), lambda: _load_call_detail(selected_id))

st.subheader(t("calls_detail"))
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

if st.button(t("calls_delete_btn"), type="secondary"):
    try:
        run_with_progress(t("calls_delete_progress"), lambda: _delete_call(selected_id))
        st.success(t("calls_deleted"))
        st.rerun()
    except Exception as e:
        st.error(f"{t('error')}: {e}")
