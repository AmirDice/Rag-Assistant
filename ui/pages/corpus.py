"""Corpus management dashboard with background ingestion + pause/resume."""

import os
import time
from datetime import datetime
from typing import Any

import httpx
import streamlit as st
from i18n import t
from progress_helpers import run_with_progress
from ui_style import banner, clean_source_display_name, page_heading, section_header, status_cards

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _fmt_bytes(num_bytes: Any) -> str:
    try:
        value = float(num_bytes or 0)
    except Exception:
        value = 0.0
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


def _fmt_timestamp(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return t("never")
    try:
        normalized = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return raw[:16]


def _fetch_corpus_bundle() -> dict:
    stats = httpx.get(f"{API_URL}/stats", timeout=10).json()
    ing = None
    try:
        ing = httpx.get(f"{API_URL}/ingest/status", timeout=5).json()
    except Exception:
        pass
    docs = None
    try:
        docs = httpx.get(f"{API_URL}/library/documents", timeout=30).json()
    except Exception:
        docs = None
    return {"stats": stats, "ingest": ing, "documents": docs}


def _summary_table(data: dict[str, int], key_col: str, val_col: str) -> list[dict[str, Any]]:
    total = max(sum(int(v or 0) for v in data.values()), 1)
    rows: list[dict[str, Any]] = []
    for key, count in sorted(data.items(), key=lambda kv: int(kv[1]), reverse=True):
        n = int(count or 0)
        rows.append(
            {
                key_col: key,
                val_col: n,
                "%": f"{(n / total) * 100:.1f}%",
            }
        )
    return rows


page_heading(t("corpus_title"), "folder_open")
st.caption(t("page_desc_corpus"))


def _manual_upload_section():
    section_header(t("upload_title"), "upload_file")
    st.caption(t("upload_help"))
    uploaded = st.file_uploader(
        t("upload_files_label"),
        type=["pdf", "docx", "pptx", "xlsx"],
        accept_multiple_files=True,
        key="corpus_upload_files",
    )
    u_col1, u_col2 = st.columns(2)
    with u_col1:
        upload_force = st.checkbox(t("ingest_force"), value=False, key="upload_force_cb")
    with u_col2:
        upload_workers = st.number_input(
            t("ingest_workers"),
            min_value=1,
            max_value=32,
            value=4,
            step=1,
            key="upload_workers_n",
        )

    if st.button(
        t("upload_ingest_btn"),
        type="primary",
        disabled=not uploaded,
        icon=":material/upload:",
        key="upload_ingest_btn",
    ):
        if not uploaded:
            return

        def _do_upload():
            parts = []
            for uf in uploaded:
                content = uf.getvalue()
                ctype = uf.type or "application/octet-stream"
                parts.append(("files", (uf.name, content, ctype)))
            r = httpx.post(
                f"{API_URL}/ingest/upload",
                files=parts,
                data={
                    "force": str(upload_force).lower(),
                    "workers": str(int(upload_workers)),
                },
                timeout=300.0,
            )
            r.raise_for_status()
            return r.json()

        try:
            out = run_with_progress(t("upload_progress"), _do_upload)
            if out.get("error"):
                banner(f"{t('upload_error')}: {out['error']}", variant="error", icon_name="error")
            else:
                banner(
                    t("upload_started").format(
                        n=len(out.get("saved") or []),
                        files=out.get("total_files", 0),
                    ),
                    variant="ok",
                    icon_name="check_circle",
                )
                time.sleep(1)
                st.rerun()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = e.response.text or str(e)
            banner(f"{t('upload_error')}: {detail}", variant="error", icon_name="error")
        except Exception as ex:
            banner(f"{t('upload_error')}: {ex}", variant="error", icon_name="error")


def _corpus_body():
    try:
        data = run_with_progress(t("page_loading"), _fetch_corpus_bundle)
    except httpx.ConnectError:
        banner(t("api_unreachable"), variant="error", icon_name="error")
        return
    except Exception as e:
        banner(f"{t('stats_error')}: {e}", variant="error", icon_name="error")
        return

    stats = data["stats"]
    ingest_status = data.get("ingest") or {}

    ingest_state = str(ingest_status.get("status") or "").lower()
    if ingest_state == "running":
        ingest_state_label = t("status_running")
        ingest_state_kind = "on"
    elif ingest_state == "paused":
        ingest_state_label = t("status_paused")
        ingest_state_kind = "off"
    else:
        ingest_state_label = t("status_idle")
        ingest_state_kind = "neutral"

    last = stats.get("last_ingestion")
    total_docs = int(stats.get("total_docs", 0) or 0)
    total_chunks = int(stats.get("total_chunks", 0) or 0)
    total_size_bytes = int(stats.get("total_docs_bytes", 0) or 0)

    status_cards(
        [
            {
                "label": t("ingest_title"),
                "show_state": False,
                "value": ingest_state_label,
                "hint": f"{ingest_status.get('processed', 0)} {t('metric_docs').lower()}",
            },
            {
                "label": t("metric_docs"),
                "show_state": False,
                "value": str(total_docs),
            },
            {
                "label": t("metric_chunks"),
                "show_state": False,
                "value": str(total_chunks),
            },
            {
                "label": t("metric_last"),
                "show_state": False,
                "value": _fmt_timestamp(last),
            },
            {
                "label": t("metric_docs_size"),
                "show_state": False,
                "value": _fmt_bytes(total_size_bytes),
            },
        ]
    )

    summary_left, summary_right = st.columns(2)
    with summary_left:
        section_header(t("by_type"), "category")
        by_type = stats.get("by_type") or {}
        if by_type:
            st.dataframe(
                _summary_table(by_type, "type", t("metric_docs")),
                hide_index=True,
                use_container_width=True,
            )
        else:
            banner(t("documents_empty"), variant="info", icon_name="info")
    with summary_right:
        section_header(t("by_module"), "grid_view")
        by_module = stats.get("by_module") or {}
        if by_module:
            st.dataframe(
                _summary_table(by_module, "module", t("metric_docs")),
                hide_index=True,
                use_container_width=True,
            )
        else:
            banner(t("documents_empty"), variant="info", icon_name="info")

    section_header(t("documents_list_title"), "list")
    doc_payload = data.get("documents")
    if doc_payload is not None:
        try:
            rows = doc_payload.get("documents", [])
            if rows:
                st.dataframe(
                    [
                        {
                            "doc_id": r.get("doc_id"),
                            "source_file": clean_source_display_name(str(r.get("source_file") or "")),
                            "doc_type": r.get("doc_type"),
                            "module_id": r.get("module_id"),
                            "images": r.get("image_count", 0),
                        }
                        for r in rows
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(t("documents_reingest_help"))
                pick = st.selectbox(
                    t("documents_pick"),
                    options=[r["doc_id"] for r in rows],
                    format_func=lambda x: next(
                        (f"{x} — {r.get('source_file', '')}" for r in rows if r["doc_id"] == x),
                        x,
                    ),
                )
                src = next((r.get("source_file") or "" for r in rows if r["doc_id"] == pick), "")
                default_path = f"/app/docs/{src}" if src else "/app/docs"
                one_path = st.text_input(t("documents_one_path"), value=default_path)
                if st.button(t("documents_reingest_btn")):
                    def _reingest():
                        return httpx.post(
                            f"{API_URL}/ingest/start",
                            json={"path": one_path.strip(), "force": True, "workers": 4},
                            timeout=30,
                        ).json()

                    try:
                        ir = run_with_progress(t("ingest_action_progress"), _reingest)
                        if ir.get("error"):
                            st.error(ir["error"])
                        else:
                            st.success(t("documents_reingest_started"))
                            time.sleep(1)
                            st.rerun()
                    except Exception as ex:
                        st.error(str(ex))
            else:
                banner(t("documents_empty"), variant="info", icon_name="info")
        except Exception as ex:
            banner(f"{t('documents_list_error')}: {ex}", variant="warn", icon_name="warning")
    else:
        banner(t("documents_list_error"), variant="warn", icon_name="warning")

    st.divider()
    _manual_upload_section()
    st.divider()
    section_header(t("ingest_title"), "dataset")

    active = ingest_status and ingest_status.get("status") in ("running", "paused")

    if active:
        status = ingest_status["status"]
        pct = float(ingest_status.get("progress_pct", 0) or 0)
        processed = ingest_status.get("processed", 0)
        skipped = ingest_status.get("skipped", 0)
        total = ingest_status.get("total_files", 0)
        chunks = ingest_status.get("chunks_created", 0)
        current = ingest_status.get("current_file", "")
        errors_count = ingest_status.get("errors_count", 0)
        last_errors = ingest_status.get("last_errors", [])
        failed_count = ingest_status.get("failed_count", 0)
        failed_files = ingest_status.get("failed_files") or []
        succeeded_recent = ingest_status.get("succeeded_recent") or []
        workers_n = ingest_status.get("workers", 1)
        resume_mode = ingest_status.get("resume_mode", False)
        ck_pending = ingest_status.get("checkpoint_failed_pending", 0)

        if status == "running":
            st.info(t("ingest_bg_running"), icon=":material/sync:")
        else:
            st.warning(t("ingest_bg_paused"), icon=":material/pause_circle:")

        st.progress(
            pct / 100.0,
            text=t("ingest_progress_pct").format(
                pct=int(round(pct)),
                processed=processed,
                skipped=skipped,
                total=total,
            ),
        )

        if current:
            st.caption(t("ingest_current").format(file=current))
        st.caption(t("ingest_chunks_so_far").format(chunks=chunks))
        st.caption(f"{t('ingest_workers')}: {workers_n}" + (f" · resume" if resume_mode else ""))
        if ck_pending:
            st.caption(f"{t('ingest_checkpoint_pending')}: {ck_pending}")

        m1, m2, m3 = st.columns(3)
        m1.metric(t("ingest_failed_list"), failed_count)
        m2.metric(t("metric_docs"), processed)
        m3.metric(t("metric_chunks"), chunks)

        if failed_files:
            with st.expander(f"{t('ingest_failed_list')} ({len(failed_files)})"):
                for row in failed_files[-20:]:
                    st.markdown(f"**{row.get('file', '?')}** — {row.get('error', '')}")

        if succeeded_recent:
            with st.expander(f"{t('ingest_succeeded_recent')} ({len(succeeded_recent)})"):
                for name in succeeded_recent[-30:]:
                    st.text(name)

        if errors_count > 0 and last_errors:
            with st.expander(f"{t('ingest_errors_title')} ({errors_count})"):
                for err in last_errors:
                    st.error(err)

        col1, col2 = st.columns(2)
        with col1:
            if status == "running":
                if st.button(t("btn_pause"), type="secondary", use_container_width=True, icon=":material/pause:"):
                    def _p():
                        httpx.post(f"{API_URL}/ingest/pause", timeout=5)

                    run_with_progress(t("ingest_action_progress"), _p)
                    st.rerun()
            else:
                if st.button(t("btn_resume"), type="primary", use_container_width=True, icon=":material/play_arrow:"):
                    def _r():
                        httpx.post(f"{API_URL}/ingest/resume", timeout=5)

                    run_with_progress(t("ingest_action_progress"), _r)
                    st.rerun()
        with col2:
            if st.button(t("btn_cancel"), type="secondary", use_container_width=True, icon=":material/stop_circle:"):
                def _c():
                    httpx.post(f"{API_URL}/ingest/cancel", timeout=5)

                run_with_progress(t("ingest_action_progress"), _c)
                st.rerun()

        time.sleep(3)
        st.rerun()

    else:
        if ingest_status:
            _detail_keys = (
                "status",
                "total_files",
                "processed",
                "skipped",
                "failed_count",
                "chunks_created",
                "progress_pct",
                "workers",
                "resume_mode",
                "root_path",
                "checkpoint_failed_pending",
                "can_resume",
                "failed_files",
                "succeeded_recent",
            )
            with st.expander(t("ingest_status_detail"), expanded=False):
                st.json({k: ingest_status[k] for k in _detail_keys if k in ingest_status})

        if ingest_status and ingest_status.get("status") == "completed":
            proc = ingest_status.get("processed", 0)
            ch = ingest_status.get("chunks_created", 0)
            fc = ingest_status.get("failed_count", 0)
            if proc > 0 or ch > 0 or fc > 0:
                msg = t("ingest_bg_completed") + f" — {proc} docs, {ch} chunks"
                if fc:
                    banner(msg + f" · {fc} {t('ingest_failed_list').lower()}", variant="warn", icon_name="warning")
                else:
                    banner(msg, variant="ok", icon_name="check_circle")

        can_resume = bool(ingest_status and ingest_status.get("can_resume"))
        resume_only = st.checkbox(t("ingest_resume_cb"), value=False, disabled=not can_resume)
        if resume_only and can_resume:
            st.caption(f"{t('ingest_checkpoint_pending')}: {ingest_status.get('checkpoint_failed_pending', 0)}")

        workers_val = st.number_input(
            t("ingest_workers"),
            min_value=1,
            max_value=32,
            value=4,
            step=1,
        )

        ingest_path_val = st.text_input(
            t("ingest_path_label"),
            value="/app/docs",
            placeholder=t("ingest_path_placeholder"),
            disabled=resume_only,
        )
        force = st.checkbox(t("ingest_force"), disabled=resume_only)

        start_disabled = resume_only is False and not (ingest_path_val or "").strip()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(t("ingest_bg_start"), type="primary", disabled=start_disabled):
                def _start():
                    body: dict = {
                        "path": ingest_path_val.strip() if not resume_only else "",
                        "force": force,
                        "workers": int(workers_val),
                    }
                    if resume_only:
                        body["resume"] = True
                    return httpx.post(f"{API_URL}/ingest/start", json=body, timeout=30).json()

                try:
                    resp = run_with_progress(t("ingest_action_progress"), _start)
                    if resp.get("error"):
                        st.error(resp["error"])
                    else:
                        st.success(
                            f"Started — {resp.get('total_files', 0)} files, workers={resp.get('workers', workers_val)}"
                        )
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    banner(f"{t('ingest_error')}: {e}", variant="error", icon_name="error")
        with col_b:
            if can_resume and st.button(t("ingest_resume_btn"), type="secondary", disabled=False):
                def _resume():
                    return httpx.post(
                        f"{API_URL}/ingest/start",
                        json={"path": "", "resume": True, "workers": int(workers_val)},
                        timeout=30,
                    ).json()

                try:
                    resp = run_with_progress(t("ingest_action_progress"), _resume)
                    if resp.get("error"):
                        st.error(resp["error"])
                    else:
                        st.success(
                            f"Resume — {resp.get('total_files', 0)} files, workers={resp.get('workers', workers_val)}"
                        )
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    banner(f"{t('ingest_error')}: {e}", variant="error", icon_name="error")

    st.divider()
    section_header(t("health_title"), "health_and_safety")

    if st.button(t("health_check")):
        def _health():
            return httpx.get(f"{API_URL}/health", timeout=5).json()

        try:
            health = run_with_progress(t("page_loading"), _health)
            status_cards(
                [
                    {
                        "label": str(service).upper(),
                        "state": "on" if str(status).lower() == "ok" else "off",
                        "state_text": t("status_ok") if str(status).lower() == "ok" else t("status_error"),
                    }
                    for service, status in health.items()
                ]
            )
        except Exception as e:
            banner(f"{t('error')}: {e}", variant="error", icon_name="error")


_corpus_body()
