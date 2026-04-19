"""WP16 §16.1 — Benchmark runner (evaluation, analysis, generation + stepped jobs)."""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st
from benchmark_jobs import (
    JOB_KEY,
    merge_analyze_cumulative,
    merge_eval_cumulative,
    rates_from_raw,
)
from i18n import t
from progress_helpers import run_with_progress
from ui_style import LOGO_PATH, page_heading

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _benchmark_httpx_timeout() -> httpx.Timeout:
    """Long reads for /benchmark/generate, eval, analyze (LLM + retrieval). Default 8h; was 10m and caused 'timed out'."""
    raw = os.getenv("BENCHMARK_HTTP_TIMEOUT", "28800").strip()
    try:
        total = float(raw)
    except ValueError:
        total = 28800.0
    total = max(60.0, min(total, 86400.0))
    return httpx.Timeout(total, connect=60.0)


if LOGO_PATH:
    st.image(LOGO_PATH, width=168)
page_heading(t("benchmark_title"), "analytics")
st.caption(t("page_desc_benchmark"))


def _httpx_client():
    return httpx.Client(timeout=_benchmark_httpx_timeout())


def _retrieval_workers_params(retrieval_workers: int) -> dict:
    w = int(retrieval_workers)
    return {"workers": w} if w > 0 else {}


def _job_control_bar():
    j = st.session_state.get(JOB_KEY)
    if not j or j.get("finished"):
        return
    st.markdown(f"**{t('benchmark_job_running')}** — `{j.get('type', '?')}`")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(t("benchmark_pause"), key="vk_job_pause", disabled=j.get("paused", False)):
            j["paused"] = True
            j["auto_run"] = False
            st.session_state[JOB_KEY] = j
            st.rerun()
    with c2:
        if st.button(t("benchmark_resume"), key="vk_job_resume", disabled=not j.get("paused", False)):
            j["paused"] = False
            j["auto_run"] = True
            st.session_state[JOB_KEY] = j
            st.rerun()
    with c3:
        if st.button(t("benchmark_cancel_job"), key="vk_job_cancel"):
            st.session_state.pop(JOB_KEY, None)
            st.rerun()
    if j.get("paused"):
        st.warning(t("benchmark_job_paused_hint"))


def _run_eval_step(j: dict) -> None:
    tid = j["tenant_id"]
    bs = max(1, int(j["batch_size"]))
    off = int(j["eval_next"])
    params = {
        "validated_only": j["validated_only"],
        "tenant_id": tid,
        "top_k": j["top_k"],
        "pair_offset": off,
        "pair_limit": bs,
        "persist_reports": True,
    }
    params.update(_retrieval_workers_params(int(j.get("retrieval_workers", 0))))
    with _httpx_client() as client:
        r = client.get(f"{API_URL}/benchmark/run", params=params)
        r.raise_for_status()
        d = r.json()
    j["eval_cum"] = merge_eval_cumulative(j.get("eval_cum"), d)
    j["eval_last"] = d
    j["eval_next"] = int(d.get("next_offset", off))
    full_n = int(d.get("full_dataset_n", 0))
    j["eval_full_n"] = full_n
    if not d.get("has_more"):
        j["auto_run"] = False
        j["finished"] = True
    st.session_state[JOB_KEY] = j
    if j.get("auto_run") and not j.get("paused"):
        st.rerun()


def _run_analyze_step(j: dict) -> None:
    tid = j["tenant_id"]
    bs = max(1, int(j["batch_size"]))
    off = int(j["anal_next"])
    cap = int(j["anal_cap"])
    chunk = min(bs, max(0, cap - off))
    if chunk <= 0:
        j["auto_run"] = False
        j["finished"] = True
        st.session_state[JOB_KEY] = j
        return
    params = {
        "tenant_id": tid,
        "top_k": j["top_k"],
        "validated_only": j["validated_only"],
        "limit": chunk,
        "offset": off,
        "misses_only": j["anal_misses"],
    }
    params.update(_retrieval_workers_params(int(j.get("retrieval_workers", 0))))
    with _httpx_client() as client:
        r = client.get(f"{API_URL}/benchmark/analyze", params=params)
        r.raise_for_status()
        d = r.json()
    j["anal_workers_used"] = d.get("workers_used")
    j["anal_items"] = (j.get("anal_items") or []) + (d.get("items") or [])
    j["anal_cum"] = merge_analyze_cumulative(j.get("anal_cum"), d)
    j["anal_next"] = int(d.get("next_offset", off + chunk))
    if not d.get("has_more") or j["anal_next"] >= cap:
        j["auto_run"] = False
        j["finished"] = True
    st.session_state[JOB_KEY] = j
    if j.get("auto_run") and not j.get("paused"):
        st.rerun()


def _run_gen_step(j: dict) -> None:
    tid = j["tenant_id"]
    bs = max(1, int(j["batch_size"]))
    target = int(j["gen_target"])
    done = int(j.get("gen_accum", 0))
    chunk = min(bs, max(0, target - done))
    if chunk <= 0:
        j["auto_run"] = False
        j["finished"] = True
        st.session_state[JOB_KEY] = j
        return
    append_param = bool(j.get("append_only")) or (done > 0)
    with _httpx_client() as client:
        r = client.post(
            f"{API_URL}/benchmark/generate",
            params={
                "max_pairs": chunk,
                "validate": j["gen_validate"],
                "tenant_id": tid,
                "append": append_param,
            },
        )
        r.raise_for_status()
        d = r.json()
    j["gen_accum"] = done + int(d.get("batch_pairs", chunk))
    j["gen_last"] = d
    if j["gen_accum"] >= target:
        j["auto_run"] = False
        j["finished"] = True
    st.session_state[JOB_KEY] = j
    if j.get("auto_run") and not j.get("paused"):
        st.rerun()


def _maybe_run_job_step():
    j = st.session_state.get(JOB_KEY)
    if not j or not j.get("auto_run") or j.get("paused"):
        return
    kind = j.get("type")
    if kind == "eval":
        _run_eval_step(j)
    elif kind == "analyze":
        _run_analyze_step(j)
    elif kind == "gen":
        _run_gen_step(j)


def _display_eval_results_from_job(j: dict) -> None:
    cum = j.get("eval_cum") or {}
    n = int(cum.get("n", 0))
    if n <= 0:
        return
    st.success(t("benchmark_done"))
    st.subheader(t("benchmark_summary"))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("HR@1", f"{cum['h1'] / n * 100:.1f}%")
    c2.metric("HR@3", f"{cum['h3'] / n * 100:.1f}%")
    c3.metric("HR@5", f"{cum['h5'] / n * 100:.1f}%")
    c4.metric("MRR", f"{cum['mrr_sum'] / n:.3f}")
    fn = j.get("eval_full_n", n)
    c5.metric(t("benchmark_progress_pairs"), f"{n} / {fn}")
    last = j.get("eval_last") or {}
    if last.get("workers_used"):
        st.caption(t("benchmark_workers_used").format(n=last["workers_used"]))
    if last.get("meets_spec") is not None and j.get("finished"):
        st.metric(
            t("benchmark_meets_spec"),
            t("benchmark_pass") if last["meets_spec"] else t("benchmark_fail"),
        )
    with st.expander(t("benchmark_by_type")):
        st.json(rates_from_raw(cum.get("by_doc_type_raw") or {}))
    with st.expander(t("benchmark_by_difficulty")):
        st.json(rates_from_raw(cum.get("by_difficulty_raw") or {}))
    if last.get("spec_compliance") and j.get("finished"):
        with st.expander(t("benchmark_spec_compliance")):
            st.json(last["spec_compliance"])
    with st.expander(t("benchmark_raw")):
        st.json(last)
    st.caption(t("benchmark_job_complete_hint"))
    if st.button(t("benchmark_dismiss_job"), key="dismiss_eval_job"):
        st.session_state.pop(JOB_KEY, None)
        st.rerun()


def _display_analyze_from_job(j: dict) -> None:
    items = j.get("anal_items") or []
    cum = j.get("anal_cum") or {}
    nq = int(cum.get("n_queries", 0))
    st.subheader(t("benchmark_analyze_summary"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queries", nq)
    c2.metric("Hits@k", cum.get("hits", 0))
    hr = (cum["hits"] / nq) if nq else 0.0
    c3.metric("HR@k", f"{hr * 100:.1f}%")
    c4.metric("Errors", cum.get("errors", 0))
    if j.get("anal_workers_used"):
        st.caption(t("benchmark_workers_used").format(n=j["anal_workers_used"]))
    payload = {
        "summary": {
            "n_queries": nq,
            "hits_at_k": cum.get("hits", 0),
            "hr_at_k": round(hr, 4),
            "n_errors": cum.get("errors", 0),
        },
        "items": items,
        "workers_used": j.get("anal_workers_used"),
    }
    st.download_button(
        label=t("benchmark_download_analysis"),
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name="benchmark_retrieval_analysis.json",
        mime="application/json",
        key="dl_analysis_job",
    )
    st.subheader(t("benchmark_analyze_table"))
    table_rows = []
    for it in items:
        r1 = (it.get("retrieved") or [{}])[0] if it.get("retrieved") else {}
        q = (it.get("question") or "").replace("\n", " ")
        table_rows.append(
            {
                "#": it.get("pair_index"),
                "hit": it.get("hit"),
                "rank": it.get("hit_rank"),
                "question": q[:200] + ("…" if len(q) > 200 else ""),
                "gold_doc": (it.get("gold") or {}).get("source_doc", ""),
                "top1_doc": r1.get("source_doc", ""),
                "top1_ok": r1.get("matches_gold"),
            }
        )
    try:
        import pandas as pd

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    except ImportError:
        st.json(table_rows)
    st.caption(t("benchmark_detail_expander"))
    for it in items[:40]:
        label = f"#{it.get('pair_index')} — {'HIT' if it.get('hit') else 'MISS'}"
        with st.expander(label):
            st.markdown("**Question**")
            st.write(it.get("question") or "")
            if it.get("error"):
                st.error(it["error"])
            g = it.get("gold") or {}
            st.markdown("**Gold (expected)**")
            st.json(
                {
                    k: g.get(k)
                    for k in (
                        "chunk_id",
                        "source_doc",
                        "source_section",
                        "doc_type",
                        "difficulty",
                    )
                    if g.get(k) is not None
                }
            )
            if g.get("expected_answer"):
                st.caption("Expected answer (excerpt)")
                st.text(g["expected_answer"][:1500])
            if g.get("chunk_preview"):
                st.caption("Gold chunk preview")
                st.text(g["chunk_preview"][:2000])
            st.markdown("**Retrieved (top-k)**")
            st.json(it.get("retrieved") or [])
    st.caption(t("benchmark_job_complete_hint"))
    if st.button(t("benchmark_dismiss_job"), key="dismiss_analyze_job"):
        st.session_state.pop(JOB_KEY, None)
        st.rerun()


def _benchmark_body():
    bench_tenant = st.text_input(t("benchmark_tenant_id"), value="demo", key="bench_tenant")
    st.caption(t("benchmark_eval_pipeline_note"))
    batch_size = st.number_input(
        t("benchmark_batch_size"),
        min_value=1,
        max_value=500,
        value=5,
        step=1,
        help=t("benchmark_batch_size_help"),
        key="bench_batch_size",
    )
    retrieval_workers = st.number_input(
        t("benchmark_retrieval_workers"),
        min_value=0,
        max_value=32,
        value=0,
        step=1,
        help=t("benchmark_retrieval_workers_help"),
        key="bench_retrieval_workers",
    )

    _job_control_bar()
    _maybe_run_job_step()

    j = st.session_state.get(JOB_KEY)
    if j and j.get("type") == "eval" and j.get("eval_cum"):
        prog_n = j["eval_cum"].get("n", 0)
        prog_full = j.get("eval_full_n") or prog_n
        if prog_full:
            st.progress(
                min(1.0, prog_n / max(prog_full, 1)),
                text=f"{t('benchmark_progress_pairs')}: {prog_n} / {prog_full}",
            )
        if j.get("finished"):
            _display_eval_results_from_job(j)

    st.subheader(t("benchmark_eval_section"))
    ec1, ec2 = st.columns(2)
    with ec1:
        if st.button(t("benchmark_run"), type="primary", icon=":material/play_arrow:"):
            tid = bench_tenant.strip() or "demo"

            def _run_eval():
                with _httpx_client() as client:
                    resp = client.get(
                        f"{API_URL}/benchmark/run",
                        params={"validated_only": True, "tenant_id": tid, "top_k": 5},
                    )
                    resp.raise_for_status()
                    return resp.json()

            try:
                data = run_with_progress(t("benchmark_running"), _run_eval)
                st.success(t("benchmark_done"))
                st.subheader(t("benchmark_summary"))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("HR@1", f"{data.get('hr_at_1', 0) * 100:.1f}%")
                c2.metric("HR@3", f"{data.get('hr_at_3', 0) * 100:.1f}%")
                c3.metric("HR@5", f"{data.get('hr_at_5', 0) * 100:.1f}%")
                c4.metric("MRR", f"{data.get('mrr', 0):.3f}")
                if data.get("workers_used"):
                    st.caption(t("benchmark_workers_used").format(n=data["workers_used"]))
                if "meets_spec" in data:
                    st.metric(
                        t("benchmark_meets_spec"),
                        t("benchmark_pass") if data["meets_spec"] else t("benchmark_fail"),
                    )
                with st.expander(t("benchmark_by_type")):
                    st.json(data.get("by_doc_type", {}))
                with st.expander(t("benchmark_by_difficulty")):
                    st.json(data.get("by_difficulty", {}))
                if data.get("spec_compliance"):
                    with st.expander(t("benchmark_spec_compliance")):
                        st.json(data["spec_compliance"])
                with st.expander(t("benchmark_raw")):
                    st.json(data)
            except Exception as e:
                st.error(f"{t('benchmark_error')}: {e}")
    with ec2:
        if st.button(t("benchmark_run_stepped"), icon=":material/view_timeline:"):
            tid = bench_tenant.strip() or "demo"
            st.session_state[JOB_KEY] = {
                "type": "eval",
                "paused": False,
                "auto_run": True,
                "tenant_id": tid,
                "top_k": 5,
                "validated_only": True,
                "batch_size": int(batch_size),
                "eval_next": 0,
                "eval_cum": None,
                "eval_last": None,
                "finished": False,
                "eval_full_n": 0,
                "retrieval_workers": int(retrieval_workers),
            }
            st.rerun()

    st.divider()
    st.subheader(t("benchmark_analyze"))
    st.caption(t("benchmark_analyze_help"))
    az1, az2 = st.columns(2)
    with az1:
        anal_limit = st.number_input(
            t("benchmark_analyze_limit"),
            min_value=1,
            max_value=500,
            value=100,
            key="anal_limit",
        )
    with az2:
        anal_misses = st.checkbox(
            t("benchmark_analyze_misses"),
            value=False,
            key="anal_misses",
        )

    if j and j.get("type") == "analyze" and (j.get("anal_items") or j.get("finished")):
        st.progress(
            min(1.0, j.get("anal_next", 0) / max(j.get("anal_cap", 1), 1)),
            text=f"{t('benchmark_progress_pairs')}: {min(j.get('anal_next', 0), j.get('anal_cap', 0))} / {j.get('anal_cap', 0)}",
        )
        if j.get("finished"):
            _display_analyze_from_job(j)

    ac1, ac2 = st.columns(2)
    with ac1:
        if st.button(t("benchmark_analyze"), key="btn_analyze_once", icon=":material/bug_report:"):
            tid = bench_tenant.strip() or "demo"

            def _analyze():
                p = {
                    "tenant_id": tid,
                    "top_k": 5,
                    "validated_only": True,
                    "limit": int(anal_limit),
                    "misses_only": anal_misses,
                }
                p.update(_retrieval_workers_params(retrieval_workers))
                with _httpx_client() as client:
                    r = client.get(f"{API_URL}/benchmark/analyze", params=p)
                    r.raise_for_status()
                    return r.json()

            try:
                adata = run_with_progress(t("benchmark_analyzing"), _analyze)
                summ = adata.get("summary") or {}
                st.subheader(t("benchmark_analyze_summary"))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Queries", summ.get("n_queries", 0))
                c2.metric("Hits@k", summ.get("hits_at_k", 0))
                c3.metric("HR@k", f"{summ.get('hr_at_k', 0) * 100:.1f}%")
                c4.metric("Errors", summ.get("n_errors", 0))
                if adata.get("workers_used"):
                    st.caption(t("benchmark_workers_used").format(n=adata["workers_used"]))
                st.download_button(
                    label=t("benchmark_download_analysis"),
                    data=json.dumps(adata, ensure_ascii=False, indent=2),
                    file_name="benchmark_retrieval_analysis.json",
                    mime="application/json",
                    key="dl_analysis",
                )
                st.subheader(t("benchmark_analyze_table"))
                items = adata.get("items") or []
                table_rows = []
                for it in items:
                    r1 = (it.get("retrieved") or [{}])[0] if it.get("retrieved") else {}
                    q = (it.get("question") or "").replace("\n", " ")
                    table_rows.append(
                        {
                            "#": it.get("pair_index"),
                            "hit": it.get("hit"),
                            "rank": it.get("hit_rank"),
                            "question": q[:200] + ("…" if len(q) > 200 else ""),
                            "gold_doc": (it.get("gold") or {}).get("source_doc", ""),
                            "top1_doc": r1.get("source_doc", ""),
                            "top1_ok": r1.get("matches_gold"),
                        }
                    )
                try:
                    import pandas as pd

                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
                except ImportError:
                    st.json(table_rows)
                st.caption(t("benchmark_detail_expander"))
                for it in items[:40]:
                    label = f"#{it.get('pair_index')} — {'HIT' if it.get('hit') else 'MISS'}"
                    with st.expander(label):
                        st.markdown("**Question**")
                        st.write(it.get("question") or "")
                        if it.get("error"):
                            st.error(it["error"])
                        g = it.get("gold") or {}
                        st.markdown("**Gold (expected)**")
                        st.json(
                            {
                                k: g.get(k)
                                for k in (
                                    "chunk_id",
                                    "source_doc",
                                    "source_section",
                                    "doc_type",
                                    "difficulty",
                                )
                                if g.get(k) is not None
                            }
                        )
                        if g.get("expected_answer"):
                            st.caption("Expected answer (excerpt)")
                            st.text(g["expected_answer"][:1500])
                        if g.get("chunk_preview"):
                            st.caption("Gold chunk preview")
                            st.text(g["chunk_preview"][:2000])
                        st.markdown("**Retrieved (top-k)**")
                        st.json(it.get("retrieved") or [])
            except Exception as e:
                st.error(f"{t('benchmark_error')}: {e}")
    with ac2:
        if st.button(t("benchmark_analyze_stepped"), key="btn_analyze_step", icon=":material/view_timeline:"):
            tid = bench_tenant.strip() or "demo"
            st.session_state[JOB_KEY] = {
                "type": "analyze",
                "paused": False,
                "auto_run": True,
                "tenant_id": tid,
                "top_k": 5,
                "validated_only": True,
                "batch_size": int(batch_size),
                "anal_next": 0,
                "anal_cap": int(anal_limit),
                "anal_misses": anal_misses,
                "anal_items": [],
                "anal_cum": None,
                "finished": False,
                "retrieval_workers": int(retrieval_workers),
            }
            st.rerun()

    st.divider()
    st.markdown(t("benchmark_generate_help"))
    gen_cols = st.columns([1, 1, 2])
    with gen_cols[0]:
        max_pairs = st.number_input(
            t("benchmark_max_pairs"),
            min_value=5,
            max_value=500,
            value=50,
            step=5,
        )
    with gen_cols[1]:
        do_validate = st.checkbox(t("benchmark_validate"), value=True)
    append_only = st.checkbox(
        t("benchmark_gen_append"),
        value=False,
        help=t("benchmark_gen_append_help"),
    )
    n_preview = int(max_pairs)
    st.caption(
        t("benchmark_generate_caption").format(
            n=n_preview,
            lo=max(2, n_preview // 20),
            hi=max(3, n_preview // 10),
        )
    )

    gj = st.session_state.get(JOB_KEY)
    if gj and gj.get("type") == "gen":
        st.progress(
            min(1.0, gj.get("gen_accum", 0) / max(gj.get("gen_target", 1), 1)),
            text=f"{t('benchmark_progress_pairs')}: {gj.get('gen_accum', 0)} / {gj.get('gen_target', 0)}",
        )
        if gj.get("finished") and gj.get("gen_last"):
            st.success(t("benchmark_gen_done"))
            st.json(gj["gen_last"])
            st.caption(t("benchmark_job_complete_hint"))
            if st.button(t("benchmark_dismiss_job"), key="dismiss_gen_job"):
                st.session_state.pop(JOB_KEY, None)
                st.rerun()

    gc1, gc2 = st.columns(2)
    with gc1:
        if st.button(t("benchmark_generate_btn"), icon=":material/auto_awesome:"):
            tenant = bench_tenant.strip() or "demo"
            n_pairs = int(max_pairs)

            def _gen():
                with _httpx_client() as client:
                    r = client.post(
                        f"{API_URL}/benchmark/generate",
                        params={
                            "max_pairs": n_pairs,
                            "validate": do_validate,
                            "tenant_id": tenant,
                            "append": append_only,
                        },
                    )
                    r.raise_for_status()
                    return r.json()

            try:
                data = run_with_progress(t("benchmark_generating"), _gen)
                st.json(data)
                dist = data.get("distribution") or {}
                warns = dist.get("warnings") or []
                if warns:
                    st.subheader(t("benchmark_warnings"))
                    for w in warns:
                        st.warning(w)
                with st.expander(t("benchmark_distribution")):
                    st.json(dist)
            except Exception as e:
                st.error(f"{t('benchmark_error')}: {e}")
    with gc2:
        if st.button(t("benchmark_generate_stepped"), icon=":material/view_timeline:"):
            tenant = bench_tenant.strip() or "demo"
            st.session_state[JOB_KEY] = {
                "type": "gen",
                "paused": False,
                "auto_run": True,
                "tenant_id": tenant,
                "batch_size": int(batch_size),
                "gen_target": int(max_pairs),
                "gen_validate": do_validate,
                "append_only": append_only,
                "gen_accum": 0,
                "gen_last": None,
                "finished": False,
            }
            st.rerun()


_benchmark_body()
