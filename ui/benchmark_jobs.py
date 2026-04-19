"""Session-state helpers for stepped benchmark jobs (pause / resume / cancel)."""

from __future__ import annotations

from typing import Any

JOB_KEY = "vk_benchmark_long_job"


def merge_eval_cumulative(cum: dict | None, batch: dict) -> dict:
    if cum is None:
        cum = {
            "h1": 0,
            "h3": 0,
            "h5": 0,
            "mrr_sum": 0.0,
            "n": 0,
            "by_doc_type_raw": {},
            "by_difficulty_raw": {},
        }
    cum["h1"] += int(batch.get("hits_1", 0))
    cum["h3"] += int(batch.get("hits_3", 0))
    cum["h5"] += int(batch.get("hits_5", 0))
    cum["mrr_sum"] += float(batch.get("mrr_sum", 0.0))
    cum["n"] += int(batch.get("total_queries", 0))
    _merge_raw_buckets(cum["by_doc_type_raw"], batch.get("by_doc_type_raw") or {})
    _merge_raw_buckets(cum["by_difficulty_raw"], batch.get("by_difficulty_raw") or {})
    return cum


def _merge_raw_buckets(dst: dict, src: dict) -> None:
    for k, g in src.items():
        if k not in dst:
            dst[k] = {"h1": 0, "h3": 0, "h5": 0, "mrr": 0.0, "total": 0}
        dst[k]["h1"] += int(g.get("h1", 0))
        dst[k]["h3"] += int(g.get("h3", 0))
        dst[k]["h5"] += int(g.get("h5", 0))
        dst[k]["total"] += int(g.get("total", 0))
        dst[k]["mrr"] += float(g.get("mrr", 0.0))


def rates_from_raw(raw: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, g in raw.items():
        t = int(g.get("total", 0))
        out[k] = {
            "total": t,
            "hr_at_1": round(g["h1"] / t, 4) if t else 0,
            "hr_at_3": round(g["h3"] / t, 4) if t else 0,
            "hr_at_5": round(g["h5"] / t, 4) if t else 0,
            "mrr": round(g["mrr"] / t, 4) if t else 0,
        }
    return out


def merge_analyze_cumulative(cum: dict | None, batch: dict) -> dict:
    if cum is None:
        cum = {"n_queries": 0, "hits": 0, "errors": 0, "mrr_sum": 0.0}
    s = batch.get("summary") or {}
    cum["n_queries"] += int(s.get("n_queries", 0))
    cum["hits"] += int(s.get("hits_at_k", 0))
    cum["errors"] += int(s.get("n_errors", 0))
    for it in batch.get("items") or []:
        cum["mrr_sum"] += float(it.get("mrr") or 0.0)
    return cum
