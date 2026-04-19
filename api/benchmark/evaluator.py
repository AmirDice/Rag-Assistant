"""WP15 — Benchmark evaluation: HR@K + MRR against live Qdrant index.

Uses the same retrieval pipeline as `/query` (tenant filter, changelog_pure,
title boost, doc diversity) so metrics match real chat behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from api.core.models import AnswerChunk, QueryRequest
from api.core.retriever import retrieve
from api.core.settings import get_settings

logger = logging.getLogger(__name__)

# WP15 §15.4 — minimum HR@3 by doc type; L3 from by_difficulty
DOC_TYPE_MIN_HR3: dict[str, float] = {
    "faq_document": 0.90,
    "structured_manual": 0.85,
    "operational_guide": 0.82,
    "module_manual": 0.80,
    "changelog_as_manual": 0.70,
}
L3_IMAGE_MIN_HR3 = 0.65


def _norm_section(value: str | None) -> str:
    """Collapse whitespace and case for stable section matching."""
    if value is None:
        return ""
    return " ".join(str(value).strip().casefold().split())


def _chunk_matches_pair(chunk: AnswerChunk, pair: dict) -> bool:
    """Hit if Qdrant id matches, else same doc + normalized section (or doc-only if no section)."""
    gold_id = (pair.get("chunk_id") or "").strip()
    if gold_id and (chunk.chunk_id or "").strip() == gold_id:
        return True

    if chunk.source_doc != pair.get("source_doc", ""):
        return False

    psec = pair.get("source_section")
    if psec is None or str(psec).strip() == "":
        return True
    return _norm_section(chunk.source_section) == _norm_section(str(psec))


@dataclass
class EvalResult:
    config_name: str
    hr_at_1: float = 0.0
    hr_at_3: float = 0.0
    hr_at_5: float = 0.0
    mrr: float = 0.0
    total_queries: int = 0
    by_doc_type: dict = field(default_factory=dict)
    by_difficulty: dict = field(default_factory=dict)
    spec_compliance: dict = field(default_factory=dict)
    meets_spec: bool = False
    # Batched evaluation (UI pause/resume)
    full_dataset_n: int = 0
    batch_offset: int = 0
    has_more: bool = False
    next_offset: int = 0
    hits_1: int = 0
    hits_3: int = 0
    hits_5: int = 0
    mrr_sum: float = 0.0
    by_doc_type_raw: dict = field(default_factory=dict)
    by_difficulty_raw: dict = field(default_factory=dict)
    workers_used: int = 1

    def to_dict(self) -> dict:
        d = {
            "config_name": self.config_name,
            "hr_at_1": round(self.hr_at_1, 4),
            "hr_at_3": round(self.hr_at_3, 4),
            "hr_at_5": round(self.hr_at_5, 4),
            "mrr": round(self.mrr, 4),
            "total_queries": self.total_queries,
            "by_doc_type": self.by_doc_type,
            "by_difficulty": self.by_difficulty,
            "spec_compliance": self.spec_compliance,
            "meets_spec": self.meets_spec,
            "full_dataset_n": self.full_dataset_n,
            "batch_offset": self.batch_offset,
            "has_more": self.has_more,
            "next_offset": self.next_offset,
            "hits_1": self.hits_1,
            "hits_3": self.hits_3,
            "hits_5": self.hits_5,
            "mrr_sum": round(self.mrr_sum, 4),
            "by_doc_type_raw": self.by_doc_type_raw,
            "by_difficulty_raw": self.by_difficulty_raw,
            "workers_used": self.workers_used,
        }
        return d


def resolve_benchmark_workers(explicit: Optional[int] = None) -> int:
    """Concurrent retrievals: query param > models.yaml retrieval.benchmark_concurrency > env."""
    if explicit is not None:
        return max(1, min(32, int(explicit)))
    settings = get_settings()
    cfg = settings.models_config().get("retrieval", {}) or {}
    raw = cfg.get("benchmark_concurrency")
    if raw is not None:
        try:
            return max(1, min(32, int(raw)))
        except (TypeError, ValueError):
            pass
    return max(1, min(32, settings.benchmark_workers))


def _load_benchmark() -> list[dict]:
    settings = get_settings()
    path = Path(settings.data_dir) / "benchmark_v1.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. Run benchmark generation first."
        )
    pairs = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            pairs.append(json.loads(line))
    return pairs


def get_benchmark_pairs_for_eval(validated_only: bool = True) -> list[dict]:
    """Pairs used for metrics: validated subset if any, else all (with warning)."""
    pairs = _load_benchmark()
    if validated_only:
        val = [p for p in pairs if p.get("validated", False)]
        if val:
            pairs = val
        elif pairs:
            logger.warning(
                "No validated benchmark pairs — using all %d pairs", len(pairs)
            )
    if not pairs:
        raise RuntimeError("No benchmark pairs available")
    return pairs


_RETRIEVED_PREVIEW_LEN = 400
_GOLD_ANSWER_PREVIEW_LEN = 800
_GOLD_CHUNK_PREVIEW_LEN = 1200


def _gold_block(pair: dict) -> dict:
    prev = (pair.get("chunk_text_preview") or "").strip()
    if len(prev) > _GOLD_CHUNK_PREVIEW_LEN:
        prev = prev[: _GOLD_CHUNK_PREVIEW_LEN - 1] + "…"
    ans = (pair.get("answer") or "").strip()
    if len(ans) > _GOLD_ANSWER_PREVIEW_LEN:
        ans = ans[: _GOLD_ANSWER_PREVIEW_LEN - 1] + "…"
    return {
        "chunk_id": (pair.get("chunk_id") or "").strip(),
        "source_doc": pair.get("source_doc", ""),
        "source_section": pair.get("source_section"),
        "doc_type": pair.get("doc_type"),
        "difficulty": pair.get("difficulty"),
        "expected_answer": ans,
        "chunk_preview": prev,
    }


def _retrieved_row(rank: int, c: AnswerChunk, pair: dict) -> dict:
    txt = (c.text or "").replace("\n", " ").strip()
    if len(txt) > _RETRIEVED_PREVIEW_LEN:
        txt = txt[: _RETRIEVED_PREVIEW_LEN - 1] + "…"
    return {
        "rank": rank,
        "chunk_id": (c.chunk_id or "").strip(),
        "source_doc": c.source_doc or "",
        "source_section": c.source_section,
        "score": round(float(c.score), 4),
        "matches_gold": _chunk_matches_pair(c, pair),
        "text_preview": txt,
    }


async def _analyze_one_pair(
    sem: asyncio.Semaphore,
    pair_index: int,
    pair: dict,
    *,
    tenant_id: str,
    eval_top_k: int,
) -> dict:
    question = pair["question"]
    record: dict = {
        "pair_index": pair_index,
        "question": question,
        "gold": _gold_block(pair),
    }
    async with sem:
        try:
            chunks = await _retrieve_for_eval(
                question, tenant_id=tenant_id, top_k=eval_top_k
            )
        except Exception as e:
            record["error"] = str(e)
            record["hit"] = False
            record["hit_rank"] = None
            record["mrr"] = 0.0
            record["retrieved"] = []
            return record

        hit_rank: int | None = None
        for j, c in enumerate(chunks):
            if _chunk_matches_pair(c, pair):
                hit_rank = j + 1
                break
        hit = hit_rank is not None
        mrr = (1.0 / hit_rank) if hit_rank else 0.0
        record["hit"] = hit
        record["hit_rank"] = hit_rank
        record["mrr"] = round(mrr, 4)
        record["retrieved"] = [
            _retrieved_row(j + 1, c, pair) for j, c in enumerate(chunks)
        ]
        return record


async def run_retrieval_analysis(
    *,
    tenant_id: str = "demo",
    eval_top_k: int = 5,
    validated_only: bool = True,
    limit: int = 100,
    offset: int = 0,
    misses_only: bool = False,
    workers: Optional[int] = None,
) -> dict:
    """Per-query breakdown: benchmark question, gold chunk metadata, retrieved top-k rows."""
    pairs = get_benchmark_pairs_for_eval(validated_only)
    total = len(pairs)
    window = pairs[offset : offset + limit]
    wn = resolve_benchmark_workers(workers)
    sem = asyncio.Semaphore(wn)

    records = await asyncio.gather(
        *(
            _analyze_one_pair(
                sem,
                offset + local_i,
                pair,
                tenant_id=tenant_id,
                eval_top_k=eval_top_k,
            )
            for local_i, pair in enumerate(window)
        )
    )
    records = sorted(records, key=lambda r: r["pair_index"])

    items: list[dict] = []
    n_run = 0
    hits = 0
    errors = 0
    for record in records:
        n_run += 1
        if record.get("error"):
            errors += 1
        elif record.get("hit"):
            hits += 1
        if misses_only and record.get("hit"):
            continue
        items.append(record)

    hr = hits / n_run if n_run else 0.0
    span = len(window)
    next_off = offset + span
    has_more = next_off < total
    return {
        "tenant_id": tenant_id,
        "top_k": eval_top_k,
        "validated_only": validated_only,
        "misses_only": misses_only,
        "offset": offset,
        "limit": limit,
        "total_pairs": total,
        "has_more": has_more,
        "next_offset": next_off,
        "workers_used": wn,
        "summary": {
            "n_queries": n_run,
            "n_errors": errors,
            "hits_at_k": hits,
            "hr_at_k": round(hr, 4),
            "n_items_returned": len(items),
        },
        "items": items,
    }


async def _retrieve_for_eval(
    question: str,
    *,
    tenant_id: str,
    top_k: int = 5,
) -> list[AnswerChunk]:
    """Same path as user chat: merged tenant, filters, rerank, boost, diversity."""
    resp = await retrieve(
        QueryRequest(question=question, tenant_id=tenant_id, top_k=top_k)
    )
    return resp.answer_chunks


async def _eval_retrieve_one(
    sem: asyncio.Semaphore,
    idx: int,
    pair: dict,
    *,
    tenant_id: str,
    eval_top_k: int,
) -> tuple[int, dict, list[AnswerChunk] | None, str | None]:
    async with sem:
        try:
            chunks = await _retrieve_for_eval(
                pair["question"], tenant_id=tenant_id, top_k=eval_top_k
            )
            return (idx, pair, chunks, None)
        except Exception as e:
            logger.warning("Retrieval failed for '%s': %s", pair["question"][:50], e)
            return (idx, pair, None, str(e))


def _is_hit(chunks: list[AnswerChunk], pair: dict, k: int) -> bool:
    for c in chunks[:k]:
        if _chunk_matches_pair(c, pair):
            return True
    return False


def _reciprocal_rank(chunks: list[AnswerChunk], pair: dict) -> float:
    for i, c in enumerate(chunks):
        if _chunk_matches_pair(c, pair):
            return 1.0 / (i + 1)
    return 0.0


async def run_evaluation(
    config_name: str = "current",
    validated_only: bool = True,
    tenant_id: str = "demo",
    eval_top_k: int = 5,
    *,
    pair_offset: int = 0,
    pair_limit: Optional[int] = None,
    persist_reports: bool = True,
    workers: Optional[int] = None,
) -> EvalResult:
    """Run evaluation on all benchmark pairs, or a slice ``[pair_offset : pair_offset + pair_limit]``."""
    pairs_all = get_benchmark_pairs_for_eval(validated_only)
    full_n = len(pairs_all)
    if pair_limit is not None:
        pairs = pairs_all[pair_offset : pair_offset + pair_limit]
        batch_off = pair_offset
        has_more = pair_offset + len(pairs) < full_n
        next_off = pair_offset + len(pairs)
    else:
        pairs = pairs_all
        batch_off = 0
        has_more = False
        next_off = full_n

    wn = resolve_benchmark_workers(workers)
    logger.info(
        "Running evaluation on %d/%d pairs (offset=%s, limit=%s, config=%s, tenant_id=%s, top_k=%d, workers=%d)",
        len(pairs),
        full_n,
        pair_offset if pair_limit is not None else 0,
        pair_limit,
        config_name,
        tenant_id,
        eval_top_k,
        wn,
    )

    hits_1 = hits_3 = hits_5 = 0
    mrr_sum = 0.0

    type_hits: dict[str, dict] = {}
    diff_hits: dict[str, dict] = {}

    sem = asyncio.Semaphore(wn)
    rows = await asyncio.gather(
        *(
            _eval_retrieve_one(sem, i, p, tenant_id=tenant_id, eval_top_k=eval_top_k)
            for i, p in enumerate(pairs)
        )
    )
    rows.sort(key=lambda x: x[0])

    for i, pair, chunks, err in rows:
        if err or chunks is None:
            continue

        h1 = _is_hit(chunks, pair, 1)
        h3 = _is_hit(chunks, pair, 3)
        h5 = _is_hit(chunks, pair, 5)
        rr = _reciprocal_rank(chunks, pair)

        hits_1 += int(h1)
        hits_3 += int(h3)
        hits_5 += int(h5)
        mrr_sum += rr

        for group_key, group_dict, group_val in [
            ("doc_type", type_hits, pair.get("doc_type", "unknown")),
            ("difficulty", diff_hits, pair.get("difficulty", "L1")),
        ]:
            if group_val not in group_dict:
                group_dict[group_val] = {"total": 0, "h1": 0, "h3": 0, "h5": 0, "mrr": 0.0}
            g = group_dict[group_val]
            g["total"] += 1
            g["h1"] += int(h1)
            g["h3"] += int(h3)
            g["h5"] += int(h5)
            g["mrr"] += rr

        if (i + 1) % 10 == 0:
            logger.info("Evaluated %d/%d (batch)", i + 1, len(pairs))

    n = len(pairs)
    result = EvalResult(
        config_name=config_name,
        hr_at_1=hits_1 / n if n else 0,
        hr_at_3=hits_3 / n if n else 0,
        hr_at_5=hits_5 / n if n else 0,
        mrr=mrr_sum / n if n else 0,
        total_queries=n,
        full_dataset_n=full_n,
        batch_offset=batch_off,
        has_more=has_more,
        next_offset=next_off,
        hits_1=hits_1,
        hits_3=hits_3,
        hits_5=hits_5,
        mrr_sum=mrr_sum,
        workers_used=wn,
    )

    result.by_doc_type_raw = {k: dict(v) for k, v in type_hits.items()}
    result.by_difficulty_raw = {k: dict(v) for k, v in diff_hits.items()}

    for name, group_dict, target_dict in [
        ("doc_type", type_hits, "by_doc_type"),
        ("difficulty", diff_hits, "by_difficulty"),
    ]:
        d = {}
        for key, g in group_dict.items():
            t = g["total"]
            d[key] = {
                "total": t,
                "hr_at_1": round(g["h1"] / t, 4) if t else 0,
                "hr_at_3": round(g["h3"] / t, 4) if t else 0,
                "hr_at_5": round(g["h5"] / t, 4) if t else 0,
                "mrr": round(g["mrr"] / t, 4) if t else 0,
            }
        setattr(result, target_dict, d)

    if has_more:
        result.spec_compliance = {}
        result.meets_spec = False
    else:
        compliance: dict[str, dict] = {}
        for dtype, min_hr in DOC_TYPE_MIN_HR3.items():
            block = result.by_doc_type.get(dtype, {})
            t = block.get("total", 0)
            if not t:
                compliance[dtype] = {"skipped": True, "reason": "no_pairs"}
                continue
            actual = block.get("hr_at_3", 0.0)
            compliance[dtype] = {
                "min_hr_at_3": min_hr,
                "actual_hr_at_3": actual,
                "pass": actual >= min_hr,
                "n": t,
            }

        l3 = result.by_difficulty.get("L3", {})
        l3_n = l3.get("total", 0)
        if l3_n:
            actual_l3 = l3.get("hr_at_3", 0.0)
            compliance["L3_image_derived"] = {
                "min_hr_at_3": L3_IMAGE_MIN_HR3,
                "actual_hr_at_3": actual_l3,
                "pass": actual_l3 >= L3_IMAGE_MIN_HR3,
                "n": l3_n,
            }
        else:
            compliance["L3_image_derived"] = {"skipped": True, "reason": "no_L3_pairs"}

        graded = [c for c in compliance.values() if "pass" in c]
        meets_types = all(c["pass"] for c in graded) if graded else True
        result.spec_compliance = compliance
        result.meets_spec = (result.hr_at_3 >= 0.85) and meets_types

    if persist_reports and not has_more and n > 0:
        _save_performance_report(result)
        _save_config_gold(result)

    logger.info(
        "Evaluation complete: HR@1=%.1f%% HR@3=%.1f%% HR@5=%.1f%% MRR=%.3f (n=%d batch, full_n=%d has_more=%s)",
        result.hr_at_1 * 100, result.hr_at_3 * 100, result.hr_at_5 * 100,
        result.mrr, n, full_n, has_more,
    )
    return result


def _save_performance_report(result: EvalResult) -> None:
    settings = get_settings()
    path = Path(settings.data_dir) / "performance_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved performance report to %s", path)


def _save_config_gold(result: EvalResult) -> None:
    """Save the current config as gold if it meets spec thresholds."""
    settings = get_settings()
    models_cfg = settings.models_config()

    config_gold = {
        "config_name": result.config_name,
        "embedding": models_cfg.get("embedding", {}),
        "reranker": models_cfg.get("reranker", {}),
        "metrics": {
            "hr_at_1": result.hr_at_1,
            "hr_at_3": result.hr_at_3,
            "hr_at_5": result.hr_at_5,
            "mrr": result.mrr,
        },
        "by_doc_type": result.by_doc_type,
        "spec_compliance": result.spec_compliance,
        "meets_spec": result.meets_spec,
    }

    path = Path(settings.data_dir) / "config_gold.yaml"
    import yaml
    path.write_text(
        yaml.dump(config_gold, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info("Saved config_gold to %s", path)
