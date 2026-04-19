"""Benchmark endpoints — generate Q/A pairs, review, evaluation (WP14 + WP15)."""

from __future__ import annotations

import json
from typing import Optional
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from api.benchmark.generator import generate_benchmark
from api.benchmark.evaluator import run_evaluation, run_retrieval_analysis
from api.core.benchmark_distribution import benchmark_distribution_report
from api.core.models import BenchmarkReviewRequest
from api.core.settings import get_settings

router = APIRouter()

_BENCHMARK_JSONL = "benchmark_v1.jsonl"
_REVIEWS_JSONL = "benchmark_reviews.jsonl"


def _benchmark_path() -> Path:
    return Path(get_settings().data_dir) / _BENCHMARK_JSONL


def _reviews_path() -> Path:
    p = Path(get_settings().data_dir) / _REVIEWS_JSONL
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_pairs_slice(offset: int, limit: int) -> tuple[list[dict], int]:
    path = _benchmark_path()
    if not path.exists():
        return [], 0
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    lines = [ln for ln in lines if ln.strip()]
    total = len(lines)
    chunk = lines[offset : offset + limit]
    pairs = []
    for i, ln in enumerate(chunk):
        try:
            obj = json.loads(ln)
            obj["_line_index"] = offset + i
            pairs.append(obj)
        except json.JSONDecodeError:
            continue
    return pairs, total


@router.post("/benchmark/generate")
async def generate_endpoint(
    max_pairs: int = Query(default=50, ge=1, le=500),
    validate: bool = Query(default=True),
    tenant_id: str = Query(default="demo"),
    append: bool = Query(
        default=False,
        description="Append new pairs to the benchmark JSONL file instead of replacing it.",
    ),
) -> dict:
    """WP14: Generate benchmark Q/A pairs from indexed chunks."""
    _pairs, stats = await generate_benchmark(
        max_pairs=max_pairs,
        validate=validate,
        tenant_id=tenant_id,
        append=append,
    )
    dist = benchmark_distribution_report(stats)
    return {
        "status": "completed",
        "tenant_id": tenant_id,
        "total_pairs": stats.total_pairs,
        "validated": stats.validated_count,
        "by_difficulty": stats.by_difficulty,
        "by_doc_type": stats.by_doc_type,
        "distribution": dist,
    }


@router.get("/benchmark/pairs")
async def list_pairs(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict:
    """Paginated benchmark pairs for human review (WP14 §14.2)."""
    pairs, total = _load_pairs_slice(offset, limit)
    return {"pairs": pairs, "total": total, "offset": offset, "limit": limit}


@router.post("/benchmark/review")
async def post_review(body: BenchmarkReviewRequest) -> dict:
    """Append a review decision (accept/reject/edit) to data/benchmark_reviews.jsonl."""
    action = body.action.strip().lower()
    if action not in ("accept", "reject"):
        raise HTTPException(status_code=400, detail="action must be accept or reject")

    path = _reviews_path()
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pair_index": body.pair_index,
        "action": action,
        "edited_answer": body.edited_answer,
        "notes": body.notes,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"status": "stored", "review": rec}


@router.get("/benchmark/run")
async def run_endpoint(
    config_name: str = Query(default="current"),
    validated_only: bool = Query(default=True),
    tenant_id: str = Query(default="demo"),
    top_k: int = Query(default=5, ge=1, le=20),
    pair_offset: int = Query(default=0, ge=0),
    pair_limit: Optional[int] = Query(
        default=None,
        ge=1,
        le=500,
        description="If set, evaluate only this many pairs starting at pair_offset (batched runs).",
    ),
    persist_reports: bool = Query(
        default=True,
        description="When true, reports are written only if this response completes the dataset (no has_more).",
    ),
    workers: Optional[int] = Query(
        default=None,
        ge=1,
        le=32,
        description="Concurrent retrieval calls (default: config/models.yaml retrieval.benchmark_concurrency or BENCHMARK_WORKERS).",
    ),
) -> dict:
    """WP15: Run evaluation against the benchmark. Returns HR@K + MRR.

    Uses the same retrieval stack as `/query` for ``tenant_id`` (filters, rerank, diversity).
    """
    result = await run_evaluation(
        config_name=config_name,
        validated_only=validated_only,
        tenant_id=tenant_id,
        eval_top_k=top_k,
        pair_offset=pair_offset,
        pair_limit=pair_limit,
        persist_reports=persist_reports,
        workers=workers,
    )
    return result.to_dict()


@router.get("/benchmark/analyze")
async def analyze_endpoint(
    tenant_id: str = Query(default="demo"),
    top_k: int = Query(default=5, ge=1, le=20),
    validated_only: bool = Query(default=True),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    misses_only: bool = Query(default=False),
    workers: Optional[int] = Query(
        default=None,
        ge=1,
        le=32,
        description="Concurrent retrieval calls (default: retrieval.benchmark_concurrency or BENCHMARK_WORKERS).",
    ),
) -> dict:
    """Per-query retrieval diagnostics: gold chunk vs top-k results (same pipeline as `/query`)."""
    return await run_retrieval_analysis(
        tenant_id=tenant_id,
        eval_top_k=top_k,
        validated_only=validated_only,
        limit=limit,
        offset=offset,
        misses_only=misses_only,
        workers=workers,
    )
