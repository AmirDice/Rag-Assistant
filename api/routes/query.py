"""POST /query — Main RAG retrieval endpoint (WP15 §15.2)."""

from __future__ import annotations

from fastapi import APIRouter

from api.core.models import QueryRequest, QueryResponse
from api.core.cache import get_cache
from api.core.retriever import retrieve
from api.core.generator import generate_answer

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    cache = await get_cache()

    cache_key_suffix = ":gen" if req.generate else ""
    cached = await cache.get(req.question + cache_key_suffix, req.tenant_id)
    if cached:
        resp = QueryResponse(**cached)
        resp.cache_hit = True
        return resp

    resp = await retrieve(req)

    if req.generate and resp.answer_chunks:
        resp.synthesized_answer = await generate_answer(
            req.question, resp.answer_chunks
        )

    await cache.set(req.question + cache_key_suffix, req.tenant_id, resp.model_dump())

    return resp
