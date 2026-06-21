"""POST /query — Main RAG retrieval endpoint (WP15 §15.2)."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from api.core.models import QueryRequest, QueryResponse
from api.core.cache import get_cache
from api.core.retriever import retrieve
from api.core.generator import generate_answer
from api.core.grounding_verifier import verify_answer_grounding
from api.core.query_preprocessor import preprocess_query
from api.core.query_intent_planner import (
    plan_query_intent,
    preliminary_top_k,
    query_refinement_enabled,
)
from api.core.settings import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Shown in place of an ungrounded answer when the grounding guard trips.
_GROUNDING_FALLBACK = (
    "No tengo suficiente evidencia en la documentación recuperada para dar una "
    "respuesta fiable. ¿Puedes concretar un poco más la consulta?"
)


def _apply_grounding_guard(resp: QueryResponse) -> None:
    """Verify the synthesized answer against retrieved chunks (in place).

    Always attaches grounding metadata. When the guard is enabled in
    config/models.yaml and the answer falls below the support threshold, the
    answer is replaced with a safe fallback so we don't surface confident
    hallucinations.
    """
    if not resp.synthesized_answer or not resp.answer_chunks:
        return
    retrieval_cfg = get_settings().models_config().get("retrieval", {}) or {}
    if not retrieval_cfg.get("grounding_enabled", True):
        return
    try:
        min_ratio = float(retrieval_cfg.get("grounding_min_ratio", 0.6))
    except (TypeError, ValueError):
        min_ratio = 0.6

    result = verify_answer_grounding(
        resp.synthesized_answer, resp.answer_chunks, min_grounded_ratio=min_ratio
    )
    resp.grounded = result.grounded
    resp.grounding_ratio = round(result.grounded_ratio, 3)
    if not result.grounded:
        logger.warning(
            "grounding_guard_rejected query_id=%s ratio=%.3f unsupported=%d/%d",
            resp.query_id,
            result.grounded_ratio,
            result.unsupported_sentences,
            result.checked_sentences,
        )
        resp.synthesized_answer = _GROUNDING_FALLBACK


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    cache = await get_cache()

    parts: list[str] = []
    if req.generate:
        parts.append("gen")
    if req.reranker:
        parts.append(f"rr:{req.reranker}")
    if req.generation_model:
        parts.append(f"gm:{req.generation_model}")
    cache_key_suffix = (":" + ":".join(parts)) if parts else ""
    cached = await cache.get(req.question + cache_key_suffix, req.tenant_id)
    if cached:
        resp = QueryResponse(**cached)
        resp.cache_hit = True
        return resp

    # Preprocess the query (glossary enrichment / optional spell-fix) so the
    # embedder sees canonical terminology. Original question still drives rerank
    # + generation.
    pre = await preprocess_query(req.question, tenant_id=req.tenant_id)

    generation_question = req.question
    retrieval_query = pre.retrieval_query
    prelim_resp = None

    # Optional intent planner: preliminary retrieval → refine the query or ask
    # ONE clarification question. Off by default; falls back to plain search.
    if req.generate and query_refinement_enabled():
        prelim_req = req.model_copy(update={"top_k": max(req.top_k, preliminary_top_k())})
        prelim_resp = await retrieve(prelim_req, retrieval_query=pre.retrieval_query)
        decision = await plan_query_intent(
            preprocessed=pre,
            chunks=prelim_resp.answer_chunks,
            prior_turns=list(req.prior_turns or []),
        )
        if decision.needs_clarification:
            # Clarification turns depend on conversation state — not cached.
            return QueryResponse(
                answer_chunks=[],
                synthesized_answer=decision.clarification_question,
                needs_clarification=True,
                message_type="clarification_question",
                original_query=pre.original,
                retrieval_query=pre.retrieval_query,
                final_query=decision.final_query,
                matched_terms=pre.matched_terms,
            )
        retrieval_query = decision.retrieval_query or pre.retrieval_query
        generation_question = decision.final_query or req.question

    # Reuse the preliminary retrieval when the query didn't change.
    if prelim_resp is not None and retrieval_query.strip().lower() == pre.retrieval_query.strip().lower():
        resp = prelim_resp.model_copy(update={"answer_chunks": prelim_resp.answer_chunks[: req.top_k]})
    else:
        resp = await retrieve(req, retrieval_query=retrieval_query)

    resp.original_query = pre.original
    resp.retrieval_query = retrieval_query
    resp.matched_terms = pre.matched_terms
    resp.final_query = generation_question

    if req.generate and resp.answer_chunks:
        gen = await generate_answer(
            generation_question,
            resp.answer_chunks,
            generation_model=req.generation_model,
        )
        resp.synthesized_answer = gen.text
        resp.total_estimated_cost = gen.estimated_cost
        resp.cost_currency = gen.cost_currency
        resp.cost_breakdown = gen.cost_breakdown
        _apply_grounding_guard(resp)

    await cache.set(req.question + cache_key_suffix, req.tenant_id, resp.model_dump())

    return resp
