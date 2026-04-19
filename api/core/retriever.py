"""Qdrant retriever with multi-tenant metadata filtering (WP15 + WP16 §16.6)."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    MinShould,
    PayloadField,
    Range,
)

from api.core.settings import get_settings
from api.core.tenant_state import merged_tenant_config
from api.core.embedder import get_embedder
from api.core.reranker import get_reranker
from api.core.models import (
    QueryRequest,
    QueryResponse,
    AnswerChunk,
    RelatedDoc,
)

logger = logging.getLogger(__name__)


def _tenant_erp_version_raw(tenant_cfg: dict) -> Any:
    """Prefer `erp_version`; fall back to `legacy_erp_version`."""
    v = tenant_cfg.get("erp_version")
    if v is not None and v != "":
        return v
    return tenant_cfg.get("legacy_erp_version")


def _build_tenant_filter(tenant_cfg: dict, lang: Optional[str] = None) -> Filter:
    """Build Qdrant filter conditions from tenant config (WP16 §16.6).

    Spec table 39:
      module_id IN contracted OR module_id IS NULL
      version_min <= tenant_version AND (version_max IS NULL OR version_max >= tenant_version)
      is_robot_doc = false OR tenant.has_robot_integration = true
      lang = tenant.preferred_lang (if set)
    """
    must: list = []

    contracted = tenant_cfg.get("contracted_modules", [])
    if contracted:
        must.append(
            Filter(
                min_should=MinShould(
                    conditions=[
                        FieldCondition(
                            key="module_id",
                            match=MatchAny(any=list(contracted) + [""]),
                        ),
                        IsEmptyCondition(is_empty=PayloadField(key="module_id")),
                    ],
                    min_count=1,
                )
            )
        )

    version = _tenant_erp_version_raw(tenant_cfg)
    if version is not None and version != "":
        try:
            v = float(version)
        except (TypeError, ValueError):
            v = None
        if v is not None:
            must.append(
                FieldCondition(key="version_min", range=Range(lte=v))
            )
            # version_max null OR version_max >= tenant (doc still applies)
            must.append(
                Filter(
                    min_should=MinShould(
                        conditions=[
                            IsEmptyCondition(is_empty=PayloadField(key="version_max")),
                            FieldCondition(key="version_max", range=Range(gte=v)),
                        ],
                        min_count=1,
                    )
                )
            )

    if not tenant_cfg.get("has_robot_integration", False):
        must.append(
            FieldCondition(key="is_robot_doc", match=MatchValue(value=False))
        )

    effective_lang = lang or tenant_cfg.get("preferred_lang")
    if effective_lang:
        must.append(
            FieldCondition(key="lang", match=MatchValue(value=effective_lang))
        )

    return Filter(must=must) if must else Filter()


def _tenant_version_float(tenant_cfg: dict) -> Optional[float]:
    v = _tenant_erp_version_raw(tenant_cfg)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def payload_visible_for_tenant(
    p: dict,
    tenant_cfg: dict,
    query_lang: Optional[str] = None,
) -> bool:
    """True iff this payload would pass the same rules as `retrieve()` after Qdrant returns it.

    Used by benchmark generation so gold chunks are always reachable for the chosen tenant.
    Mirrors `_build_tenant_filter` + `_include_changelog_pure` (changelog is applied post-search).
    """
    contracted = tenant_cfg.get("contracted_modules", [])
    if contracted:
        mid = p.get("module_id")
        if mid is not None and str(mid).strip() != "" and mid not in contracted:
            return False

    version = _tenant_erp_version_raw(tenant_cfg)
    if version is not None and str(version).strip() != "":
        try:
            v = float(version)
        except (TypeError, ValueError):
            v = None
        if v is not None:
            try:
                p_vmin = float(p.get("version_min", 0) or 0)
            except (TypeError, ValueError):
                p_vmin = 0.0
            if p_vmin > v:
                return False
            vmax = p.get("version_max")
            if vmax is not None and str(vmax).strip() != "":
                try:
                    vmax_f = float(vmax)
                except (TypeError, ValueError):
                    pass
                else:
                    if vmax_f < v:
                        return False

    if not tenant_cfg.get("has_robot_integration", False):
        if p.get("is_robot_doc", False):
            return False

    effective_lang = query_lang or tenant_cfg.get("preferred_lang")
    if effective_lang:
        if (p.get("lang") or "es") != effective_lang:
            return False

    tv = _tenant_version_float(tenant_cfg)
    if not _include_changelog_pure(p, tv):
        return False

    return True


def _include_changelog_pure(p: dict, tenant_version: Optional[float]) -> bool:
    """WP12: changelog_pure only when tenant version is within doc range."""
    if p.get("doc_type") != "changelog_pure":
        return True
    if tenant_version is None:
        return False
    try:
        vmin = float(p.get("version_min") or 0)
    except (TypeError, ValueError):
        vmin = 0.0
    if tenant_version < vmin:
        return False
    vmax = p.get("version_max")
    if vmax is None or vmax == "":
        return True
    try:
        vmax_f = float(vmax)
    except (TypeError, ValueError):
        return True
    return tenant_version <= vmax_f


_TITLE_BOOST = 0.06
_NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")


def _normalize(text: str) -> str:
    """Lowercase, strip accents, remove non-alphanumeric."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return _NON_ALPHA_RE.sub(" ", text).strip()


def _doc_stem(source_doc: str) -> str:
    """'CashFarma.pdf' → 'cashfarma'"""
    stem = source_doc.rsplit(".", 1)[0] if "." in source_doc else source_doc
    return _normalize(stem)


def _apply_title_boost(
    query: str,
    rerank_results: list,
    payloads: list[dict],
) -> list[tuple[int, float]]:
    """Boost chunks whose source document name overlaps with the query.

    Splits the doc filename into tokens and checks how many appear in the
    query. Boost is proportional to the fraction of matching tokens.
    """
    q_norm = _normalize(query)
    q_tokens = set(q_norm.split())

    scored: list[tuple[int, float]] = []
    for rr in rerank_results:
        p = payloads[rr.index]
        doc_name = _doc_stem(p.get("source_doc", ""))
        doc_tokens = set(doc_name.replace("_", " ").split())
        doc_tokens.discard("")

        if doc_tokens:
            overlap = len(q_tokens & doc_tokens) / len(doc_tokens)
            boost = _TITLE_BOOST * overlap
        else:
            boost = 0.0

        scored.append((rr.index, rr.score + boost))

    scored.sort(key=lambda x: -x[1])
    return scored


def _apply_doc_diversity(
    scored: list[tuple[int, float]],
    payloads: list[dict],
    max_per_doc: int = 2,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Limit chunks per document to ensure diverse results."""
    doc_counts: defaultdict[str, int] = defaultdict(int)
    result: list[tuple[int, float]] = []

    for idx, score in scored:
        doc = payloads[idx].get("source_doc", "")
        if doc_counts[doc] < max_per_doc:
            result.append((idx, score))
            doc_counts[doc] += 1
        if len(result) >= top_k:
            break

    return result


async def retrieve(request: QueryRequest) -> QueryResponse:
    """Full retrieval pipeline: embed → Qdrant search → rerank → format."""
    settings = get_settings()
    tenant_cfg = merged_tenant_config(request.tenant_id)
    models_cfg = settings.models_config()

    initial_top_k = models_cfg.get("retrieval", {}).get("initial_top_k", 20)

    embedder = get_embedder()
    query_vec = await embedder.embed_query(request.question)

    # Search Qdrant
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    tenant_filter = _build_tenant_filter(tenant_cfg, request.lang)

    try:
        results = await qdrant.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vec,
            query_filter=tenant_filter,
            limit=initial_top_k,
            with_payload=True,
        )
    finally:
        await qdrant.close()

    if not results.points:
        return QueryResponse(answer_chunks=[], reranked=False)

    tv = _tenant_version_float(tenant_cfg)
    payloads = []
    texts = []
    point_ids: list[str] = []
    for point in results.points:
        p = dict(point.payload or {})
        if not _include_changelog_pure(p, tv):
            continue
        payloads.append(p)
        texts.append(p.get("text", ""))
        point_ids.append(str(point.id))

    if not payloads:
        return QueryResponse(answer_chunks=[], reranked=False)

    # Rerank every vector candidate (not a truncated prefix). Previously we used
    # top_n=min(initial_top_k, 15), which dropped the last 5 of 20 hits — correct
    # chunks often sat at 16–20 by embedding score and could never surface.
    reranker = get_reranker(request.reranker)
    rerank_top_n = len(texts)
    rerank_results = await reranker.rerank(
        request.question, texts, top_n=rerank_top_n
    )

    # Post-rerank: title-matching boost + document diversity
    boosted = _apply_title_boost(request.question, rerank_results, payloads)
    diverse = _apply_doc_diversity(boosted, payloads, max_per_doc=2, top_k=request.top_k)

    # Build answer chunks
    answer_chunks: list[AnswerChunk] = []
    for idx, score in diverse:
        p = payloads[idx]
        answer_chunks.append(AnswerChunk(
            text=p.get("text", ""),
            score=score,
            source_doc=p.get("source_doc", ""),
            source_page=p.get("source_page"),
            source_section=p.get("source_section"),
            has_image_caption=p.get("has_image_caption", False),
            chunk_id=point_ids[idx] if idx < len(point_ids) else "",
        ))

    # WP15 §15.2: unique source_doc from top-(pre-rerank) set, minus answer chunk docs
    answer_doc_names = {c.source_doc for c in answer_chunks if c.source_doc}
    related_docs: list[RelatedDoc] = []
    for p in payloads:
        doc_name = p.get("source_doc", "")
        if doc_name and doc_name not in answer_doc_names:
            related_docs.append(RelatedDoc(
                doc=doc_name,
                relevance=f"Sección: {p.get('source_section', 'N/A')}",
            ))
            answer_doc_names.add(doc_name)
        if len(related_docs) >= 5:
            break

    return QueryResponse(
        answer_chunks=answer_chunks,
        related_docs=related_docs,
        reranked=True,
    )
