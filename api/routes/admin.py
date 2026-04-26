"""Admin endpoints: /health, /stats, /docs/{doc_id}, /corpus (WP15 §15.2)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Header
from qdrant_client import AsyncQdrantClient

from api.core.models import HealthResponse, StatsResponse
from api.core.settings import get_settings
from api.core.pipeline_config import build_pipeline_config
from api.pipeline.indexer import get_collection_stats

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    result = HealthResponse(api="ok")

    # Check Qdrant
    try:
        qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        await qdrant.get_collections()
        result.qdrant = "ok"
        await qdrant.close()
    except Exception:
        result.qdrant = "error"

    # Check Redis
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        result.redis = "ok"
        await r.close()
    except Exception:
        result.redis = "error"

    return result


@router.get("/stats", response_model=StatsResponse)
async def stats_endpoint() -> StatsResponse:
    settings = get_settings()
    docs_dir = Path(settings.docs_dir)
    corpus_dir = Path(settings.corpus_dir)

    qdrant_stats = await get_collection_stats()
    chunks_n = int(qdrant_stats.get("total_chunks", 0) or 0)
    # Rough vector footprint (1024-d float32) + ~15% payload / HNSW overhead (WP16 dashboard)
    vec_bytes = chunks_n * 1024 * 4
    approximate_index_mb = round(vec_bytes * 1.15 / (1024**2), 2)

    # Count docs from corpus directory metadata files
    by_type: dict[str, int] = {}
    by_module: dict[str, int] = {}
    total_docs = 0
    total_docs_bytes = 0
    last_ingestion = None

    for meta_file in corpus_dir.rglob("metadata.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            total_docs += 1
            dt = meta.get("doc_type", "unknown")
            by_type[dt] = by_type.get(dt, 0) + 1
            mod = meta.get("module_id") or "core"
            by_module[mod] = by_module.get(mod, 0) + 1

            src_raw = str(meta.get("source_file") or "").strip()
            candidates: list[Path] = []
            if src_raw:
                src_path = Path(src_raw)
                if src_path.is_absolute():
                    candidates.append(src_path)
                candidates.append(docs_dir / src_raw)
                if src_raw.startswith("/app/docs/"):
                    candidates.append(docs_dir / src_raw.replace("/app/docs/", "", 1))
                elif src_raw.startswith("app/docs/"):
                    candidates.append(docs_dir / src_raw.replace("app/docs/", "", 1))
            folder_guess = meta_file.parent / src_raw
            if src_raw:
                candidates.append(folder_guess)

            seen: set[str] = set()
            picked_size = 0
            for candidate in candidates:
                key = str(candidate.resolve()) if candidate.exists() else str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                if candidate.exists() and candidate.is_file():
                    picked_size = int(candidate.stat().st_size)
                    break
            total_docs_bytes += picked_size
        except Exception:
            pass

    # Last ingestion from log
    log_path = Path(settings.data_dir) / "ingestion_log.jsonl"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        if lines:
            try:
                last = json.loads(lines[-1])
                last_ingestion = last.get("timestamp")
            except Exception:
                pass

    return StatsResponse(
        total_docs=total_docs,
        total_chunks=qdrant_stats.get("total_chunks", 0),
        by_type=by_type,
        by_module=by_module,
        last_ingestion=last_ingestion,
        approximate_index_mb=approximate_index_mb,
        total_docs_bytes=total_docs_bytes,
    )


@router.get("/config/pipeline")
async def pipeline_config_endpoint() -> dict:
    """WP12 pipeline_config.json v2 — YAML sources merged for export."""
    return build_pipeline_config()


@router.get("/library/documents")
async def list_corpus_documents() -> dict:
    """Indexed documents under corpus/ (metadata + optional media file names)."""
    settings = get_settings()
    corpus_dir = Path(settings.corpus_dir)
    items: list[dict] = []
    for d in sorted(corpus_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        media_dir = d / "media"
        media_files: list[str] = []
        if media_dir.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                media_files.extend(sorted(p.name for p in media_dir.glob(ext)))
        items.append({
            "doc_id": d.name,
            "source_file": meta.get("source_file", ""),
            "doc_type": meta.get("doc_type", ""),
            "module_id": meta.get("module_id"),
            "format": meta.get("format", ""),
            "image_count": meta.get("image_count", 0),
            "media_files": media_files,
        })
    return {"documents": items, "count": len(items)}


@router.get("/docs/{doc_id}")
async def get_document(doc_id: str) -> dict:
    settings = get_settings()
    meta_path = Path(settings.corpus_dir) / doc_id / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


@router.delete("/corpus")
async def delete_corpus(
    x_admin_token: str = Header(alias="X-Admin-Token"),
) -> dict:
    settings = get_settings()
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    try:
        qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        await qdrant.delete_collection(settings.qdrant_collection)
        await qdrant.close()
    except Exception:
        pass

    return {"status": "deleted", "collection": settings.qdrant_collection}
