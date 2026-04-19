"""Qdrant indexer — upsert chunks with full metadata payload (WP15 §15.1)."""

from __future__ import annotations

import logging
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from api.core.settings import get_settings
from api.core.embedder import get_embedder
from api.core.models import ChunkPayload

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024


async def ensure_collection() -> None:
    """Create the Qdrant collection if it doesn't exist."""
    settings = get_settings()
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        collections = await qdrant.get_collections()
        names = [c.name for c in collections.collections]
        if settings.qdrant_collection not in names:
            await qdrant.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", settings.qdrant_collection)
    finally:
        await qdrant.close()


_ZERO_VEC = [0.0] * 1024


async def index_chunks(chunks: list[ChunkPayload]) -> int:
    """Embed and upsert chunks into Qdrant. Returns number indexed."""
    if not chunks:
        return 0

    settings = get_settings()
    embedder = get_embedder()

    try:
        texts = [c.text for c in chunks]
        embeddings = await embedder.embed(texts)
    except Exception as e:
        logger.error("Embedding failed for %d chunks: %s", len(chunks), e)
        return 0

    points = []
    skipped = 0
    for chunk, vector in zip(chunks, embeddings):
        if vector == _ZERO_VEC:
            skipped += 1
            continue
        payload = chunk.model_dump()
        if payload.get("module_id") is None:
            payload["module_id"] = ""
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload,
        ))

    if skipped:
        logger.warning("Skipped %d chunks with zero vectors", skipped)

    if not points:
        logger.error("No valid embeddings produced for %d chunks", len(chunks))
        return 0

    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await qdrant.upsert(
                collection_name=settings.qdrant_collection,
                points=batch,
            )
        logger.info("Indexed %d chunks into Qdrant (skipped %d)", len(points), skipped)
    finally:
        await qdrant.close()

    return len(points)


async def delete_doc_chunks(doc_id: str) -> int:
    """Remove all chunks for a document (for re-ingestion)."""
    settings = get_settings()
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        result = await qdrant.delete(
            collection_name=settings.qdrant_collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        logger.info("Deleted chunks for doc_id=%s", doc_id)
        return 0  # Qdrant delete doesn't return count
    finally:
        await qdrant.close()


async def get_collection_stats() -> dict:
    """Return collection info for /stats endpoint."""
    settings = get_settings()
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        info = await qdrant.get_collection(settings.qdrant_collection)
        total = int(info.points_count or 0)
        # Some Qdrant builds leave points_count unset; exact count is reliable.
        if total == 0:
            try:
                cr = await qdrant.count(
                    collection_name=settings.qdrant_collection,
                    exact=True,
                )
                total = int(getattr(cr, "count", 0) or 0)
            except Exception:
                pass
        if total == 0:
            iv = getattr(info, "indexed_vectors_count", None)
            if iv is not None:
                total = int(iv or 0)
        vc = info.vectors_count
        if (vc is None or vc == 0) and total:
            vc = total
        return {
            "total_chunks": total,
            "vectors_count": int(vc or 0),
            "status": info.status.value if info.status else "unknown",
        }
    except Exception:
        return {"total_chunks": 0, "vectors_count": 0, "status": "not_found"}
    finally:
        await qdrant.close()
