"""Phase 4 — index CallAnalysis JSONs into Qdrant (RAG pairs + issue summary).

Reads ``output/{sha256}/CALL-*.json`` from the Phase 3 layout. Skips files whose
hash is already recorded in the SQLite dedup store.

Usage::

    python -m modules.audio_pipeline.indexer --output-dir modules/audio_pipeline/output

Configuration: ``api.core.settings.Settings`` (``.env`` / pydantic-settings), same Qdrant URL
and active embedding model as document ingestion:

- ``AUDIO_RAG_QDRANT_COLLECTION`` (default: ``audio_calls``)
- ``AUDIO_INDEX_SQLITE_PATH`` (optional; default under ``DATA_DIR``)
- ``AUDIO_PIPELINE_DEDUP_THRESHOLD``, ``AUDIO_ISSUE_DEDUP_THRESHOLD``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from modules.audio_pipeline._env import load_repo_dotenv
from modules.audio_pipeline.call_index_db import is_hash_indexed, mark_indexed, open_db
from modules.audio_pipeline.calls_catalog import open_catalog_db, upsert_call
from modules.audio_pipeline.schemas import CallAnalysis

DEFAULT_CALLS_AGENT_ID = "call_audio"

load_repo_dotenv()

# Ensure repo root importable when run as ``python modules/audio_pipeline/indexer.py``
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.core.embedder import get_active_embedding_dimensions, get_embedder  # noqa: E402
from api.core.settings import get_settings  # noqa: E402

logger = logging.getLogger(__name__)

_POINT_NS = uuid.UUID("8e58f1c8-4c8d-5b2e-9a1f-0c3e7d2b6a01")


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "output"


def _audio_sqlite_path(settings) -> Path:
    raw = (settings.audio_index_sqlite_path or "").strip()
    if raw:
        return Path(raw).resolve()
    return Path(settings.data_dir).resolve() / "audio_indexed_calls.sqlite"


def _calls_catalog_path(settings) -> Path:
    raw = (getattr(settings, "audio_calls_catalog_sqlite_path", "") or "").strip()
    if raw:
        return Path(raw).resolve()
    return Path(settings.data_dir).resolve() / "audio_calls_catalog.sqlite"


def _point_id_rag(source_file_hash: str, call_id: str, index: int) -> str:
    return str(
        uuid.uuid5(_POINT_NS, f"rag:{source_file_hash}:{call_id}:{index}")
    )


def _point_id_issue(source_file_hash: str, call_id: str) -> str:
    return str(uuid.uuid5(_POINT_NS, f"issue:{source_file_hash}:{call_id}"))


async def ensure_audio_collection(collection_name: str, *, dim: int) -> None:
    """Create a dense-only cosine collection for call embeddings (same dim as corpus embedder)."""
    settings = get_settings()
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        cols = await qdrant.get_collections()
        names = [c.name for c in cols.collections]
        if collection_name in names:
            return
        await qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection %s (dim=%d, COSINE)", collection_name, dim)
    finally:
        await qdrant.close()


async def _max_similarity(
    qdrant,
    *,
    collection: str,
    vector: list[float],
    point_kind: str,
    limit: int = 3,
) -> float:
    """Return max similarity score among top-``limit`` neighbors with same ``point_kind``."""
    flt = Filter(
        must=[
            FieldCondition(key="point_kind", match=MatchValue(value=point_kind)),
        ]
    )
    res = await qdrant.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        query_filter=flt,
        with_payload=False,
    )
    points = list(res.points)
    if not points:
        return 0.0
    scores = [float(p.score or 0.0) for p in points]
    return max(scores)


async def _upsert_point(
    qdrant,
    *,
    collection: str,
    point_id: str,
    vector: list[float],
    payload: dict[str, Any],
) -> None:
    await qdrant.upsert(
        collection_name=collection,
        points=[
            PointStruct(id=point_id, vector=vector, payload=payload),
        ],
    )


def _safe_confidence(v: Any) -> float:
    try:
        x = float(v)
        return max(0.0, min(1.0, x))
    except (TypeError, ValueError):
        return 0.5


def _payload_base(ca: CallAnalysis) -> dict[str, Any]:
    return {
        "agent_id": "call_audio",
        "kind": "call_analysis",
        "call_id": ca.call_id,
        "source_file": ca.source_file,
        "source_file_hash": ca.source_file_hash,
        "farmacia": ca.farmacia,
        "timestamp_start": ca.timestamp_start,
        "timestamp_end": ca.timestamp_end,
        "tags": list(ca.tags),
        "software_features": list(ca.software_features),
    }


async def index_call_analysis(
    ca: CallAnalysis,
    *,
    collection: str,
    embedder,
    qdrant,
    rag_dedup_threshold: float,
    issue_dedup_threshold: float,
) -> tuple[int, int, int, int]:
    """Index one CallAnalysis. Returns (rag_total, rag_indexed, rag_skipped, issue_upserted).

    ``issue_upserted`` is 1 if an issue-summary point was written, 0 if skipped by dedup.
    """
    rag_total = len(ca.rag_qa)
    rag_indexed = 0
    rag_skipped = 0
    issue_upserted = 0

    base = _payload_base(ca)

    for i, pair in enumerate(ca.rag_qa):
        q = (pair.question or "").strip()
        a = (pair.answer or "").strip()
        if not q and not a:
            continue
        full_text = f"{q} {a}".strip()
        if not full_text:
            continue

        dedup_q = q if q else full_text[:2000]
        q_vecs = await embedder.embed([dedup_q])
        q_vec = q_vecs[0]
        sim = await _max_similarity(
            qdrant,
            collection=collection,
            vector=q_vec,
            point_kind="rag_qa",
        )
        if sim > rag_dedup_threshold:
            rag_skipped += 1
            logger.debug(
                "Dedup skip rag_qa idx=%d call=%s sim=%.4f",
                i,
                ca.call_id,
                sim,
            )
            continue

        full_vecs = await embedder.embed([full_text])
        f_vec = full_vecs[0]

        pid = _point_id_rag(ca.source_file_hash, ca.call_id, i)
        payload = {
            **base,
            "point_kind": "rag_qa",
            "category": pair.category,
            "confidence": _safe_confidence(pair.confidence),
            "rag_pair_index": i,
            "indexed_text": full_text,
        }
        await _upsert_point(
            qdrant,
            collection=collection,
            point_id=pid,
            vector=f_vec,
            payload=payload,
        )
        rag_indexed += 1

    p_short = (ca.problema_corto or "").strip()
    p_res = (ca.resolucion or "").strip()
    issue_text = f"{p_short}\n\n{p_res}".strip()
    if issue_text:
        issue_vecs = await embedder.embed([issue_text])
        issue_vec = issue_vecs[0]
        sim_issue = await _max_similarity(
            qdrant,
            collection=collection,
            vector=issue_vec,
            point_kind="issue_summary",
        )
        if sim_issue > issue_dedup_threshold:
            logger.debug(
                "Dedup skip issue_summary call=%s sim=%.4f",
                ca.call_id,
                sim_issue,
            )
        else:
            pid_i = _point_id_issue(ca.source_file_hash, ca.call_id)
            payload_i = {
                **base,
                "point_kind": "issue_summary",
                "category": "",
                "confidence": None,
                "rag_pair_index": None,
                "indexed_text": issue_text,
            }
            await _upsert_point(
                qdrant,
                collection=collection,
                point_id=pid_i,
                vector=issue_vec,
                payload=payload_i,
            )
            issue_upserted = 1

    return rag_total, rag_indexed, rag_skipped, issue_upserted


async def _index_one_hash_dir(
    hash_dir: Path,
    *,
    collection: str,
    embedder,
    qdrant,
    conn,
    rag_dedup_threshold: float,
    issue_dedup_threshold: float,
    catalog_conn=None,
    agent_id: str = DEFAULT_CALLS_AGENT_ID,
) -> int:
    """Index all CALL-*.json under hash_dir. Returns total Qdrant points upserted."""
    call_files = sorted(hash_dir.glob("CALL-*.json"))
    if not call_files:
        logger.warning("No CALL-*.json under %s", hash_dir)
        return 0

    source_file_hash = hash_dir.name
    already_indexed = is_hash_indexed(conn, source_file_hash)

    total_points = 0
    first_source_file = ""

    for path in call_files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        ca = CallAnalysis.model_validate(raw)
        if not first_source_file:
            first_source_file = ca.source_file

        # Keep the catalog in sync even if Qdrant dedup already marked this
        # hash as indexed (covers backfill of pre-existing calls).
        if catalog_conn is not None:
            try:
                upsert_call(catalog_conn, agent_id=agent_id, ca=ca)
            except Exception:  # pragma: no cover - catalog must never break indexing
                logger.exception("Failed to upsert call %s into catalog", ca.call_id)

        if already_indexed:
            continue

        r_tot, r_idx, r_skip, iss_up = await index_call_analysis(
            ca,
            collection=collection,
            embedder=embedder,
            qdrant=qdrant,
            rag_dedup_threshold=rag_dedup_threshold,
            issue_dedup_threshold=issue_dedup_threshold,
        )
        points_this = r_idx + iss_up
        total_points += points_this

        logger.info(
            '{"call_id": %r, "rag_pairs_total": %d, "rag_pairs_indexed": %d, "rag_pairs_skipped_dedup": %d, "issue_summary_upserted": %d}',
            ca.call_id,
            r_tot,
            r_idx,
            r_skip,
            iss_up,
        )

    if already_indexed:
        logger.info("Skip Qdrant (already indexed), catalog synced: %s", source_file_hash)
    else:
        mark_indexed(
            conn,
            source_file_hash=source_file_hash,
            source_file=first_source_file or "",
            points_upserted=total_points,
        )
        logger.info(
            "Marked indexed hash=%s points=%d source_file=%r",
            source_file_hash,
            total_points,
            first_source_file,
        )

    return total_points


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Index Phase 3 CallAnalysis JSON into Qdrant")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Phase 3 output root (contains {sha256}/ subdirs)",
    )
    parser.add_argument(
        "--hash",
        type=str,
        default="",
        help="Process only this subdirectory (hex hash), instead of all children",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    out_root = args.output_dir.resolve()
    if not out_root.is_dir():
        logger.error("Output directory does not exist: %s", out_root)
        return 1

    settings = get_settings()
    collection = settings.audio_rag_qdrant_collection
    dim = get_active_embedding_dimensions(settings)
    db_path = _audio_sqlite_path(settings)
    catalog_path = _calls_catalog_path(settings)
    logger.info(
        "Audio RAG indexer: collection=%s embed_dim=%d sqlite=%s catalog=%s",
        collection,
        dim,
        db_path,
        catalog_path,
    )

    await ensure_audio_collection(collection, dim=dim)

    embedder = get_embedder()
    conn = open_db(db_path)
    catalog_conn = open_catalog_db(catalog_path)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)

    try:
        if args.hash.strip():
            sub = out_root / args.hash.strip()
            if not sub.is_dir():
                logger.error("Hash directory not found: %s", sub)
                return 2
            await _index_one_hash_dir(
                sub,
                collection=collection,
                embedder=embedder,
                qdrant=qdrant,
                conn=conn,
                rag_dedup_threshold=settings.audio_pipeline_dedup_threshold,
                issue_dedup_threshold=settings.audio_issue_dedup_threshold,
                catalog_conn=catalog_conn,
                agent_id=DEFAULT_CALLS_AGENT_ID,
            )
        else:
            subdirs = [p for p in out_root.iterdir() if p.is_dir()]
            if not subdirs:
                logger.warning("No hash subdirectories under %s", out_root)
                return 0
            for hdir in sorted(subdirs, key=lambda p: p.name):
                await _index_one_hash_dir(
                    hdir,
                    collection=collection,
                    embedder=embedder,
                    qdrant=qdrant,
                    conn=conn,
                    rag_dedup_threshold=settings.audio_pipeline_dedup_threshold,
                    issue_dedup_threshold=settings.audio_issue_dedup_threshold,
                    catalog_conn=catalog_conn,
                    agent_id=DEFAULT_CALLS_AGENT_ID,
                )
    finally:
        await qdrant.close()
        conn.close()
        catalog_conn.close()

    logger.info("Done.")
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
