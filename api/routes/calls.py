"""Endpoints for analysed call audio (list/detail/audio/upload/delete).

Exposes the SQLite catalog maintained by the audio pipeline indexer so the UI
can list and view calls, stream the original audio for in-browser playback,
upload new calls (orchestrates WhisperX → analyser → indexer in a background
task), and delete analysed calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from api.core.settings import get_settings
from modules.audio_pipeline.calls_catalog import (
    CallCatalogRow,
    catalog_stats,
    delete_call as catalog_delete_call,
    distinct_farmacias,
    distinct_tags,
    get_call,
    list_calls,
    open_catalog_db,
)
from modules.audio_pipeline.schemas import CallAnalysis

logger = logging.getLogger(__name__)

router = APIRouter()

# Upper bound for audio uploads. WhisperX large-v3 on a 200MB file is already a
# long job — bigger files should go through the CLI / batch path.
UPLOAD_SIZE_LIMIT_BYTES = 200 * 1024 * 1024

# Extensions we serve as audio. Must match what the upload endpoint accepts.
_AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4")

CALL_STAGE_PROGRESS: dict[str, float] = {
    "uploaded":     5.0,
    "transcribing": 25.0,
    "analyzing":    65.0,
    "indexing":     90.0,
    "completed":    100.0,
    "failed":       100.0,
}


@dataclass
class CallUploadJob:
    job_id: str
    agent_id: str
    file_name: str
    source_file_hash: str
    status: str = "queued"        # queued | running | completed | failed
    stage: str = "uploaded"
    progress_pct: float = 0.0
    calls_created: int = 0
    error: str | None = None
    call_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


_call_jobs: dict[str, CallUploadJob] = {}
_call_jobs_lock = threading.Lock()


def _set_job_stage(job: CallUploadJob, stage: str, *, status: str | None = None, error: str | None = None) -> None:
    with _call_jobs_lock:
        job.stage = stage
        job.progress_pct = CALL_STAGE_PROGRESS.get(stage, job.progress_pct)
        if status:
            job.status = status
        if error:
            job.error = error
        job.updated_at = datetime.now(timezone.utc).isoformat()

# MIME map for FileResponse. HTML5 <audio> picks the right decoder from this.
_AUDIO_MIME = {
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    ".m4a":  "audio/mp4",
    ".mp4":  "audio/mp4",
    ".ogg":  "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


def _catalog_path() -> Path:
    settings = get_settings()
    raw = (getattr(settings, "audio_calls_catalog_sqlite_path", "") or "").strip()
    if raw:
        return Path(raw).resolve()
    return Path(settings.data_dir).resolve() / "audio_calls_catalog.sqlite"


def _audio_uploads_dir() -> Path:
    settings = get_settings()
    raw = (getattr(settings, "audio_uploads_dir", "") or "").strip()
    if raw:
        return Path(raw).resolve()
    return Path(settings.data_dir).resolve() / "audio_uploads"


def _audio_output_dir() -> Path:
    settings = get_settings()
    raw = (getattr(settings, "audio_output_dir", "") or "").strip()
    if raw:
        return Path(raw).resolve()
    # Default to the pipeline's own output/ directory.
    return (Path(__file__).resolve().parents[2] / "modules" / "audio_pipeline" / "output").resolve()


def _row_to_json(row: CallCatalogRow) -> dict:
    return asdict(row)


def _find_audio_file(source_file_hash: str, source_file: str) -> Path | None:
    """Locate the audio file on disk.

    Order of precedence:
      1. ``{AUDIO_UPLOADS_DIR}/{hash}{ext}`` for any accepted extension.
      2. ``source_file`` if it is an absolute path that still exists.
    """
    if source_file_hash:
        uploads = _audio_uploads_dir()
        for ext in _AUDIO_EXTS:
            candidate = uploads / f"{source_file_hash}{ext}"
            if candidate.is_file():
                return candidate

    if source_file:
        p = Path(source_file)
        if p.is_file():
            return p

    return None


def _read_call_json(source_file_hash: str, call_id: str) -> CallAnalysis | None:
    """Read and validate the CallAnalysis JSON for a given (hash, call_id)."""
    root = _audio_output_dir()
    hash_dir = root / source_file_hash
    if not hash_dir.is_dir():
        return None
    # Fast path: by call_id naming convention (CALL-001.json).
    candidate = hash_dir / f"{call_id}.json"
    paths = [candidate] if candidate.is_file() else sorted(hash_dir.glob("CALL-*.json"))
    for path in paths:
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if raw.get("call_id") != call_id:
            continue
        try:
            return CallAnalysis.model_validate(raw)
        except Exception:
            return None
    return None


def _authorised_agent(request: Request, agent_id: str | None) -> str | None:
    del request
    return agent_id


@router.get("/calls")
async def list_calls_endpoint(
    request: Request,
    agent_id: str | None = Query(default=None),
    tag: str | None = Query(default=None),
    farmacia: str | None = Query(default=None),
    resolved: bool | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """List analysed calls (thin rows, no transcript)."""
    effective_agent = _authorised_agent(request, agent_id)

    conn = open_catalog_db(_catalog_path())
    try:
        rows = list_calls(
            conn,
            agent_id=effective_agent,
            tag=tag,
            farmacia=farmacia,
            resolved=resolved,
            search=search,
            limit=limit,
            offset=offset,
        )
        return {
            "calls": [_row_to_json(r) for r in rows],
            "count": len(rows),
            "limit": limit,
            "offset": offset,
        }
    finally:
        conn.close()


@router.get("/calls/stats")
async def calls_stats_endpoint(
    request: Request,
    agent_id: str | None = Query(default=None),
) -> dict:
    """Aggregate stats for StatCards in the UI."""
    effective_agent = _authorised_agent(request, agent_id)

    conn = open_catalog_db(_catalog_path())
    try:
        return catalog_stats(conn, agent_id=effective_agent)
    finally:
        conn.close()


@router.get("/calls/filters")
async def calls_filters_endpoint(
    request: Request,
    agent_id: str | None = Query(default=None),
) -> dict:
    """Distinct values for filter dropdowns (tags, farmacias)."""
    effective_agent = _authorised_agent(request, agent_id)

    conn = open_catalog_db(_catalog_path())
    try:
        return {
            "farmacias": distinct_farmacias(conn, agent_id=effective_agent),
            "tags": distinct_tags(conn, agent_id=effective_agent),
        }
    finally:
        conn.close()


@router.get("/calls/{id}")
async def call_detail_endpoint(id: str, request: Request) -> dict:
    """Full CallAnalysis payload (with transcript) + catalog row metadata."""

    conn = open_catalog_db(_catalog_path())
    try:
        row = get_call(conn, id=id)
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Call not found")
    _authorised_agent(request, row.agent_id)

    ca = _read_call_json(row.source_file_hash, row.call_id)
    if ca is None:
        raise HTTPException(
            status_code=404,
            detail="Call analysis JSON missing on disk; catalog row is stale",
        )
    payload = ca.model_dump()
    audio_file = _find_audio_file(row.source_file_hash, row.source_file)
    payload["id"] = row.id
    payload["audio_available"] = audio_file is not None
    payload["agent_id"] = row.agent_id
    payload["indexed_at"] = row.indexed_at
    return payload


@router.get("/calls/{id}/audio")
async def call_audio_endpoint(id: str, request: Request) -> FileResponse:
    """Stream the original audio file. Supports Range requests natively via FileResponse."""

    conn = open_catalog_db(_catalog_path())
    try:
        row = get_call(conn, id=id)
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Call not found")
    _authorised_agent(request, row.agent_id)

    audio_file = _find_audio_file(row.source_file_hash, row.source_file)
    if audio_file is None:
        raise HTTPException(status_code=404, detail="Audio file not available")
    mime = _AUDIO_MIME.get(audio_file.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path=str(audio_file),
        media_type=mime,
        filename=f"{row.call_id}{audio_file.suffix}",
    )


async def _run_call_pipeline(
    job_id: str,
    *,
    audio_path: Path,
    source_file_hash: str,
    agent_id: str,
    tenant_id: str,
) -> None:
    """Background worker: transcribe → segment → analyse → index a single audio file."""
    from modules.audio_pipeline.analyzer import analyze_conversation
    from modules.audio_pipeline.calls_catalog import upsert_call
    from modules.audio_pipeline.indexer import (
        DEFAULT_CALLS_AGENT_ID,
        ensure_audio_collection,
        index_call_analysis,
    )
    from modules.audio_pipeline.segmenter import format_mm_ss, split_conversations
    from modules.audio_pipeline.transcriber import (
        ffprobe_duration_seconds,
        transcribe,
    )
    from api.core.embedder import get_active_embedding_dimensions, get_embedder

    with _call_jobs_lock:
        job = _call_jobs.get(job_id)
    if job is None:
        logger.error("Call job %s vanished before starting", job_id)
        return

    _set_job_stage(job, "transcribing", status="running")
    try:
        out_root = _audio_output_dir() / source_file_hash
        out_root.mkdir(parents=True, exist_ok=True)
        whisperx_dir = out_root / "whisperx_raw"
        whisperx_dir.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_running_loop()
        duration_sec = await loop.run_in_executor(
            None, ffprobe_duration_seconds, audio_path
        )
        if duration_sec is None:
            raise RuntimeError("ffprobe could not read audio duration")

        tr = await loop.run_in_executor(None, lambda: transcribe(audio_path, whisperx_dir, tenant_id=tenant_id))
        segments = tr.get("segments") or []
        if not segments:
            raise RuntimeError("WhisperX returned no segments")

        _set_job_stage(job, "analyzing")
        chunks = split_conversations(segments)
        if not chunks:
            raise RuntimeError("No conversations detected above minimum duration")

        settings = get_settings()
        analyses = []
        for i, ch in enumerate(chunks):
            call_id = f"CALL-{i + 1:03d}"
            rel = ch.segments_relative
            ts_start = format_mm_ss(0.0)
            ts_end = format_mm_ss(rel[-1]["end"]) if rel else "00:00"
            ca, _stats = await loop.run_in_executor(
                None,
                lambda rel=rel, call_id=call_id, ts_end=ts_end: analyze_conversation(
                    segments_relative=rel,
                    call_id=call_id,
                    source_file=audio_path.name,
                    source_file_hash=source_file_hash,
                    timestamp_start=ts_start,
                    timestamp_end=ts_end,
                    tenant_id=tenant_id,
                ),
            )
            ca = ca.model_copy(
                update={
                    "call_id": call_id,
                    "source_file": audio_path.name,
                    "source_file_hash": source_file_hash,
                    "timestamp_start": ts_start,
                    "timestamp_end": ts_end,
                }
            )
            out_path = out_root / f"{call_id}.json"
            out_path.write_text(
                json.dumps(ca.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            analyses.append(ca)

        _set_job_stage(job, "indexing")
        collection = settings.audio_rag_qdrant_collection
        dim = get_active_embedding_dimensions(settings)
        await ensure_audio_collection(collection, dim=dim)

        embedder = get_embedder()
        qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        catalog_conn = open_catalog_db(_catalog_path())
        catalog_ids: list[str] = []
        try:
            for ca in analyses:
                await index_call_analysis(
                    ca,
                    collection=collection,
                    embedder=embedder,
                    qdrant=qdrant,
                    rag_dedup_threshold=settings.audio_pipeline_dedup_threshold,
                    issue_dedup_threshold=settings.audio_issue_dedup_threshold,
                )
                catalog_ids.append(
                    upsert_call(
                        catalog_conn,
                        agent_id=agent_id or DEFAULT_CALLS_AGENT_ID,
                        ca=ca,
                    )
                )
        finally:
            await qdrant.close()
            catalog_conn.close()

        with _call_jobs_lock:
            job.call_ids = catalog_ids
            job.calls_created = len(catalog_ids)
        _set_job_stage(job, "completed", status="completed")
        logger.info(
            "Call upload job %s done — hash=%s calls=%d",
            job_id,
            source_file_hash,
            len(catalog_ids),
        )
    except Exception as exc:
        logger.exception("Call upload job %s failed", job_id)
        _set_job_stage(job, "failed", status="failed", error=str(exc))


@router.post("/calls/upload")
async def call_upload_endpoint(
    request: Request,
    file: UploadFile = File(...),
    agent_id: str = Form(...),
) -> dict:
    """Upload an audio file and kick off the transcribe+analyse+index pipeline."""
    _authorised_agent(request, agent_id)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio extension {suffix!r}. Accepted: {', '.join(_AUDIO_EXTS)}",
        )

    uploads_dir = _audio_uploads_dir()
    uploads_dir.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    total_bytes = 0
    tmp_path = uploads_dir / f".incoming-{uuid4().hex}{suffix}"
    try:
        with tmp_path.open("wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > UPLOAD_SIZE_LIMIT_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio exceeds {UPLOAD_SIZE_LIMIT_BYTES // (1024 * 1024)}MB limit",
                    )
                hasher.update(chunk)
                out.write(chunk)
    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    if total_bytes == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload")

    source_file_hash = hasher.hexdigest()
    final_path = uploads_dir / f"{source_file_hash}{suffix}"
    if final_path.exists():
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.rename(final_path)

    job = CallUploadJob(
        job_id=str(uuid4()),
        agent_id=agent_id,
        file_name=file.filename or final_path.name,
        source_file_hash=source_file_hash,
    )
    tenant_id = "default"
    with _call_jobs_lock:
        _call_jobs[job.job_id] = job

    asyncio.create_task(
        _run_call_pipeline(
            job.job_id,
            audio_path=final_path,
            source_file_hash=source_file_hash,
            agent_id=agent_id,
            tenant_id=tenant_id,
        )
    )

    return {"job_id": job.job_id, **job.to_dict()}


@router.get("/calls/jobs/{job_id}")
async def call_job_endpoint(job_id: str, request: Request) -> dict:
    """Poll the status of a background upload job."""
    with _call_jobs_lock:
        job = _call_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    _authorised_agent(request, job.agent_id)
    return job.to_dict()


@router.delete("/calls/{id}")
async def call_delete_endpoint(id: str, request: Request) -> dict:
    """Remove a call from Qdrant + catalog + JSON. Audio is kept unless orphaned.

    Vector deletion runs first; if it fails the catalog/JSON are untouched and a 5xx
    is returned so the caller can retry — this prevents orphaned vectors that would
    keep surfacing in retrieval but be unreachable from the UI.
    """

    catalog_path = _catalog_path()
    conn = open_catalog_db(catalog_path)
    try:
        row = get_call(conn, id=id)
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Call not found")
    _authorised_agent(request, row.agent_id)

    settings = get_settings()
    collection = settings.audio_rag_qdrant_collection

    qdrant: AsyncQdrantClient | None = None
    try:
        qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        await qdrant.delete(
            collection_name=collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="call_id", match=MatchValue(value=row.call_id)),
                        FieldCondition(
                            key="source_file_hash",
                            match=MatchValue(value=row.source_file_hash),
                        ),
                    ]
                )
            ),
        )
    except Exception as exc:
        logger.exception("Qdrant delete failed for call %s", id)
        raise HTTPException(
            status_code=502,
            detail=f"Vector deletion failed; call not deleted: {exc}",
        ) from exc
    finally:
        try:
            if qdrant is not None:
                await qdrant.close()
        except Exception:
            pass

    conn = open_catalog_db(catalog_path)
    try:
        catalog_delete_call(conn, id=id)
        cur = conn.execute(
            "SELECT COUNT(*) AS n FROM calls_catalog WHERE source_file_hash = ?",
            (row.source_file_hash,),
        ).fetchone()
        siblings_remaining = int(cur["n"] or 0) if cur else 0
    finally:
        conn.close()

    out_root = _audio_output_dir() / row.source_file_hash
    json_path = out_root / f"{row.call_id}.json"
    try:
        json_path.unlink(missing_ok=True)
    except OSError:
        logger.exception("Failed to delete %s", json_path)

    audio_removed = False
    if siblings_remaining == 0:
        audio_file = _find_audio_file(row.source_file_hash, row.source_file)
        if audio_file is not None and audio_file.parent == _audio_uploads_dir():
            try:
                audio_file.unlink(missing_ok=True)
                audio_removed = True
            except OSError:
                logger.exception("Failed to delete audio %s", audio_file)

    return {
        "id": id,
        "call_id": row.call_id,
        "deleted": True,
        "audio_removed": audio_removed,
        "siblings_remaining": siblings_remaining,
    }
