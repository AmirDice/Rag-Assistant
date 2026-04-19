"""Ingestion endpoints with background processing + pause/resume."""

from __future__ import annotations

from fastapi import APIRouter

from api.core.models import IngestRequest, IngestResponse
from api.pipeline.ingest import (
    ingest_path,
    start_background_ingestion,
    get_ingestion_state,
)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    """Synchronous single-file ingestion (original behavior)."""
    return await ingest_path(req.path, force=req.force)


@router.post("/ingest/start")
async def ingest_start(req: IngestRequest) -> dict:
    """Start background ingestion of a folder with pause/resume support."""
    return await start_background_ingestion(
        req.path,
        force=req.force,
        resume=req.resume,
        workers=req.workers,
    )


@router.get("/ingest/status")
async def ingest_status() -> dict:
    """Get current ingestion progress."""
    return get_ingestion_state().to_dict()


@router.post("/ingest/pause")
async def ingest_pause() -> dict:
    state = get_ingestion_state()
    if state.status != "running":
        return {"error": f"Cannot pause: status is {state.status}"}
    state.pause()
    return {"status": "paused"}


@router.post("/ingest/resume")
async def ingest_resume() -> dict:
    state = get_ingestion_state()
    if state.status != "paused":
        return {"error": f"Cannot resume: status is {state.status}"}
    state.resume()
    return {"status": "running"}


@router.post("/ingest/cancel")
async def ingest_cancel() -> dict:
    state = get_ingestion_state()
    if state.status not in ("running", "paused"):
        return {"error": f"Cannot cancel: status is {state.status}"}
    state.cancel()
    return {"status": "cancelling"}
