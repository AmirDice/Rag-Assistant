"""Ingestion endpoints with background processing + pause/resume."""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.core.models import IngestRequest, IngestResponse
from api.core.settings import get_settings
from api.pipeline.ingest import (
    SUPPORTED_EXTENSIONS,
    ingest_path,
    start_background_ingestion,
    get_ingestion_state,
)

router = APIRouter()

_MAX_UPLOAD_BYTES = 80 * 1024 * 1024
_MAX_FILES_PER_REQUEST = 40


def _safe_upload_filename(name: str) -> str:
    raw = Path(name).name.replace("..", "_").strip() or "document"
    stem = Path(raw).stem
    suf = Path(raw).suffix.lower()
    stem_safe = (
        re.sub(r"[^\w\s\-]", "_", stem, flags=re.UNICODE).strip().replace(" ", "_")[:160]
        or "document"
    )
    return f"{stem_safe}{suf}"[:200]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    """Synchronous single-file ingestion (original behavior)."""
    return await ingest_path(req.path, force=req.force)


@router.post("/ingest/upload")
async def ingest_upload(
    files: list[UploadFile] = File(...),
    force: bool = Form(False),
    workers: int = Form(4),
) -> dict:
    """Save uploaded PDF/DOCX/PPTX/XLSX under docs_dir/uploads/<batch>/ and start background ingest."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    settings = get_settings()
    docs_root = Path(settings.docs_dir)
    docs_root.mkdir(parents=True, exist_ok=True)
    batch = docs_root / "uploads" / uuid.uuid4().hex
    batch.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    n_workers = max(1, min(32, workers))
    try:
        for uf in files[:_MAX_FILES_PER_REQUEST]:
            raw_name = uf.filename or "document"
            safe = _safe_upload_filename(raw_name)
            suf = Path(safe).suffix.lower()
            if suf not in SUPPORTED_EXTENSIONS:
                continue
            body = await uf.read()
            if len(body) > _MAX_UPLOAD_BYTES:
                shutil.rmtree(batch, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MiB: {raw_name}",
                )
            dest = batch / f"{uuid.uuid4().hex[:12]}_{safe}"
            dest.write_bytes(body)
            saved.append(dest.name)
        if not saved:
            shutil.rmtree(batch, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail="No supported files. Allowed: "
                + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
            )
        result = await start_background_ingestion(
            str(batch.resolve()), force=force, workers=n_workers
        )
        result["saved"] = saved
        result["batch_dir"] = str(batch.resolve())
        return result
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(batch, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


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
