"""Async job queue API — enqueue background work and poll its status."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.core.job_queue import get_job_queue

router = APIRouter()


@router.get("/jobs")
def list_jobs() -> dict:
    """All tracked jobs (most recent first)."""
    return {"jobs": [j.to_dict() for j in get_job_queue().list()]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = get_job_queue().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@router.post("/jobs/refresh-stats")
def enqueue_refresh_stats() -> dict:
    """Enqueue a background job that recomputes Qdrant collection stats."""
    async def _task() -> dict:
        from api.pipeline.indexer import get_collection_stats

        return await get_collection_stats()

    job = get_job_queue().enqueue("refresh_stats", _task)
    return job.to_dict()
