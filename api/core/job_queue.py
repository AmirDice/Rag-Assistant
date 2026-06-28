"""In-process async job queue with bounded workers and per-job status.

A lightweight background-task queue for the FastAPI event loop: enqueue an async
callable, a pool of workers pulls jobs FIFO and runs them, and each job's status
(queued → running → done/error) + result is tracked in a registry you can poll.

Used for fire-and-forget work (re-indexing, batch embedding, etc.) without an
external broker. For multi-process / durable queues, swap this for Celery/RQ +
Redis behind the same enqueue/get interface.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# A job body: a zero-arg async callable returning a JSON-serializable dict.
JobFunc = Callable[[], Awaitable[dict]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | error
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }


class JobQueue:
    def __init__(self, workers: int = 2, max_history: int = 200) -> None:
        self._workers = max(1, int(workers))
        self._max_history = int(max_history)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._funcs: dict[str, JobFunc] = {}
        self._tasks: list[asyncio.Task] = []
        self._started = False

    def start(self) -> None:
        """Launch worker tasks on the running event loop (idempotent)."""
        if self._started:
            return
        for i in range(self._workers):
            self._tasks.append(asyncio.create_task(self._worker(i), name=f"jobqueue-worker-{i}"))
        self._started = True
        logger.info("JobQueue started with %d workers", self._workers)

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()
        self._started = False

    async def _worker(self, idx: int) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                job = self._jobs.get(job_id)
                func = self._funcs.pop(job_id, None)
                if job is None or func is None:
                    continue
                job.status = "running"
                job.started_at = _now()
                try:
                    job.result = await func()
                    job.status = "done"
                except Exception as exc:  # noqa: BLE001 - record and continue
                    job.error = str(exc)
                    job.status = "error"
                    logger.warning("Job %s (%s) failed: %s", job.id, job.kind, exc)
                finally:
                    job.finished_at = _now()
            finally:
                self._queue.task_done()

    def enqueue(self, kind: str, func: JobFunc) -> Job:
        if not self._started:
            self.start()
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        self._jobs[job.id] = job
        self._funcs[job.id] = func
        # Trim oldest finished jobs beyond the history cap.
        while len(self._jobs) > self._max_history:
            self._jobs.popitem(last=False)
        self._queue.put_nowait(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(reversed(self._jobs.values()))

    async def join(self) -> None:
        """Wait until all queued jobs have been processed (mainly for tests)."""
        await self._queue.join()


_instance: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    global _instance
    if _instance is None:
        _instance = JobQueue(workers=2)
    return _instance
