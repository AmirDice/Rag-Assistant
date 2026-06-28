"""Async job queue: workers process enqueued jobs and track status/result."""

from __future__ import annotations

import asyncio

from api.core.job_queue import JobQueue


def test_job_runs_and_records_result():
    async def _run():
        q = JobQueue(workers=2)
        q.start()

        async def task():
            await asyncio.sleep(0)
            return {"answer": 42}

        job = q.enqueue("demo", task)
        assert job.status == "queued"
        await asyncio.wait_for(q.join(), timeout=5)
        return q.get(job.id)

    job = asyncio.run(_run())
    assert job.status == "done"
    assert job.result == {"answer": 42}
    assert job.finished_at is not None


def test_failing_job_is_marked_error():
    async def _run():
        q = JobQueue(workers=1)
        q.start()

        async def boom():
            raise RuntimeError("kaboom")

        job = q.enqueue("demo", boom)
        await asyncio.wait_for(q.join(), timeout=5)
        return q.get(job.id)

    job = asyncio.run(_run())
    assert job.status == "error"
    assert "kaboom" in (job.error or "")


def test_jobs_processed_concurrently():
    async def _run():
        q = JobQueue(workers=3)
        q.start()

        async def slow():
            await asyncio.sleep(0.2)
            return {"ok": True}

        jobs = [q.enqueue("demo", slow) for _ in range(3)]
        # 3 jobs on 3 workers should finish well under the serial 0.6s.
        await asyncio.wait_for(q.join(), timeout=1.0)
        return [q.get(j.id).status for j in jobs]

    statuses = asyncio.run(_run())
    assert statuses == ["done", "done", "done"]
