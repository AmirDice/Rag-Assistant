"""Full ingestion orchestrator: parallel workers, pause/resume, failure checkpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from api.core.settings import get_settings
from api.core.models import DocumentMeta, IngestResponse
from api.pipeline.classifier import classify_document
from api.pipeline.converter import convert_document
from api.pipeline.chunker import chunk_document
from api.pipeline.image_pipeline import process_images
from api.pipeline.indexer import index_chunks, delete_doc_chunks, ensure_collection

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}
CHECKPOINT_NAME = "ingest_checkpoint.json"
_RECENT_SUCCESS = 30
_RECENT_ERRORS = 50


def _checkpoint_path() -> Path:
    return Path(get_settings().data_dir) / CHECKPOINT_NAME


def load_checkpoint() -> Optional[dict[str, Any]]:
    p = _checkpoint_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_checkpoint(data: dict[str, Any]) -> None:
    p = _checkpoint_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_checkpoint() -> None:
    p = _checkpoint_path()
    if p.exists():
        p.unlink()


def _append_failed(root_path: str, fp: Path, error: str) -> None:
    ck = load_checkpoint() or {"root_path": root_path, "failed": []}
    ck["root_path"] = root_path
    resolved = str(fp.resolve())
    failed = ck.setdefault("failed", [])
    failed[:] = [e for e in failed if e.get("path") != resolved]
    failed.append({
        "path": resolved,
        "name": fp.name,
        "error": error,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    _write_checkpoint(ck)


def _checkpoint_remove_success(fp: Path) -> None:
    ck = load_checkpoint()
    if not ck or not ck.get("failed"):
        return
    resolved = str(fp.resolve())
    failed = ck["failed"]
    failed[:] = [e for e in failed if e.get("path") != resolved]
    if not failed:
        clear_checkpoint()
    else:
        _write_checkpoint(ck)


class IngestionState:
    """Shared mutable state for background ingestion with pause/resume."""

    def __init__(self) -> None:
        self.status: str = "idle"
        self.total_files: int = 0
        self.processed: int = 0
        self.skipped: int = 0
        self.failed_count: int = 0
        self.chunks_created: int = 0
        self.errors: list[str] = []
        self.failed_files: list[dict[str, str]] = []
        self.succeeded_recent: deque[str] = deque(maxlen=_RECENT_SUCCESS)
        self.current_file: str = ""
        self.workers_used: int = 1
        self.resume_mode: bool = False
        self.root_path: str = ""
        self.pause_event: asyncio.Event = asyncio.Event()
        self.pause_event.set()
        self._cancel: bool = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

    def pause(self) -> None:
        self.status = "paused"
        self.pause_event.clear()

    def resume(self) -> None:
        self.status = "running"
        self.pause_event.set()

    def cancel(self) -> None:
        self._cancel = True
        self.pause_event.set()

    def to_dict(self) -> dict[str, Any]:
        total = self.total_files
        done = self.processed + self.skipped + self.failed_count
        pct = round(done / total * 100, 1) if total else 0.0
        ck = load_checkpoint()
        ck_failed = len(ck.get("failed", [])) if ck else 0
        return {
            "status": self.status,
            "total_files": total,
            "available_files": total,
            "processed": self.processed,
            "skipped": self.skipped,
            "succeeded": self.processed,
            "failed_count": self.failed_count,
            "failed_files": list(self.failed_files)[-_RECENT_ERRORS:],
            "succeeded_recent": list(self.succeeded_recent),
            "chunks_created": self.chunks_created,
            "current_file": self.current_file,
            "workers": self.workers_used,
            "resume_mode": self.resume_mode,
            "root_path": self.root_path,
            "errors_count": len(self.errors),
            "last_errors": self.errors[-5:] if self.errors else [],
            "progress_pct": pct,
            "checkpoint_failed_pending": ck_failed,
            "can_resume": ck_failed > 0,
        }


_state = IngestionState()


def get_ingestion_state() -> IngestionState:
    return _state


def _collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if path.is_dir():
        return sorted(
            fp for fp in path.rglob("*")
            if fp.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return []


async def ingest_path(path_str: str, force: bool = False) -> IngestResponse:
    """Ingest a single file or folder synchronously (blocking API call)."""
    path = Path(path_str)
    errors: list[str] = []
    docs_processed = 0
    chunks_created = 0

    await ensure_collection()

    if path.is_file():
        result = await _ingest_file(path, force)
        if result["error"]:
            errors.append(result["error"])
        else:
            docs_processed = 1
            chunks_created = result["chunks"]
    elif path.is_dir():
        for fp in sorted(path.rglob("*")):
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            result = await _ingest_file(fp, force)
            if result["error"]:
                errors.append(result["error"])
            else:
                docs_processed += 1
                chunks_created += result["chunks"]
    else:
        errors.append(f"Path not found: {path_str}")

    _save_ingestion_log(docs_processed, chunks_created, errors)

    status = "completed" if not errors else "completed_with_errors"
    return IngestResponse(
        status=status,
        docs_processed=docs_processed,
        chunks_created=chunks_created,
        errors=errors,
    )


async def start_background_ingestion(
    path_str: str,
    force: bool = False,
    resume: bool = False,
    workers: Optional[int] = None,
) -> dict:
    """Launch ingestion as a background task with pause/resume and parallel workers."""
    global _state

    if _state.status in ("running", "paused"):
        return {"error": "Ingestion already in progress. Pause or cancel first."}

    if not resume and not (path_str or "").strip():
        return {"error": "path is required unless resume=true"}

    settings = get_settings()
    n_workers = workers if workers is not None else settings.ingestion_workers
    n_workers = max(1, min(32, n_workers))

    files: list[Path] = []
    root_display = path_str
    use_force = force

    if resume:
        ck = load_checkpoint()
        if not ck or not ck.get("failed"):
            return {"error": "No failed files in checkpoint — nothing to resume."}
        root_display = ck.get("root_path", path_str)
        for entry in ck["failed"]:
            p = Path(entry["path"])
            if p.exists():
                files.append(p)
        if not files:
            missing = [e.get("path", "") for e in ck["failed"]]
            ck["failed"] = []
            clear_checkpoint()
            return {
                "error": "Checkpoint paths no longer exist on disk.",
                "pruned_paths": missing,
            }
        use_force = True
    else:
        path = Path(path_str)
        if not path.exists():
            return {"error": f"Path not found: {path_str}"}
        files = _collect_files(path)
        if not files:
            return {"error": "No supported files found"}
        clear_checkpoint()

    _state = IngestionState()
    _state.total_files = len(files)
    _state.status = "running"
    _state.workers_used = n_workers
    _state.resume_mode = resume
    _state.root_path = root_display

    async def _run() -> None:
        try:
            await ensure_collection()
            settings_inner = get_settings()
            corpus_dir = Path(settings_inner.corpus_dir)
            sem = asyncio.Semaphore(n_workers)

            async def process_one(fp: Path) -> None:
                async with sem:
                    if _state._cancel:
                        return
                    await _state.pause_event.wait()
                    if _state._cancel:
                        return

                    with _state._lock:
                        doc_id = fp.stem
                        meta_path = corpus_dir / doc_id / "metadata.json"
                        if not use_force and meta_path.exists():
                            _state.skipped += 1
                            logger.info("Skipping (already ingested): %s", fp.name)
                            return
                        _state.current_file = fp.name

                    result = await _ingest_file(fp, use_force)

                    with _state._lock:
                        if result["error"]:
                            err = result["error"]
                            _state.errors.append(err)
                            _state.failed_count += 1
                            _state.failed_files.append({"file": fp.name, "error": err})
                            _append_failed(root_display, fp, err)
                            logger.error("Ingest failed: %s — %s", fp.name, err)
                        elif result["chunks"] > 0:
                            _state.processed += 1
                            _state.chunks_created += result["chunks"]
                            _state.succeeded_recent.append(fp.name)
                            if _state.resume_mode:
                                _checkpoint_remove_success(fp)
                        else:
                            _state.skipped += 1
                            if _state.resume_mode:
                                _checkpoint_remove_success(fp)

            await asyncio.gather(*(process_one(fp) for fp in files))

            if _state._cancel:
                _state.status = "cancelled"
            elif _state.status == "running":
                _state.status = "completed"
            _state.current_file = ""

            if _state.status == "completed" and _state.failed_count == 0:
                clear_checkpoint()

            _save_ingestion_log(
                _state.processed, _state.chunks_created, _state.errors
            )
            logger.info(
                "Background ingestion finished: %d ok, %d skipped, %d failed (workers=%d)",
                _state.processed, _state.skipped, _state.failed_count, n_workers,
            )
        except Exception as e:
            _state.status = "error"
            _state.errors.append(str(e))
            logger.exception("Background ingestion failed")

    _state._task = asyncio.create_task(_run())
    return {"status": "started", "total_files": len(files), "workers": n_workers, "resume": resume}


async def _ingest_file(filepath: Path, force: bool) -> dict:
    """Process a single document through the full pipeline."""
    try:
        settings = get_settings()
        corpus_dir = Path(settings.corpus_dir)
        doc_id = filepath.stem
        output_dir = corpus_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Ingesting: %s", filepath.name)

        markdown, images = convert_document(filepath, output_dir)
        if not markdown.strip():
            return {"chunks": 0, "error": f"Empty conversion result: {filepath.name}"}

        (output_dir / "content.md").write_text(markdown, encoding="utf-8")

        meta = classify_document(filepath, markdown)
        meta.image_count = len(images)

        if meta.discard:
            logger.info("Discarding %s (doc_type=%s)", filepath.name, meta.doc_type)
            return {"chunks": 0, "error": None}

        if images:
            markdown, caption_count = await process_images(images, markdown, doc_id=doc_id)
            if caption_count:
                (output_dir / "content_with_captions.md").write_text(
                    markdown, encoding="utf-8"
                )

        chunks = chunk_document(markdown, meta)
        if not chunks:
            return {"chunks": 0, "error": f"No chunks produced: {filepath.name}"}

        chunks_path = output_dir / "chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")

        if force:
            await delete_doc_chunks(doc_id)

        indexed = await index_chunks(chunks)

        meta_path = output_dir / "metadata.json"
        meta_path.write_text(
            meta.model_dump_json(indent=2),
            encoding="utf-8",
        )

        logger.info("Indexed %s → %d chunks", filepath.name, indexed)
        return {"chunks": indexed, "error": None}

    except Exception as e:
        logger.exception("Failed to ingest %s", filepath.name)
        return {"chunks": 0, "error": f"{filepath.name}: {e}"}


def _save_ingestion_log(docs: int, chunks: int, errors: list[str]) -> None:
    settings = get_settings()
    log_path = Path(settings.data_dir) / "ingestion_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs_processed": docs,
        "chunks_created": chunks,
        "errors": errors,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
