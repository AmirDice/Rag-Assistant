"""SQLite backing store: which Phase 3 output directories (by SHA-256) are already indexed."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class IndexedCallRecord:
    source_file_hash: str
    indexed_at: str
    source_file: str
    points_upserted: int


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS indexed_calls (
            source_file_hash TEXT PRIMARY KEY,
            indexed_at TEXT NOT NULL,
            source_file TEXT NOT NULL DEFAULT '',
            points_upserted INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()


def is_hash_indexed(conn: sqlite3.Connection, source_file_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM indexed_calls WHERE source_file_hash = ? LIMIT 1",
        (source_file_hash,),
    ).fetchone()
    return row is not None


def mark_indexed(
    conn: sqlite3.Connection,
    *,
    source_file_hash: str,
    source_file: str,
    points_upserted: int,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO indexed_calls (source_file_hash, indexed_at, source_file, points_upserted)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(source_file_hash) DO UPDATE SET
            indexed_at = excluded.indexed_at,
            source_file = excluded.source_file,
            points_upserted = excluded.points_upserted
        """,
        (source_file_hash, now, source_file, points_upserted),
    )
    conn.commit()


def open_db(path: Path) -> sqlite3.Connection:
    conn = _connect(path)
    init_schema(conn)
    return conn
