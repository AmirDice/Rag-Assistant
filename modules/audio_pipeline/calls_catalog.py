"""SQLite catalog of analysed calls — the read-side source of truth for the UI.

This is additive to ``call_index_db`` (which only tracks *which hashes* have
been indexed into Qdrant). The catalog stores one row per ``CallAnalysis`` so
the frontend can list and filter calls without reading JSON files or querying
Qdrant.

Schema intentionally denormalises a few fields (``farmacia``, ``llamante``,
``agent``, ``problema_corto``, ``resolucion_exitosa``, ``tags``) that the
``/calls`` list view renders, keeping ``list_calls`` a single indexed query.
The full ``CallAnalysis`` payload still lives on disk as ``output/{hash}/CALL-*.json``
and is served by the ``/calls/{call_id}`` detail endpoint.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from modules.audio_pipeline.schemas import CallAnalysis


@dataclass(frozen=True)
class CallCatalogRow:
    id: str
    call_id: str
    agent_id: str
    source_file_hash: str
    source_file: str
    farmacia: str
    llamante: str
    agent: str
    problema_corto: str
    resolucion_exitosa: bool
    tags: list[str]
    timestamp_start: str
    timestamp_end: str
    indexed_at: str


def make_catalog_id(source_file_hash: str, call_id: str) -> str:
    """Globally-unique catalog ID.

    ``call_id`` (e.g. ``CALL-001``) is only unique within one analysed audio
    file — every upload restarts the numbering. We prefix with the first 12
    hex chars of the source-file hash to make the row uniquely addressable
    across uploads and safe to use as a URL path segment.
    """
    short = (source_file_hash or "")[:12]
    cid = call_id or ""
    return f"{short}-{cid}" if short else cid


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS calls_catalog (
    id               TEXT PRIMARY KEY,
    call_id          TEXT NOT NULL DEFAULT '',
    agent_id         TEXT NOT NULL DEFAULT '',
    source_file_hash TEXT NOT NULL DEFAULT '',
    source_file      TEXT NOT NULL DEFAULT '',
    farmacia         TEXT NOT NULL DEFAULT '',
    llamante         TEXT NOT NULL DEFAULT '',
    agent            TEXT NOT NULL DEFAULT '',
    problema_corto   TEXT NOT NULL DEFAULT '',
    resolucion_exitosa INTEGER NOT NULL DEFAULT 0,
    tags_json        TEXT NOT NULL DEFAULT '[]',
    timestamp_start  TEXT NOT NULL DEFAULT '',
    timestamp_end    TEXT NOT NULL DEFAULT '',
    indexed_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_catalog_agent ON calls_catalog(agent_id);
CREATE INDEX IF NOT EXISTS idx_calls_catalog_hash  ON calls_catalog(source_file_hash);
CREATE INDEX IF NOT EXISTS idx_calls_catalog_resol ON calls_catalog(resolucion_exitosa);
CREATE INDEX IF NOT EXISTS idx_calls_catalog_when  ON calls_catalog(indexed_at);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    conn.commit()


def open_catalog_db(path: Path) -> sqlite3.Connection:
    conn = _connect(path)
    init_schema(conn)
    return conn


def _row_to_record(row: sqlite3.Row) -> CallCatalogRow:
    try:
        tags = json.loads(row["tags_json"] or "[]")
        if not isinstance(tags, list):
            tags = []
    except (TypeError, ValueError):
        tags = []
    return CallCatalogRow(
        id=row["id"],
        call_id=row["call_id"] or "",
        agent_id=row["agent_id"] or "",
        source_file_hash=row["source_file_hash"] or "",
        source_file=row["source_file"] or "",
        farmacia=row["farmacia"] or "",
        llamante=row["llamante"] or "",
        agent=row["agent"] or "",
        problema_corto=row["problema_corto"] or "",
        resolucion_exitosa=bool(row["resolucion_exitosa"]),
        tags=[str(t) for t in tags],
        timestamp_start=row["timestamp_start"] or "",
        timestamp_end=row["timestamp_end"] or "",
        indexed_at=row["indexed_at"] or "",
    )


def upsert_call(
    conn: sqlite3.Connection,
    *,
    agent_id: str,
    ca: CallAnalysis,
) -> str:
    """Insert or replace a single call row from a CallAnalysis. Returns the catalog id."""
    now = datetime.now(timezone.utc).isoformat()
    tags_json = json.dumps(list(ca.tags), ensure_ascii=False)
    catalog_id = make_catalog_id(ca.source_file_hash or "", ca.call_id or "")
    conn.execute(
        """
        INSERT INTO calls_catalog (
            id, call_id, agent_id, source_file_hash, source_file,
            farmacia, llamante, agent, problema_corto,
            resolucion_exitosa, tags_json,
            timestamp_start, timestamp_end, indexed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            call_id = excluded.call_id,
            agent_id = excluded.agent_id,
            source_file_hash = excluded.source_file_hash,
            source_file = excluded.source_file,
            farmacia = excluded.farmacia,
            llamante = excluded.llamante,
            agent = excluded.agent,
            problema_corto = excluded.problema_corto,
            resolucion_exitosa = excluded.resolucion_exitosa,
            tags_json = excluded.tags_json,
            timestamp_start = excluded.timestamp_start,
            timestamp_end = excluded.timestamp_end,
            indexed_at = excluded.indexed_at
        """,
        (
            catalog_id,
            ca.call_id or "",
            agent_id or "",
            ca.source_file_hash or "",
            ca.source_file or "",
            ca.farmacia or "",
            ca.llamante or "",
            ca.agent or "",
            ca.problema_corto or "",
            1 if ca.resolucion_exitosa else 0,
            tags_json,
            ca.timestamp_start or "",
            ca.timestamp_end or "",
            now,
        ),
    )
    conn.commit()
    return catalog_id


def get_call(conn: sqlite3.Connection, *, id: str) -> CallCatalogRow | None:
    row = conn.execute(
        "SELECT * FROM calls_catalog WHERE id = ? LIMIT 1",
        (id,),
    ).fetchone()
    return _row_to_record(row) if row else None


def list_calls(
    conn: sqlite3.Connection,
    *,
    agent_id: str | None = None,
    tag: str | None = None,
    farmacia: str | None = None,
    resolved: bool | None = None,
    search: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> list[CallCatalogRow]:
    clauses: list[str] = []
    params: list[object] = []
    if agent_id:
        clauses.append("agent_id = ?")
        params.append(agent_id)
    if farmacia:
        clauses.append("farmacia = ?")
        params.append(farmacia)
    if resolved is not None:
        clauses.append("resolucion_exitosa = ?")
        params.append(1 if resolved else 0)
    if search:
        clauses.append(
            "(problema_corto LIKE ? OR source_file LIKE ? OR call_id LIKE ? OR id LIKE ?)"
        )
        like = f"%{search}%"
        params.extend([like, like, like, like])
    if tag:
        # Match tag as a JSON array member. Stored as `["a","b"]`, so the
        # canonical quoted form is safe against substrings of other tags.
        clauses.append("tags_json LIKE ?")
        params.append(f'%"{tag}"%')

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        f"SELECT * FROM calls_catalog {where} "
        "ORDER BY indexed_at DESC LIMIT ? OFFSET ?"
    )
    params.extend([int(limit), int(offset)])
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_record(r) for r in rows]


def distinct_farmacias(conn: sqlite3.Connection, *, agent_id: str | None = None) -> list[str]:
    if agent_id:
        rows = conn.execute(
            "SELECT DISTINCT farmacia FROM calls_catalog WHERE agent_id = ? AND farmacia != '' ORDER BY farmacia",
            (agent_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT DISTINCT farmacia FROM calls_catalog WHERE farmacia != '' ORDER BY farmacia"
        ).fetchall()
    return [r["farmacia"] for r in rows]


def distinct_tags(conn: sqlite3.Connection, *, agent_id: str | None = None) -> list[str]:
    rows = conn.execute(
        "SELECT tags_json FROM calls_catalog"
        + (" WHERE agent_id = ?" if agent_id else ""),
        (agent_id,) if agent_id else (),
    ).fetchall()
    out: set[str] = set()
    for row in rows:
        try:
            arr = json.loads(row["tags_json"] or "[]")
            if isinstance(arr, list):
                for t in arr:
                    if isinstance(t, str) and t.strip():
                        out.add(t.strip())
        except (TypeError, ValueError):
            continue
    return sorted(out)


def catalog_stats(conn: sqlite3.Connection, *, agent_id: str | None = None) -> dict:
    where = "WHERE agent_id = ?" if agent_id else ""
    params: tuple = (agent_id,) if agent_id else ()
    total_row = conn.execute(
        f"SELECT COUNT(*) AS total FROM calls_catalog {where}", params
    ).fetchone()
    resolved_row = conn.execute(
        f"SELECT COUNT(*) AS resolved FROM calls_catalog {where + (' AND' if where else 'WHERE')} resolucion_exitosa = 1",
        params,
    ).fetchone()
    last_row = conn.execute(
        f"SELECT MAX(indexed_at) AS last_at FROM calls_catalog {where}", params
    ).fetchone()
    return {
        "total": int(total_row["total"] or 0),
        "resolved": int(resolved_row["resolved"] or 0),
        "last_indexed_at": last_row["last_at"] or "",
    }


def delete_call(conn: sqlite3.Connection, *, id: str) -> int:
    cur = conn.execute("DELETE FROM calls_catalog WHERE id = ?", (id,))
    conn.commit()
    return int(cur.rowcount or 0)


def delete_by_hash(conn: sqlite3.Connection, *, source_file_hash: str) -> int:
    cur = conn.execute(
        "DELETE FROM calls_catalog WHERE source_file_hash = ?",
        (source_file_hash,),
    )
    conn.commit()
    return int(cur.rowcount or 0)


def upsert_many(
    conn: sqlite3.Connection,
    *,
    agent_id: str,
    calls: Iterable[CallAnalysis],
) -> int:
    count = 0
    for ca in calls:
        upsert_call(conn, agent_id=agent_id, ca=ca)
        count += 1
    return count
