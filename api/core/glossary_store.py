"""Domain glossary — SQLite CRUD + YAML seed.

Editable from the UI by account admins, so terminology mappings can be fixed
without a code deploy. Used by the query preprocessor to enrich short/shorthand
queries with canonical terms (and fix domain-specific typos a generic spell
checker never catches) before retrieval.

Schema
------
glossary_entries(
    id                INTEGER PK,
    tenant_id         TEXT,           -- "__global__" or real tenant id
    term              TEXT NOT NULL,  -- canonical form
    aliases_json      TEXT NOT NULL,  -- JSON list[str] — normalized on read
    expansion         TEXT NOT NULL DEFAULT '',
    description       TEXT NOT NULL DEFAULT '',
    enabled           INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    updated_by_user_id TEXT
)

Resolution
----------
Per-tenant entries override global entries with the same normalized term.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from api.core.settings import get_settings

logger = logging.getLogger(__name__)

GLOBAL_TENANT = "__global__"


def _db_path() -> Path:
    return Path(get_settings().data_dir) / "glossary.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    """Accent-insensitive, case-insensitive, whitespace-collapsed form.

    Used for matching aliases against the raw user question. Kept permissive on
    purpose — punctuation is stripped out, but internal characters like dots in
    ``h.c.p.`` are preserved as spaces to keep whole-word semantics.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFD", str(text).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    out: list[str] = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


@dataclass
class GlossaryEntry:
    id: Optional[int]
    tenant_id: str
    term: str
    aliases: list[str] = field(default_factory=list)
    expansion: str = ""
    description: str = ""
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""
    updated_by_user_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "term": self.term,
            "aliases": list(self.aliases),
            "expansion": self.expansion,
            "description": self.description,
            "enabled": bool(self.enabled),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "updated_by_user_id": self.updated_by_user_id,
        }


def init_glossary_table() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS glossary_entries (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id          TEXT NOT NULL DEFAULT '__global__',
                term               TEXT NOT NULL,
                aliases_json       TEXT NOT NULL DEFAULT '[]',
                expansion          TEXT NOT NULL DEFAULT '',
                description        TEXT NOT NULL DEFAULT '',
                enabled            INTEGER NOT NULL DEFAULT 1,
                created_at         TEXT NOT NULL DEFAULT '',
                updated_at         TEXT NOT NULL DEFAULT '',
                updated_by_user_id TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_glossary_tenant ON glossary_entries(tenant_id)"
        )
        conn.commit()
    _seed_glossary_from_yaml_if_empty()


def _seed_glossary_from_yaml_if_empty() -> None:
    try:
        with _connect() as conn:
            count = int(
                conn.execute(
                    "SELECT COUNT(1) FROM glossary_entries WHERE tenant_id = ?",
                    (GLOBAL_TENANT,),
                ).fetchone()[0]
                or 0
            )
    except Exception as exc:  # pragma: no cover
        logger.warning("Glossary count query failed: %s", exc)
        return
    if count > 0:
        return

    settings = get_settings()
    path = settings.config_path / "glossary.yaml"
    if not path.exists():
        return
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Glossary seed load failed (%s): %s", path, exc)
        return

    entries = data.get("entries") or []
    if not isinstance(entries, list) or not entries:
        return

    now = _now_iso()
    rows: list[tuple[Any, ...]] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        term = str(raw.get("term") or "").strip()
        if not term:
            continue
        aliases = [str(a).strip() for a in (raw.get("aliases") or []) if str(a).strip()]
        expansion = str(raw.get("expansion") or "").strip()
        description = str(raw.get("description") or "").strip()
        rows.append(
            (
                GLOBAL_TENANT,
                term,
                json.dumps(aliases, ensure_ascii=False),
                expansion,
                description,
                1,
                now,
                now,
                None,
            )
        )
    if not rows:
        return
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO glossary_entries
                (tenant_id, term, aliases_json, expansion, description, enabled,
                 created_at, updated_at, updated_by_user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    logger.info("Seeded %d global glossary entries from %s", len(rows), path)


def _row_to_entry(row: sqlite3.Row) -> GlossaryEntry:
    try:
        aliases = json.loads(row["aliases_json"] or "[]")
        if not isinstance(aliases, list):
            aliases = []
    except Exception:
        aliases = []
    return GlossaryEntry(
        id=int(row["id"]),
        tenant_id=str(row["tenant_id"] or GLOBAL_TENANT),
        term=str(row["term"] or ""),
        aliases=[str(a) for a in aliases if str(a).strip()],
        expansion=str(row["expansion"] or ""),
        description=str(row["description"] or ""),
        enabled=bool(row["enabled"]),
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
        updated_by_user_id=row["updated_by_user_id"],
    )


def list_entries(tenant_id: Optional[str] = None, *, include_global: bool = True) -> list[GlossaryEntry]:
    """Return entries for a tenant + global (deduped by normalized term, tenant wins)."""
    init_glossary_table()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM glossary_entries ORDER BY tenant_id, term"
        ).fetchall()
    entries = [_row_to_entry(r) for r in rows]

    if tenant_id is None:
        return entries

    picked: dict[str, GlossaryEntry] = {}
    # Global first so tenant-specific overrides win.
    if include_global:
        for e in entries:
            if e.tenant_id == GLOBAL_TENANT and e.enabled:
                picked[normalize_text(e.term)] = e
    for e in entries:
        if e.tenant_id == tenant_id and e.enabled:
            picked[normalize_text(e.term)] = e
    return list(picked.values())


def get_entry(entry_id: int) -> Optional[GlossaryEntry]:
    init_glossary_table()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM glossary_entries WHERE id = ?",
            (int(entry_id),),
        ).fetchone()
    return _row_to_entry(row) if row else None


def create_entry(
    *,
    tenant_id: str,
    term: str,
    aliases: Iterable[str] = (),
    expansion: str = "",
    description: str = "",
    enabled: bool = True,
    updated_by_user_id: Optional[str] = None,
) -> GlossaryEntry:
    init_glossary_table()
    tid = (tenant_id or "").strip() or GLOBAL_TENANT
    term_clean = (term or "").strip()
    if not term_clean:
        raise ValueError("term is required")
    aliases_clean = [str(a).strip() for a in aliases if str(a).strip()]
    now = _now_iso()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO glossary_entries
                (tenant_id, term, aliases_json, expansion, description, enabled,
                 created_at, updated_at, updated_by_user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tid,
                term_clean,
                json.dumps(aliases_clean, ensure_ascii=False),
                (expansion or "").strip(),
                (description or "").strip(),
                1 if enabled else 0,
                now,
                now,
                updated_by_user_id,
            ),
        )
        new_id = int(cur.lastrowid)
        conn.commit()
    entry = get_entry(new_id)
    if entry is None:
        raise RuntimeError("Failed to create glossary entry")
    return entry


def update_entry(
    entry_id: int,
    patch: dict[str, Any],
    *,
    updated_by_user_id: Optional[str] = None,
) -> GlossaryEntry:
    init_glossary_table()
    existing = get_entry(int(entry_id))
    if existing is None:
        raise KeyError(f"Glossary entry not found: {entry_id}")

    columns: list[str] = []
    values: list[Any] = []
    allowed = {"term", "aliases", "expansion", "description", "enabled", "tenant_id"}
    for k, v in patch.items():
        if k not in allowed:
            continue
        if k == "aliases":
            values.append(json.dumps([str(a).strip() for a in (v or []) if str(a).strip()], ensure_ascii=False))
            columns.append("aliases_json = ?")
        elif k == "enabled":
            values.append(1 if bool(v) else 0)
            columns.append("enabled = ?")
        elif k == "tenant_id":
            values.append(str(v or "").strip() or GLOBAL_TENANT)
            columns.append("tenant_id = ?")
        else:
            values.append((str(v) if v is not None else "").strip())
            columns.append(f"{k} = ?")

    if not columns:
        return existing

    now = _now_iso()
    columns.append("updated_at = ?")
    values.append(now)
    columns.append("updated_by_user_id = ?")
    values.append(updated_by_user_id)
    values.append(int(entry_id))
    with _connect() as conn:
        conn.execute(
            f"UPDATE glossary_entries SET {', '.join(columns)} WHERE id = ?",
            values,
        )
        conn.commit()
    entry = get_entry(int(entry_id))
    if entry is None:
        raise RuntimeError("Glossary entry disappeared after update")
    return entry


def delete_entry(entry_id: int) -> bool:
    init_glossary_table()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM glossary_entries WHERE id = ?",
            (int(entry_id),),
        )
        conn.commit()
    return cur.rowcount > 0


def glossary_fingerprint(tenant_id: Optional[str] = None) -> str:
    """Short hash of the effective glossary for cache invalidation.

    Any create/update/delete (including ``enabled`` toggle) changes the hash,
    so cached answers keyed on this fingerprint are invalidated automatically.
    """
    entries = list_entries(tenant_id=tenant_id, include_global=True)
    payload = sorted(
        (
            e.term.strip(),
            tuple(sorted(a.strip() for a in e.aliases)),
            e.expansion.strip(),
            int(bool(e.enabled)),
            e.updated_at,
        )
        for e in entries
    )
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def resolve_effective_entries(tenant_id: Optional[str]) -> list[GlossaryEntry]:
    """Enabled entries for runtime preprocessing (tenant overrides global)."""
    return [e for e in list_entries(tenant_id=tenant_id, include_global=True) if e.enabled]
