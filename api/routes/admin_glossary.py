"""Admin CRUD for the query-preprocessing glossary (api/core/glossary_store.py).

Lets an operator fix terminology mappings (canonical term + aliases/typos +
recall-widening expansion) from the UI without a code deploy. Mounted behind the
admin-token guard in api/main.py.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.core import glossary_store

router = APIRouter()


class GlossaryEntryCreate(BaseModel):
    term: str
    aliases: list[str] = []
    expansion: str = ""
    description: str = ""
    enabled: bool = True
    tenant_id: str = ""   # "" → global entry


class GlossaryEntryUpdate(BaseModel):
    term: str | None = None
    aliases: list[str] | None = None
    expansion: str | None = None
    description: str | None = None
    enabled: bool | None = None
    tenant_id: str | None = None


@router.get("/admin/glossary")
def list_glossary(tenant_id: str | None = None) -> dict:
    """List glossary entries.

    With ``tenant_id`` → effective set for that tenant (global + tenant overrides,
    deduped by canonical term). Without it → all raw entries.
    """
    entries = glossary_store.list_entries(tenant_id=tenant_id)
    return {"entries": [e.to_dict() for e in entries], "count": len(entries)}


@router.post("/admin/glossary")
def create_glossary_entry(body: GlossaryEntryCreate) -> dict:
    term = body.term.strip()
    if not term:
        raise HTTPException(status_code=422, detail="term is required")
    entry = glossary_store.create_entry(
        tenant_id=body.tenant_id,
        term=term,
        aliases=body.aliases,
        expansion=body.expansion,
        description=body.description,
        enabled=body.enabled,
    )
    return entry.to_dict()


@router.patch("/admin/glossary/{entry_id}")
def update_glossary_entry(entry_id: int, body: GlossaryEntryUpdate) -> dict:
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    if not patch:
        raise HTTPException(status_code=422, detail="no fields to update")
    try:
        entry = glossary_store.update_entry(entry_id, patch)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Glossary entry not found: {entry_id}")
    return entry.to_dict()


@router.delete("/admin/glossary/{entry_id}")
def delete_glossary_entry(entry_id: int) -> dict:
    deleted = glossary_store.delete_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Glossary entry not found: {entry_id}")
    return {"status": "deleted", "id": entry_id}
