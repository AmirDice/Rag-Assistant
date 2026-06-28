"""Glossary store CRUD (isolated SQLite DB) backing the /admin/glossary route."""

from __future__ import annotations

import pytest

from api.core import glossary_store
from api.core.glossary_store import (
    GLOBAL_TENANT,
    create_entry,
    delete_entry,
    get_entry,
    glossary_fingerprint,
    list_entries,
    update_entry,
)


@pytest.fixture()
def isolated_glossary(monkeypatch, tmp_path):
    db = tmp_path / "glossary.db"
    monkeypatch.setattr(glossary_store, "_db_path", lambda: db)
    return db


def test_create_and_get(isolated_glossary):
    entry = create_entry(tenant_id="acme", term="cartera", aliases=["carteras"], expansion="cartera compras")
    assert entry.id is not None
    fetched = get_entry(entry.id)
    assert fetched is not None
    assert fetched.term == "cartera"
    assert "carteras" in fetched.aliases
    assert fetched.tenant_id == "acme"


def test_blank_tenant_becomes_global(isolated_glossary):
    entry = create_entry(tenant_id="", term="HCP")
    assert entry.tenant_id == GLOBAL_TENANT


def test_update_entry(isolated_glossary):
    entry = create_entry(tenant_id="acme", term="encargo")
    updated = update_entry(entry.id, {"expansion": "encargo cliente", "enabled": False})
    assert updated.expansion == "encargo cliente"
    assert updated.enabled is False


def test_update_missing_raises(isolated_glossary):
    with pytest.raises(KeyError):
        update_entry(999999, {"term": "x"})


def test_delete_entry(isolated_glossary):
    entry = create_entry(tenant_id="acme", term="temporal")
    assert delete_entry(entry.id) is True
    assert get_entry(entry.id) is None
    assert delete_entry(entry.id) is False


def test_tenant_entry_listed_for_tenant(isolated_glossary):
    create_entry(tenant_id="acme", term="solo-acme")
    terms = {e.term for e in list_entries(tenant_id="acme")}
    assert "solo-acme" in terms
    # Global YAML seed is also visible to the tenant.
    assert "SSO" in terms


def test_fingerprint_changes_on_create(isolated_glossary):
    before = glossary_fingerprint("acme")
    create_entry(tenant_id="acme", term="nuevo-termino")
    after = glossary_fingerprint("acme")
    assert before != after
