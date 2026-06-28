"""Query preprocessing + glossary enrichment (deterministic path, no LLM)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from api.core import glossary_store
from api.core.glossary_store import normalize_text, resolve_effective_entries
from api.core import query_preprocessor
from api.core.query_preprocessor import preprocess_query


@pytest.fixture()
def isolated_glossary(monkeypatch, tmp_path):
    """Point the glossary DB at a temp file so the YAML seed loads fresh."""
    db = tmp_path / "glossary.db"
    monkeypatch.setattr(glossary_store, "_db_path", lambda: db)
    return db


def test_normalize_text_strips_accents_and_punct():
    assert normalize_text("¿Cómo  HAGO un Encargo?") == "como hago un encargo"
    assert normalize_text("h.c.p.") == "h c p"


def test_glossary_seeds_from_yaml(isolated_glossary):
    entries = resolve_effective_entries("demo")
    terms = {e.term for e in entries}
    assert "SSO" in terms
    assert "2FA" in terms


def test_preprocess_enriches_with_canonical_term(isolated_glossary):
    pre = asyncio.run(preprocess_query("how do I set up single sign on", tenant_id="demo"))
    # "single sign on" is an alias of canonical "SSO".
    assert "SSO" in pre.matched_terms
    assert "SSO" in pre.retrieval_query
    assert pre.ai_used is False
    assert pre.original == "how do I set up single sign on"


def test_preprocess_acronym_expansion(isolated_glossary):
    pre = asyncio.run(preprocess_query("error with 2fa", tenant_id="demo"))
    assert "2FA" in pre.matched_terms
    # expansion tokens widen lexical recall
    assert "two" in normalize_text(pre.retrieval_query)


def test_preprocess_no_match_returns_original(isolated_glossary):
    pre = asyncio.run(preprocess_query("xyzzy plugh frobnicate", tenant_id="demo"))
    assert pre.matched_terms == []
    assert pre.retrieval_query == "xyzzy plugh frobnicate"


def test_preprocess_disabled_passthrough(isolated_glossary, monkeypatch):
    monkeypatch.setattr(query_preprocessor, "_preprocessor_cfg", lambda: {"enabled": False})
    pre = asyncio.run(preprocess_query("how do I set up single sign on", tenant_id="demo"))
    assert pre.retrieval_query == "how do I set up single sign on"
    assert pre.matched_terms == []


def test_short_query_passthrough(isolated_glossary):
    pre = asyncio.run(preprocess_query("a", tenant_id="demo"))
    assert pre.retrieval_query == "a"
