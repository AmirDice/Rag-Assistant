"""LLM doc-type classification: JSON parsing + classifier mapping/heuristics.

No network: the 'too short' guard returns before any API call, and the
classify_document test forces the heuristic fallback via monkeypatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.pipeline import classifier
from api.pipeline.classifier import _heuristic_type, _normalize_type, classify_document
from api.pipeline.doc_type_llm import _parse_json_response, classify_document_text


def test_parse_json_plain():
    out = _parse_json_response('{"type_id":"faq_document","confidence":0.9}')
    assert out["type_id"] == "faq_document"


def test_parse_json_with_fences():
    out = _parse_json_response('```json\n{"type_id":"operational_guide","confidence":0.8}\n```')
    assert out["type_id"] == "operational_guide"


def test_parse_json_embedded_object():
    out = _parse_json_response('Sure! {"type_id":"technical_spec"} done')
    assert out["type_id"] == "technical_spec"


def test_parse_json_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse_json_response("no json here")


def test_classify_too_short_returns_none():
    # Below _MIN_TEXT_CHARS → returns before any API call regardless of keys.
    assert classify_document_text(filename="x.pdf", text="too short") is None


def test_normalize_type_aliases():
    # Aliases not already in the canonical set get remapped.
    assert _normalize_type("faq") == "faq_document"
    assert _normalize_type("user_manual") == "structured_manual"
    assert _normalize_type("installation_manual") == "module_manual"
    assert _normalize_type("release_notes") == "changelog_pure"
    # Canonical ids pass through unchanged; unknowns default to structured_manual.
    assert _normalize_type("structured_manual") == "structured_manual"
    assert _normalize_type("totally_unknown") == "structured_manual"


def test_heuristic_type_by_filename():
    assert _heuristic_type("faq_clientes", "") == "faq_document"
    assert _heuristic_type("changelog_v5", "") == "changelog_pure"


def test_classify_document_heuristic_fallback(monkeypatch):
    # Force the heuristic path so the test never calls an LLM.
    monkeypatch.setattr(classifier, "_llm_classify", lambda *a, **k: None)
    meta = classify_document(
        Path("FAQ_clientes.pdf"),
        "¿Cómo hago una consulta? Respuesta: así. ¿Y la otra? Respuesta: asá.",
    )
    assert meta.doc_type == "faq_document"
    assert meta.doc_id == "FAQ_clientes"
    assert meta.format == "pdf"
    assert meta.lang in {"es", "ca"}
