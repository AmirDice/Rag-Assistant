"""Query intent planner — pure parsing + decision logic (no LLM calls)."""

from __future__ import annotations

from api.core.query_preprocessor import PreprocessedQuery
from api.core.query_intent_planner import (
    QueryRefinementDecision,
    _decision_from_parsed,
    fallback_decision,
    parse_planner_json,
)


def _pre() -> PreprocessedQuery:
    return PreprocessedQuery(
        original="como cierro la caja",
        corrected="como cierro la caja",
        retrieval_query="como cierro la caja cierre diario",
        matched_terms=[],
        expansions=[],
    )


def test_parse_planner_json_plain():
    parsed = parse_planner_json('{"action":"search","final_query":"cerrar caja"}')
    assert parsed["action"] == "search"
    assert parsed["final_query"] == "cerrar caja"


def test_parse_planner_json_strips_fences_and_thoughts():
    raw = '<think>hmm</think>\n```json\n{"action":"search","confidence":0.9}\n```'
    parsed = parse_planner_json(raw)
    assert parsed["action"] == "search"
    assert parsed["confidence"] == 0.9


def test_parse_planner_json_garbage_returns_none():
    assert parse_planner_json("not json at all") is None


def test_decision_search_passthrough():
    parsed = {"action": "search", "final_query": "cerrar caja diaria", "confidence": 0.9}
    d = _decision_from_parsed(parsed, preprocessed=_pre(), model_id="m", parent_query_id=None)
    assert d.action == "search"
    assert d.needs_clarification is False
    assert d.final_query == "cerrar caja diaria"
    assert d.used is True


def test_clarification_requires_high_confidence():
    # Low-confidence clarification is downgraded to search.
    parsed = {
        "action": "ask_clarification",
        "clarification_question": "¿Te refieres a la caja diaria o mensual?",
        "confidence": 0.4,
    }
    d = _decision_from_parsed(parsed, preprocessed=_pre(), model_id="m", parent_query_id=None)
    assert d.action == "search"
    assert d.needs_clarification is False


def test_high_confidence_clarification_is_kept():
    parsed = {
        "action": "ask_clarification",
        "clarification_question": "¿Caja diaria o mensual?",
        "confidence": 0.95,
    }
    d = _decision_from_parsed(parsed, preprocessed=_pre(), model_id="m", parent_query_id=None)
    assert d.action == "ask_clarification"
    assert d.needs_clarification is True


def test_clarification_without_question_downgrades():
    parsed = {"action": "ask_clarification", "clarification_question": "", "confidence": 0.99}
    d = _decision_from_parsed(parsed, preprocessed=_pre(), model_id="m", parent_query_id=None)
    assert d.action == "search"


def test_fallback_decision_uses_preprocessed():
    d = fallback_decision(_pre(), reason="no_model_id")
    assert isinstance(d, QueryRefinementDecision)
    assert d.used is False
    assert d.fallback_reason == "no_model_id"
    assert d.retrieval_query == "como cierro la caja cierre diario"
    assert d.needs_clarification is False
