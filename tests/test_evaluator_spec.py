from api.benchmark.evaluator import (
    DOC_TYPE_MIN_HR3,
    L3_IMAGE_MIN_HR3,
    _chunk_matches_pair,
    _norm_section,
)
from api.core.models import AnswerChunk


def test_wp15_doc_type_thresholds_present():
    assert DOC_TYPE_MIN_HR3["faq_document"] == 0.90
    assert DOC_TYPE_MIN_HR3["structured_manual"] == 0.85
    assert L3_IMAGE_MIN_HR3 == 0.65


def test_norm_section_collapses_case_and_whitespace():
    assert _norm_section("  Foo  Bar ") == "foo bar"
    assert _norm_section(None) == ""


def test_chunk_matches_by_qdrant_id():
    pair = {"chunk_id": "abc-123", "source_doc": "x.pdf", "source_section": "S1"}
    c = AnswerChunk(
        text="t", score=1.0, source_doc="other.pdf", chunk_id="abc-123",
    )
    assert _chunk_matches_pair(c, pair) is True


def test_chunk_matches_doc_and_normalized_section():
    pair = {"chunk_id": "", "source_doc": "Manual.pdf", "source_section": "  Intro  "}
    c = AnswerChunk(
        text="t",
        score=1.0,
        source_doc="Manual.pdf",
        source_section="intro",
        chunk_id="",
    )
    assert _chunk_matches_pair(c, pair) is True
