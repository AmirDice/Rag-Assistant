"""Tests for benchmark retrieval analysis (gold vs retrieved breakdown)."""

from __future__ import annotations

import pytest

from api.benchmark.evaluator import _gold_block, run_retrieval_analysis
from api.core.models import AnswerChunk


def test_gold_block_truncates_answer():
    pair = {
        "chunk_id": "id-1",
        "source_doc": "a.pdf",
        "source_section": "S",
        "answer": "x" * 900,
        "chunk_text_preview": "preview",
        "doc_type": "t",
        "difficulty": "L1",
    }
    g = _gold_block(pair)
    assert g["chunk_id"] == "id-1"
    assert len(g["expected_answer"]) < 900


@pytest.mark.asyncio
async def test_run_retrieval_analysis_miss_and_hit(monkeypatch):
    sample = [
        {
            "question": "q1",
            "chunk_id": "gold-1",
            "source_doc": "right.pdf",
            "source_section": "Intro",
            "answer": "a1",
            "chunk_text_preview": "body",
            "doc_type": "operational_guide",
            "difficulty": "L1",
        },
        {
            "question": "q2",
            "chunk_id": "gold-2",
            "source_doc": "b.pdf",
            "source_section": None,
            "answer": "a2",
            "chunk_text_preview": "b2",
            "doc_type": "x",
            "difficulty": "L1",
        },
    ]
    monkeypatch.setattr(
        "api.benchmark.evaluator.get_benchmark_pairs_for_eval",
        lambda validated_only: sample,
    )

    calls = {"n": 0}

    async def fake_retrieve(question, *, tenant_id, top_k=5):
        calls["n"] += 1
        if question == "q1":
            return [
                AnswerChunk(
                    text="other",
                    score=0.9,
                    source_doc="wrong.pdf",
                    chunk_id="w1",
                ),
            ]
        return [
            AnswerChunk(
                text="ok",
                score=0.9,
                source_doc="b.pdf",
                chunk_id="gold-2",
            ),
        ]

    monkeypatch.setattr(
        "api.benchmark.evaluator._retrieve_for_eval",
        fake_retrieve,
    )

    out = await run_retrieval_analysis(
        tenant_id="demo",
        eval_top_k=5,
        validated_only=False,
        limit=10,
        offset=0,
        misses_only=False,
    )
    assert out["summary"]["n_queries"] == 2
    assert out["summary"]["hits_at_k"] == 1
    assert out["summary"]["hr_at_k"] == 0.5
    assert len(out["items"]) == 2
    assert out["items"][0]["hit"] is False
    assert out["items"][0]["retrieved"][0]["matches_gold"] is False
    assert out["items"][1]["hit"] is True
    assert out["items"][1]["hit_rank"] == 1
    assert out.get("has_more") is False
    assert out.get("next_offset") == 2

    out_m = await run_retrieval_analysis(
        tenant_id="demo",
        eval_top_k=5,
        validated_only=False,
        limit=10,
        offset=0,
        misses_only=True,
    )
    assert len(out_m["items"]) == 1
    assert out_m["items"][0]["hit"] is False
