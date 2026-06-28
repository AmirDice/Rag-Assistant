"""Generator model-fallback chain: primary errors → fallback model is used."""

from __future__ import annotations

import asyncio

import api.core.backends as backends_mod
import api.core.cost_meter as cost_meter
import api.core.generator as gen
from api.core.backends.base import GenerationResult
from api.core.models import AnswerChunk


class _FakeBackend:
    def __init__(self, model: str, fail: bool):
        self._model = model
        self._fail = fail

    async def generate(self, prompt, chunks, images):
        if self._fail:
            raise RuntimeError("503 UNAVAILABLE")
        return GenerationResult(
            text=f"answer from {self._model}",
            model=self._model,
            provider="google",
            units_in=10.0,
            units_out=5.0,
            units_thoughts=0.0,
            units_basis="real",
        )


def _chunk():
    return AnswerChunk(text="Acme Cloud supports SSO on the Business plan.", score=1.0, source_doc="d.pdf")


def test_falls_back_when_primary_model_errors(monkeypatch):
    # Primary model fails, fallback model succeeds.
    monkeypatch.setattr(gen, "_fallback_model_id", lambda: "gemini-2.5-flash")
    monkeypatch.setattr(cost_meter, "record_cost_event", lambda event: None)

    def fake_build(settings, model_cfg):
        return _FakeBackend(model_cfg.model, fail=(model_cfg.model == "gemini-2.5-flash-lite"))

    monkeypatch.setattr(backends_mod, "build_generation_backend", fake_build)

    result = asyncio.run(
        gen.generate_answer("Does it support SSO?", [_chunk()], generation_model="gemini-2.5-flash-lite")
    )
    assert result.text == "answer from gemini-2.5-flash"
    assert result.model == "gemini-2.5-flash"
    assert result.cost_breakdown.get("fell_back") is True


def test_no_fallback_when_primary_succeeds(monkeypatch):
    monkeypatch.setattr(gen, "_fallback_model_id", lambda: "gemini-2.5-flash")
    monkeypatch.setattr(cost_meter, "record_cost_event", lambda event: None)
    monkeypatch.setattr(backends_mod, "build_generation_backend", lambda s, c: _FakeBackend(c.model, fail=False))

    result = asyncio.run(
        gen.generate_answer("Does it support SSO?", [_chunk()], generation_model="gemini-2.5-flash-lite")
    )
    assert result.model == "gemini-2.5-flash-lite"
    assert result.cost_breakdown.get("fell_back") is False
