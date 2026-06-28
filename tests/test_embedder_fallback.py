"""Embedder fallback wrapper — primary → fallback on failure (same dimension)."""

from __future__ import annotations

import asyncio

from api.core.embedder import Embedder, FallbackEmbedder


class _Boom(Embedder):
    async def embed(self, texts):
        raise RuntimeError("provider down")

    async def embed_query(self, query):
        raise RuntimeError("provider down")


class _Ok(Embedder):
    def __init__(self, value: float):
        self.value = value

    async def embed(self, texts):
        return [[self.value] * 4 for _ in texts]

    async def embed_query(self, query):
        return [self.value] * 4


def test_fallback_used_when_primary_fails():
    fe = FallbackEmbedder(_Boom(), _Ok(0.1), primary_name="p", fallback_name="f")
    assert asyncio.run(fe.embed_query("x")) == [0.1] * 4
    assert asyncio.run(fe.embed(["a", "b"])) == [[0.1] * 4, [0.1] * 4]


def test_primary_used_when_healthy():
    fe = FallbackEmbedder(_Ok(0.9), _Boom(), primary_name="p", fallback_name="f")
    assert asyncio.run(fe.embed_query("x")) == [0.9] * 4
    assert asyncio.run(fe.embed(["a"])) == [[0.9] * 4]
