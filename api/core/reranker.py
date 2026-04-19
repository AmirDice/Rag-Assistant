"""Reranker abstraction — Voyage / Cohere (cloud) / BGE (self-hosted) / None."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from api.core.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    index: int
    score: float


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, documents: list[str], top_n: int = 5
    ) -> list[RerankResult]:
        """Return reranked indices + scores, sorted by relevance."""


class VoyageReranker(Reranker):
    """Voyage rerank API (rerank-2.5, etc.)."""

    def __init__(self) -> None:
        settings = get_settings()
        cfg = settings.models_config()["reranker"]["voyage"]
        self.model = cfg["model"]
        self.api_base = (cfg.get("api_base") or "https://api.voyageai.com/v1").rstrip("/")
        self.api_key = settings.voyage_api_key

    async def rerank(
        self, query: str, documents: list[str], top_n: int = 5
    ) -> list[RerankResult]:
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY is required when reranker.active is voyage")
        if not documents:
            return []
        top_k = min(max(1, top_n), len(documents))
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.api_base}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_k": top_k,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                RerankResult(index=r["index"], score=float(r["relevance_score"]))
                for r in data["data"]
            ]


class CohereReranker(Reranker):
    def __init__(self) -> None:
        settings = get_settings()
        cfg = settings.models_config()["reranker"]["cohere"]
        self.model = cfg["model"]
        self.api_key = settings.cohere_api_key

    async def rerank(
        self, query: str, documents: list[str], top_n: int = 5
    ) -> list[RerankResult]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.cohere.com/v2/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                RerankResult(index=r["index"], score=r["relevance_score"])
                for r in data["results"]
            ]


class BGEReranker(Reranker):
    """BGE reranker via OpenAI-compatible API (e.g., vLLM, TEI)."""

    def __init__(self) -> None:
        settings = get_settings()
        cfg = settings.models_config()["reranker"]["bge"]
        self.model = cfg["model"]
        self.api_base = cfg["api_base"]

    async def rerank(
        self, query: str, documents: list[str], top_n: int = 5
    ) -> list[RerankResult]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.api_base}/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                RerankResult(index=r["index"], score=r["score"])
                for r in data["results"]
            ]


class NoReranker(Reranker):
    """Passthrough — preserves full vector order so downstream diversity sees all candidates."""

    async def rerank(
        self, query: str, documents: list[str], top_n: int = 5
    ) -> list[RerankResult]:
        _ = top_n
        return [
            RerankResult(index=i, score=1.0 - i * 0.01)
            for i in range(len(documents))
        ]


_reranker_instance: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker_instance
    if _reranker_instance is None:
        settings = get_settings()
        cfg = settings.models_config()
        active = cfg["reranker"]["active"]
        if active == "voyage":
            _reranker_instance = VoyageReranker()
        elif active == "cohere":
            _reranker_instance = CohereReranker()
        elif active == "bge":
            _reranker_instance = BGEReranker()
        elif active == "none":
            _reranker_instance = NoReranker()
        else:
            raise ValueError(f"Unknown reranker: {active}")
        logger.info("Reranker initialized: %s", active)
    return _reranker_instance
