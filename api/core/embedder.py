"""Embedding abstraction — Voyage (cloud ceiling) / Qwen3 (local baseline).

Both expose the same interface so the retriever doesn't care which is active.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import httpx
import tiktoken

from api.core.settings import get_settings

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")
VOYAGE_MAX_TOKENS = 16000
_MAX_RETRIES = 3
_RETRY_BACKOFF = [2, 5, 15]


def _truncate_to_tokens(text: str, max_tokens: int = VOYAGE_MAX_TOKENS) -> str:
    """Truncate text to fit within the token budget."""
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    logger.warning("Truncating text from %d to %d tokens", len(tokens), max_tokens)
    return _enc.decode(tokens[:max_tokens])


class Embedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts."""

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Return a single embedding for a search query."""


class VoyageEmbedder(Embedder):
    def __init__(self) -> None:
        settings = get_settings()
        cfg = settings.models_config()["embedding"]["voyage"]
        self.model = cfg["model"]
        self.batch_size = cfg.get("batch_size", 128)
        self.api_key = settings.voyage_api_key
        self.api_base = cfg.get("api_base", "https://api.voyageai.com/v1")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        safe_texts = [_truncate_to_tokens(t) for t in texts]
        all_embeddings: list[list[float]] = []
        for i in range(0, len(safe_texts), self.batch_size):
            batch = safe_texts[i:i + self.batch_size]
            try:
                resp = await self._call(batch, input_type="document")
                all_embeddings.extend(resp)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400 and len(batch) > 1:
                    logger.warning(
                        "Batch of %d failed with 400, falling back to one-by-one",
                        len(batch),
                    )
                    for j, single in enumerate(batch):
                        try:
                            r = await self._call([single], input_type="document")
                            all_embeddings.extend(r)
                        except httpx.HTTPStatusError:
                            logger.error(
                                "Skipping chunk %d (offset %d) — Voyage 400",
                                i + j, i + j,
                            )
                            all_embeddings.append([0.0] * 1024)
                else:
                    raise
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        result = await self._call([_truncate_to_tokens(query)], input_type="query")
        return result[0]

    async def _call(self, texts: list[str], input_type: str) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{self.api_base}/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "input": texts,
                            "input_type": input_type,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return [item["embedding"] for item in data["data"]]
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
                last_err = e
                is_retryable = isinstance(e, (httpx.ConnectError, httpx.ReadTimeout))
                if isinstance(e, httpx.HTTPStatusError):
                    is_retryable = e.response.status_code in (429, 500, 502, 503, 504)
                if not is_retryable:
                    raise
                wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                logger.warning(
                    "Voyage API error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, e, wait,
                )
                await asyncio.sleep(wait)
        raise last_err  # type: ignore[misc]


class OpenAICompatibleEmbedder(Embedder):
    """Works with any OpenAI-compatible embedding API (Qwen3 via vLLM, SiliconFlow, etc.)."""

    def __init__(self) -> None:
        settings = get_settings()
        cfg = settings.models_config()["embedding"]["qwen3"]
        self.model = cfg["model"]
        self.batch_size = cfg.get("batch_size", 32)
        self.api_base = cfg["api_base"]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = await self._call(batch)
            all_embeddings.extend(resp)
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        result = await self._call([query])
        return result[0]

    async def _call(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.api_base}/embeddings",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]


_embedder_instance: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        settings = get_settings()
        cfg = settings.models_config()
        active = cfg["embedding"]["active"]
        if active == "voyage":
            _embedder_instance = VoyageEmbedder()
        elif active == "qwen3":
            _embedder_instance = OpenAICompatibleEmbedder()
        else:
            raise ValueError(f"Unknown embedding provider: {active}")
        logger.info("Embedder initialized: %s", active)
    return _embedder_instance
